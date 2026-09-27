"""
hedging_env.py

A production-grade Gymnasium environment for simulating the dynamic hedging
of a short European call option position under proportional transaction
costs.

This module is Phase 1 infrastructure for a Deep Deterministic Policy
Gradient (DDPG) agent whose objective is to learn a hedging policy that
outperforms a classical Black-Scholes delta-hedging benchmark net of
transaction costs.

Design notes
------------
* The underlying follows discrete Geometric Brownian Motion (GBM) under the
  real-world (physical) measure, parameterised by a drift ``mu`` and
  volatility ``sigma``. Option values used for portfolio marking are priced
  under the risk-neutral Black-Scholes model with rate ``r``.
* The environment maintains a *self-financing* replicating portfolio:
    - a stock position of ``h_t`` shares (the agent's action / hedge ratio),
    - a cash account that accrues interest at the risk-free rate and
      absorbs the cost of rebalancing the stock position,
    - a short liability equal to the Black-Scholes value of the option.
  The portfolio's mark-to-market value is
      Pi_t = h_t * S_t + B_t - C_t
  where ``B_t`` is the cash/bond account and ``C_t`` is the BS call price.
* The reward penalises the *change* in portfolio value (hedging error)
  plus the transaction costs paid to rebalance, following
      R_t = -(Pi_t - Pi_{t-1})**2 - kappa * costs_t
  which is a mean-variance-style objective: minimise the squared P&L swing
  of the hedged book while penalising the cost of trading.

Author: Quant Dev
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Black-Scholes analytics
# ---------------------------------------------------------------------------

def bs_call_price(s: float, k: float, tau: float, r: float, sigma: float) -> float:
    """Analytical Black-Scholes price of a European call option.

    Parameters
    ----------
    s : float
        Current spot price of the underlying.
    k : float
        Strike price.
    tau : float
        Time to maturity in years. Must be >= 0.
    r : float
        Continuously compounded, annualised risk-free rate.
    sigma : float
        Annualised volatility of the underlying.

    Returns
    -------
    float
        The Black-Scholes call price. At/after expiry (tau <= 0) or with
        zero volatility, returns the intrinsic value ``max(s - k, 0)``.
    """
    if tau <= 0.0 or sigma <= 0.0:
        return max(s - k, 0.0)

    sqrt_tau = math.sqrt(tau)
    d1 = (math.log(s / k) + (r + 0.5 * sigma * sigma) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    return s * norm.cdf(d1) - k * math.exp(-r * tau) * norm.cdf(d2)


def bs_call_delta(s: float, k: float, tau: float, r: float, sigma: float) -> float:
    """Analytical Black-Scholes delta of a European call option.

    Parameters
    ----------
    s, k, tau, r, sigma : see :func:`bs_call_price`.

    Returns
    -------
    float
        d(Call)/d(S), the hedge ratio implied by Black-Scholes. At expiry
        this collapses to the indicator ``1.0`` if in the money else
        ``0.0`` (the sub-gradient of the payoff kink is not returned).
    """
    if tau <= 0.0 or sigma <= 0.0:
        return 1.0 if s > k else 0.0

    sqrt_tau = math.sqrt(tau)
    d1 = (math.log(s / k) + (r + 0.5 * sigma * sigma) * tau) / (sigma * sqrt_tau)
    return float(norm.cdf(d1))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HedgingEnvConfig:
    """Immutable configuration for :class:`HedgingEnv`.

    Attributes
    ----------
    s0 : float
        Initial underlying price.
    strike : float
        Strike price of the European call being hedged (short position).
    mu : float
        Real-world (physical measure) annualised drift of the underlying,
        used to *simulate* the GBM path.
    sigma : float
        Annualised volatility, used both for simulating the path and for
        pricing/hedging under Black-Scholes (assumed known/constant).
    r : float
        Continuously compounded annualised risk-free rate, used for option
        pricing and cash-account accrual.
    maturity : float
        Time to maturity in years (e.g. ``30 / 252`` for a 30 trading-day
        option).
    dt : float
        Length of a single simulation step in years. Defaults to one
        trading day, ``1 / 252``.
    transaction_cost : float
        Proportional transaction cost rate ``c`` applied as
        ``c * |delta_h| * S_t`` whenever the hedge is rebalanced.
    cost_penalty : float
        Weight ``kappa`` applied to the transaction cost term inside the
        reward function.
    initial_hedge : float
        Hedge ratio held entering the first step (typically ``0.0``).
    n_contracts : float
        Number of option contracts (multiplier) the agent is short. Kept
        at 1.0 by default; scales P&L and cost terms linearly.
    """

    s0: float = 100.0
    strike: float = 100.0
    mu: float = 0.05
    sigma: float = 0.20
    r: float = 0.02
    maturity: float = 30.0 / 252.0
    dt: float = 1.0 / 252.0
    transaction_cost: float = 0.001
    cost_penalty: float = 1.0
    initial_hedge: float = 0.0
    n_contracts: float = 1.0

    def __post_init__(self) -> None:
        if self.s0 <= 0.0:
            raise ValueError("s0 must be positive.")
        if self.strike <= 0.0:
            raise ValueError("strike must be positive.")
        if self.sigma <= 0.0:
            raise ValueError("sigma must be positive.")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self.maturity <= 0.0:
            raise ValueError("maturity must be positive.")
        if self.transaction_cost < 0.0:
            raise ValueError("transaction_cost must be non-negative.")
        if not (0.0 <= self.initial_hedge <= 1.0):
            raise ValueError("initial_hedge must lie in [0, 1].")

    @property
    def n_steps(self) -> int:
        """Number of discrete simulation steps to maturity (rounded)."""
        return max(1, round(self.maturity / self.dt))


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class HedgingEnv(gym.Env):
    """Gymnasium environment for hedging a short European call under costs.

    Observation
    -----------
    ``Box(low=[0, 0, 0], high=[inf, maturity, 1], shape=(3,), dtype=float32)``
        ``[0]`` Moneyness ``S_t / K``.
        ``[1]`` Time to maturity ``tau = T - t`` (years).
        ``[2]`` Current hedge position ``h_{t-1}`` held entering the step.

    Action
    ------
    ``Box(low=0.0, high=1.0, shape=(1,), dtype=float32)``
        Target hedge ratio ``h_t`` (fraction of a share held per unit of
        option short, e.g. ``0.5`` shares per option). Clipped to
        ``[0, 1]`` by the space itself; values are also clipped inside
        :meth:`step` as a defensive measure against out-of-bound inputs.

    Reward
    ------
    ``R_t = -(Pi_t - Pi_{t-1})**2 - cost_penalty * costs_t``
        The squared change in mark-to-market portfolio value (hedging
        error) is penalised, together with the (weighted) dollar cost of
        rebalancing. Both terms are normalised by ``S_0**2`` / ``S_0`` to
        keep the reward scale roughly O(1) regardless of the initial spot
        level, which stabilises DDPG training.

    Episode termination
    --------------------
    An episode always ``terminated``\\ s once ``tau <= 0`` (option
    maturity is reached); ``truncated`` is only ``True`` if an external
    time-limit wrapper is used, and is otherwise always ``False`` since the
    environment has a natural, finite horizon.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: HedgingEnvConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config or HedgingEnvConfig()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode!r}")
        self.render_mode = render_mode

        cfg = self.config
        # Observation: [moneyness, tau, prev_hedge]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([np.inf, cfg.maturity, 1.0], dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32,
        )

        # Normalisation constants used to keep the reward well-scaled.
        self._pnl_scale = cfg.s0 ** 2
        self._cost_scale = cfg.s0

        # Episode state, initialised properly in reset().
        self._np_random_seeded = False
        self._step_idx: int = 0
        self._spot: float = cfg.s0
        self._hedge: float = cfg.initial_hedge
        self._cash: float = 0.0
        self._prev_portfolio_value: float = 0.0
        self._option_price: float = 0.0
        self._episode_costs: float = 0.0

    # ------------------------------------------------------------------
    # Core Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to the start of a new hedging episode.

        Parameters
        ----------
        seed : int, optional
            Seed for the environment's internal RNG (Gymnasium standard).
        options : dict, optional
            Optional overrides. Supported keys:
                ``"s0"`` : float, override the initial spot price.
                ``"initial_hedge"`` : float, override the starting hedge.

        Returns
        -------
        obs : np.ndarray
            Initial observation, shape ``(3,)``.
        info : dict
            Auxiliary diagnostic info (empty at reset besides option price).
        """
        super().reset(seed=seed)
        cfg = self.config
        options = options or {}

        self._step_idx = 0
        self._spot = float(options.get("s0", cfg.s0))
        self._hedge = float(options.get("initial_hedge", cfg.initial_hedge))
        self._cash = 0.0
        self._episode_costs = 0.0

        tau = cfg.maturity
        self._option_price = bs_call_price(self._spot, cfg.strike, tau, cfg.r, cfg.sigma)

        # Mark-to-market value of the initial (short-call + hedge) book.
        self._prev_portfolio_value = (
            self._hedge * self._spot * cfg.n_contracts - self._option_price * cfg.n_contracts
        )

        obs = self._get_obs()
        info = self._get_info(cost=0.0, delta_h=0.0)
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the simulation by one time step.

        Parameters
        ----------
        action : np.ndarray
            Shape ``(1,)`` array with the target hedge ratio in ``[0, 1]``.

        Returns
        -------
        obs : np.ndarray
            Next observation, shape ``(3,)``.
        reward : float
            Scalar reward for this transition.
        terminated : bool
            ``True`` once the option has reached maturity.
        truncated : bool
            Always ``False``; no artificial truncation is imposed here.
        info : dict
            Diagnostics: spot, option price, portfolio value, costs, etc.
        """
        cfg = self.config
        target_hedge = float(np.clip(action, 0.0, 1.0).item())

        # --- 1. Rebalance the hedge and pay transaction costs -----------
        delta_h = target_hedge - self._hedge
        cost = cfg.transaction_cost * abs(delta_h) * self._spot * cfg.n_contracts

        # Cash flow from trading shares (buying costs cash, selling raises
        # cash), net of the transaction cost, then accrue interest for dt.
        trade_cashflow = -delta_h * self._spot * cfg.n_contracts - cost
        self._cash = self._cash * math.exp(cfg.r * cfg.dt) + trade_cashflow
        self._hedge = target_hedge
        self._episode_costs += cost

        # --- 2. Evolve the underlying under real-world GBM --------------
        z = self.np_random.standard_normal()
        drift = (cfg.mu - 0.5 * cfg.sigma ** 2) * cfg.dt
        diffusion = cfg.sigma * math.sqrt(cfg.dt) * z
        self._spot = self._spot * math.exp(drift + diffusion)

        # --- 3. Re-mark the option and portfolio -------------------------
        self._step_idx += 1
        tau = max(cfg.maturity - self._step_idx * cfg.dt, 0.0)
        self._option_price = bs_call_price(self._spot, cfg.strike, tau, cfg.r, cfg.sigma)

        portfolio_value = (
            self._hedge * self._spot * cfg.n_contracts
            + self._cash
            - self._option_price * cfg.n_contracts
        )
        pnl_change = portfolio_value - self._prev_portfolio_value
        self._prev_portfolio_value = portfolio_value

        # --- 4. Reward: penalise squared hedging error + trading cost ----
        normalised_pnl_penalty = (pnl_change ** 2) / self._pnl_scale
        normalised_cost = cost / self._cost_scale
        reward = -normalised_pnl_penalty - cfg.cost_penalty * normalised_cost

        terminated = tau <= 0.0
        truncated = False

        # Note on settlement: at tau == 0, bs_call_price collapses to the
        # intrinsic payoff max(S_T - K, 0), so portfolio_value computed
        # above already reflects the option's final cash settlement — no
        # separate terminal adjustment step is required.

        obs = self._get_obs()
        info = self._get_info(cost=cost, delta_h=delta_h)
        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:  # pragma: no cover - human-readable only
        """Print a human-readable summary of the current state."""
        cfg = self.config
        tau = max(cfg.maturity - self._step_idx * cfg.dt, 0.0)
        print(
            f"step={self._step_idx:3d} | S={self._spot:8.3f} | "
            f"tau={tau:6.4f} | h={self._hedge:5.3f} | "
            f"C={self._option_price:8.4f} | cash={self._cash:10.4f} | "
            f"Pi={self._prev_portfolio_value:10.4f}"
        )

    def close(self) -> None:
        """No external resources are held; provided for API compliance."""
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        cfg = self.config
        tau = max(cfg.maturity - self._step_idx * cfg.dt, 0.0)
        moneyness = self._spot / cfg.strike
        return np.array([moneyness, tau, self._hedge], dtype=np.float32)

    def _get_info(self, *, cost: float, delta_h: float) -> dict[str, Any]:
        cfg = self.config
        tau = max(cfg.maturity - self._step_idx * cfg.dt, 0.0)
        bs_delta = bs_call_delta(self._spot, cfg.strike, tau, cfg.r, cfg.sigma)
        return {
            "spot": self._spot,
            "option_price": self._option_price,
            "hedge_ratio": self._hedge,
            "bs_delta": bs_delta,
            "delta_h": delta_h,
            "transaction_cost": cost,
            "cumulative_cost": self._episode_costs,
            "cash": self._cash,
            "portfolio_value": self._prev_portfolio_value,
            "tau": tau,
        }


__all__ = [
    "HedgingEnv",
    "HedgingEnvConfig",
    "bs_call_price",
    "bs_call_delta",
]
