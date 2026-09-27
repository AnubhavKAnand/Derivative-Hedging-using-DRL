"""
baseline_bs_agent.py

Monte Carlo evaluation of a classical Black-Scholes delta-hedging strategy
inside `HedgingEnv`.

This script establishes the Phase 1 performance baseline that a learned
(e.g. DDPG) hedging policy must beat: at every step the agent simply takes
the analytical Black-Scholes delta already exposed by the environment via
`info["bs_delta"]`, incurring the same proportional transaction costs as
any other policy.

Run with:  python3 baseline_bs_agent.py
Options:   python3 baseline_bs_agent.py --episodes 5000 --seed 0
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from hedging_env import HedgingEnv, HedgingEnvConfig


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

def bs_delta_action(info: dict[str, float]) -> np.ndarray:
    """Return the deterministic Black-Scholes delta-hedging action.

    Parameters
    ----------
    info : dict
        The `info` dictionary returned by `HedgingEnv.reset()` or
        `HedgingEnv.step()`, which already contains the analytical
        Black-Scholes delta under the key ``"bs_delta"``.

    Returns
    -------
    np.ndarray
        Shape ``(1,)`` float32 array holding the target hedge ratio,
        clipped defensively to the environment's ``[0, 1]`` action bounds.
    """
    delta = float(np.clip(info["bs_delta"], 0.0, 1.0))
    return np.array([delta], dtype=np.float32)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EpisodeResult:
    """Container for the three metrics tracked per Monte Carlo episode."""

    terminal_pnl: float
    cumulative_cost: float
    episodic_reward: float


def run_episode(env: HedgingEnv, seed: int) -> EpisodeResult:
    """Run a single episode under the Black-Scholes delta-hedging policy.

    Parameters
    ----------
    env : HedgingEnv
        The environment instance to step through (reused across episodes).
    seed : int
        Seed passed to `env.reset` for this episode's GBM path.

    Returns
    -------
    EpisodeResult
        Terminal portfolio value, cumulative transaction cost, and the
        sum of step-wise rewards for this episode.
    """
    obs, info = env.reset(seed=seed)
    episodic_reward = 0.0

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = bs_delta_action(info)
        obs, reward, terminated, truncated, info = env.step(action)
        episodic_reward += reward

    return EpisodeResult(
        terminal_pnl=info["portfolio_value"],
        cumulative_cost=info["cumulative_cost"],
        episodic_reward=episodic_reward,
    )


def run_monte_carlo(
    env: HedgingEnv, n_episodes: int, base_seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run `n_episodes` independent episodes and collect per-episode metrics.

    Each episode is seeded deterministically (``base_seed + i``) so the
    full run is reproducible while still sampling ``n_episodes`` distinct
    GBM paths.

    Returns
    -------
    tuple of np.ndarray
        ``(terminal_pnls, cumulative_costs, episodic_rewards)``, each of
        shape ``(n_episodes,)`` and dtype ``float64``.
    """
    terminal_pnls = np.empty(n_episodes, dtype=np.float64)
    cumulative_costs = np.empty(n_episodes, dtype=np.float64)
    episodic_rewards = np.empty(n_episodes, dtype=np.float64)

    for i in range(n_episodes):
        result = run_episode(env, seed=base_seed + i)
        terminal_pnls[i] = result.terminal_pnl
        cumulative_costs[i] = result.cumulative_cost
        episodic_rewards[i] = result.episodic_reward

    return terminal_pnls, cumulative_costs, episodic_rewards


# ---------------------------------------------------------------------------
# Metrics / tear sheet
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TearSheet:
    """Summary statistics for a Monte Carlo batch of hedging episodes."""

    n_episodes: int
    mean_terminal_pnl: float
    pnl_std: float
    mean_transaction_cost: float
    cost_std: float
    sharpe_ratio: float
    mean_episodic_reward: float
    reward_std: float
    pnl_5th_pct: float
    pnl_95th_pct: float
    win_rate: float


def compute_tear_sheet(
    terminal_pnls: np.ndarray,
    cumulative_costs: np.ndarray,
    episodic_rewards: np.ndarray,
) -> TearSheet:
    """Vectorised computation of the baseline performance tear sheet.

    Parameters
    ----------
    terminal_pnls, cumulative_costs, episodic_rewards : np.ndarray
        Per-episode metrics of shape ``(n_episodes,)``, as returned by
        :func:`run_monte_carlo`.

    Returns
    -------
    TearSheet
        Aggregated statistics. The Sharpe ratio is a simple proxy
        ``mean(PnL) / std(PnL)`` (no risk-free adjustment, since the
        environment's cash account already accrues at the risk-free
        rate). If ``std(PnL)`` is numerically zero, the Sharpe ratio is
        reported as ``0.0`` rather than raising a division error.
    """
    n_episodes = terminal_pnls.shape[0]
    mean_pnl = float(np.mean(terminal_pnls))
    pnl_std = float(np.std(terminal_pnls, ddof=1))
    mean_cost = float(np.mean(cumulative_costs))
    cost_std = float(np.std(cumulative_costs, ddof=1))
    mean_reward = float(np.mean(episodic_rewards))
    reward_std = float(np.std(episodic_rewards, ddof=1))

    sharpe = mean_pnl / pnl_std if pnl_std > 1e-12 else 0.0

    pnl_5th, pnl_95th = np.percentile(terminal_pnls, [5.0, 95.0])
    win_rate = float(np.mean(terminal_pnls > 0.0))

    return TearSheet(
        n_episodes=n_episodes,
        mean_terminal_pnl=mean_pnl,
        pnl_std=pnl_std,
        mean_transaction_cost=mean_cost,
        cost_std=cost_std,
        sharpe_ratio=sharpe,
        mean_episodic_reward=mean_reward,
        reward_std=reward_std,
        pnl_5th_pct=float(pnl_5th),
        pnl_95th_pct=float(pnl_95th),
        win_rate=win_rate,
    )


def print_tear_sheet(sheet: TearSheet, config: HedgingEnvConfig, elapsed_s: float) -> None:
    """Print the tear sheet in a clean, fixed-width terminal report."""
    width = 62
    rule = "-" * width

    print(rule)
    print("BLACK-SCHOLES DELTA HEDGING BASELINE — MONTE CARLO TEAR SHEET")
    print(rule)
    print(f"{'Episodes simulated':<32}{sheet.n_episodes:>30,d}")
    print(f"{'Spot / Strike':<32}{f'{config.s0:.2f} / {config.strike:.2f}':>30}")
    print(f"{'Maturity (trading days)':<32}{round(config.maturity * 252):>30d}")
    print(f"{'Volatility (annualised)':<32}{f'{config.sigma:.2%}':>30}")
    print(f"{'Real-world drift mu':<32}{f'{config.mu:.2%}':>30}")
    print(f"{'Risk-free rate r':<32}{f'{config.r:.2%}':>30}")
    print(f"{'Transaction cost rate c':<32}{f'{config.transaction_cost:.4%}':>30}")
    print(f"{'Wall-clock time':<32}{f'{elapsed_s:.2f}s':>30}")
    print(rule)
    print(f"{'Mean Terminal PnL':<32}{sheet.mean_terminal_pnl:>30.4f}")
    print(f"{'PnL Volatility (std)':<32}{sheet.pnl_std:>30.4f}")
    print(f"{'PnL 5th / 95th pct':<32}{f'{sheet.pnl_5th_pct:.4f} / {sheet.pnl_95th_pct:.4f}':>30}")
    print(f"{'Win Rate (PnL > 0)':<32}{f'{sheet.win_rate:.2%}':>30}")
    print(rule)
    print(f"{'Avg Transaction Costs':<32}{sheet.mean_transaction_cost:>30.4f}")
    print(f"{'Transaction Cost Std':<32}{sheet.cost_std:>30.4f}")
    print(rule)
    print(f"{'Approx. Sharpe Ratio':<32}{sheet.sharpe_ratio:>30.4f}")
    print(rule)
    print(f"{'Mean Episodic Reward':<32}{sheet.mean_episodic_reward:>30.6f}")
    print(f"{'Episodic Reward Std':<32}{sheet.reward_std:>30.6f}")
    print(rule)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the baseline evaluation run."""
    parser = argparse.ArgumentParser(
        description="Monte Carlo baseline evaluation of BS delta hedging."
    )
    parser.add_argument(
        "--episodes", type=int, default=5_000, help="Number of Monte Carlo episodes."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Base seed; episode i uses seed + i."
    )
    parser.add_argument(
        "--s0", type=float, default=100.0, help="Initial underlying spot price."
    )
    parser.add_argument("--strike", type=float, default=100.0, help="Option strike.")
    parser.add_argument(
        "--maturity-days", type=int, default=30, help="Maturity in trading days."
    )
    parser.add_argument("--mu", type=float, default=0.07, help="Real-world drift.")
    parser.add_argument("--sigma", type=float, default=0.25, help="Annualised volatility.")
    parser.add_argument("--r", type=float, default=0.02, help="Risk-free rate.")
    parser.add_argument(
        "--cost", type=float, default=0.001, help="Proportional transaction cost rate."
    )
    return parser.parse_args()


def main() -> None:
    """Run the full Monte Carlo baseline evaluation and print the results."""
    args = parse_args()

    config = HedgingEnvConfig(
        s0=args.s0,
        strike=args.strike,
        mu=args.mu,
        sigma=args.sigma,
        r=args.r,
        maturity=args.maturity_days / 252.0,
        dt=1.0 / 252.0,
        transaction_cost=args.cost,
        cost_penalty=1.0,
    )
    env = HedgingEnv(config=config)

    start = time.perf_counter()
    terminal_pnls, cumulative_costs, episodic_rewards = run_monte_carlo(
        env, n_episodes=args.episodes, base_seed=args.seed
    )
    elapsed = time.perf_counter() - start
    env.close()

    sheet = compute_tear_sheet(terminal_pnls, cumulative_costs, episodic_rewards)
    print_tear_sheet(sheet, config, elapsed)


if __name__ == "__main__":
    main()
