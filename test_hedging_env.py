"""
test_hedging_env.py

Standalone verification script for `HedgingEnv`.

Instantiates the environment, runs a full episode of random actions,
and checks:
    * observation/action shapes and dtypes against the declared spaces,
    * that observations and actions stay within their declared bounds,
    * that transaction costs are computed and are non-negative,
    * that the episode terminates exactly at maturity,
    * basic sanity of the reward (finite, real-valued).

Run with:  python3 test_hedging_env.py
"""

from __future__ import annotations

import math

import numpy as np

from hedging_env import HedgingEnv, HedgingEnvConfig, bs_call_delta, bs_call_price


def check_bs_analytics() -> None:
    """Sanity-check the closed-form Black-Scholes helpers."""
    price = bs_call_price(s=100.0, k=100.0, tau=30 / 252, r=0.02, sigma=0.20)
    delta = bs_call_delta(s=100.0, k=100.0, tau=30 / 252, r=0.02, sigma=0.20)
    assert price > 0.0, "ATM call price should be strictly positive."
    assert 0.0 <= delta <= 1.0, "Call delta must lie in [0, 1]."
    # Deep ITM call should have delta close to 1.
    deep_itm_delta = bs_call_delta(s=200.0, k=100.0, tau=30 / 252, r=0.02, sigma=0.20)
    assert deep_itm_delta > 0.95, "Deep ITM call delta should be close to 1."
    # Deep OTM call should have delta close to 0.
    deep_otm_delta = bs_call_delta(s=50.0, k=100.0, tau=30 / 252, r=0.02, sigma=0.20)
    assert deep_otm_delta < 0.05, "Deep OTM call delta should be close to 0."
    print(f"[OK] BS analytics: ATM price={price:.4f}, ATM delta={delta:.4f}")


def run_random_episode(env: HedgingEnv, seed: int = 42, verbose: bool = True) -> None:
    """Run one episode of uniformly random actions and validate outputs."""
    obs, info = env.reset(seed=seed)

    assert env.observation_space.contains(obs), (
        f"Initial observation {obs} out of bounds "
        f"{env.observation_space.low}..{env.observation_space.high}"
    )
    assert obs.shape == (3,), f"Expected obs shape (3,), got {obs.shape}"
    assert obs.dtype == np.float32, f"Expected float32 obs, got {obs.dtype}"

    total_reward = 0.0
    total_cost = 0.0
    step_count = 0
    max_steps = env.config.n_steps + 5  # safety margin against infinite loops

    if verbose:
        print("\n--- Running random-action episode ---")
        env.render()

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample()
        assert env.action_space.contains(action), f"Sampled action {action} invalid."

        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

        # --- Structural checks -----------------------------------------
        assert isinstance(reward, float), f"Reward must be a float, got {type(reward)}"
        assert np.isfinite(reward), f"Reward must be finite, got {reward}"
        assert isinstance(terminated, bool), "terminated must be a bool"
        assert isinstance(truncated, bool), "truncated must be a bool"
        assert obs.shape == (3,), f"Expected obs shape (3,), got {obs.shape}"
        assert env.observation_space.contains(obs), (
            f"Observation {obs} out of bounds at step {step_count}"
        )
        assert info["transaction_cost"] >= 0.0, "Transaction cost cannot be negative."
        assert 0.0 <= info["hedge_ratio"] <= 1.0, "Hedge ratio must lie in [0, 1]."
        assert info["tau"] >= 0.0, "Time to maturity cannot be negative."

        total_reward += reward
        total_cost += info["transaction_cost"]

        if verbose:
            env.render()
            print(
                f"    action={float(action[0]):.3f} | reward={reward: .6f} | "
                f"cost={info['transaction_cost']:.5f} | bs_delta={info['bs_delta']:.4f} | "
                f"terminated={terminated}"
            )

        assert step_count <= max_steps, (
            "Episode did not terminate within the expected number of steps "
            f"({max_steps}); possible bug in maturity handling."
        )

    assert terminated, "Episode should terminate (reach maturity) by the end of the loop."
    assert math.isclose(info["tau"], 0.0, abs_tol=1e-9), (
        f"Expected tau == 0 at termination, got {info['tau']}"
    )
    assert step_count == env.config.n_steps, (
        f"Expected exactly {env.config.n_steps} steps to maturity, got {step_count}"
    )

    print(
        f"\n[OK] Episode finished in {step_count} steps | "
        f"total_reward={total_reward:.6f} | total_cost={total_cost:.5f} | "
        f"final_spot={info['spot']:.3f} | final_portfolio_value={info['portfolio_value']:.4f}"
    )


def check_reset_reproducibility(env: HedgingEnv) -> None:
    """Same seed should produce the same first-step transition."""
    obs1, _ = env.reset(seed=123)
    action = np.array([0.5], dtype=np.float32)
    next_obs1, reward1, *_ = env.step(action)

    obs2, _ = env.reset(seed=123)
    next_obs2, reward2, *_ = env.step(action)

    np.testing.assert_allclose(obs1, obs2, err_msg="Reset with same seed should be identical.")
    np.testing.assert_allclose(
        next_obs1, next_obs2, err_msg="Step after same-seed reset should be identical."
    )
    assert reward1 == reward2, "Rewards after same-seed reset+step should match exactly."
    print("[OK] Seeded reset() is reproducible.")


def check_transaction_cost_math(env: HedgingEnv) -> None:
    """Directly verify the transaction cost formula c * |delta_h| * S_t."""
    env.reset(seed=7, options={"initial_hedge": 0.0})
    spot_before = env.unwrapped._spot  # noqa: SLF001 - white-box test check
    action = np.array([0.4], dtype=np.float32)
    _, _, _, _, info = env.step(action)

    expected_cost = env.config.transaction_cost * abs(0.4 - 0.0) * spot_before
    # rel_tol relaxed to accommodate the float32 round-trip of the action
    # through `gymnasium.spaces.Box` (actions are stored/clipped as float32).
    assert math.isclose(info["transaction_cost"], expected_cost, rel_tol=1e-6), (
        f"Cost mismatch: expected {expected_cost}, got {info['transaction_cost']}"
    )
    print(
        f"[OK] Transaction cost formula verified: "
        f"c*|dh|*S = {expected_cost:.6f} == reported {info['transaction_cost']:.6f}"
    )


if __name__ == "__main__":
    check_bs_analytics()

    config = HedgingEnvConfig(
        s0=100.0,
        strike=100.0,
        mu=0.07,
        sigma=0.25,
        r=0.02,
        maturity=30 / 252,
        dt=1 / 252,
        transaction_cost=0.001,
        cost_penalty=1.0,
    )
    env = HedgingEnv(config=config)

    check_reset_reproducibility(env)
    check_transaction_cost_math(env)
    run_random_episode(env, seed=42, verbose=True)

    env.close()
    print("\nAll checks passed.")
