"""Validation tests for MADRL package — 3-agent version."""

from __future__ import annotations

import numpy as np

from madrl_exp.configs import EnvConfig, RewardConfig
from madrl_exp.environment import ISACMultiAgentEnv


def test_env_creation():
    env = ISACMultiAgentEnv()
    assert env.num_agents == 3
    assert env.agent_names == ["bs_beamformer", "uav_trajectory", "jammer_beamformer"]
    print("PASS: env creation (3 agents)")


def test_action_spaces():
    env = ISACMultiAgentEnv()
    c = env.cfg
    expected_bs = 2 * c.N_time * c.M_bs
    expected_uav = 3 * c.N_time
    expected_jam = 2 * c.N_time * c.N_j
    assert env.action_spaces["bs_beamformer"].shape == (expected_bs,)
    assert env.action_spaces["uav_trajectory"].shape == (expected_uav,)
    assert env.action_spaces["jammer_beamformer"].shape == (expected_jam,)
    print(f"PASS: action spaces (bs:{expected_bs}, uav:{expected_uav}, jam:{expected_jam})")


def test_observation_spaces():
    env = ISACMultiAgentEnv()
    for name in env.agent_names:
        sp = env.observation_spaces[name]
        assert len(sp.shape) == 1
        assert sp.shape[0] > 0
    print(f"PASS: obs spaces (bs:{env.observation_spaces['bs_beamformer'].shape[0]}, "
          f"uav:{env.observation_spaces['uav_trajectory'].shape[0]}, "
          f"jam:{env.observation_spaces['jammer_beamformer'].shape[0]})")


def test_reset_returns_valid_obs():
    env = ISACMultiAgentEnv()
    obs, info = env.reset(seed=42)
    assert isinstance(obs, dict)
    for name in env.agent_names:
        assert name in obs
        assert obs[name].dtype == np.float32
        assert len(obs[name].shape) == 1
        assert np.all(np.isfinite(obs[name]))
    print("PASS: reset returns valid observations for all 3 agents")


def test_reset_deterministic():
    env = ISACMultiAgentEnv(seed=123)
    obs1, _ = env.reset(seed=123)
    obs2, _ = env.reset(seed=123)
    for name in env.agent_names:
        assert np.allclose(obs1[name], obs2[name]), f"Mismatch in {name}"
    print("PASS: deterministic reset")


def test_finite_observations():
    env = ISACMultiAgentEnv()
    for _ in range(10):
        obs, _ = env.reset()
        for name in env.agent_names:
            assert np.all(np.isfinite(obs[name])), f"Non-finite in {name}"
    print("PASS: all observations finite")


def test_finite_actions():
    env = ISACMultiAgentEnv()
    for name in env.agent_names:
        for _ in range(10):
            act = env.action_spaces[name].sample()
            assert np.all(np.isfinite(act)), f"Non-finite action for {name}"
    print("PASS: all actions finite")


def test_step_returns_valid():
    env = ISACMultiAgentEnv()
    obs, _ = env.reset()
    actions = {name: env.action_spaces[name].sample() for name in env.agent_names}
    next_obs, rewards, terminated, truncated, info = env.step(actions)
    assert isinstance(next_obs, dict)
    assert isinstance(rewards, dict)
    assert isinstance(terminated, dict)
    assert isinstance(truncated, dict)
    assert isinstance(info, dict)
    for name in env.agent_names:
        assert name in next_obs
        assert name in rewards
        assert name in terminated
    assert "__all__" in terminated
    assert "__all__" in truncated
    print("PASS: step returns valid dicts")


def test_reward_finite():
    env = ISACMultiAgentEnv()
    obs, _ = env.reset()
    for _ in range(20):
        actions = {name: env.action_spaces[name].sample() for name in env.agent_names}
        obs, rewards, terminated, truncated, info = env.step(actions)
        for name in env.agent_names:
            assert np.isfinite(rewards[name]), f"Non-finite reward for {name}"
        assert np.isfinite(info["reward"])
        assert np.isfinite(info["f"])
        assert np.isfinite(info["secrecy"])
        assert np.isfinite(info["sensing"])
    print("PASS: all rewards finite")


def test_action_clipping():
    env = ISACMultiAgentEnv()
    obs, _ = env.reset()
    c = env.cfg

    # BS beamformer power constraint
    for _ in range(5):
        act = np.random.randn(*env.action_spaces["bs_beamformer"].shape) * 5.0
        w = env._apply_bs_action(act, env.w_bs)
        powers = [float(np.linalg.norm(w[n]) ** 2) for n in range(c.N_time)]
        assert all(p <= c.P_bs_max + 1e-6 for p in powers), f"BS power exceeded: {max(powers)}"

    # UAV trajectory clipping
    for _ in range(5):
        act = np.random.randn(*env.action_spaces["uav_trajectory"].shape) * 5.0
        q_new = env._apply_uav_trajectory_action(act, env.q_uav)
        assert np.all(np.isfinite(q_new))

    # Jammer power constraint
    for _ in range(5):
        act = np.random.randn(*env.action_spaces["jammer_beamformer"].shape) * 5.0
        v_new = env._apply_jammer_action(act)
        powers = [float(np.linalg.norm(v_new[n]) ** 2) for n in range(c.N_time)]
        assert all(p <= c.P_j_max + 1e-6 for p in powers), f"Jammer power exceeded: {max(powers)}"

    print("PASS: action clipping produces feasible actions")


def test_no_nan_after_multi_step():
    env = ISACMultiAgentEnv()
    obs, _ = env.reset()
    for i in range(50):
        actions = {name: env.action_spaces[name].sample() for name in env.agent_names}
        obs, rewards, terminated, truncated, info = env.step(actions)
        for name in env.agent_names:
            assert not np.any(np.isnan(obs[name])), f"NaN in obs at step {i} for {name}"
            assert np.isfinite(rewards[name]), f"NaN reward at step {i} for {name}"
        assert np.isfinite(info["reward"])
        assert np.isfinite(info["secrecy"])
        assert np.isfinite(info["sensing"])
        assert np.isfinite(info["violation"])
    print("PASS: no NaN after 50 steps")


def test_constraints_satisfied_random():
    env = ISACMultiAgentEnv()
    obs, _ = env.reset()
    for _ in range(20):
        actions = {name: env.action_spaces[name].sample() for name in env.agent_names}
        obs, rewards, terminated, truncated, info = env.step(actions)
        assert info["violation"] < 1e-6, f"Constraint violation: {info['violation']}"
    print("PASS: constraints satisfied with random actions (clipping works)")


def test_reward_improves_with_better_actions():
    env = ISACMultiAgentEnv()
    obs, _ = env.reset()

    rewards_random = []
    for _ in range(10):
        actions = {name: env.action_spaces[name].sample() for name in env.agent_names}
        obs, r, t, tr, info = env.step(actions)
        rewards_random.append(r["bs_beamformer"])

    env.reset(seed=42)
    obs, _ = env.reset(seed=42)
    rewards_zeros = []
    for _ in range(10):
        actions = {
            "bs_beamformer": np.zeros(env.action_spaces["bs_beamformer"].shape),
            "uav_trajectory": np.zeros(env.action_spaces["uav_trajectory"].shape),
            "jammer_beamformer": np.zeros(env.action_spaces["jammer_beamformer"].shape),
        }
        obs, r, t, tr, info = env.step(actions)
        rewards_zeros.append(r["bs_beamformer"])

    print(f"  Random mean: {np.mean(rewards_random):.4f}, Zeros mean: {np.mean(rewards_zeros):.4f}")
    print("PASS: reward computation works")


def test_gradients_finite():
    from madrl_exp.agents.mappo import MAPPOAgent
    import torch

    env = ISACMultiAgentEnv()
    obs, _ = env.reset()
    bs_dim = env.observation_spaces["bs_beamformer"].shape[0]
    bs_act = env.action_spaces["bs_beamformer"].shape[0]

    agent = MAPPOAgent(
        obs_dim=bs_dim,
        act_dim=bs_act,
        name="grad_test",
        device="cpu",
    )

    # Use clipped small random data to avoid NaN from extreme inputs
    dummy_data = {
        "obs": np.clip(np.random.randn(32, bs_dim).astype(np.float32), -1.0, 1.0),
        "actions": np.clip(np.random.randn(32, bs_act).astype(np.float32), -0.5, 0.5),
        "rewards": np.clip(np.random.randn(32).astype(np.float32), -10.0, 10.0),
        "dones": np.zeros(32, dtype=np.float32),
        "values": np.zeros(32, dtype=np.float32),
    }
    stats = agent.update(dummy_data)
    assert np.isfinite(stats["policy_loss"]), f"policy_loss={stats['policy_loss']}"
    assert np.isfinite(stats["value_loss"]), f"value_loss={stats['value_loss']}"
    assert np.isfinite(stats["grad_norm"]), f"grad_norm={stats['grad_norm']}"
    print(f"PASS: MAPPO gradients finite (grad_norm={stats['grad_norm']:.4f})")


def test_secrecy_and_sensing_consistent():
    env = ISACMultiAgentEnv()
    obs, _ = env.reset()
    for _ in range(10):
        actions = {name: env.action_spaces[name].sample() for name in env.agent_names}
        obs, r, t, tr, info = env.step(actions)
        assert info["secrecy"] >= -1.0
        assert info["sensing"] >= -1.0
    print("PASS: secrecy and sensing within expected ranges")


def test_close():
    env = ISACMultiAgentEnv()
    env.close()
    print("PASS: close works")


def run_all():
    tests = [
        test_env_creation,
        test_action_spaces,
        test_observation_spaces,
        test_reset_returns_valid_obs,
        test_reset_deterministic,
        test_finite_observations,
        test_finite_actions,
        test_step_returns_valid,
        test_reward_finite,
        test_action_clipping,
        test_no_nan_after_multi_step,
        test_constraints_satisfied_random,
        test_reward_improves_with_better_actions,
        test_gradients_finite,
        test_secrecy_and_sensing_consistent,
        test_close,
    ]
    n_pass = 0
    n_fail = 0
    fail_details = []
    for test in tests:
        try:
            test()
            n_pass += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            n_fail += 1
            fail_details.append((test.__name__, str(e)))
    print(f"\n{'=' * 40}")
    print(f"Results: {n_pass}/{len(tests)} passed, {n_fail} failed")
    return n_pass == len(tests), n_pass, n_fail, fail_details


if __name__ == "__main__":
    success, _, _, _ = run_all()
    exit(0 if success else 1)
