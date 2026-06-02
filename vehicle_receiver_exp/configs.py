"""Experiment configurations for vehicle receiver experiments."""

from dataclasses import dataclass


@dataclass
class VehicleExperimentConfig:
    episodes: int = 100
    hidden_dim: int = 128
    seed: int = 42
    device: str = "cpu"
    fading_model: str = "rician"
    mobility_mode: str = "straight_road"
    vehicle_max_speed: float = 10.0
    output_root: str = "outputs/vehicle_receiver"


def build_run_name(algorithm: str, fading_model: str) -> str:
    return f"{algorithm}_{fading_model}"


def build_output_dir(algorithm: str, fading_model: str, output_root: str = "outputs/vehicle_receiver") -> str:
    return f"{output_root}/{algorithm}/{fading_model}"


def _base_dqn_kwargs(episodes: int, fading_model: str, seed: int):
    return dict(
        episodes=episodes,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        replay_size=50000,
        min_replay_size=1000,
        target_update_tau=0.005,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=episodes * 200,
        grad_clip_norm=5.0,
        td_target_clip=20.0,
        hidden_dim=128,
        seed=seed,
        device="cpu",
        fading_model=fading_model,
        rician_k=5.0,
        evaluation_episodes=10,
        eval_interval=0,
        train_eval_episodes=5,
        control_mode="velocity",
        user_mobile=True,
        use_los_model=False,
        observation_mode="full",
        normalize_observations=True,
        enable_ntn=False,
        satellite_altitude_km=500.0,
        satellite_horizontal_offset_km=100.0,
        ntn_carrier_frequency_hz=2e9,
        ntn_atmospheric_loss_db=0.5,
        ntn_rician_k_db=10.0,
    )


def build_vehicle_dqn_config(fading_model: str, episodes: int = 100, seed: int = 42):
    from rl.dqn_train import DQNConfig
    return DQNConfig(**_base_dqn_kwargs(episodes, fading_model, seed))


def build_vehicle_ddpg_config(fading_model: str, episodes: int = 100, seed: int = 42):
    from rl.ddpg_train import DDPGConfig
    kw = _base_dqn_kwargs(episodes, fading_model, seed)
    kw.pop("lr")
    kw.pop("epsilon_start")
    kw.pop("epsilon_end")
    kw.pop("epsilon_decay_steps")
    kw.pop("target_update_tau")
    kw.pop("td_target_clip")
    return DDPGConfig(
        episodes=episodes,
        gamma=0.99,
        tau=0.005,
        actor_lr=1e-3,
        critic_lr=1e-3,
        batch_size=64,
        replay_size=50000,
        min_replay_size=1000,
        hidden_dim=128,
        noise_type="ou",
        noise_std_start=0.25,
        noise_std_end=0.05,
        noise_decay_steps=episodes * 200,
        ou_theta=0.15,
        ou_dt=1.0,
        seed=seed,
        device="cpu",
        fading_model=fading_model,
        rician_k=5.0,
        evaluation_episodes=10,
        eval_interval=0,
        train_eval_episodes=5,
        control_mode="velocity",
        role_switching=False,
        user_mobile=True,
        use_los_model=False,
        observation_mode="full",
        normalize_observations=True,
        enable_ntn=False,
        satellite_altitude_km=500.0,
        satellite_horizontal_offset_km=100.0,
        ntn_carrier_frequency_hz=2e9,
        ntn_atmospheric_loss_db=0.5,
        ntn_rician_k_db=10.0,
    )


def build_vehicle_d3qn_config(fading_model: str, episodes: int = 100, seed: int = 42):
    from d3qn_study.train_d3qn import D3QNConfig
    return D3QNConfig(
        episodes=episodes,
        gamma=0.99,
        lr=8e-4,
        batch_size=64,
        replay_size=50000,
        min_replay_size=2000,
        hidden_dim=64,
        epsilon_start=1.0,
        epsilon_end=0.03,
        epsilon_decay_steps=episodes * 120,
        target_update_tau=0.005,
        grad_clip_norm=5.0,
        td_target_clip=20.0,
        seed=seed,
        device="cpu",
        fading_model=fading_model,
        rician_k=5.0,
        eval_interval=0,
        train_eval_episodes=5,
        control_mode="velocity",
        user_mobile=True,
        use_los_model=False,
        observation_mode="full",
        normalize_observations=True,
    )


def _base_adv_kwargs(episodes: int, fading_model: str, seed: int):
    return dict(
        episodes=episodes,
        gamma=0.99,
        tau=0.005,
        actor_lr=1e-3,
        critic_lr=1e-3,
        batch_size=64,
        replay_size=50000,
        min_replay_size=1000,
        hidden_dim=128,
        seed=seed,
        device="cpu",
        fading_model=fading_model,
        rician_k=5.0,
        evaluation_episodes=10,
        control_mode="velocity",
        role_switching=False,
        user_mobile=True,
        use_los_model=False,
        observation_mode="full",
        normalize_observations=True,
        enable_ntn=False,
        satellite_altitude_km=500.0,
        satellite_horizontal_offset_km=100.0,
        ntn_carrier_frequency_hz=2e9,
        ntn_atmospheric_loss_db=0.5,
        ntn_rician_k_db=10.0,
    )


def build_vehicle_ppo_config(fading_model: str, episodes: int = 100, seed: int = 42):
    from rl.advanced_rl_train import AdvancedRLConfig
    kw = _base_adv_kwargs(episodes, fading_model, seed)
    kw["ppo_clip"] = 0.2
    kw["ppo_epochs"] = 4
    kw["td3_policy_delay"] = 2
    kw["td3_target_noise"] = 0.2
    kw["td3_noise_clip"] = 0.5
    kw["exploration_noise"] = 0.2
    kw["sac_alpha"] = 0.2
    return AdvancedRLConfig(**kw)


def build_vehicle_sac_config(fading_model: str, episodes: int = 100, seed: int = 42):
    kw = _base_adv_kwargs(episodes, fading_model, seed)
    from sac_study.configs import SACStudyConfig
    return SACStudyConfig(
        episodes=episodes,
        gamma=0.99,
        tau=0.005,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        batch_size=64,
        replay_size=100000,
        min_replay_size=1024,
        hidden_dim=64,
        seed=seed,
        device="cpu",
        fading_model=fading_model,
        rician_k=5.0,
        init_alpha=0.20,
        auto_entropy_tuning=True,
        eval_interval=0,
        train_eval_episodes=5,
        final_eval_episodes=10,
        grad_clip_norm=5.0,
        control_mode="velocity",
        role_switching=False,
        user_mobile=True,
        use_los_model=False,
        observation_mode="full",
        normalize_observations=True,
        output_root="outputs/vehicle_receiver/sac",
    )


def build_vehicle_td3pg_config(fading_model: str, episodes: int = 100, seed: int = 42):
    from td3pg_study.configs import TD3PGStudyConfig
    return TD3PGStudyConfig(
        episodes=episodes,
        gamma=0.99,
        tau=0.005,
        actor_lr=1e-3,
        critic_lr=1e-3,
        batch_size=64,
        replay_size=100000,
        min_replay_size=1024,
        hidden_dim=64,
        seed=seed,
        device="cpu",
        fading_model=fading_model,
        rician_k=5.0,
        exploration_noise_start=0.30,
        exploration_noise_end=0.05,
        exploration_noise_decay_steps=episodes * 200,
        target_policy_noise=0.20,
        target_noise_clip=0.50,
        policy_delay=2,
        grad_clip_norm=5.0,
        eval_interval=0,
        train_eval_episodes=5,
        final_eval_episodes=10,
        control_mode="velocity",
        role_switching=False,
        user_mobile=True,
        use_los_model=False,
        observation_mode="full",
        normalize_observations=True,
        output_root="outputs/vehicle_receiver/td3pg",
    )
