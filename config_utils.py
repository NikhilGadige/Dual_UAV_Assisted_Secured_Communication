from environment import EnvConfig

def build_env_config(
    seed: int | None = None,
    fading_model: str = "rician",
    rician_k: float = 5.0,
    control_mode: str = "velocity",
    role_switching: bool = False,
    user_mobile: bool = False,
    use_los_model: bool = False,
    observation_mode: str = "full",
    normalize_observations: bool = True,
) -> EnvConfig:
    return EnvConfig(
        seed=seed,
        fading_model=fading_model,
        rician_k=rician_k,
        control_mode=control_mode,
        role_switching=role_switching,
        user_mobile=user_mobile,
        use_los_model=use_los_model,
        observation_mode=observation_mode,
        normalize_observations=normalize_observations,
    )
