from td3pg_study.configs import TD3PGStudyConfig
from td3pg_study.td3pg_train import train_td3pg


def main() -> None:
    for fading_model in ("rician", "rayleigh"):
        train_td3pg(TD3PGStudyConfig(fading_model=fading_model, episodes=4000, hidden_dim=64))


if __name__ == "__main__":
    main()
