from sac_study.configs import SACStudyConfig
from sac_study.sac_train import train_sac


def main() -> None:
    for fading_model in ("rician", "rayleigh"):
        train_sac(SACStudyConfig(fading_model=fading_model, episodes=4000, hidden_dim=64))


if __name__ == "__main__":
    main()

