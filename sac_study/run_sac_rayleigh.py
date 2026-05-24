from sac_study.configs import SACStudyConfig
from sac_study.sac_train import train_sac


if __name__ == "__main__":
    train_sac(SACStudyConfig(fading_model="rayleigh", episodes=4000, hidden_dim=64))

