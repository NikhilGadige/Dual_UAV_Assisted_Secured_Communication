from td3pg_study.configs import TD3PGStudyConfig
from td3pg_study.td3pg_train import train_td3pg


if __name__ == "__main__":
    train_td3pg(TD3PGStudyConfig(fading_model="rayleigh", episodes=4000, hidden_dim=64))
