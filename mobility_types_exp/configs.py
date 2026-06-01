from dataclasses import dataclass, field
from typing import Optional


MOBILITY_MODELS = ["random_walk", "random_waypoint", "gauss_markov", "constant_velocity"]
ALGORITHMS = ["dqn", "ddpg", "d3qn", "ppo", "sac", "td3pg"]
CHANNELS = ["rician", "rayleigh"]

OUTPUT_ROOT = "outputs/mobility_experiments"

TRAINING_PARAMS = {
    "dqn": {"episodes": 100, "hidden_dim": 32, "learning_rate": 5e-4},
    "ddpg": {"episodes": 100, "hidden_dim": 32, "learning_rate": 5e-4},
    "d3qn": {"episodes": 100, "hidden_dim": 64, "learning_rate": 8e-4},
    "ppo": {"episodes": 100, "hidden_dim": 64, "learning_rate": 3e-4},
    "sac": {"episodes": 100, "hidden_dim": 64, "learning_rate": 3e-4},
    "td3pg": {"episodes": 100, "hidden_dim": 64, "learning_rate": 1e-3},
}


@dataclass
class MobilityExperimentConfig:
    mobility_model: str = "random_walk"
    algorithm: str = "dqn"
    channel: str = "rician"
    episodes: int = 100
    seed: int = 42
    device: str = "cpu"

    output_root: str = OUTPUT_ROOT

    @property
    def output_dir(self) -> str:
        return f"{self.output_root}/{self.mobility_model}/{self.algorithm}_{self.channel}"

    @property
    def run_key(self) -> str:
        return f"{self.mobility_model}/{self.algorithm}_{self.channel}"
