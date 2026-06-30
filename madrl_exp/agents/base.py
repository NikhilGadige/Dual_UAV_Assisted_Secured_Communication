"""Base agent interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class BaseAgent(ABC):
    """Common interface for all MARL agents."""

    @abstractmethod
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        ...

    @abstractmethod
    def update(self, *args, **kwargs) -> dict:
        ...

    @abstractmethod
    def save(self, path: str):
        ...

    @abstractmethod
    def load(self, path: str):
        ...

    @abstractmethod
    def train_mode(self):
        ...

    @abstractmethod
    def eval_mode(self):
        ...


@dataclass
class Experience:
    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool
    log_prob: float | None = None
    value: float | None = None
