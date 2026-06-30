"""Base agent interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

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
