"""Agent base interface"""
from abc import ABC, abstractmethod
from typing import Any

class Agent(ABC):
    @abstractmethod
    def act(self, obs: Any, deterministic: bool = False) -> Any:
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> dict:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass