# agents/base.py
from abc import ABC, abstractmethod
from typing import List

class Agent(ABC):
    def __init__(self, name: str, capabilities: List[str]):
        self.name = name
        self.capabilities = capabilities

    @abstractmethod
    def execute(self, x: float, y: float) -> float:
        pass

    def can_handle(self, query: str) -> bool:
        return any(capability.lower() in query.lower() for capability in self.capabilities)