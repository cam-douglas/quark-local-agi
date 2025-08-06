#!/usr/bin/env python3
from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None

    @abstractmethod
    def load_model(self):
        """Load and return the model/pipeline."""
        pass

    def _ensure_model(self):
        if self.model is None:
            self.model = self.load_model()
        return self.model

    @abstractmethod
    def generate(self, prompt: str, **kwargs):
        """Run the model on `prompt` (and optional router/agents)."""
        pass

