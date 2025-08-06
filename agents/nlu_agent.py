#!/usr/bin/env python3
from transformers import pipeline
from agents.base import Agent

class NLUAgent(Agent):
    def load_model(self):
        return pipeline(
            "zero-shot-classification",
            model=self.model_name,
            hypothesis_template="This example is {}."
        )

    def generate(self, prompt: str, **kwargs):
        self._ensure_model()
        return self.model(sequences=prompt, candidate_labels=["question", "statement", "command"])

