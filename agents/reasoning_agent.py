#!/usr/bin/env python3
from transformers import pipeline
from agents.base import Agent

class ReasoningAgent(Agent):
    def load_model(self):
        return pipeline("text2text-generation", model=self.model_name)

    def generate(self, prompt: str, **kwargs):
        self._ensure_model()
        # Generate reasoning about the prompt
        reasoning_prompt = f"Think step by step about: {prompt}"
        return self.model(reasoning_prompt, max_length=200)[0]["generated_text"]

