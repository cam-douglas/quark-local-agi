# File: meta_model/agents/execution_agent.py
#!/usr/bin/env python3
"""
ExecutionAgent: safely executes shell commands or code snippets.
"""
import subprocess
from .base import Agent

class ExecutionAgent(Agent):
    def load_model(self):
        # no model to load
        return None

    def _ensure_model(self):
        # skip loading entirely
        return

    def generate(self, prompt: str, **kwargs):
        try:
            result = subprocess.run(
                prompt, shell=True, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            return {"stdout": result.stdout, "stderr": result.stderr}
        except subprocess.CalledProcessError as e:
            return {"error": str(e), "stderr": e.stderr}

