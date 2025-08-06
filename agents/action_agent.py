#!/usr/bin/env python3
"""
ActionExecutionAgent: executes shell commands, orchestrates API calls, and runs code snippets.
"""
import subprocess
import requests
from meta_model.agents.base import Agent

class ActionExecutionAgent(Agent):
    def __init__(self, model_name: str = None, model_path: str = None):
        super().__init__(model_name, model_path)

    def load_model(self):
        return True

    def generate(self, prompt: str, **kwargs):
        if prompt.lower().startswith("run:"):
            cmd = prompt[len("run:"):].strip()
            try:
                out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
                return {"type": "shell", "command": cmd, "output": out}
            except subprocess.CalledProcessError as e:
                return {"type": "shell_error", "command": cmd, "error": e.output}

        if prompt.startswith(("http://", "https://")):
            try:
                resp = requests.get(prompt)
                return {"type": "http", "url": prompt, "status_code": resp.status_code, "text": resp.text[:200]}
            except Exception as e:
                return {"type": "http_error", "url": prompt, "error": str(e)}

        if prompt.lower().startswith("exec:"):
            code = prompt[len("exec:"):].strip()
            try:
                local_vars = {}
                exec(code, {}, local_vars)
                return {"type": "code", "result": local_vars}
            except Exception as e:
                return {"type": "code_error", "error": str(e)}

        return {"type": "noop", "message": "Use 'run:', URL, or 'exec:' prefix."}

