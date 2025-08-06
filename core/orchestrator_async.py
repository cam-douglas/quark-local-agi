#!/usr/bin/env python3
import asyncio
from router import Router
from agents.nlu_agent import NLUAgent
from agents.memory_agent import MemoryAgent
from agents.action_agent import ActionExecutionAgent

class AsyncOrchestrator:
    def __init__(self):
        self.router = Router()
        self.agents = {
            "NLU":    NLUAgent(model_name="typeform/distilbert-base-uncased-mnli"),
            "Memory": MemoryAgent(db_path="memory_db"),
            "Action": ActionExecutionAgent(),
        }

    async def handle(self, prompt: str) -> dict:
        # run NLU and Memory recall in parallel
        nlu_task = asyncio.create_task(self.agents["NLU"].generate(prompt))
        mem_task = asyncio.create_task(self.agents["Memory"].recall(prompt))
        intent, memories = await asyncio.gather(nlu_task, mem_task)

        # route based on top intent label
        top_intent = intent.get("labels", [None])[0]
        agent = self.agents.get(top_intent) or self.agents["Action"]

        # dispatch the routed agent (possibly blocking) in a thread
        result = await asyncio.to_thread(agent.generate, prompt)
        return {
            "intent":   intent,
            "memories": memories,
            "result":   result,
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("orchestrator_async:app", port=9000, reload=True)

