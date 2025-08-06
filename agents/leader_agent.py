from .base import Agent

# map high‐level pillars to the sub‐tasks they should run
PILLAR_TO_TASKS = {
    "Natural Language Understanding": [
        "Intent classification",
        "Entity recognition",
        "Sentiment analysis",
        "Syntax parsing",  # if you add a syntax parser model
    ],
    "Knowledge Retrieval": [
        "Keyword-based document search",  # placeholder
        "Vector-based semantic retrieval",
        "FAQ lookup"  # placeholder
    ],
    "Reasoning": [
        "Chain-of-thought reasoning",
        "Deduction",        # if you wire in a deduction model
        "Induction",        # etc.
    ],
    "Planning": [
        "Task decomposition",
        "Sequence planning",
        "Resource allocation",
    ],
}

class LeaderAgent(Agent):
    def __init__(self, model_manager):
        # model_manager: an instance of ModelManager
        self.manager = model_manager

    def load_model(self):
        # nothing to load here—we delegate to ModelManager
        return True

    def generate(self, intent_obj: dict, **kwargs):
        """
        intent_obj = {"category": "Reasoning", "params": {"text": "..."}}
        """
        cat = intent_obj.get("category")
        if not cat:
            return {"error": intent_obj.get("error", "No category.")}
        tasks = PILLAR_TO_TASKS.get(cat, [])
        text = intent_obj["params"]["text"]
        results = {}
        for task in tasks:
            pipe = self.manager.get(task)
            if not pipe:
                results[task] = {"error": f"Model for '{task}' not loaded."}
                continue
            try:
                # most pipelines expect a single string input
                out = pipe(text)
                results[task] = out
            except Exception as e:
                results[task] = {"error": str(e)}
        return results

