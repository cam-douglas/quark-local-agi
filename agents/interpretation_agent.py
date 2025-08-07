#!/usr/bin/env python3
from quark.use_cases_tasks import list_categories

class InterpretationAgent:
    """
    Takes raw user text and normalizes into a single intent label + params.
    """
    def interpret(self, text: str) -> dict:
        # Very simple placeholder: pick first keyword match
        cats = list_categories()
        lowered = text.lower()
        for cat in cats:
            if cat.split()[0].lower() in lowered:
                return {"category": cat, "params": {"text": text}}
        # fallback
        return {"category": None, "error": "Could not interpret intent."}

