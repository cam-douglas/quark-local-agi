#!/usr/bin/env python3
from transformers import pipeline
from core.use_cases_tasks import list_categories

class IntentClassifier:
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            hypothesis_template="This example is {}."
        )

    def classify(self, text: str) -> str:
        res = self.classifier(text, candidate_labels=list_categories())
        return res["labels"][0]

