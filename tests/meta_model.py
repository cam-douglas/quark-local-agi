import pytest
from meta_model.agents.intent_classifier import IntentClassifierAgent
from meta_model.use_cases_tasks import list_categories

@pytest.mark.parametrize("model_name", [
    "facebook/bart-large-mnli",
])
def test_intent_classifier(model_name):
    agent = IntentClassifierAgent(model_name)
    res = agent.generate("Is this positive or negative?", candidate_labels=list_categories())
    assert "labels" in res and "scores" in res

