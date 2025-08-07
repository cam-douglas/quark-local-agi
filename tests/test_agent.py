import pytest
from quark.agents.intent_classifier import IntentClassifierAgent
from quark.use_cases_tasks import list_categories

@pytest.mark.parametrize("model_name", [
    "facebook/bart-large-mnli",
])
def test_intent_classifier_smoke(model_name):
    agent = IntentClassifierAgent(model_name)
    # We pass candidate_labels explicitly here
    res = agent.generate(
        "Is this positive or negative?",
        candidate_labels=list_categories()
    )
    assert "labels" in res and "scores" in res

