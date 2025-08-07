import pytest
from quark.agents.intent_classifier import IntentClassifier
from quark.core.use_cases_tasks import list_categories

@pytest.mark.parametrize("model_name", [
    "facebook/bart-large-mnli",
])
def test_intent_classifier_smoke(model_name):
    agent = IntentClassifier(model_name)
    # We pass candidate_labels explicitly here
    res = agent.classify(
        "Is this positive or negative?"
    )
    assert isinstance(res, str)

