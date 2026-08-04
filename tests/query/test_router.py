from query.generation.router import classify_route
from query.schemas import RouteType


class FakeLLMClient:
    def __init__(self, route: str):
        self._route = route

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        return {"route": self._route}


def test_classifies_clause_lookup():
    result = classify_route("What does the termination clause say?", llm_client=FakeLLMClient("clause_lookup"))
    assert result == RouteType.CLAUSE_LOOKUP


def test_classifies_comparison():
    result = classify_route("Compare the liability caps in v1 and v2.", llm_client=FakeLLMClient("comparison"))
    assert result == RouteType.COMPARISON


def test_classifies_contradiction_check():
    result = classify_route(
        "Does Section 4 conflict with Section 9?", llm_client=FakeLLMClient("contradiction_check")
    )
    assert result == RouteType.CONTRADICTION_CHECK


def test_invalid_route_value_falls_back_to_ambiguous():
    result = classify_route("asdf", llm_client=FakeLLMClient("not_a_real_route"))
    assert result == RouteType.AMBIGUOUS_OUT_OF_SCOPE
