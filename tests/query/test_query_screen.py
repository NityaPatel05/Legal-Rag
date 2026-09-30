"""The input screen: jailbreak verdict + route from one call, failing closed
(review finding B9) and matching route labels case-insensitively (B18)."""

import pytest

from query.filters.query_screen import screen_query
from query.schemas import RouteType


class FakeLLM:
    def __init__(self, response):
        self._response = response
        self.calls = 0

    def generate_json(self, system_prompt, user_prompt):
        self.calls += 1
        assert "input screen" in system_prompt
        return self._response


def test_clean_query_is_routed_in_the_same_call():
    llm = FakeLLM({"is_flagged": False, "reason": None, "route": "comparison"})
    check, route = screen_query("Compare sections 3 and 7", llm_client=llm)
    assert not check.is_flagged and check.reason is None
    assert route == RouteType.COMPARISON
    assert llm.calls == 1


@pytest.mark.parametrize("response", [{}, {"route": "clause_lookup"}, {"is_flagged": None}, {"is_flagged": "maybe"}])
def test_missing_or_malformed_verdict_fails_closed(response):
    check, _ = screen_query("q", llm_client=FakeLLM(response))
    assert check.is_flagged


def test_string_false_is_not_flagged():
    """bool("false") is True -- the old classifier flagged this."""
    check, _ = screen_query("q", llm_client=FakeLLM({"is_flagged": "false", "route": "clause_lookup"}))
    assert not check.is_flagged


def test_non_string_reason_does_not_500():
    check, _ = screen_query("q", llm_client=FakeLLM({"is_flagged": True, "reason": {"why": 1}}))
    assert check.is_flagged and isinstance(check.reason, str)


@pytest.mark.parametrize(
    "label,expected",
    [
        ("Comparison ", RouteType.COMPARISON),
        ("CONTRADICTION_CHECK", RouteType.CONTRADICTION_CHECK),
        ("clause lookup", RouteType.CLAUSE_LOOKUP),
        ("something else", RouteType.AMBIGUOUS_OUT_OF_SCOPE),
        (None, RouteType.AMBIGUOUS_OUT_OF_SCOPE),
    ],
)
def test_route_labels_are_normalised(label, expected):
    _, route = screen_query("q", llm_client=FakeLLM({"is_flagged": False, "route": label}))
    assert route == expected
