from query.generation.answer_generator import generate_answer
from query.generation.query_rewriter import rewrite_query
from query.generation.sufficiency_evaluator import SufficiencyDecision, evaluate_sufficiency
from query.schemas import RetrievedClause
from tests.query.factories import make_clause, make_metadata


class FakeLLMClient:
    def __init__(self, response: dict):
        self._response = response
        self.last_user_prompt = None

    def generate_json(self, system_prompt, user_prompt):
        self.last_user_prompt = user_prompt
        return self._response


def _shown_clause():
    clause = make_clause(
        "doc::c4", section_id="4", heading="Governing Law", text="Governed by Delaware law."
    )
    return RetrievedClause(clause=clause, metadata=make_metadata(), score=1.0, source="hybrid")


# --- generate_answer ---


def test_generate_answer_parses_claims():
    llm_client = FakeLLMClient(
        {
            "answer_text": "This Agreement is governed by Delaware law.",
            "claims": [{"claim_text": "Governed by Delaware law.", "citation_section_id": "4"}],
        }
    )
    result = generate_answer("What law governs?", [_shown_clause()], llm_client=llm_client)

    assert result.answer_text == "This Agreement is governed by Delaware law."
    assert len(result.claims) == 1
    assert result.claims[0].citation_section_id == "4"
    assert "not legal advice" in result.disclaimer


def test_generate_answer_handles_no_claims():
    llm_client = FakeLLMClient({"answer_text": "Not addressed.", "claims": []})
    result = generate_answer("What about indemnification?", [_shown_clause()], llm_client=llm_client)
    assert result.claims == []


# --- evaluate_sufficiency ---


def test_evaluate_sufficiency_parses_each_decision():
    for value in ("sufficient", "ambiguous", "insufficient"):
        llm_client = FakeLLMClient({"sufficiency": value})
        result = evaluate_sufficiency("query", [_shown_clause()], llm_client=llm_client)
        assert result == SufficiencyDecision(value)


def test_evaluate_sufficiency_empty_clauses_is_insufficient_without_calling_llm():
    class ExplodingLLMClient:
        def generate_json(self, *args, **kwargs):
            raise AssertionError("should not call the LLM with no retrieved clauses")

    result = evaluate_sufficiency("query", [], llm_client=ExplodingLLMClient())
    assert result == SufficiencyDecision.INSUFFICIENT


def test_evaluate_sufficiency_malformed_response_defaults_to_ambiguous():
    llm_client = FakeLLMClient({"sufficiency": "definitely maybe"})
    result = evaluate_sufficiency("query", [_shown_clause()], llm_client=llm_client)
    assert result == SufficiencyDecision.AMBIGUOUS


# --- rewrite_query ---


def test_rewrite_query_returns_llm_rewrite():
    llm_client = FakeLLMClient({"rewritten_query": "What jurisdiction governs disputes?"})
    result = rewrite_query("What law governs?", "What law governs?", llm_client=llm_client)
    assert result == "What jurisdiction governs disputes?"


def test_rewrite_query_falls_back_to_current_query_on_empty_response():
    llm_client = FakeLLMClient({})
    result = rewrite_query("original", "current", llm_client=llm_client)
    assert result == "current"
