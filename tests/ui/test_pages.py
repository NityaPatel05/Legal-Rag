"""The Streamlit pages driven end to end: AppTest runs the real UI scripts,
whose ApiClient talks to the real FastAPI app (TestClient) over in-memory
Qdrant with the scripted LLM from tests/app -- so a UI/API contract drift
(a renamed field, a changed status) fails here, not in front of a user."""

from pathlib import Path

import pytest

pytest.importorskip("qdrant_client")

from fastapi.testclient import TestClient  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from api import ApiClient, Scope  # noqa: E402
from app.dependencies import get_dependencies  # noqa: E402
from app.main import app  # noqa: E402
from ingestion.config import Settings  # noqa: E402
from tests.app.test_api import _QUOTE, CONTRACT, FakeDeps  # noqa: E402

ENTRY = str(Path(__file__).parents[2] / "ui" / "streamlit_app.py")
SCOPE = Scope("t-1", "m-1")


@pytest.fixture
def backend():
    deps = FakeDeps(Settings(api_keys={}, dev_mode=True))
    app.dependency_overrides[get_dependencies] = lambda: deps
    yield ApiClient(TestClient(app, client=("127.0.0.1", 50000))), deps
    app.dependency_overrides.clear()


@pytest.fixture
def ui(backend):
    client, _ = backend

    def _open(page: str | None = None, *, scope: Scope | None = SCOPE, **state) -> AppTest:
        at = AppTest.from_file(ENTRY, default_timeout=60)
        at.session_state["api_client"] = client
        at.session_state["tenant_id"] = scope.tenant_id if scope else ""
        at.session_state["matter_id"] = scope.matter_id if scope else ""
        for key, value in state.items():
            at.session_state[key] = value
        if page:
            at.switch_page(page)
        at.run()
        assert not at.exception, at.exception
        return at

    return _open


def _html(at: AppTest) -> str:
    return "\n".join(el.proto.body for el in at.get("html"))


def _button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def _ask(at: AppTest, question: str, **filters) -> AppTest:
    at.text_area(key="question").input(question)
    for label, value in filters.items():
        next(t for t in at.text_input if t.label == label).input(value)
    _button(at, "Ask").click()
    at.run()
    assert not at.exception, at.exception
    return at


def _ingest(client: ApiClient) -> None:
    client.ingest_text(SCOPE, "msa-1", CONTRACT, "confidential")


def test_without_a_scope_the_pages_ask_for_one(ui):
    at = ui(scope=None)
    assert any("tenant" in i.value for i in at.info)


def test_empty_matter_points_to_documents(ui):
    at = ui()
    assert any("No contracts" in i.value for i in at.info)


def test_answer_shows_each_claim_with_its_quote_highlighted_in_the_clause(ui, backend):
    _ingest(backend[0])
    at = _ask(ui(), "Which law governs this agreement?")

    assert any("Answered" in s.value for s in at.success)
    html = _html(at)
    assert '<span class="num">1</span>' in html  # numbered claim
    assert f'<mark class="q"><span class="qn">1</span>{_QUOTE}</mark>' in html  # located in the stored clause
    assert any("not legal advice" in c.value for c in at.caption)


def test_not_found_is_queued_and_the_asker_can_poll_without_opening_the_case(ui, backend):
    client, deps = backend
    _ingest(client)
    at = _ask(ui(), "Which law governs this agreement?", **{"Governing law": "California"})

    assert any("Not found" in i.value for i in at.info)
    [review] = client.list_reviews(SCOPE)
    _button(at, "Check review status").click()
    at.run()
    assert any("Still waiting" in i.value for i in at.info)
    assert deps.review_store.get(review["review_id"]).first_viewed_at is None


def test_reviewer_adds_a_checked_claim_and_releases_it_to_the_asker(ui, backend):
    client, deps = backend
    _ingest(client)
    law_clause = client.query(SCOPE, "msa-1", "Which law governs this agreement?")["citations"][0]["clause_id"]
    asker = _ask(ui(), "Which law governs this agreement?", **{"Governing law": "California"})
    [review] = client.list_reviews(SCOPE)

    at = ui("views/review.py", open_review_id=review["review_id"])
    assert deps.review_store.get(review["review_id"]).first_viewed_at is not None  # the reviewer opened it
    _button(at, "Add a claim").click()
    at.run()
    next(t for t in at.text_area if t.label == "Claim").input("Delaware law governs the agreement.")
    next(t for t in at.text_input if t.label == "Cited clause ID").input(law_clause)
    next(t for t in at.text_area if t.label.startswith("Supporting quote")).input(_QUOTE)
    _button(at, "Check claims").click()
    at.run()
    assert not at.exception, at.exception
    assert any("passes the citation check" in m.value for m in at.markdown)
    assert deps.review_store.get(review["review_id"]).outcome is None  # checking decides nothing

    _button(at, "Submit decision").click()
    at.run()
    assert any("Enter your name" in e.value for e in at.error)

    at.text_input(key="reviewer").input("a.lawyer")
    _button(at, "Submit decision").click()
    at.run()
    assert not at.exception, at.exception
    assert any("Answer released" in s.value for s in at.success)
    assert deps.review_store.get(review["review_id"]).outcome.final_status == "answered_after_review"

    _button(asker, "Check review status").click()
    asker.run()
    assert any("Released by a lawyer" in s.value for s in asker.success)
    assert "Delaware law governs the agreement." in _html(asker)


def test_draft_claims_each_need_a_decision_and_a_failing_approval_needs_a_reason(ui, backend):
    client, deps = backend
    _ingest(client)

    class OneBadClaim:  # generator returns a claim whose quote isn't in the contract
        def __init__(self, inner):
            self.inner = inner
            self.user_prompts = inner.user_prompts

        def generate_json(self, system_prompt, user_prompt):
            if "legal contract analysis assistant" in system_prompt:
                return {
                    "sufficiency": "sufficient",
                    "claims": [{"claim_text": "California law governs.", "citation_section_id": "msa-1::clause-0001",
                                "supporting_quote": "governed by the laws of the State of California"}],
                }
            return self.inner.generate_json(system_prompt, user_prompt)

    deps.llm_client = OneBadClaim(deps.llm_client)
    response = client.query(SCOPE, "msa-1", "Which law governs this agreement?")
    assert response["final_status"] == "flagged_for_review"

    at = ui("views/review.py", open_review_id=response["review_id"], reviewer="a.lawyer")
    assert any("Failed the citation check" in m.value for m in at.markdown)
    _button(at, "Submit decision").click()
    at.run()
    assert any("Claim 1 has no decision" in e.value for e in at.error)

    at.button_group(key=f"action:{response['review_id']}:0").set_value("approve")
    at.run()
    _button(at, "Submit decision").click()
    at.run()
    assert any("override reason" in e.value for e in at.error)

    at.button_group(key=f"action:{response['review_id']}:0").set_value("reject")
    at.run()
    _button(at, "Submit decision").click()
    at.run()
    assert not at.exception, at.exception
    assert any("No answer released" in e.value for e in at.error)


def test_documents_page_ingests_pasted_text_and_lists_it(ui):
    at = ui("views/documents.py")
    at.button_group(key="ingest_source").set_value("Paste text")
    at.run()
    next(t for t in at.text_input if t.label == "Document ID").input("msa-9")
    next(t for t in at.text_area if t.label == "Contract text").input(CONTRACT)
    _button(at, "Ingest contract").click()
    at.run()
    assert not at.exception, at.exception
    assert any("msa-9" in s.value for s in at.success)
    assert {m.label: m.value for m in at.metric}["Clauses"] == "7"

    at = ui("views/documents.py")
    assert at.dataframe[0].value["Document"].tolist() == ["msa-9"]


def test_documents_page_rejects_a_bad_document_id_before_calling_the_api(ui, backend):
    at = ui("views/documents.py")
    at.button_group(key="ingest_source").set_value("Paste text")
    at.run()
    next(t for t in at.text_input if t.label == "Document ID").input("bad id!")
    next(t for t in at.text_area if t.label == "Contract text").input(CONTRACT)
    _button(at, "Ingest contract").click()
    at.run()
    assert any("Document ID" in e.value for e in at.error)
    assert backend[1].llm_client.user_prompts == []
