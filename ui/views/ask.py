"""Ask: one question about one contract (queries are per-document,
CLAUDE.md #9). The answer is shown as numbered claims, each next to its
verified quote, with the quote highlighted in the stored clause text."""

from __future__ import annotations

import streamlit as st

from api import ApiError
from render import STATUS, claim_html, plain_html, show_clauses, show_status, show_verdict
from state import document, documents, get_client, require_scope, show_error

EXAMPLES = [
    "Which law governs this agreement?",
    "When can either party terminate?",
    "What are the payment terms?",
    "Do the limitation of liability and indemnification clauses conflict?",
]

scope = require_scope()
client = get_client()

st.title("Ask a contract")

try:
    docs = documents(scope)
except ApiError as exc:
    show_error(exc)
    st.stop()
if not docs:
    st.info("No contracts in this matter yet.", icon=":material/folder_off:")
    st.page_link("views/documents.py", label="Add a contract", icon=":material/upload_file:")
    st.stop()

by_id = {d["document_id"]: d for d in docs}
if st.session_state.get("selected_document") not in by_id:
    st.session_state["selected_document"] = docs[0]["document_id"]


def _doc_label(document_id: str) -> str:
    d = by_id[document_id]
    extras = [d.get("contract_type"), (d.get("governing_law") or "").title() or None]
    return " · ".join([document_id, *[e for e in extras if e]])


document_id = st.selectbox("Contract", list(by_id), key="selected_document", format_func=_doc_label)
doc = by_id[document_id]

facts = [f"{doc['num_clauses']} clauses"]
if doc.get("parties"):
    facts.append("Parties: " + ", ".join(doc["parties"]))
if doc.get("jurisdiction"):
    facts.append("Jurisdiction: " + doc["jurisdiction"].title())
if doc.get("privilege_level"):
    facts.append(f"Privilege: {doc['privilege_level']}")
st.caption(" · ".join(facts))
if doc.get("privilege_level") == "privileged":
    st.warning(
        "Privileged contract: questions go only to the self-hosted LLM. If the API has none configured, "
        "it will refuse (403).",
        icon=":material/lock:",
    )
if doc.get("injection_flagged_clauses"):
    st.warning(
        f"{doc['injection_flagged_clauses']} clause(s) contain text that reads like instructions to an AI. "
        "They are still searched and quoted as evidence, but never followed.",
        icon=":material/shield:",
    )


def _use_example() -> None:
    if st.session_state.get("example"):
        st.session_state["question"] = st.session_state["example"]
        st.session_state["example"] = None


st.pills("Try", EXAMPLES, key="example", on_change=_use_example, label_visibility="collapsed")

with st.form("ask_form", border=True):
    question = st.text_area(
        "Question",
        key="question",
        max_chars=4000,
        height=90,
        placeholder="Ask about this contract's clauses, e.g. “What notice is required to terminate?”",
    )
    with st.expander("Filters (optional)"):
        f1, f2 = st.columns(2)
        governing_law = f1.text_input("Governing law", placeholder="e.g. Delaware or DE")
        jurisdiction = f2.text_input("Jurisdiction / venue", placeholder="e.g. New York")
    submitted = st.form_submit_button("Ask", type="primary", icon=":material/send:")

history: list[dict] = st.session_state.setdefault("history", {}).setdefault((scope, document_id), [])

if submitted:
    if not question.strip():
        st.error("Type a question first.")
    else:
        with st.spinner("Screening the question, retrieving clauses, drafting an answer and checking every citation…"):
            try:
                response = client.query(
                    scope, document_id, question.strip(), jurisdiction=jurisdiction, governing_law=governing_law
                )
            except ApiError as exc:
                show_error(exc)
            else:
                history.insert(0, response)


def _clause_label(clause: dict | None, fallback: str) -> str:
    if not clause:
        return fallback
    parts = [clause.get("section_id") and f"§ {clause['section_id']}", clause.get("heading")]
    return " ".join(p for p in parts if p) or clause["clause_id"]


def _show_evidence(citations: list[dict]) -> dict | None:
    """Renders the cited clauses with their quotes highlighted; returns
    clause_id -> clause for labelling the claims, or None on error."""
    try:
        detail = document(scope, document_id)
    except ApiError as exc:
        show_error(exc)
        return
    clauses = detail["clauses"]
    by_clause = {c["clause_id"]: c for c in clauses}
    quotes: dict[str, list[tuple[str, str | None]]] = {}
    for n, c in enumerate(citations, 1):
        if c.get("clause_id"):
            quotes.setdefault(c["clause_id"], []).append((str(n), c.get("supporting_quote")))
    cited = [by_clause[cid] for cid in quotes if cid in by_clause]

    tab_cited, tab_all = st.tabs([f"Cited clauses ({len(cited)})", "Whole contract"])
    with tab_cited:
        show_clauses(cited, quotes)
    with tab_all:
        with st.container(height=640, border=False):
            show_clauses(clauses, quotes)
    return by_clause


def _show_review_panel(response: dict, key: str) -> None:
    review_id = response["review_id"]
    state_key = f"review_status:{review_id}"
    c1, c2 = st.columns([3, 2], vertical_alignment="center")
    c1.caption(f"Review case `{review_id}` is in the lawyers' queue.")
    if c2.button("Check review status", key=f"check-{key}", icon=":material/refresh:"):
        try:
            st.session_state[state_key] = client.review_status(scope, review_id)
        except ApiError as exc:
            show_error(exc)
    status = st.session_state.get(state_key)
    if not status:
        return
    outcome = status.get("outcome")
    if status["status"] == "open" or not outcome:
        st.info("Still waiting for a lawyer's decision.", icon=":material/hourglass_top:")
        return
    if outcome["final_status"] == "answered_after_review":
        show_status("success", "Released by a lawyer", f"Reviewed by {outcome['reviewer']}.")
        accepted = [c for c in outcome["claims"] if c["action"] != "reject"]
        st.html("".join(claim_html(f"R{n}", c["claim_text"], c.get("supporting_quote")) for n, c in enumerate(accepted, 1)))
        for n, c in enumerate(accepted, 1):
            if not c["validator_passed"]:
                st.caption(f"R{n} was released on a lawyer override: {c.get('override_reason') or ''}")
    else:
        show_status("error", "No answer released", f"{outcome['reviewer']} reviewed this and rejected every claim.")
    if outcome.get("note"):
        st.html(plain_html("Reviewer note: " + outcome["note"], "meta"))


def show_result(response: dict, key: str) -> None:
    status = response["final_status"]
    review_id = response.get("review_id")
    tone, title, body = STATUS.get(
        "flagged_no_review" if status == "flagged_for_review" and not review_id else status,
        ("info", status, ""),
    )
    show_status(tone, title, body)
    if response.get("flagged_reason"):
        st.html(plain_html("Reason: " + response["flagged_reason"], "meta"))
    show_verdict(response.get("verdict"))

    citations = response.get("citations") or []
    if citations:
        left, right = st.columns([5, 4], gap="large")
        with right:
            st.markdown("##### Evidence")
            by_clause = _show_evidence(citations) or {}
        with left:
            st.markdown("##### Answer")
            parts = []
            for n, c in enumerate(citations, 1):
                parts.append(claim_html(str(n), c["claim_text"], c.get("supporting_quote")))
                label = _clause_label(by_clause.get(c.get("clause_id")), c.get("clause_id") or c["section_id"])
                parts.append(plain_html(f"Cited: {label}", "meta"))
            st.html("".join(parts))
    elif status == "not_found" and response.get("answer_text"):
        st.html(plain_html(response["answer_text"]))

    if review_id:
        _show_review_panel(response, key)
    st.caption(response.get("disclaimer") or "")


if not history:
    st.caption("Ask a question to see an answer where every sentence is backed by a quote from the contract.")
    st.stop()

latest, earlier = history[0], history[1:]
st.divider()
st.html(plain_html(latest["query_text"], "claim"))
show_result(latest, "latest")

if earlier:
    st.divider()
    st.markdown("##### Earlier questions on this contract")
    for i, response in enumerate(earlier):
        with st.expander(response["query_text"][:120]):
            show_result(response, f"h{i}")
