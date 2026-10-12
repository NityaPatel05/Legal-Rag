"""Review queue: the lawyer's side of the human-review workflow (H1/H2).

Mirrors the backend's rules instead of softening them: every draft claim
needs its own decision (no approve-all), an approved or edited claim that
fails the citation check needs an override reason, and "Check claims" runs
the same validator as the decision without deciding anything.

The open case is tracked by review id, captured when the row is clicked,
not by table row index: the list can change between reruns (a decision
here, another reviewer elsewhere), and a stale index would silently open a
different case -- starting its time-to-decision clock (GET /reviews/{id})."""

from __future__ import annotations

import uuid

import streamlit as st

from api import ApiError
from render import claim_html, plain_html, show_clauses, show_status
from state import cached, document, get_client, invalidate, require_scope, show_error, store

scope = require_scope()
client = get_client()

ACTIONS = {"approve": "Approve", "edit": "Edit", "reject": "Reject"}

st.title("Review queue")
st.caption(
    "Answers the system would not release on its own. Decide every draft claim individually; "
    "approved and edited claims are re-checked against the contract as stored now."
)


def _when(iso: str | None) -> str:
    return (iso or "")[:16].replace("T", " ")


# --- queue ------------------------------------------------------------------

status_filter = st.segmented_control(
    "Show", ["open", "resolved", "all"], default="open", format_func=str.title, key="review_filter"
) or "open"
try:
    reviews = client.list_reviews(scope, status_filter)
except ApiError as exc:
    show_error(exc)
    st.stop()

table_key = f"review_table:{status_filter}:{st.session_state.get('review_table_version', 0)}"
st.session_state["review_table_ids"] = [r["review_id"] for r in reviews]


def _on_select() -> None:
    rows = st.session_state[table_key].selection.rows
    ids = st.session_state.get("review_table_ids", [])
    if rows and rows[0] < len(ids):
        st.session_state["open_review_id"] = ids[rows[0]]


if reviews:
    st.dataframe(
        [
            {
                "Queued": _when(r["created_at"]),
                "Document": r["document_id"],
                "Question": r["question"],
                "Why": r["reason"],
                "Failing claims": f"{r['failing_claims']}/{r['total_claims']}",
                "Injection flags": r["injection_flagged_clauses"],
                "Status": r["status"],
            }
            for r in reviews
        ],
        hide_index=True,
        use_container_width=True,
        on_select=_on_select,
        selection_mode="single-row",
        key=table_key,
        column_config={"Question": st.column_config.TextColumn(width="large")},
    )
elif status_filter == "open":
    st.success("Nothing is waiting for review.", icon=":material/task_alt:")
else:
    st.info("No cases.")

review_id = st.session_state.get("open_review_id")
if not review_id:
    if reviews:
        st.caption("Select a case to open it.")
    st.stop()

try:
    # Opening the case as its reviewer (starts the decision clock); fetched
    # once per session, not on every rerun of this page.
    case = cached(("review", scope, review_id), lambda: client.get_review(scope, review_id))
except ApiError as exc:
    show_error(exc)
    st.session_state.pop("open_review_id", None)
    st.stop()

# --- case -------------------------------------------------------------------

st.divider()
head, close = st.columns([6, 1], vertical_alignment="top")
with head:
    st.html(plain_html(case["question"], "claim"))
    st.caption(
        f"Document **{case['document_id']}** · route `{case['route']}` · system outcome "
        f"`{case['original_final_status']}` · queued {_when(case['created_at'])} UTC · case `{case['review_id']}`"
    )
    st.html(plain_html("Why queued: " + case["reason"], "meta"))
if close.button("Close", icon=":material/close:", key="close_case"):
    st.session_state.pop("open_review_id", None)
    st.session_state["review_table_version"] = st.session_state.get("review_table_version", 0) + 1
    st.rerun()

flagged = [c for c in case["shown_clauses"] if c.get("injection_suspected")]
if flagged:
    st.error(
        f"{len(flagged)} clause(s) the model read contain instruction-like text "
        f"({', '.join(c['clause_id'] for c in flagged)}). Check that the drafts weren't steered by them.",
        icon=":material/shield:",
    )


def _show_outcome(outcome: dict) -> None:
    if outcome["final_status"] == "answered_after_review":
        show_status("success", "Answer released", f"by {outcome['reviewer']} at {_when(outcome['decided_at'])} UTC.")
    else:
        show_status("error", "No answer released", f"{outcome['reviewer']} rejected every claim.")
    if outcome.get("seconds_to_decision") is not None:
        st.caption(f"Decided {outcome['seconds_to_decision']:.0f} s after the case was first opened.")
    st.dataframe(
        [
            {
                "Claim": c["index"] + 1 if c["index"] is not None else "added",
                "Decision": ACTIONS[c["action"]],
                "Text": c["claim_text"],
                "Cites": c.get("resolved_clause_id") or c["citation"],
                "Check": "passed" if c["validator_passed"] else ("—" if c["action"] == "reject" else "failed"),
                "Override reason": c.get("override_reason") or "",
            }
            for c in outcome["claims"]
        ],
        hide_index=True,
        use_container_width=True,
    )
    if outcome.get("answer_text"):
        st.markdown("##### Released answer")
        st.html(plain_html(outcome["answer_text"]))
    if outcome.get("note"):
        st.html(plain_html("Reviewer note: " + outcome["note"], "meta"))


def _show_evidence() -> None:
    quotes: dict[str, list[tuple[str, str | None]]] = {}
    for d in case["draft_claims"]:
        cid = d.get("resolved_clause_id") or d["citation"]
        quotes.setdefault(cid, []).append((str(d["index"] + 1), d.get("supporting_quote")))
    shown_tab, whole_tab, trace_tab = st.tabs(["Shown to the model", "Whole contract", "Retrieval trace"])
    with shown_tab:
        st.caption("Highlights mark where each draft's quote occurs. A draft with no highlight quoted text that isn't there.")
        with st.container(height=700, border=False):
            show_clauses(case["shown_clauses"], quotes)
    with whole_tab:
        st.caption("You may cite any clause of the contract, not only those the model saw.")
        if st.toggle("Load the whole contract", key=f"whole:{review_id}"):
            try:
                detail = document(scope, case["document_id"])
            except ApiError as exc:
                show_error(exc)
            else:
                with st.container(height=700, border=False):
                    show_clauses(detail["clauses"], quotes)
    with trace_tab:
        for step in case.get("trace") or []:
            st.markdown(f"**{step['stage']}** · {len(step['clause_ids'])} clause(s)")
            st.code("\n".join(step["clause_ids"]) or "(none)", language=None)


if case["status"] != "open" or case.get("outcome"):
    if case.get("outcome"):
        _show_outcome(case["outcome"])
    with st.expander("Evidence"):
        _show_evidence()
    st.stop()

# --- decision form ------------------------------------------------------------

added_key = f"added:{review_id}"
added: list[str] = st.session_state.setdefault(added_key, [])
checks: dict = st.session_state.setdefault(f"checks:{review_id}", {})


def _k(field: str, ident) -> str:
    return f"{field}:{review_id}:{ident}"


def _clear_check(ident) -> None:
    checks.pop(str(ident), None)


def _check_badge(ident) -> None:
    result = checks.get(str(ident))
    if result is None:
        return
    if result["validator_passed"]:
        st.badge("Edited claim passes the citation check", icon=":material/check:", color="green")
    else:
        st.badge("Edited claim fails the citation check", icon=":material/close:", color="red")
        st.caption(result.get("failure_reason") or "")


left, right = st.columns([5, 4], gap="large")
with right:
    st.markdown("##### Evidence")
    _show_evidence()

with left:
    st.markdown(f"##### Draft claims ({len(case['draft_claims'])})")
    if not case["draft_claims"]:
        st.info("The model drafted no claims. Add a claim below with a verbatim quote, or submit with none to "
                "record that no answer can be released.")
    for d in case["draft_claims"]:
        ident = d["index"]
        with st.container(border=True):
            st.html(
                claim_html(str(ident + 1), d["claim_text"], d.get("supporting_quote"))
                + plain_html(f"Cites {d.get('resolved_clause_id') or d['citation']}", "meta")
            )
            if d["is_valid"]:
                st.badge("Passed the citation check", icon=":material/check:", color="green")
            else:
                st.badge("Failed the citation check", icon=":material/close:", color="red")
                st.html(plain_html(d.get("failure_reason") or "", "meta"))
            action = st.segmented_control(
                "Decision", list(ACTIONS), format_func=ACTIONS.get, key=_k("action", ident)
            )
            if action == "edit":
                changed = {"on_change": _clear_check, "args": (ident,)}
                st.text_area("Claim", value=d["claim_text"], key=_k("claim", ident), **changed)
                st.text_input(
                    "Cited clause ID", value=d.get("resolved_clause_id") or d["citation"], key=_k("cite", ident), **changed
                )
                st.text_area(
                    "Supporting quote — verbatim from that clause",
                    value=d.get("supporting_quote") or "",
                    key=_k("quote", ident),
                    **changed,
                )
                _check_badge(ident)
            if action in ("approve", "edit"):
                required = action == "approve" and not d["is_valid"]
                st.text_area(
                    "Override reason" + (" (required: this claim fails the citation check)" if required else ""),
                    key=_k("override", ident),
                    placeholder="Only needed if the claim fails the check. Recorded in the audit log.",
                    height=68,
                )

    for ident in list(added):
        with st.container(border=True):
            top, remove = st.columns([5, 1])
            top.markdown("**Added claim**")
            if remove.button("Remove", key=_k("remove", ident)):
                added.remove(ident)
                st.rerun()
            changed = {"on_change": _clear_check, "args": (ident,)}
            st.text_area("Claim", key=_k("claim", ident), **changed)
            st.text_input(
                "Cited clause ID", key=_k("cite", ident), placeholder=f"{case['document_id']}::clause-0007", **changed
            )
            st.text_area("Supporting quote — verbatim from that clause", key=_k("quote", ident), **changed)
            _check_badge(ident)
            st.text_area("Override reason", key=_k("override", ident), height=68,
                         placeholder="Only needed if the claim fails the check.")

    if st.button("Add a claim", icon=":material/add:"):
        added.append(uuid.uuid4().hex[:8])
        st.rerun()


def _decisions() -> tuple[list[dict], list[str], list[str]]:
    """(decisions, ident of each decision, problems found before sending)."""
    decisions, idents, problems = [], [], []

    def text(field, ident) -> str:
        return (st.session_state.get(_k(field, ident)) or "").strip()

    for d in case["draft_claims"]:
        ident = d["index"]
        action = st.session_state.get(_k("action", ident))
        if action is None:
            problems.append(f"Claim {ident + 1} has no decision.")
            continue
        decision = {"index": ident, "action": action, "override_reason": text("override", ident) or None}
        if action == "edit":
            decision |= {"claim_text": text("claim", ident), "citation": text("cite", ident),
                         "supporting_quote": text("quote", ident)}
            if not all((decision["claim_text"], decision["citation"], decision["supporting_quote"])):
                problems.append(f"Claim {ident + 1}: an edit needs a claim, a clause ID and a quote.")
        if action == "approve" and not d["is_valid"] and not decision["override_reason"]:
            problems.append(f"Claim {ident + 1} fails the citation check: edit it, reject it, or give an override reason.")
        decisions.append(decision)
        idents.append(str(ident))
    for ident in added:
        decision = {"index": None, "action": "edit", "claim_text": text("claim", ident),
                    "citation": text("cite", ident), "supporting_quote": text("quote", ident),
                    "override_reason": text("override", ident) or None}
        if not all((decision["claim_text"], decision["citation"], decision["supporting_quote"])):
            problems.append("An added claim needs a claim, a clause ID and a quote (or remove it).")
        decisions.append(decision)
        idents.append(ident)
    return decisions, idents, problems


st.divider()
st.markdown("##### Decision")
r1, r2 = st.columns([2, 3])
reviewer = r1.text_input("Your name", key="reviewer", help="Recorded with the decision and in the audit log.")
note = r2.text_input("Note (optional)", key=_k("note", "case"))

b1, b2, _ = st.columns([1, 1, 3])
check_clicked = b1.button("Check claims", icon=":material/fact_check:", help="Run the citation check on edited and added claims without deciding.")
submit_clicked = b2.button("Submit decision", type="primary", icon=":material/gavel:")

if check_clicked:
    decisions, idents, _ = _decisions()
    to_check = [(i, d) for i, d in zip(idents, decisions) if d["action"] == "edit" and d.get("claim_text")
                and d.get("citation") and d.get("supporting_quote")]
    if not to_check:
        st.info("Nothing to check: the check runs on edited and added claims that have a claim, clause ID and quote.")
    else:
        try:
            results = client.check_claims(scope, review_id, [d for _, d in to_check])
        except ApiError as exc:
            show_error(exc)
        else:
            st.session_state[f"checks:{review_id}"] = {i: r for (i, _), r in zip(to_check, results)}
            st.rerun()

if submit_clicked:
    decisions, _, problems = _decisions()
    if not reviewer.strip():
        problems.insert(0, "Enter your name.")
    if problems:
        st.error("\n".join(f"- {p}" for p in problems), icon=":material/error:")
    else:
        try:
            decided = client.decide(scope, review_id, reviewer.strip(), decisions, note)
        except ApiError as exc:
            if exc.status == 409:
                invalidate("review")
                st.warning(
                    "Another reviewer already decided this case. Their decision shows when the page next refreshes.",
                    icon=":material/sync:",
                )
            else:
                show_error(exc)
        else:
            store(("review", scope, review_id), decided)
            st.session_state.pop(added_key, None)
            st.session_state.pop(f"checks:{review_id}", None)
            st.session_state["review_table_version"] = st.session_state.get("review_table_version", 0) + 1
            st.toast("Decision recorded.", icon=":material/check_circle:")
            st.rerun()
