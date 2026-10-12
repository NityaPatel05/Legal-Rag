"""Streamlit UI for the contract clause-analysis API.

    cd ui && streamlit run streamlit_app.py

Environment: RAG_API_URL (default http://127.0.0.1:8000), RAG_API_KEY (sent
as X-API-Key from this server process; leave unset against a DEV_MODE API),
RAG_TENANT_ID / RAG_MATTER_ID (sidebar defaults). See ui/api.py for why the
key stays server-side and what that implies."""

from __future__ import annotations

import os

import streamlit as st

from api import ApiError
from render import inject_css
from state import current_scope, get_client

st.set_page_config(page_title="Contract Clause Analysis", page_icon=":material/gavel:", layout="wide")
inject_css()

st.session_state.setdefault("tenant_id", os.environ.get("RAG_TENANT_ID", ""))
st.session_state.setdefault("matter_id", os.environ.get("RAG_MATTER_ID", ""))


def _open_review_count() -> int | None:
    scope = current_scope()
    if scope is None:
        return None
    try:
        return len(get_client().list_reviews(scope, "open"))
    except ApiError:
        return None


with st.sidebar:
    st.subheader("Workspace")
    st.text_input("Tenant", key="tenant_id", placeholder="e.g. acme-legal")
    st.text_input("Matter", key="matter_id", placeholder="e.g. m-2026-014")
    if current_scope() is None and (st.session_state.tenant_id or st.session_state.matter_id):
        st.caption(":red[Letters, digits, `.`, `_` and `-` only.]")

    client = get_client()
    try:
        client.health()
        auth = "API key" if client.has_api_key else "dev mode, no key"
        st.caption(f":green[●] API connected · {auth}")
    except ApiError as exc:
        st.caption(f":red[●] {exc.message}")

open_reviews = _open_review_count()
review_title = f"Review queue ({open_reviews})" if open_reviews else "Review queue"

page = st.navigation(
    [
        st.Page("views/ask.py", title="Ask", icon=":material/quick_reference_all:", default=True),
        st.Page("views/review.py", title=review_title, icon=":material/gavel:", url_path="review"),
        st.Page("views/documents.py", title="Documents", icon=":material/description:", url_path="documents"),
    ]
)
page.run()

with st.sidebar:
    st.divider()
    st.caption(
        "Answers are informational summaries, not legal advice. Every claim shown carries a quote "
        "verified against the stored contract text; anything that can't be grounded is refused "
        "and sent to a lawyer."
    )
