"""Session-level plumbing shared by the pages: the API client, the active
tenant + matter scope, and a per-session cache of what the API returned.

The cache lives in st.session_state rather than st.cache_data: responses
are scoped to a credential and a matter, and must never be served to a
different session."""

from __future__ import annotations

import re

import streamlit as st

from api import ID_PATTERN, ApiClient, ApiError, Scope


@st.cache_resource
def _default_client() -> ApiClient:
    return ApiClient.from_env()


def get_client() -> ApiClient:
    # Tests put an ApiClient over FastAPI's TestClient here; no widget uses
    # this key, so a viewer can't set it.
    return st.session_state.get("api_client") or _default_client()


def current_scope() -> Scope | None:
    tenant = (st.session_state.get("tenant_id") or "").strip()
    matter = (st.session_state.get("matter_id") or "").strip()
    if re.match(ID_PATTERN, tenant) and re.match(ID_PATTERN, matter):
        return Scope(tenant, matter)
    return None


def require_scope() -> Scope:
    scope = current_scope()
    if scope is None:
        st.info("Enter a **tenant** and **matter** in the sidebar to begin.", icon=":material/arrow_back:")
        st.stop()
    return scope


def _cache() -> dict:
    return st.session_state.setdefault("api_cache", {})


def cached(key: tuple, fetch):
    """fetch() once per session per key; ApiError propagates uncached."""
    cache = _cache()
    if key not in cache:
        cache[key] = fetch()
    return cache[key]


def store(key: tuple, value) -> None:
    _cache()[key] = value


def invalidate(*prefixes: str) -> None:
    cache = _cache()
    for key in [k for k in cache if k[0] in prefixes]:
        del cache[key]


def documents(scope: Scope) -> list[dict]:
    return cached(("documents", scope), lambda: get_client().list_documents(scope))


def document(scope: Scope, document_id: str) -> dict:
    return cached(("document", scope, document_id), lambda: get_client().get_document(scope, document_id))


def show_error(error: ApiError) -> None:
    st.error(error.message, icon=":material/error:")
