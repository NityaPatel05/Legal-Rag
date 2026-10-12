"""Documents: the matter's contract library, and adding a contract to it."""

from __future__ import annotations

import re

import streamlit as st

from api import ID_PATTERN, ApiError
from state import documents, get_client, invalidate, require_scope, show_error

scope = require_scope()
client = get_client()

st.title("Documents")
st.caption(f"Contracts in matter **{scope.matter_id}** (tenant {scope.tenant_id}).")

PRIVILEGE_HELP = (
    "**public / confidential** — text is sent to the hosted LLM with structured PII redacted.  \n"
    "**privileged** — text only ever goes to the self-hosted LLM (PRIVILEGED_LLM_BASE_URL). Without one, the "
    "contract is stored with structural metadata only and can't be queried. A re-ingest can't lower it."
)

library, add = st.tabs([":material/folder_open: Library", ":material/upload_file: Add a contract"])

with library:
    try:
        docs = documents(scope)
    except ApiError as exc:
        show_error(exc)
        docs = []
    if st.button("Refresh", icon=":material/refresh:"):
        invalidate("documents", "document")
        st.rerun()
    if not docs:
        st.info("No contracts in this matter yet. Add one in the **Add a contract** tab.")
    else:
        rows = [
            {
                "Document": d["document_id"],
                "Type": d.get("contract_type") or "—",
                "Parties": ", ".join(d.get("parties") or []) or "—",
                "Governing law": (d.get("governing_law") or "—").title(),
                "Jurisdiction": (d.get("jurisdiction") or "—").title(),
                "Privilege": d.get("privilege_level") or "—",
                "Clauses": d["num_clauses"],
                "Injection flags": d.get("injection_flagged_clauses", 0),
            }
            for d in docs
        ]
        selection = st.dataframe(
            rows,
            hide_index=True,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            key="library_table",
            column_config={
                "Clauses": st.column_config.NumberColumn(width="small"),
                "Injection flags": st.column_config.NumberColumn(
                    width="small", help="Clauses whose text reads like instructions to an AI model"
                ),
            },
        )
        picked = selection.selection.rows if selection else []
        if picked:
            chosen = docs[picked[0]]["document_id"]
            if st.button(f"Ask about **{chosen}**", type="primary", icon=":material/arrow_forward:"):
                st.session_state["selected_document"] = chosen
                st.switch_page("views/ask.py")
        else:
            st.caption("Select a row to ask questions about that contract.")

with add:
    source = st.segmented_control(
        "Source", ["Upload a file", "Paste text"], default="Upload a file", key="ingest_source"
    )
    with st.form("ingest", border=True):
        document_id = st.text_input(
            "Document ID",
            placeholder="e.g. msa-acme-2026",
            help="Your identifier for this contract. Re-using an ID replaces that contract's stored text.",
        )
        if source == "Paste text":
            text = st.text_area("Contract text", height=320, placeholder="Paste the full contract text…")
            upload = None
        else:
            upload = st.file_uploader(
                "Contract file",
                type=["pdf", "docx", "txt", "md"],
                help="PDF and DOCX need Docling installed on the API server; .txt/.md must be UTF-8.",
            )
            text = None
        privilege = st.radio(
            "Privilege level",
            ["confidential", "public", "privileged"],
            horizontal=True,
            help=PRIVILEGE_HELP,
        )
        submitted = st.form_submit_button("Ingest contract", type="primary", icon=":material/upload:")

    if submitted:
        document_id = document_id.strip()
        if not re.match(ID_PATTERN, document_id):
            st.error("Document ID: 1–128 letters, digits, `.`, `_` or `-`.")
        elif source == "Paste text" and not (text or "").strip():
            st.error("Paste the contract text first.")
        elif source == "Upload a file" and upload is None:
            st.error("Choose a file first.")
        else:
            existing = {d["document_id"] for d in docs}
            with st.status(f"Ingesting **{document_id}**…", expanded=True) as status:
                st.write("Parsing clauses, tagging metadata (about one LLM call per 8 passages) and embedding.")
                st.write("Large contracts can take a few minutes on a rate-limited LLM tier.")
                try:
                    if upload is not None:
                        result = client.ingest_file(scope, document_id, upload.name, upload.getvalue(), privilege)
                    else:
                        result = client.ingest_text(scope, document_id, text, privilege)
                except ApiError as exc:
                    status.update(label="Ingest failed", state="error")
                    show_error(exc)
                else:
                    status.update(label=f"Ingested **{document_id}**", state="complete", expanded=False)
                    invalidate("documents", "document")
                    st.session_state["selected_document"] = document_id
                    verb = "Re-ingested" if document_id in existing else "Added"
                    st.success(f"{verb} **{document_id}**.", icon=":material/check_circle:")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Clauses", result["num_clauses"])
                    c2.metric("Defined terms", result["num_defined_terms"])
                    c3.metric("Cross-references", result["num_references"])
                    if result["num_clauses"] <= 2:
                        st.warning(
                            "Only a couple of clauses were detected, so citations will point at large blocks. "
                            "The parser relies on numbered or ALL-CAPS headings; check the source formatting.",
                            icon=":material/warning:",
                        )
                    if privilege == "privileged":
                        st.info("Privileged: questions need the self-hosted LLM configured on the API.")
                    st.page_link("views/ask.py", label="Ask about this contract", icon=":material/arrow_forward:")
