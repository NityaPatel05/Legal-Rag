"""Prompted-LLM metadata extraction — CLAUDE.md #3 field list. jurisdiction and
governing_law are requested as two distinct prompt fields (CLAUDE.md #2): the
prompt explicitly warns the model not to collapse them even when the source
clause states both in one sentence ("governed by Delaware law, exclusive
jurisdiction in New York courts").

tenant_id / matter_id / privilege_level are NOT sent to or trusted from the
LLM — see the docstring on ClauseMetadata in ingestion/schemas.py for why."""

from __future__ import annotations

from ingestion.metadata.llm_client import LLMClient
from ingestion.schemas import Clause, ClauseMetadata, PrivilegeLevel

SYSTEM_PROMPT = """You are a legal contract metadata extraction assistant. Given \
a single clause from a contract, extract structured metadata as JSON with \
exactly these fields:

- contract_type: string or null (e.g. "Master Services Agreement", "NDA")
- clause_type: string or null (e.g. "Termination", "Governing Law", \
"Confidentiality")
- parties: array of strings (party names mentioned), [] if none
- effective_date: string "YYYY-MM-DD" or null
- signature_date: string "YYYY-MM-DD" or null
- jurisdiction: string or null — the court/venue with authority to hear \
disputes (e.g. "New York")
- governing_law: string or null — the body of law that interprets the \
contract (e.g. "Delaware")
- contains_pii: boolean — true if the clause text contains personal data \
(names, SSNs, addresses, signatures)

jurisdiction and governing_law are DIFFERENT fields. A clause can say \
"governed by Delaware law, exclusive jurisdiction in the courts of New York" \
— that is governing_law="Delaware" and jurisdiction="New York", two separate \
values. Never copy one into the other, and never leave one out because the \
other was easier to find.

Respond with JSON only, no prose, matching exactly the field names above."""


def tag_clause(
    clause: Clause,
    *,
    tenant_id: str,
    matter_id: str,
    privilege_level: PrivilegeLevel,
    llm_client: LLMClient,
) -> ClauseMetadata:
    user_prompt = f"Clause heading: {clause.heading}\n\nClause text:\n{clause.text}"
    result = llm_client.generate_json(SYSTEM_PROMPT, user_prompt)

    return ClauseMetadata(
        contract_type=result.get("contract_type"),
        clause_type=result.get("clause_type"),
        parties=result.get("parties") or [],
        effective_date=result.get("effective_date") or None,
        signature_date=result.get("signature_date") or None,
        jurisdiction=result.get("jurisdiction"),
        governing_law=result.get("governing_law"),
        matter_id=matter_id,
        tenant_id=tenant_id,
        privilege_level=privilege_level,
        contains_pii=bool(result.get("contains_pii", False)),
        supersession_links=[],
    )
