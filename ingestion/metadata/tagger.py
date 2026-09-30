"""Prompted-LLM metadata extraction — CLAUDE.md #3 field list. jurisdiction and
governing_law are requested as two distinct prompt fields (CLAUDE.md #2): the
prompt explicitly warns the model not to collapse them even when the source
clause states both in one sentence ("governed by Delaware law, exclusive
jurisdiction in New York courts").

tenant_id / matter_id / privilege_level are NOT sent to or trusted from the
LLM — see the docstring on ClauseMetadata in ingestion/schemas.py for why.

clause_type / contract_type are constrained to the controlled vocabularies in
vocabulary.py: listed in the prompt, and mapped onto the vocabulary in code
whatever the model returns.

tag_and_enrich_clause (R12) asks for the metadata AND the contextual-retrieval
note (context_enrichment.py) in one call: ingestion used to make two LLM calls
per clause, which on a free-tier provider is what hits the rate limit
mid-document. Same per-field parsing either way (_metadata_from_result)."""

from __future__ import annotations

from ingestion.embedding.context_enrichment import (
    CONTEXT_NOTE_INSTRUCTIONS,
    context_user_prompt,
    parse_context_note,
)
from ingestion.metadata.document_profile import parse_iso_date
from ingestion.metadata.llm_client import LLMClient
from ingestion.metadata.normalize import normalize_place
from ingestion.metadata.vocabulary import (
    CLAUSE_TYPES,
    CONTRACT_TYPES,
    normalize_clause_type,
    normalize_contract_type,
    prompt_list,
)
from ingestion.schemas import Clause, ClauseMetadata, PrivilegeLevel

_FIELDS = f"""- contract_type: one of {prompt_list(CONTRACT_TYPES)}, or null
- clause_type: the single best label from {prompt_list(CLAUSE_TYPES)}, or null
- parties: array of strings (party names mentioned), [] if none
- effective_date: string "YYYY-MM-DD" or null
- signature_date: string "YYYY-MM-DD" or null
- jurisdiction: string or null — the court/venue with authority to hear \
disputes (e.g. "New York")
- governing_law: string or null — the body of law that interprets the \
contract (e.g. "Delaware")
- contains_pii: boolean — true if the clause text contains personal data \
(names, SSNs, addresses, signatures)"""

_JURISDICTION_WARNING = """jurisdiction and governing_law are DIFFERENT fields. A clause can say \
"governed by Delaware law, exclusive jurisdiction in the courts of New York" \
— that is governing_law="Delaware" and jurisdiction="New York", two separate \
values. Never copy one into the other, and never leave one out because the \
other was easier to find."""

SYSTEM_PROMPT = f"""You are a legal contract metadata extraction assistant. Given \
a single clause from a contract, extract structured metadata as JSON with \
exactly these fields:

{_FIELDS}

{_JURISDICTION_WARNING}

Respond with JSON only, no prose, matching exactly the field names above."""

TAG_AND_ENRICH_SYSTEM_PROMPT = f"""You are a legal contract metadata extraction \
assistant that also writes a context note. Given one clause from a contract \
plus the document's summary and outline, respond with JSON with exactly these \
fields:

{_FIELDS}
- context_note: string — see below

{_JURISDICTION_WARNING}

{CONTEXT_NOTE_INSTRUCTIONS}

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
    return _metadata_from_result(
        result, tenant_id=tenant_id, matter_id=matter_id, privilege_level=privilege_level
    )


def tag_and_enrich_clause(
    clause: Clause,
    *,
    tenant_id: str,
    matter_id: str,
    privilege_level: PrivilegeLevel,
    llm_client: LLMClient,
    document_summary: str | None = None,
    document_outline: str | None = None,
) -> tuple[ClauseMetadata, str | None]:
    """(metadata, context_note) from a single LLM call."""
    result = llm_client.generate_json(
        TAG_AND_ENRICH_SYSTEM_PROMPT, context_user_prompt(clause, document_summary, document_outline)
    )
    metadata = _metadata_from_result(
        result, tenant_id=tenant_id, matter_id=matter_id, privilege_level=privilege_level
    )
    return metadata, parse_context_note(result)


def _metadata_from_result(
    result: dict, *, tenant_id: str, matter_id: str, privilege_level: PrivilegeLevel
) -> ClauseMetadata:
    parties = result.get("parties") or []
    # Every LLM field is untrusted input: bad dates become None and place
    # names are normalized to the same canonical form query filters use
    # (see normalize.py) rather than failing ingestion or never matching.
    return ClauseMetadata(
        contract_type=normalize_contract_type(result.get("contract_type")),
        clause_type=normalize_clause_type(result.get("clause_type")),
        parties=[str(p) for p in parties] if isinstance(parties, list) else [],
        effective_date=parse_iso_date(result.get("effective_date")),
        signature_date=parse_iso_date(result.get("signature_date")),
        jurisdiction=normalize_place(result.get("jurisdiction")),
        governing_law=normalize_place(result.get("governing_law")),
        matter_id=matter_id,
        tenant_id=tenant_id,
        privilege_level=privilege_level,
        contains_pii=bool(result.get("contains_pii", False)),
        supersession_links=[],
    )
