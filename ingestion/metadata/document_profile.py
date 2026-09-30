"""Document-level metadata, extracted once per contract with the whole
document in view -- not per clause.

Why this exists: contract_type, parties, effective_date, governing_law and
jurisdiction are properties of the *contract*, but the per-clause tagger
(tagger.py) only ever sees one clause. Tagged per clause, governing_law is
set on the governing-law clause and null everywhere else, so a
`governing_law=delaware` filter returns only that one clause and hides the
termination / indemnity / liability clauses of the very contract the user
asked about. merge_document_profile() fills each clause's null document-level
fields from this profile; a clause-level value that is already set wins
(e.g. an exhibit with its own governing-law clause), so nothing is silently
overwritten.

The profile's `summary` is also the "document-level summary" of
Summary-Augmented Chunking (Reuter et al. 2025, arXiv:2510.06999), which
roughly halved Document-Level Retrieval Mismatch -- retrieving near-identical
boilerplate from the wrong contract -- on CUAD/ContractNLI. It's prepended to
every clause's context note before embedding (see pipeline.py)."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

from ingestion.metadata.llm_client import LLMClient
from ingestion.metadata.normalize import normalize_place, place_mentioned
from ingestion.metadata.vocabulary import CONTRACT_TYPES, normalize_contract_type, prompt_list
from ingestion.schemas import ClauseMetadata

# The head of a contract (title, recitals, parties, effective date) plus the
# tail (governing law, signatures) carry nearly all document-level facts.
_HEAD_CHARS = 12_000
_TAIL_CHARS = 6_000

SYSTEM_PROMPT = f"""You extract document-level metadata from a legal \
contract. You are shown the beginning and end of the contract. Respond with \
JSON only, with exactly these fields:

- summary: string, ONE sentence (max ~30 words) naming the contract type, \
the parties, and its subject matter
- contract_type: one of {prompt_list(CONTRACT_TYPES)}, or null
- parties: array of strings, the legal names of the contracting parties
- effective_date: string "YYYY-MM-DD" or null
- governing_law: string or null — the body of law that interprets the \
contract (e.g. "Delaware")
- jurisdiction: string or null — the court/venue with authority to hear \
disputes (e.g. "New York")

governing_law and jurisdiction are DIFFERENT fields and frequently differ. \
Never copy one into the other."""


class DocumentProfile(BaseModel):
    summary: str | None = None
    contract_type: str | None = None
    parties: list[str] = Field(default_factory=list)
    effective_date: date | None = None
    governing_law: str | None = None
    jurisdiction: str | None = None


def _document_excerpt(raw_text: str) -> str:
    if len(raw_text) <= _HEAD_CHARS + _TAIL_CHARS:
        return raw_text
    return f"{raw_text[:_HEAD_CHARS]}\n\n[... middle of contract omitted ...]\n\n{raw_text[-_TAIL_CHARS:]}"


def parse_iso_date(value) -> date | None:
    """LLM date output is untrusted: anything that isn't a valid ISO date
    becomes None instead of failing the whole ingestion with a
    ValidationError."""
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


# The profile's governing_law / jurisdiction are merged onto EVERY clause
# and become filter values for the whole contract, so they are only kept
# when the contract text names that place in a sentence that is about
# governing law / venue (review finding S4): an injected instruction or a
# hallucination can't retarget the filters of the whole document.
_GOVERNING_LAW_CUES = ("govern", "laws of", "law of", "construed")
_JURISDICTION_CUES = ("jurisdiction", "court", "venue", "forum", "arbitrat")


def _grounded(value, raw_text: str, cues: tuple[str, ...]) -> str | None:
    place = normalize_place(value)
    return place if place and place_mentioned(place, raw_text, near=cues) else None


def extract_document_profile(raw_text: str, *, llm_client: LLMClient) -> DocumentProfile:
    result = llm_client.generate_json(SYSTEM_PROMPT, _document_excerpt(raw_text))
    parties = result.get("parties") or []
    summary = result.get("summary")
    return DocumentProfile(
        summary=summary if isinstance(summary, str) and summary.strip() else None,
        contract_type=normalize_contract_type(result.get("contract_type")),
        parties=[str(p) for p in parties] if isinstance(parties, list) else [],
        effective_date=parse_iso_date(result.get("effective_date")),
        governing_law=_grounded(result.get("governing_law"), raw_text, _GOVERNING_LAW_CUES),
        jurisdiction=_grounded(result.get("jurisdiction"), raw_text, _JURISDICTION_CUES),
    )


def merge_document_profile(metadata: ClauseMetadata, profile: DocumentProfile) -> ClauseMetadata:
    """Fills null document-level fields from the profile. Never overwrites a
    value the clause-level tagger already set."""
    return metadata.model_copy(
        update={
            "contract_type": metadata.contract_type or profile.contract_type,
            "parties": metadata.parties or list(profile.parties),
            "effective_date": metadata.effective_date or profile.effective_date,
            "governing_law": metadata.governing_law or profile.governing_law,
            "jurisdiction": metadata.jurisdiction or profile.jurisdiction,
        }
    )
