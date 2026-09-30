"""Contextual retrieval enrichment — architecture.md §4: prepend a short
LLM-generated context note to each clause before embedding, so an isolated
chunk still carries enough context to be retrieved/understood correctly
(architecture.md §1.2, cross-referential language).

The note is only useful if the model writing it can see where the clause
sits in the *document* (Anthropic's Contextual Retrieval generates it with
the full document in the prompt). Sending the whole contract once per clause
is too expensive on a free-tier provider, so the caller passes a compact
stand-in instead: the document-level summary (document_profile.py) plus the
section outline (every heading, in order). build_document_outline() builds
the latter deterministically."""

from __future__ import annotations

from ingestion.schemas import Clause

_MAX_OUTLINE_CHARS = 4_000

# Used by tagger.tag_and_enrich_clauses, which asks for the note in the
# same (batched) call as the clause metadata (R12).
CONTEXT_NOTE_INSTRUCTIONS = """The context note is one to two sentences, to be \
prepended before the clause text when it's embedded for retrieval. It should \
situate the clause within THIS contract: which contract and parties it \
belongs to, what section it's in, what topic it covers, and how it relates \
to other sections. Use the document summary and outline provided. Do not \
repeat the clause text verbatim."""


def build_document_outline(clauses: list[Clause]) -> str:
    lines: list[str] = []
    seen_parents: set[str] = set()
    for clause in clauses:
        # A split clause's passages share one heading -- list it once.
        if clause.parent_clause_id:
            if clause.parent_clause_id in seen_parents:
                continue
            seen_parents.add(clause.parent_clause_id)
        if clause.heading or clause.section_id:
            lines.append(f"{clause.section_id or '-'} {clause.heading or ''}".strip())
    outline = "\n".join(lines)
    return outline[:_MAX_OUTLINE_CHARS]


def parse_context_note(result: dict) -> str | None:
    note = result.get("context_note")
    return note if isinstance(note, str) and note.strip() else None
