"""Golden set for end-to-end generation eval (roadmap R2).

A golden item is a question about one ingested contract plus what a lawyer
says a correct response looks like: which clause(s) may be cited, a phrase
the answer must contain, or that the system must refuse (flag for review).

Citation accuracy is the BLOCKING metric: the share of returned citations
whose clause_id is one of the item's acceptable clause_ids. A wrong
citation is worse than no answer (a refusal costs the lawyer time; a
confidently mis-cited answer costs them the case), so the default gate is
1.0 -- a single wrong citation fails the run. Answer coverage and refusal
accuracy are reported alongside, but don't gate.

Nothing is certifiable without review: is_certifiable() refuses unless the
set has at least MIN_CERTIFIABLE_ITEMS items and every one carries a
reviewed_by. The shipped example set is deliberately unreviewed.

Usage against a running service:

    python -m eval.golden_set.golden eval/golden_set/golden_set.example.jsonl \\
        --base-url http://localhost:8000 --api-key KEY

The contracts must already be ingested under the item's tenant/matter/
document_id."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field

MIN_CERTIFIABLE_ITEMS = 40
ANSWERED_STATUSES = {"answered", "answered_with_stripped_claims"}


class GoldenItem(BaseModel):
    id: str
    tenant_id: str
    matter_id: str
    document_id: str
    question: str
    # Clauses a correct answer may cite (a passage's own clause_id; any
    # passage of a split clause is listed individually).
    acceptable_clause_ids: list[str] = Field(default_factory=list)
    # Case-insensitive phrases the composed answer must contain.
    answer_must_contain: list[str] = Field(default_factory=list)
    # True when the contract doesn't answer the question: the only correct
    # response is flagged_for_review.
    expect_refusal: bool = False
    jurisdiction: str | None = None
    governing_law: str | None = None
    reviewed_by: str | None = None
    reviewed_on: date | None = None
    notes: str | None = None


class ItemResult(BaseModel):
    id: str
    final_status: str
    cited_clause_ids: list[str]
    wrong_citations: list[str]
    refusal_correct: bool
    answer_contains_ok: bool | None  # None when not applicable


class GoldenReport(BaseModel):
    items: int
    citations: int
    wrong_citations: int
    citation_accuracy: float | None  # None when nothing was cited
    refusal_accuracy: float
    answer_coverage: float | None
    certifiable: bool
    gate_threshold: float
    passed: bool
    results: list[ItemResult]


def load_golden_set(path: str | Path) -> list[GoldenItem]:
    items = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip() and not line.lstrip().startswith("#"):
            try:
                items.append(GoldenItem.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate item ids")
    return items


def is_certifiable(items: list[GoldenItem]) -> bool:
    return len(items) >= MIN_CERTIFIABLE_ITEMS and all(item.reviewed_by for item in items)


def score_item(item: GoldenItem, response: dict) -> ItemResult:
    status = response.get("final_status", "unknown")
    cited = [c.get("clause_id") or c.get("section_id") or "" for c in response.get("citations") or []]
    acceptable = set(item.acceptable_clause_ids)
    wrong = [c for c in cited if c not in acceptable]
    answered = status in ANSWERED_STATUSES

    contains_ok: bool | None = None
    if item.answer_must_contain and not item.expect_refusal:
        text = (response.get("answer_text") or "").lower()
        contains_ok = answered and all(phrase.lower() in text for phrase in item.answer_must_contain)

    return ItemResult(
        id=item.id,
        final_status=status,
        cited_clause_ids=cited,
        wrong_citations=wrong,
        refusal_correct=(not answered) == item.expect_refusal,
        answer_contains_ok=contains_ok,
    )


def evaluate(
    items: Iterable[GoldenItem],
    answer_fn: Callable[[GoldenItem], dict],
    *,
    gate_threshold: float = 1.0,
) -> GoldenReport:
    items = list(items)
    results = [score_item(item, answer_fn(item)) for item in items]

    citations = sum(len(r.cited_clause_ids) for r in results)
    wrong = sum(len(r.wrong_citations) for r in results)
    citation_accuracy = (citations - wrong) / citations if citations else None
    applicable = [r.answer_contains_ok for r in results if r.answer_contains_ok is not None]
    certifiable = is_certifiable(items)
    return GoldenReport(
        items=len(items),
        citations=citations,
        wrong_citations=wrong,
        citation_accuracy=citation_accuracy,
        refusal_accuracy=sum(r.refusal_correct for r in results) / len(results) if results else 0.0,
        answer_coverage=sum(applicable) / len(applicable) if applicable else None,
        certifiable=certifiable,
        gate_threshold=gate_threshold,
        # A run that cited nothing hasn't demonstrated citation accuracy.
        passed=certifiable and citation_accuracy is not None and citation_accuracy >= gate_threshold,
        results=results,
    )


def http_answer_fn(base_url: str, api_key: str | None = None) -> Callable[[GoldenItem], dict]:
    import httpx  # lazy import

    headers = {"X-API-Key": api_key} if api_key else {}

    def _answer(item: GoldenItem) -> dict:
        response = httpx.post(
            f"{base_url.rstrip('/')}/query",
            json={
                "document_id": item.document_id,
                "query_text": item.question,
                "tenant_id": item.tenant_id,
                "matter_id": item.matter_id,
                "jurisdiction": item.jurisdiction,
                "governing_law": item.governing_law,
            },
            headers=headers,
            timeout=300,
        )
        response.raise_for_status()
        return response.json()

    return _answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("golden_set")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key")
    parser.add_argument("--gate", type=float, default=1.0)
    args = parser.parse_args()

    items = load_golden_set(args.golden_set)
    report = evaluate(items, http_answer_fn(args.base_url, args.api_key), gate_threshold=args.gate)
    print(report.model_dump_json(indent=2))
    if not report.certifiable:
        print(
            f"NOT CERTIFIABLE: needs >= {MIN_CERTIFIABLE_ITEMS} items, every one with reviewed_by "
            f"(have {report.items}, reviewed {sum(bool(i.reviewed_by) for i in items)})."
        )
    raise SystemExit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
