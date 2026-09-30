"""CUAD chunk-quality report: boundary containment AND the numbers that
containment alone hides.

Why containment isn't enough: "fraction of annotated spans that fall inside
a single chunk" is maximized by NOT splitting at all -- a document parsed as
one giant clause contains 100% of its spans. So it must be read alongside:

- span-to-chunk precision: annotated span length / containing chunk length.
  How much of what retrieval returns is the thing that was asked about
  (LegalBench-RAG grades retrieval at exactly this character-span level).
- chunk-size distribution vs. model limits: BGE-M3 truncates input past
  8192 tokens; a clause longer than that is partially unembeddable.

Usage (CUAD_v1.json from https://www.atticusprojectai.org/cuad in
CUAD_DATA_DIR, default ./eval/cuad_preflight/data):

    python -m eval.cuad_preflight.chunk_quality                 # clauses + passage split
    python -m eval.cuad_preflight.chunk_quality --max-chars 0   # clauses only (no split)

Token counts are approximated as chars / 4.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from ingestion.config import get_settings
from ingestion.parsing.clause_parser import (
    DEFAULT_MAX_PASSAGE_CHARS,
    parse_clauses,
    split_oversize_clauses,
)

_CHARS_PER_TOKEN = 4
_BGE_M3_MAX_TOKENS = 8192


def _percentile(sorted_values: list[float], p: float) -> float:
    return sorted_values[int(p * (len(sorted_values) - 1))]


def chunk_quality_report(cuad: dict, *, max_chars: int) -> dict:
    sizes: list[float] = []
    precisions: list[float] = []
    spans = contained = 0
    under_segmented = docs_over_limit = 0

    for i, doc in enumerate(cuad["data"]):
        paragraph = doc["paragraphs"][0]
        context = paragraph["context"]
        clauses = parse_clauses(context, document_id=f"cuad-{i}")
        if len(clauses) <= 2:
            under_segmented += 1
        if max_chars > 0:
            clauses = split_oversize_clauses(clauses, max_chars=max_chars)

        doc_sizes = [len(c.text) / _CHARS_PER_TOKEN for c in clauses]
        sizes.extend(doc_sizes)
        if max(doc_sizes) > _BGE_M3_MAX_TOKENS:
            docs_over_limit += 1

        for qa in paragraph["qas"]:
            for answer in qa["answers"]:
                start = answer["answer_start"]
                end = start + len(answer["text"])
                spans += 1
                for clause in clauses:
                    if clause.start_offset <= start and end <= clause.end_offset:
                        contained += 1
                        precisions.append(len(answer["text"]) / max(1, len(clause.text)))
                        break

    sizes.sort()
    return {
        "max_passage_chars": max_chars or None,
        "documents": len(cuad["data"]),
        "chunks": len(sizes),
        "documents_under_segmented_by_parser": under_segmented,
        "spans": spans,
        "boundary_containment": round(contained / spans, 4),
        "span_to_chunk_precision_median": round(statistics.median(precisions), 4),
        "span_to_chunk_precision_mean": round(statistics.mean(precisions), 4),
        "chunk_tokens_median": round(_percentile(sizes, 0.5)),
        "chunk_tokens_p90": round(_percentile(sizes, 0.9)),
        "chunk_tokens_p99": round(_percentile(sizes, 0.99)),
        "chunk_tokens_max": round(sizes[-1]),
        "chunks_over_bge_m3_limit": sum(s > _BGE_M3_MAX_TOKENS for s in sizes),
        "documents_with_chunk_over_bge_m3_limit": docs_over_limit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_PASSAGE_CHARS)
    parser.add_argument("--data", default=str(Path(get_settings().cuad_data_dir) / "CUAD_v1.json"))
    args = parser.parse_args()

    cuad = json.loads(Path(args.data).read_text(encoding="utf-8"))
    print(json.dumps(chunk_quality_report(cuad, max_chars=args.max_chars), indent=2))


if __name__ == "__main__":
    main()
