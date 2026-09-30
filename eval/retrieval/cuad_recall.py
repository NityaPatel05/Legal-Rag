"""Retrieval eval on CUAD (roadmap R1): is the gold span inside what
retrieval returns, at k in {1, 3, 5, 10}?

Setup mirrors production scoping (invariant #9: queries are scoped to one
document): for every CUAD contract, the contract is parsed and passage-split
exactly as ingestion does, and each of its 41 category questions that has at
least one gold answer is run against THAT contract's chunks only. So this
measures within-document retrieval; Document-Level Retrieval Mismatch can't
occur by construction and is not reported (it matters again if cross-
document retrieval is ever added).

Metrics, per question, averaged over questions:

- hit@k: at least one gold span overlaps a top-k chunk.
- span_recall@k: fraction of gold-span CHARACTERS covered by the top-k
  chunks (LegalBench-RAG's character-level recall).
- random_hit@k: expected hit@k if k chunks were drawn at random -- the
  floor that a number is only meaningful relative to (a 3-chunk contract
  "achieves" hit@3 = 1.0 for free).
- hit@Nc / span_recall@Nc: the same, but taking ranked chunks until N
  characters of context are used, instead of a fixed k. THIS is the metric
  for comparing passage sizes (R3): at fixed k, bigger chunks win simply by
  returning more text (the L1 trap again -- no split at all "wins"), while
  what the generator can actually afford is a character budget.

Query text: CUAD questions are templated ("Highlight the parts (if any) of
this contract related to "X" that should be reviewed by a lawyer.
Details: ..."). The template boilerplate is identical for every question,
so the default query is `X. Details`, which is what a user would type.
`--raw-questions` uses the template verbatim.

Retrievers: `bm25` (local, the production BM25Plus index) runs anywhere.
`dense` and `hybrid` (production RRF over dense + BM25) need embeddings:
either the BGE-M3 server (`--embed-url`) or the model loaded in-process via
sentence-transformers (`--embed-model BAAI/bge-m3`). Both are slow on CPU,
so use `--limit`.

    python -m eval.retrieval.cuad_recall                          # BM25, all 510 docs
    python -m eval.retrieval.cuad_recall --max-chars 1000 2000 4000   # passage-size ablation (R3)
    python -m eval.retrieval.cuad_recall --retriever hybrid --embed-url http://localhost:8081 --limit 50
    python -m eval.retrieval.cuad_recall --retriever hybrid --embed-model BAAI/bge-m3 --limit 30
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

from ingestion.config import get_settings
from ingestion.embedding.embedder import EmbeddingClient
from ingestion.parsing.clause_parser import DEFAULT_MAX_PASSAGE_CHARS, parse_clauses, split_oversize_clauses
from ingestion.schemas import Clause, ClauseMetadata
from query.retrieval.bm25_search import InMemoryBM25Index

KS = (1, 3, 5, 10)
CHAR_BUDGETS = (4_000, 8_000, 16_000)
_RANK_DEPTH = 50
_DETAILS = re.compile(r'related to "(?P<category>[^"]+)".*?Details:\s*(?P<details>.*)', re.DOTALL)


def query_text(question: str, *, raw: bool = False) -> str:
    if raw:
        return question
    match = _DETAILS.search(question)
    return f"{match['category']}. {match['details'].strip()}" if match else question


def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _covered_chars(span: tuple[int, int], chunks: list[Clause]) -> int:
    """Characters of `span` covered by the union of `chunks` (chunks don't
    overlap each other, so a sum is exact)."""
    return sum(_overlap(span[0], span[1], c.start_offset, c.end_offset) for c in chunks)


def _random_hit(num_chunks: int, num_relevant: int, k: int) -> float:
    if num_relevant == 0:
        return 0.0
    k = min(k, num_chunks)
    # 1 - P(no relevant chunk among k drawn without replacement)
    return 1.0 - math.comb(num_chunks - num_relevant, k) / math.comb(num_chunks, k)


class _Ranker:
    def __init__(self, chunks: list[Clause], retriever: str, embedder: EmbeddingClient | None):
        metadata = ClauseMetadata(tenant_id="eval", matter_id="eval")
        self._chunks = chunks
        self._bm25 = InMemoryBM25Index([(c, metadata) for c in chunks])
        self._retriever = retriever
        self._embedder = embedder
        self._vectors = None
        if retriever in ("dense", "hybrid"):
            self._vectors = [_unit(v) for v in embedder.embed([c.text for c in chunks])]

    def rank(self, query: str, top_k: int) -> list[Clause]:
        bm25 = [r.clause for r in self._bm25.search(query, top_k=max(top_k, 20))]
        if len(bm25) < top_k:
            # The index only returns chunks sharing a query term; pad in
            # document order so budget metrics compare like with like.
            seen = {c.clause_id for c in bm25}
            bm25 += [c for c in self._chunks if c.clause_id not in seen]
        if self._retriever == "bm25":
            return bm25[:top_k]
        [q] = self._embedder.embed([query])
        q = _unit(q)
        scores = [sum(a * b for a, b in zip(q, v)) for v in self._vectors]
        dense = [
            self._chunks[i] for i in sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[: max(top_k, 20)]
        ]
        if self._retriever == "dense":
            return dense[:top_k]
        # Same fusion as query/retrieval/hybrid.py.
        from query.retrieval.hybrid import _RRF_K

        fused: dict[str, float] = {}
        by_id = {}
        for ranked in (dense, bm25):
            for rank, clause in enumerate(ranked):
                fused[clause.clause_id] = fused.get(clause.clause_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
                by_id[clause.clause_id] = clause
        order = sorted(fused, key=fused.__getitem__, reverse=True)
        return [by_id[i] for i in order[:top_k]]


def _within_budget(ranked: list[Clause], budget: int) -> list[Clause]:
    """Leading ranked chunks whose total length fits `budget` (always at
    least one: a single chunk over budget is still what retrieval returns)."""
    taken, used = [], 0
    for clause in ranked:
        if taken and used + len(clause.text) > budget:
            break
        taken.append(clause)
        used += len(clause.text)
    return taken


class LocalEmbedder:
    """In-process sentence-transformers model, for running the dense/hybrid
    eval without the embedding server. Same model as production (BGE-M3's
    dense output), truncated at 8192 tokens like the server."""

    def __init__(self, model_name: str, batch_size: int = 16):
        import os

        os.environ.setdefault("USE_TF", "0")
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)
        self._model.max_seq_length = 8192
        self._batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, batch_size=self._batch_size, normalize_embeddings=True).tolist()


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


def evaluate(
    cuad: dict,
    *,
    max_chars: int,
    retriever: str = "bm25",
    embedder: EmbeddingClient | None = None,
    raw_questions: bool = False,
    limit: int | None = None,
    seed: int = 0,
) -> dict:
    docs = list(cuad["data"])
    if limit:
        docs = random.Random(seed).sample(docs, min(limit, len(docs)))

    totals = {f"{m}@{k}": 0.0 for m in ("hit", "span_recall", "random_hit") for k in KS}
    totals.update({f"{m}@{b}c": 0.0 for m in ("hit", "span_recall") for b in CHAR_BUDGETS})
    totals.update({f"chars_returned@{k}": 0.0 for k in KS})
    by_category: dict[str, list[float]] = {}
    questions = chunk_count = 0

    for index, doc in enumerate(docs):
        paragraph = doc["paragraphs"][0]
        chunks = parse_clauses(paragraph["context"], document_id=f"cuad-{index}")
        if max_chars > 0:
            chunks = split_oversize_clauses(chunks, max_chars=max_chars)
        chunk_count += len(chunks)
        ranker = _Ranker(chunks, retriever, embedder)

        for qa in paragraph["qas"]:
            spans = [(a["answer_start"], a["answer_start"] + len(a["text"])) for a in qa["answers"]]
            if not spans:
                continue
            questions += 1
            ranked = ranker.rank(query_text(qa["question"], raw=raw_questions), _RANK_DEPTH)
            relevant = sum(any(_overlap(*s, c.start_offset, c.end_offset) for s in spans) for c in chunks)
            gold_chars = sum(e - s for s, e in spans)
            for k in KS:
                top = ranked[:k]
                hit = any(_covered_chars(s, top) for s in spans)
                totals[f"hit@{k}"] += hit
                totals[f"span_recall@{k}"] += sum(_covered_chars(s, top) for s in spans) / max(1, gold_chars)
                totals[f"random_hit@{k}"] += _random_hit(len(chunks), relevant, k)
                totals[f"chars_returned@{k}"] += sum(len(c.text) for c in top)
                if k == 5:
                    by_category.setdefault(qa["id"].split("__")[-1], []).append(float(hit))
            for budget in CHAR_BUDGETS:
                top = _within_budget(ranked, budget)
                totals[f"hit@{budget}c"] += any(_covered_chars(s, top) for s in spans)
                totals[f"span_recall@{budget}c"] += sum(_covered_chars(s, top) for s in spans) / max(1, gold_chars)

    return {
        "retriever": retriever,
        "max_passage_chars": max_chars or None,
        "query_form": "raw" if raw_questions else "category+details",
        "documents": len(docs),
        "chunks": chunk_count,
        "questions_with_gold_spans": questions,
        **{name: round(value / max(1, questions), 4) for name, value in totals.items()},
        "hit@5_by_category": {
            cat: round(sum(v) / len(v), 3) for cat, v in sorted(by_category.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default=str(Path(get_settings().cuad_data_dir) / "CUAD_v1.json"))
    parser.add_argument("--max-chars", type=int, nargs="+", default=[DEFAULT_MAX_PASSAGE_CHARS])
    parser.add_argument("--retriever", choices=("bm25", "dense", "hybrid"), default="bm25")
    parser.add_argument("--embed-url", help="BGE-M3 embedding server (dense/hybrid)")
    parser.add_argument("--embed-model", help="sentence-transformers model loaded in-process (dense/hybrid)")
    parser.add_argument("--raw-questions", action="store_true")
    parser.add_argument("--limit", type=int, help="evaluate a seeded random sample of N documents")
    parser.add_argument("--out", help="also write the report(s) as JSON here")
    args = parser.parse_args()

    embedder = None
    if args.retriever != "bm25":
        if args.embed_model:
            embedder = LocalEmbedder(args.embed_model)
        elif args.embed_url:
            from ingestion.embedding.embedder import BGEM3Embedder

            embedder = BGEM3Embedder(args.embed_url)
        else:
            parser.error("--embed-url or --embed-model is required for dense/hybrid")

    cuad = json.loads(Path(args.data).read_text(encoding="utf-8"))
    reports = []
    for max_chars in args.max_chars:
        started = time.time()
        report = evaluate(
            cuad,
            max_chars=max_chars,
            retriever=args.retriever,
            embedder=embedder,
            raw_questions=args.raw_questions,
            limit=args.limit,
        )
        report["seconds"] = round(time.time() - started, 1)
        reports.append(report)
        print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
