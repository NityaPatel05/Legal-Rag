"""R1: retrieval-eval metrics on a tiny synthetic CUAD-format document."""

import pytest

from eval.retrieval.cuad_recall import _random_hit, evaluate, query_text

CONTEXT = (
    "1. Term\n\nThis Agreement lasts five years from the Effective Date.\n\n"
    "2. Governing Law\n\nThis Agreement is governed by the laws of the State of Delaware.\n\n"
    "3. Insurance\n\nSupplier shall maintain general liability insurance.\n"
)
GOLD = "governed by the laws of the State of Delaware"


def _cuad(question_category="Governing Law", answers=True):
    start = CONTEXT.index(GOLD)
    return {
        "data": [
            {
                "title": "doc",
                "paragraphs": [
                    {
                        "context": CONTEXT,
                        "qas": [
                            {
                                "id": f"doc__{question_category}",
                                "question": f'Highlight the parts (if any) of this contract related to '
                                f'"{question_category}" that should be reviewed by a lawyer. '
                                f"Details: Which state's laws govern the contract?",
                                "answers": [{"text": GOLD, "answer_start": start}] if answers else [],
                            }
                        ],
                    }
                ],
            }
        ]
    }


def test_query_text_strips_template_boilerplate():
    q = _cuad()["data"][0]["paragraphs"][0]["qas"][0]["question"]
    assert query_text(q) == "Governing Law. Which state's laws govern the contract?"
    assert query_text(q, raw=True) == q


def test_bm25_finds_the_gold_clause_first():
    report = evaluate(_cuad(), max_chars=2000)
    assert report["questions_with_gold_spans"] == 1
    assert report["hit@1"] == 1.0
    assert report["span_recall@1"] == 1.0
    assert report["random_hit@1"] == pytest.approx(1 / 3, abs=1e-3)


def test_questions_without_gold_spans_are_not_scored():
    assert evaluate(_cuad(answers=False), max_chars=2000)["questions_with_gold_spans"] == 0


class _KeywordEmbedder:
    VOCAB = ("delaware", "governed", "insurance", "term")

    def embed(self, texts):
        return [[float(w in t.lower()) + 1e-6 for w in self.VOCAB] for t in texts]


def test_hybrid_retriever_runs_with_an_embedding_client():
    report = evaluate(_cuad(), max_chars=2000, retriever="hybrid", embedder=_KeywordEmbedder())
    assert report["hit@1"] == 1.0


def test_random_hit_baseline():
    assert _random_hit(10, 1, 1) == pytest.approx(0.1)
    assert _random_hit(3, 1, 10) == 1.0
    assert _random_hit(10, 0, 5) == 0.0


def test_char_budget_takes_leading_chunks_that_fit():
    from eval.retrieval.cuad_recall import _within_budget
    from tests.query.factories import make_clause

    chunks = [make_clause(f"d::c{i}", text="x" * n) for i, n in enumerate((300, 500, 400))]
    assert [c.clause_id for c in _within_budget(chunks, 800)] == ["d::c0", "d::c1"]
    # A single chunk larger than the budget is still returned.
    assert [c.clause_id for c in _within_budget(chunks, 100)] == ["d::c0"]


def test_report_has_budget_metrics():
    report = evaluate(_cuad(), max_chars=2000)
    assert report["hit@4000c"] == 1.0
