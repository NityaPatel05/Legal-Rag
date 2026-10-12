# Legal Contract Clause Analysis RAG

A retrieval-augmented generation system for analyzing clauses in legal contracts —
clause lookup, comparison, and contradiction-checking with **deterministically
validated citations**.

A support-docs bot that's 85% right is useful. A contract-review bot that's 85%
right is a liability. Every design decision here follows from that: **assume
every generated claim is wrong until a code-level check proves it's grounded in
the actual clause text.** The citation validator isn't a nice-to-have layered on
top of generation — it's a first-class pipeline stage with its own test suite.

**Status:** pipeline implemented end to end (ingestion, hybrid retrieval, CRAG,
citation validation, FastAPI, Docker Compose). Run `pytest tests/` for the
current count; the suite includes a round-trip against a real (in-memory)
Qdrant client and HTTP tests of the real app (`tests/app/`). A design review in
2026-09 found and fixed several production-breaking and logic bugs; the
findings, what changed, and the prioritized roadmap (with per-item status)
live in [`CLAUDE.md`](CLAUDE.md). Retrieval is measured on CUAD
(`eval/retrieval/`); the golden-set gate exists but Stage 2 eval still needs
an actual lawyer (see [Known gaps](#known-gaps)).

Full design rationale: [`docs/architecture.md`](docs/architecture.md) (v3 spec).
Context for AI coding sessions: [`CLAUDE.md`](CLAUDE.md).

---

## Table of contents

- [Why this isn't generic RAG](#why-this-isnt-generic-rag)
- [Architecture](#architecture)
  - [Ingestion pipeline](#ingestion-pipeline)
  - [Query pipeline + CRAG evaluator](#query-pipeline--crag-evaluator)
  - [Deployment topology](#deployment-topology)
- [Tech stack](#tech-stack)
- [Project layout](#project-layout)
- [Build order & status](#build-order--status)
- [Known gaps](#known-gaps)
- [Local development](#local-development)
- [Web UI (Streamlit)](#web-ui-streamlit)
- [Deployment (Docker Compose)](#deployment-docker-compose)
- [API reference](#api-reference)
- [Testing](#testing)
- [Evaluation](#evaluation)
- [Guardrails & security](#guardrails--security)

---

## Why this isn't generic RAG

Five decisions that would be unusual in a support-docs RAG system, and why they
exist here:

| Decision | Why |
|---|---|
| **Clause-level chunking**, not token-count chunking | A retriever chunk that straddles two clauses grounds a confident answer in the wrong half of a contract. Validated against CUAD's 510 annotated contracts, not just assumed correct. |
| **`jurisdiction` and `governing_law` as two independent fields** | A contract can say "governed by Delaware law, exclusive jurisdiction in New York courts" — same clause, two different values. Merging them silently produces false negatives on jurisdiction-scoped queries. Applied as two independent filter predicates, never coupled. |
| **Deterministic citation validator, not LLM self-critique** | Three checks per citation: *exists* (the section ID is real), *in-context* (the model actually saw it), *span-supported* (the clause text actually says what the claim says — not just the same topic). The third check is the one most RAG stacks skip. |
| **CUAD as a pre-flight gate, not just a benchmark reference** | Run the clause parser against 510 real contracts *before* touching a private corpus. This caught a real bug — see [below](#evaluation). |
| **Three-way CRAG evaluator with an explicit ambiguous path** | `insufficient` → rewrite and retry (capped at 2). `ambiguous` → flag for attorney review, not a guess. Most CRAG implementations only have two branches. |

## Architecture

### Ingestion pipeline

Offline, once per document. `ingestion/pipeline.py` orchestrates all of it.

```mermaid
flowchart LR
    A[Contract text /<br/>PDF via Docling] --> B[Clause Parser<br/>regex, CUAD-validated]
    B --> C[Defined Terms<br/>extraction]
    B --> D[Section-Reference<br/>graph]
    B --> E[Metadata Tagger<br/>Groq LLM]
    E --> F[Context Enrichment<br/>Groq LLM]
    F --> G[BGE-M3<br/>Embedder]
    G --> H[(Qdrant<br/>one collection)]
    D --> H

    style H fill:#2d5,stroke:#333
```

Every clause point in Qdrant carries the full `ClauseMetadata` payload
(`contract_type`, `clause_type`, `parties`, dates, `jurisdiction`,
`governing_law`, `matter_id`/`tenant_id`, `privilege_level`, `contains_pii`) plus
`start_offset`/`end_offset` for CUAD-span validation. References are stored as
separate points in the *same* collection (`record_type: "reference"` vs.
`"clause"`) with a zero-vector — structured lookup data, not something that
needs semantic search of its own. See
[`ingestion/embedding/indexer.py`](ingestion/embedding/indexer.py) for why (and
for the point-ID subtlety: Qdrant requires int/UUID IDs, so `clause_id` strings
are hashed via `uuid5`, with the real ID kept as a payload field).

### Query pipeline + CRAG evaluator

Online, per question. `query/pipeline.py` handles retrieval (B1–B7);
`query/generation/crag_graph.py` (LangGraph) picks up from there (B8–B15).

```mermaid
flowchart TD
    Q[User Query] --> S[PII Redaction<br/>regex, always on]
    S --> J{Jailbreak<br/>Classifier}
    J -->|flagged| X[Blocked<br/>audit logged]
    J -->|clean| P[Permission Filter<br/>tenant + matter]
    P --> JG[Jurisdiction /<br/>Governing-Law Filter]
    JG --> R[Router]
    R --> H[Hybrid Retrieval<br/>Vector + BM25, RRF fusion]
    H --> RR[Cohere Rerank<br/>top 3-5]
    RR --> CE[Cross-Reference<br/>Expansion, one hop]
    CE --> RO[Long-Context<br/>Reorder]
    RO --> EV{CRAG<br/>Evaluator}
    EV -->|sufficient| GEN[Generate Answer<br/>+ structured citations]
    EV -->|ambiguous| FLAG[Flag for<br/>Attorney Review]
    EV -->|insufficient,<br/>budget left| RW[Rewrite Query]
    EV -->|insufficient,<br/>exhausted| FLAG
    RW --> H
    GEN --> CV{Citation<br/>Validator}
    CV -->|all valid| DONE[Return Answer<br/>+ Verified Citations]
    CV -->|invalid,<br/>budget left| RW
    CV -->|invalid,<br/>exhausted| STRIP[Strip Unsupported<br/>Claims]
    STRIP --> DONE

    style X fill:#c33,stroke:#333,color:#fff
    style FLAG fill:#c93,stroke:#333
    style DONE fill:#2d5,stroke:#333
```

The retry budget (max 2 attempts) is **shared** between the evaluator's
"insufficient → rewrite" branch and the citation validator's "invalid →
re-retrieve" branch — one counter, two triggers, so a query can't retry 2+2
times. Every retry re-runs retrieval *and* the sufficiency check, not just
generation, since a rewritten query can surface a completely different set of
clauses.

The citation validator (`query/validation/citation_validator.py`) checks three
things per citation, deterministically — no LLM call:

1. **Exists** — is the cited section ID real in the source document?
2. **In-context** — was that clause actually shown to the model?
3. **Span-supported** — does the clause's text actually support the specific
   claim? Bag-of-words overlap + a dedicated numeric-token check (catches
   "90 days" cited against a clause that says "30 days" — high word overlap,
   wrong fact) + a negation-marker check (catches polarity flips).

### Deployment topology

```mermaid
flowchart LR
    Client -->|HTTP| APP[app<br/>FastAPI :8000]
    subgraph "docker compose"
        APP
        QD[(qdrant<br/>:6333)]
        EMB[embedding-server<br/>BGE-M3 :8080]
    end
    APP <-->|vector search + upsert| QD
    APP -->|embed| EMB
    APP -->|LLM calls| Groq[Groq API]
    APP -->|rerank| Cohere[Cohere API]
    APP -.->|traces, optional| LangSmith[LangSmith]
```

The embedding server is a separate image on purpose — it's the one service
that needs `torch` + `FlagEmbedding` (~2GB of ML dependencies), so the main
app image stays small and starts fast. It's written in-house (not a
third-party embedding-server image) so both sides of the HTTP contract are
owned by this repo.

## Tech stack

| Layer | Choice | Where |
|---|---|---|
| API | FastAPI + uvicorn | `app/` |
| Orchestration | LangChain + LangGraph (CRAG state machine) | `query/generation/crag_graph.py` |
| Vector DB | Qdrant | `ingestion/embedding/indexer.py`, `query/retrieval/vector_search.py` |
| Embeddings | BGE-M3, self-hosted | `embedding_server/`, `ingestion/embedding/embedder.py` |
| Keyword search | BM25 (`rank_bm25`) | `query/retrieval/bm25_search.py` |
| Reranker | Cohere Rerank | `query/retrieval/reranker.py` |
| Document parsing | Docling | `ingestion/parsing/document_loader.py` |
| LLM provider | Groq (`openai/gpt-oss-120b`) | `ingestion/metadata/llm_client.py` |
| Eval | CUAD (structural) + golden set (lawyer-reviewed) | `eval/` |
| Observability | LangSmith | `query/observability/langsmith_tracer.py` |

Every external integration sits behind a Protocol (`LLMClient`,
`EmbeddingClient`, `VectorStore`, `Reranker`, ...) with the real adapter
lazy-importing its SDK. That's what makes the test suite runnable with zero live
services — every test injects a fake implementing the same interface.

## Project layout

```
ingestion/                    Offline, one-time per document
  parsing/                     Clause parser (CUAD-validated), defined-terms + reference extraction
  metadata/                     Groq LLM client, contract/clause-type + jurisdiction/governing_law tagging
  embedding/                     Context enrichment, BGE-M3 embed, Qdrant index, document_reader
query/                        Online, per user question
  filters/                      Jailbreak classifier, PII sanitizer, permission + jurisdiction filters, audit log
  retrieval/                     Hybrid retrieval (vector + BM25 + RRF), reranking, reference expansion, reorder
  generation/                     Router, structured-citation generation, CRAG evaluator (LangGraph)
  validation/                      Citation validator, strip-unsupported-claims fallback
  observability/                    LangSmith trace export
eval/
  cuad_preflight/               Stage 1: chunk_quality.py (containment + span precision + size vs. model limits)
  retrieval/                     cuad_recall.py — hit@k / span recall@k / char-budget recall on CUAD (R1, R3)
  golden_set/                    Stage 2: golden.py gate + unreviewed example set (lawyer review required)
app/                          FastAPI service — /health, /documents*, /query, /reviews*
ui/                           Streamlit UI (ask / review queue / documents); talks to the API over HTTP only
embedding_server/            Standalone BGE-M3 service, own Dockerfile
tests/
  ingestion/ query/ integration/  unit tests on fakes + real in-memory Qdrant round-trip
  app/                            HTTP tests: real app + in-memory Qdrant via dependency_overrides
  ui/                             Streamlit pages driven by AppTest against the real app; quote-highlight unit tests
  eval/                           eval metric / gate tests
docs/
  architecture.md               Full v3 design spec + as-built notes
Dockerfile · docker-compose.yml · embedding_server/Dockerfile
```

## Build order & status

Per `docs/architecture.md` §11 — all six steps implemented:

- [x] **1. Ingestion** — clause parser + defined-terms/reference extraction +
      Groq metadata tagging (split `jurisdiction`/`governing_law`) + context enrichment
- [x] **2. CUAD pre-flight** — real pass against all 510 contracts, 99.9%
      boundary-containment ([details](#evaluation))
- [x] **3. Query-time baseline** — PII redaction → jailbreak → permission →
      jurisdiction/governing-law → hybrid retrieval → rerank → expansion → reorder
- [x] **4. Citation validator** — exists / in-context / span-supported, with
      numeric-swap and negation-flip checks
- [x] **5. CRAG evaluator** — LangGraph, three-way branch, shared retry budget
- [x] **6. Eval harness** — golden-set schemas/metrics/harness;
      `is_certifiable()` blocks on lawyer sign-off regardless of metrics

Guardrails (CLAUDE.md #8) also implemented: PII redaction, immutable
append-only audit log, full per-query retrieval trace.

**Parked, not built** (per CLAUDE.md's explicit scope): GraphRAG,
Self-Querying Retriever, the recent case-law / external legal authority module.

## Known gaps

- **Missing from the repo despite being described below:** the original CUAD
  pre-flight runner (`run_preflight`, `classification_check.py`).
  `boundary_report.json` exists locally but the code that produced it does
  not; `eval/cuad_preflight/chunk_quality.py` re-implements the boundary
  measurement and adds the metrics it was missing.
- **No lawyer-reviewed golden set.** `eval/golden_set/golden.py` enforces the
  gate (citation accuracy blocking, `is_certifiable()` needs >= 40 reviewed
  items); `golden_set.example.jsonl` is an unreviewed template.
- **Retrieval is measured for BM25 only.** On CUAD (510 contracts, 6,702
  questions, per-document), BM25 hit@5 = 0.69 vs a 0.21 random baseline
  (`eval/retrieval/`). Dense/hybrid numbers need the BGE-M3 server running,
  and LegalBench-RAG hasn't been run — see CLAUDE.md R1/R3.

- **RAGAS faithfulness scoring** and the **CUAD classification-accuracy pass**
  aren't wired to run live — both need a real corpus / API budget to mean
  anything, not synthetic fixtures.
- **~10% of CUAD documents still under-segment** (≤2 clauses, 53/510). Down
  from 36% after R13 (indented headings + a guarded inline pass for flattened
  text); 39 of the 53 use roman-numeral, "(a)"-style or no numbering — see
  `ingestion/parsing/clause_parser.py`'s docstring for the measured before/after.
- **PDF/DOCX upload is wired but unverified** — `POST /documents/file` sends
  `.pdf`/`.docx` through `DoclingLoader`, but Docling isn't installed on the
  dev machine, so that path is tested with a fake loader only (`.txt`/`.md`
  uploads are tested for real). Without Docling the endpoint returns 501.
- **The Streamlit UI has no sign-in.** It calls the API with one server-side
  key (`RAG_API_KEY`), so anyone who can open it acts with that key; it binds
  to localhost for that reason. Real sign-in is R8 (OIDC).
- **Docker Compose wasn't run end-to-end** in the environment this was built
  in (Docker CLI present, daemon not running) — syntactically validated via
  `docker compose config`, but build/run should be verified before relying on it.

## Local development

```bash
cp .env.example .env      # fill in GROQ_API_KEY at minimum
pip install -r requirements.txt
pytest tests/             # no live services needed
```

## Web UI (Streamlit)

```bash
pip install -r ui/requirements.txt
uvicorn app.main:app                  # the API, as above (DEV_MODE=true for local, keyless use)
cd ui && streamlit run streamlit_app.py   # http://localhost:8501
```

| Env var | Meaning |
|---|---|
| `RAG_API_URL` | API base URL (default `http://127.0.0.1:8000`) |
| `RAG_API_KEY` | Sent as `X-API-Key` **from the Streamlit server**, never the browser. Leave unset against a `DEV_MODE` API. |
| `RAG_TENANT_ID`, `RAG_MATTER_ID` | Sidebar defaults |

- **Ask** — one question about one contract. The answer is numbered claims;
  each claim's verbatim quote is highlighted inside the stored clause text
  (located the same way the citation validator matches it). Refusals say why:
  queued for a lawyer (with a status check that doesn't touch the reviewer's
  clock), not found, or blocked / out of scope (rephrase).
- **Review queue** — the lawyer's side of H1/H2: evidence the model saw
  (injection-flagged clauses marked), retrieval trace, a decision per draft
  claim (no approve-all), edits and added claims checked against the
  validator before submitting, override reasons required where the backend
  requires them, 409 when another reviewer decided first.
- **Documents** — the matter's library, and ingest by upload or paste.

Untrusted text (contract clauses, model output) is never rendered as
Markdown — a planted image link would be fetched by the browser — only as
escaped HTML (`ui/render.py`).

## Deployment (Docker Compose)

```bash
cp .env.example .env      # GROQ_API_KEY required; COHERE_API_KEY optional
                           # (reranking degrades to pass-through without it)
docker compose up --build
```

- First `/embed` call downloads the ~2GB BGE-M3 model (3-minute healthcheck
  start period; the first real request through the app may still be slow).
- Qdrant has no compose healthcheck (third-party image, tooling unverified);
  the app retries its own Qdrant connection on startup instead (10× / 3s apart).
- **Verify the build yourself** — see [Known gaps](#known-gaps).

## API reference

### `POST /documents/text`

Ingests already-extracted contract text (parses, tags, embeds, indexes).

```jsonc
// Request
{
  "document_id": "demo-doc",
  "source_text": "1. Governing Law\n\nThis Agreement is governed by the laws of Delaware.",
  "tenant_id": "tenant-1",
  "matter_id": "matter-1",
  "privilege_level": "confidential"   // optional, default "confidential"
}
```

```jsonc
// Response
{
  "document_id": "demo-doc",
  "num_clauses": 1,
  "num_defined_terms": 0,
  "num_references": 0
}
```

### `POST /query`

Runs the full retrieval → generation → citation-validation → CRAG pipeline.

```jsonc
// Request
{
  "document_id": "demo-doc",
  "query_text": "What law governs this Agreement?",
  "tenant_id": "tenant-1",
  "matter_id": "matter-1",
  "jurisdiction": null,       // optional filter, independent of governing_law
  "governing_law": null       // optional filter, independent of jurisdiction
}
```

```jsonc
// Response
{
  "query_text": "What law governs this Agreement?",
  "route": "clause_lookup",
  "final_status": "answered",  // | "answered_with_stripped_claims" | "flagged_for_review"
  "answer_text": "Delaware law governs this Agreement.",
  "citations": [
    { "claim_text": "This Agreement is governed by Delaware law.", "section_id": "1" }
  ],
  "flagged_reason": null,
  "disclaimer": "This is an informational summary, not legal advice, and requires attorney review before being relied upon."
}
```

Returns `404` if the tenant/matter combination has no matching document —
including when a *different* tenant asks about a document that exists for
someone else. That's the permission-first filter, verified through the actual
HTTP surface (`tests/app/`).

### Documents and review endpoints used by the UI

| Endpoint | Purpose |
|---|---|
| `POST /documents/file` | Multipart upload (`file`, `document_id`, `tenant_id`, `matter_id`, `privilege_level`): `.txt`/`.md` (UTF-8), `.pdf`/`.docx` via Docling. 415 wrong type or magic bytes, 413 over 25 MB, 501 without Docling. |
| `GET /documents?tenant_id&matter_id` | The matter's documents with document-level metadata (governing law, parties, privilege, clause count, injection flags). |
| `GET /documents/{id}?tenant_id&matter_id` | Stored clauses in document order — the text citations are validated against — plus defined terms. Audited as `document_read`. |
| `GET /reviews`, `GET /reviews/{id}`, `POST /reviews/{id}/decision` | Human review queue (see CLAUDE.md, R15). `GET /reviews/{id}` is the reviewer opening the case. |
| `GET /reviews/{id}/status?tenant_id&matter_id` | For the asker: status + released outcome, without starting the reviewer's time-to-decision clock. |
| `POST /reviews/{id}/check` | Validator verdict for each proposed decision, without deciding, saving or auditing. |

All are matter-scoped like `/query`; a review outside the caller's scope is a 404.

### `GET /health`

Liveness check, `{"status": "ok"}`.

## Testing

```bash
pytest tests/                          # everything, ~6s, no live services
pytest tests/ingestion/                # clause parsing, tagging, indexing
pytest tests/query/                    # retrieval, CRAG, validation, guardrails
pytest tests/integration/              # real qdrant-client (in-memory) round-trip
pytest tests/ui/                       # Streamlit pages via AppTest against the real app
python -m eval.cuad_preflight.chunk_quality   # 510-contract CUAD chunk-quality pass (needs the dataset)
```

Every test runs against a fake implementing the same Protocol as the real
adapter — no test in this repo needs Groq, Qdrant, Cohere, or Docling actually
running. The FastAPI tests use `dependency_overrides` with the same style of
fakes, exercising the real HTTP layer end-to-end (ingest → query → cited
answer, plus a cross-tenant isolation check).

## Evaluation

### Stage 1: CUAD pre-flight (done, real data)

`eval/cuad_preflight/` runs the clause parser against all 510 CUAD contracts
(13,823 ground-truth annotated spans). Results in
[`reports/boundary_report.json`](eval/cuad_preflight/reports/boundary_report.json):

| Metric | Value |
|---|---|
| Documents | 510 |
| Avg. clauses/document | 33.6 |
| Ground-truth spans | 13,823 |
| Boundary containment | **99.7%** (13,784 / 13,823) |
| Straddled (clause-boundary miss) | 39 |
| Documents still under-segmented (≤2 clauses) | 53 / 510 (10%) |

(Clause level, after R13. `boundary_report.json` predates R13; the current
passage-level numbers are in `reports/chunk_quality_r13.json`.)

This pass caught two real bugs, fixed and re-validated in place rather than
just written up:

1. The first version left over half the corpus as one giant unsegmented
   clause because it only recognized headings alone on their own line — most
   real SEC-filed contracts put the heading and its first sentence on the
   same line. Fixing it: avg clauses/doc 12.0 → 20.7.
2. A second pass added ALL-CAPS-standalone and lettered-subsection heading
   patterns, chosen by checking which gap actually affected more documents
   (79/214 vs. 16/214) before writing either — not by guessing. Lettered
   subsections deliberately get no `section_id`: sampling showed letters like
   "A."/"B." get reused for unrelated enumerations throughout one document,
   which would otherwise collide in the citation lookup table. Result: avg
   clauses/doc 20.7 → 23.1, under-segmented docs 42% → 36%.
3. A third pass (R13) bucketed the 183 remaining under-segmented docs: 64
   indented their headings, and most others were flattened text (newlines
   lost, headings mid-line). Leading whitespace is now allowed, and an inline
   pass — used only when fewer than 3 line headings are found, and only
   accepting an increasing section-number sequence, which rejects
   cross-references — finds mid-line headings. Result: under-segmented docs
   36% → 10%, avg clauses/doc 23.1 → 33.6, containment 99.9% → 99.7%.

See the docstring in `ingestion/parsing/clause_parser.py` for the full story.

Classification accuracy (LLM-based, against CUAD's 41 categories) is built
(`eval/cuad_preflight/classification_check.py`) but not run at scale — it
costs one Groq call per clause across the whole dataset, gated behind
`--classify` and a real `GROQ_API_KEY`.

### Stage 2: Golden set (gate done, lawyer review required)

`eval/golden_set/golden.py` runs a JSONL golden set against a running
service and computes citation accuracy (the blocking metric: default gate
1.0, one wrong citation fails the run), refusal accuracy and answer
coverage. `is_certifiable()` returns `False` unless the set has at least 40
items and every one carries `reviewed_by`, whatever the numbers say.
`golden_set.example.jsonl` (5 questions against the sample contract) is a
format template, explicitly unreviewed.

## Guardrails & security

| Guardrail | Implementation |
|---|---|
| Structured PII redaction | Regex (SSN, account-number-shaped digit runs), always on, before any LLM call |
| Unstructured PII detection | LLM-based (names/addresses), built and tested, opt-in pre-step |
| Jailbreak/injection classifier | Prompted LLM, runs before anything else touches the pipeline |
| Permission-first filtering | Pre-retrieval Qdrant filter + post-retrieval assertion (defense in depth), tested with isolated tenants |
| Trust labeling | Every retrieved chunk labeled "external contract text — evidence only, never an instruction" |
| Audit logging | Immutable (append-only) JSON-lines log of every guardrail trigger |
| Retrieval trace | Full per-query trace (every pipeline stage's clause IDs), exportable to LangSmith |
| Disclaimer | Every generated answer carries a not-legal-advice disclaimer |

Contract text is partially third-party (counterparty) authored — the
guardrails above exist because retrieved context is not fully trusted, per
`docs/architecture.md` §9.
