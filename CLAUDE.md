# CLAUDE.md — Legal Contract Clause-Analysis RAG

Context for AI coding sessions (and humans). Read this before changing
anything. `docs/architecture.md` is the original v3 design spec; **where it
disagrees with this file, this file is current** (see "Review findings").

## What this is

Clause-level RAG over legal contracts: clause lookup, comparison and
contradiction-checking, where **every claim returned to the user must be
grounded in contract text by a deterministic, code-level check**. A 85%-right
contract bot is a liability, so the system is designed to refuse
(`flagged_for_review`) rather than guess.

## Commands

```bash
pip install -r requirements.txt
pytest tests/                                   # full suite, no live services (~6s)
pytest tests/query/test_citation_validator.py   # single file while iterating
pytest tests/integration/                       # real qdrant-client, in-memory mode
pytest tests/app/                               # HTTP: real app + in-memory Qdrant, fake LLM/embedder
python -m eval.cuad_preflight.chunk_quality     # CUAD chunk quality (needs CUAD_v1.json in CUAD_DATA_DIR)
python -m eval.retrieval.cuad_recall            # CUAD retrieval hit@k / span recall@k (BM25; --retriever hybrid --embed-url for dense)
python -m eval.golden_set.golden <set.jsonl> --base-url http://localhost:8000   # golden-set gate vs a running service
uvicorn app.main:app --reload                   # needs Qdrant + embedding server + GROQ_API_KEY
docker compose up --build                       # full stack (never verified end to end — see Open issues)
```

Windows note: Python 3.13 (Anaconda) on this machine; tests run fine from Git Bash.

## Map

| Stage | File |
|---|---|
| Clause parsing + passage split | `ingestion/parsing/clause_parser.py` (`parse_clauses`, `split_oversize_clauses`) |
| Defined terms / cross-refs | `ingestion/parsing/defined_terms.py`, `references.py` |
| Document-level metadata + summary | `ingestion/metadata/document_profile.py` |
| Per-clause metadata | `ingestion/metadata/tagger.py`; place-name gazetteer `normalize.py`; clause/contract-type vocabularies `vocabulary.py` |
| Injection scan (ingestion) | `ingestion/metadata/injection_scan.py` |
| Context note | `ingestion/embedding/context_enrichment.py` |
| Index / read back | `ingestion/embedding/indexer.py`, `document_reader.py` |
| Orchestration (ingest) | `ingestion/pipeline.py` |
| Guardrails | `query/filters/` (PII, jailbreak, permission, jurisdiction, audit log) |
| Retrieval | `query/retrieval/` (vector, BM25, RRF hybrid, rerank, expansion, reorder, sibling-passage context `passage_context.py`) |
| Retrieval orchestration | `query/pipeline.py` (`process_query`, B1–B7) |
| Generation + CRAG loop | `query/generation/crag_graph.py` (`run_crag`, B8–B15) |
| Citation validation | `query/validation/citation_validator.py` (+ optional NLI `EntailmentChecker`), `strip_unsupported.py` |
| Human review queue | `query/review/` (`workflow.py` queue + decisions, `store.py` SQLite/in-memory, `schemas.py` evidence packet); routes `/reviews` in `app/main.py` |
| HTTP | `app/main.py`, `app/dependencies.py` (auth + privileged-LLM selection live here) |
| Eval | `eval/retrieval/cuad_recall.py` (R1), `eval/golden_set/golden.py` (R2), `eval/cuad_preflight/chunk_quality.py` |

## Invariants — do not break these

1. **A chunk never crosses a clause boundary.** Oversize clauses are split into
   passages *inside* the clause (`parent_clause_id` set); a passage is a
   sub-range of exactly one clause.
2. **Citations are keyed by `clause_id`**, never by `section_id` alone.
   `section_id` is `None` for preambles, ALL-CAPS headings and lettered
   sub-clauses, and is not unique (exhibits restart numbering). A bare
   section_id is accepted only if it resolves to exactly one clause.
3. **Every claim carries a verbatim `supporting_quote`**, checked by string
   match against the cited clause. Numbers/negation are checked against the
   quote's enclosing sentence, not the whole clause.
4. **Only validated claims reach the user.** `QueryResponse.answer_text` is
   composed from kept claims (`final_answer_text`); the model's free-text
   `answer_text` is never returned. Zero claims ⇒ `flagged_for_review`.
5. **`jurisdiction` and `governing_law` are separate fields**, extracted and
   filtered independently, stored **normalized** (`normalize_place`), and
   are **document-level** (merged onto every clause from the document profile;
   a clause-level value always wins).
6. **Permission-first.** Every Qdrant read filters `tenant_id` + `matter_id`.
   Vector-store results are *not* silently post-filtered — a mismatch means the
   primary control failed and must raise `PermissionViolation` (audited). BM25
   hits (no native filter) are post-filtered with the same predicate list.
7. **Tenant and matter identity come from the credential**, not the request
   body, whenever `API_KEYS` is configured (`app/dependencies.authorize_request`;
   a scoped key lists its matters, deny by default).
8. **Point IDs are scoped by tenant+matter** (`uuid5(tenant, matter, clause_id)`),
   and re-ingest is delete-then-write.
9. **Queries are scoped to one `document_id`** (`QueryFilters.document_id`) —
   BM25, the validator's `exists` check and expansion are all per-document.
10. **Retry budget is shared** (`MAX_RETRIES=2`) across "insufficient →
    rewrite+retrieve" and "citation invalid → regenerate with feedback".
11. **LLM output is untrusted input**: parse defensively (bad dates → `None`,
    malformed claims → empty citation that fails validation, never a 500).
12. **What leaves the boundary.** Structured PII is redacted from every prompt
    to the default provider (`RedactingLLMClient`, ingestion and query) — stored
    text is never redacted, citations are verbatim against it.
    `privilege_level=privileged` text only ever goes to `PRIVILEGED_LLM_BASE_URL`
    (self-hosted); without one, privileged docs ingest with structural metadata
    only and `/query` on them returns 403.
13. **Metadata vocabularies are closed.** `clause_type` / `contract_type` are
    always a label from `vocabulary.py` (or `None`); place names always go
    through `normalize_place` at both ingest and query time.
14. **Auth fails closed.** Empty `API_KEYS` ⇒ the app refuses to start unless
    `DEV_MODE=true`, and dev mode serves loopback clients only (checked per
    request against the client address; 503/403 otherwise).
15. **A reviewer can't release an ungrounded claim silently.** Review
    decisions (`query/review/workflow.py`) re-run the citation validator on
    every approved/edited claim against the document as stored *now*; a
    failing claim needs an `override_reason` and is audited as
    `review_override`. Every draft claim needs its own decision (no approve-all).

## Conventions

- Every external service sits behind a `Protocol` (`LLMClient`, `EmbeddingClient`,
  `VectorStore`, `VectorSearchClient`, `BM25Index`, `Reranker`, `TraceExporter`,
  `AuditLogger`); real adapters lazy-import their SDK inside methods.
- Tests use hand-written fakes that implement the Protocol. **Fakes cannot catch
  SDK API drift** — anything touching Qdrant also needs a case in
  `tests/integration/` using `QdrantClient(":memory:")`.
- Module docstrings explain *why* (with measured numbers where they exist).
  Keep that style; update the docstring when the reasoning changes.
- Fake LLM clients dispatch on system-prompt phrases. Keep these phrases stable
  or update the fakes: `"legal contract analysis assistant"` (generator),
  `"enough to answer"` (sufficiency), `"Rewrite the question"` (rewriter),
  `"context note"` (enrichment).

## Configuration added in the review (add to `.env.example`)

`.env.example` couldn't be edited from the review session (permission-denied
path); these settings exist in `ingestion/config.py`:

```
COHERE_RERANK_MODEL=rerank-v3.5
RERANK_ENABLED=false         # default off (A6); turn on only if the R3 ablation shows a gain
API_KEYS={}                  # JSON {"key": {"tenant_id": "t", "matters": ["m"]}} (scoped, deny by default)
                             #   or legacy {"key": "tenant_id"} (tenant-wide); empty = app won't start unless DEV_MODE
DEV_MODE=false               # true + empty API_KEYS = unauthenticated, loopback clients only (A5). Useless
                             #   against docker compose (host traffic arrives from the bridge IP) -- set API_KEYS there.
REVIEW_DB_PATH=./reviews/reviews.sqlite3   # human review queue (H1); must persist across restarts
MAX_PASSAGE_CHARS=1000       # goes with GENERATION_TOP_K — change together (R3)
GENERATION_TOP_K=10
PRIVILEGED_LLM_BASE_URL=     # OpenAI-compatible self-hosted endpoint for privileged docs (R9); empty = no LLM for them
PRIVILEGED_LLM_MODEL=
PRIVILEGED_LLM_API_KEY=
NLI_MODEL=                   # e.g. cross-encoder/nli-deberta-v3-base; empty = entailment layer off (R6)
NLI_ENTAILMENT_THRESHOLD=0.5
```

---

## Review findings (2026-09-23)

Full review of every source file, checked against the installed libraries,
the local CUAD data and the literature. **F** = fixed in this pass, **O** = open
(see Roadmap).

### Production-breaking (all fixed; the fake-only test suite hid every one)

| # | Finding | Evidence | Status |
|---|---|---|---|
| B1 | Vector search called `QdrantClient.search`, which **doesn't exist** in the installed qdrant-client 1.18 → every `/query` raised `AttributeError`. | `hasattr(QdrantClient(":memory:"), "search") == False` | F → `query_points` |
| B2 | Reference points (zero vector, same collection) **are returned by vector search** with score 0.0; their payload has no `clause_id` → `KeyError`. Hits any small matter (< top_k clauses). | reproduced in-memory | F → `must_not record_type` |
| B3 | Point id = `uuid5(clause_id)`, and `clause_id` derives from the caller-chosen `document_id`: two tenants ingesting `contract-1` **overwrite each other's points**. | — | F → tenant/matter-scoped ids |
| B4 | Re-ingesting a revised contract with fewer clauses left the old tail **retrievable and citable**. | — | F → `delete_document` first |
| B5 | 7 tests failed on a clean checkout: `tests/ingestion/fixtures/sample_contract.txt` was never committed. README claimed "163 passing"; actual was 122/129. | `pytest` | F → fixture recreated |

### Security

| # | Finding | Status |
|---|---|---|
| S1 | **No authentication.** `tenant_id`/`matter_id` came from the JSON body, so "permission-first filtering" enforced whatever identity the caller typed. | F → `API_KEYS` with matter-scoped keys (`authorize_request`, deny by default); **O**: real IdP (R8) |
| S2 | Vector search had no `document_id` filter while BM25/validator/expansion were per-document → another contract's clause could be shown and **cited as this contract's**. | F |
| S3 | BM25 hits bypassed the jurisdiction/governing-law filters (only tenant/matter re-checked). | F |
| S4 | PII guardrail covers queries only; **full contract text (with PII) is sent to Groq at ingestion** (tagging, enrichment, profile). Contradicts the "never reaches a third-party model" claim. | F (structured PII + privileged; invariant #12) → `RedactingLLMClient`, self-hosted route; **O**: unstructured PII (names/addresses) still reaches Groq for non-privileged docs |
| S5 | Indirect prompt injection: contract text is counterparty-authored; the only defense is a prompt-level trust label. No ingestion-time scan. | F (signal) → `injection_scan.py`: flagged in payload, warning in prompt, trace + audit; **O**: guard model |
| S6 | Audit log is append-only, not tamper-evident. | **O** (low) — hash-chain lines |

### Approach / logic faults

| # | Finding | Evidence | Status |
|---|---|---|---|
| L1 | **"99.9% boundary containment" rewards under-segmentation**: one giant clause contains every span. The metric hid that the median annotated span was **2%** of its chunk, **97/510 docs had a clause beyond BGE-M3's 8,192-token limit** (tail silently truncated), max ~48k tokens. | `eval/cuad_preflight/chunk_quality.py` | F → passage split: precision 2.0% → **13.2%**, 0 chunks over limit; containment 99.86% → 98.44% (long spans now cross *intra-clause* passage cuts — see R4) |
| L2 | `governing_law`/`jurisdiction`/`contract_type`/`parties` tagged **per clause**, so only the governing-law clause carried "Delaware" → a `governing_law` filter hid every other clause of that contract. | — | F → document profile, merged down |
| L3 | Filter values were raw LLM strings; exact-match filters never matched "State of Delaware" vs "Delaware". | — | F → `normalize_place` + gazetteer (R7) |
| L4 | Citation key was `section_id`: `None` for preamble/ALL-CAPS/lettered clauses (prompt rendered `[Section None]` → uncitable), duplicates collided. | — | F → `clause_id` |
| L5 | Span check compared negation/numbers against the **whole clause**: correct claims rejected when "not" appears elsewhere; wrong numbers accepted when they appear elsewhere; "ninety days" vs "thirty (30) days" **passed** (digits only). | tests in `tests/query/test_review_fixes.py` | F → verbatim quote + sentence scope + number words |
| L6 | **Stripped claims still reached the user**: `answer_text` returned unfiltered after stripping. | — | F → composed from kept claims |
| L7 | Zero-claim answers returned as `answered` (uncited text passes trivially). | — | F → flagged |
| L8 | Citation failure triggered a **retrieval rewrite** ("search didn't return enough") — wrong diagnosis; discards good evidence. | — | F → regenerate with validator feedback |
| L9 | Router output was computed and **ignored**: out-of-scope queries answered; contradiction checks flagged as "ambiguous" because clauses conflict (the thing asked). | — | F |
| L10 | Defined terms extracted but **never used at query time** (architecture §1.2 unimplemented); regex missed curly quotes and the common `(the "Company")` form. | — | F → expansion + patterns |
| L11 | Context note generated from the clause alone ("inferable from the heading") — not Contextual Retrieval, which needs document context. | Anthropic 2024 | F (summary + outline); **O**: contextual BM25 (R5) |
| L12 | BM25: whitespace tokenization ("Delaware." ≠ "delaware"); `BM25Okapi` IDF ≤ 0 for terms in ≥ half of a small contract's clauses → matches dropped by `score > 0`. | reproduced | F → regex tokenizer + BM25Plus + overlap filter |
| L13 | Reference expansion: "Section 5.2" missed when only "5" was parsed; plurals/lists ("Sections 5 and 7") ignored; `by_section_id` last-write-wins on duplicates; no expansion cap. | — | F |
| L14 | `document_reader` returned clauses in UUID order, not document order. | — | F |
| L15 | Cohere reranker assumed beneficial; **LegalBench-RAG found it hurt** legal retrieval. | arXiv:2408.10343 | Made switchable (`RERANK_ENABLED`); **O**: decide by ablation (R3, needs a Cohere key) |
| L16 | LLM client: no retries/backoff/timeouts; ~2 calls per clause guarantees free-tier 429s mid-ingest. Invalid LLM dates crashed ingestion. | — | F |
| L17 | Embedding server forced fp16 (bad on CPU), no `max_length`/batch control. | — | F |
| L18 | Docs claimed files that don't exist: `eval/golden_set/`, CUAD pre-flight runner, `tests/eval/`, `tests/app/`, and this CLAUDE.md. | — | F → `eval/retrieval/`, `eval/golden_set/`, `tests/eval/`, `tests/app/` built; pre-flight runner still absent |
| L19 | Negation check is scoped to the quote's whole sentence, so a correct claim is **rejected** when an unrelated clause of the same sentence is negated: "Either party may terminate … for material breach" fails on "…, and termination does **not** affect …" (fixture §3). Deliberate trade-off (stops a claim dropping a "not" just outside the quote), but it costs recall. Candidate fix: scope negation to the quote's own clause segment (split on `, and` / `;`), or let the R6 NLI layer own negation. | `tests/app` found it | **O** |

---

## Roadmap (priority order)

**R1 — Retrieval eval before any more retrieval features.** Nothing here has a
measured recall@k. Build `eval/retrieval/`: CUAD QA (Hugging Face
`theatticusproject/cuad-qa`) → for each question, is the gold span inside a
retrieved chunk at k ∈ {1,3,5,10}? Then run **LegalBench-RAG**
(github.com/zeroentropy-ai/legalbenchrag — CUAD + MAUD + ContractNLI + PrivacyQA,
character-span ground truth) for comparability with published numbers. Report
Document-Level Retrieval Mismatch rate too.
*Status 2026-09-23:* CUAD part **done** — `eval/retrieval/cuad_recall.py`
(uses local `CUAD_v1.json`, same data as cuad-qa). BM25, all 510 docs, 6,702
questions with gold spans, after R13's parser change, 2000-char passages:
hit@1 .426 / @5 .690 / @10 .801 (random @5 .205); span recall@5 .637.
Weakest categories at hit@5: Volume Restriction, Document Name, Parties,
Exclusivity (< .45). Reports in `eval/retrieval/reports/`. **Open:** dense
and hybrid (the production path) need the BGE-M3 server running
(`--retriever hybrid --embed-url ...`); LegalBench-RAG not run. DRM isn't
applicable while queries are per-document (invariant #9).

**R2 — Golden set + generation eval.** Restore `eval/golden_set/` (40–80
lawyer-reviewed questions; citation accuracy as a blocking metric;
`is_certifiable()` refuses without `reviewed_by`). Add a `tests/app/` suite
using `dependency_overrides`.
*Status:* harness and `tests/app/` **done** — `eval/golden_set/golden.py`
(gate 1.0 on citation accuracy; `is_certifiable` needs >= 40 reviewed items)
and HTTP tests on the real app with in-memory Qdrant. **Open:** the
lawyer-reviewed questions themselves (the example set is unreviewed).

**R3 — Ablations (only after R1).** Toggle and measure: reranker on/off
(LegalBench-RAG says off may win); passage size 1000/2000/4000; summary prefix
on/off; defined-term expansion on/off; RRF k. Keep what the numbers support.
*Status:* passage size measured (BM25). **Compare at a fixed character
budget, not fixed k** — at fixed k bigger chunks win just by returning more
text (random hit@5 rises .14 → .38 from 1000 → no split). At equal context:

| passages | hit@4k chars | @8k | @16k |
|---|---|---|---|
| 1000 (current) | **.641** | **.747** | **.835** |
| 2000 (previous) | .584 | .710 | .827 |
| 4000 | .525 | .669 | .802 |
| no split | .586 | .660 | .761 |

**Switched 2026-09-24** to 1000-char passages × `GENERATION_TOP_K=10`
(`DEFAULT_MAX_PASSAGE_CHARS`, `Settings.max_passage_chars`,
`Settings.generation_top_k` → `rerank_top_k`): at about the same context
(~7.2k vs ~6.5k chars), hit .741 vs .690 and span recall .681 vs .637 against
the previous 2000 × 5. Costs, measured at 1000: boundary containment 98.4% →
**95.2%** (more spans cross a passage cut; R4 puts the neighbor passage back
in front of the model, and the hit / recall numbers above already count
overlap), span-to-chunk precision median 14.7% → 25.6%
(`eval/cuad_preflight/reports/chunk_quality_1000.json`), and 57% more chunks
(41,858 vs 26,600), so 57% more ingestion LLM calls (R12's single call per
chunk more than offsets it). **Decided on BM25 only** — an in-process
BGE-M3 hybrid run was killed for low memory on this machine; re-run
`--retriever hybrid` when the embedding server is up and revert (both
settings) if hybrid disagrees. Also measured: stripping CUAD's question template helps BM25
(hit@5 .690 vs .638 raw). Reranker, summary prefix, expansion and RRF k
need the embedding server / Cohere — still open.

**R4 — Small-to-big at generation.** Retrieve passages, but hand the generator
the neighboring passages of the same `parent_clause_id` (bounded), so a span
cut by a passage boundary is still fully visible. Recovers the 1.4pp
containment lost in L1 without giving back precision.
*Status:* **done** — `query/retrieval/passage_context.py`, after reorder, ±1
passage, max 4 added, neighbors placed around their hit in document order;
also on the CRAG rewrite path. Trace stage `passage_context`.

**R5 — Server-side sparse retrieval.** Replace the per-request in-memory BM25
(rebuilt from a full scroll every query, per-document only) with BGE-M3's
sparse output or Qdrant BM25 as a named sparse vector, fused server-side via
`query_points(prefetch=[dense, sparse], query=FusionQuery(RRF))`. Removes the
BM25 permission caveat entirely and enables contextual BM25 (index
`summary + context_note + text`, per Anthropic).

**R6 — Entailment layer in the validator.** After the verbatim-quote check,
run an NLI model on (quote → claim) to catch paraphrases that reuse the
quote's words but change meaning (swapped obligor, "may" vs "shall"). Keep it
deterministic (fixed model, fixed threshold). Candidate: a DeBERTa-v3 NLI
cross-encoder; validate on ContractNLI.
*Status:* hook **done**, off by default — `EntailmentChecker` Protocol and
`CrossEncoderEntailmentChecker` (label index read from the model's
`id2label`), enabled by `NLI_MODEL`. Smoke-tested with
`cross-encoder/nli-deberta-v3-xsmall`: rejects the swapped obligor, but
**misses "may" → "must"** — so **Open:** pick the model and threshold on
ContractNLI before enabling it.

**R7 — Controlled vocabularies.** Map `clause_type` to CUAD's 41 categories
(the spec says so; the prompt doesn't enforce it) and `contract_type` to
CUAD's 25 types; add a jurisdiction gazetteer (US states / ISO 3166) so
"NY" = "New York" and hierarchy ("New York, New York") resolves.
*Status:* **done** — `vocabulary.py`: clause_type = CUAD's 41 (checked
against `CUAD_v1.json`) + 23 general types (CUAD's categories are review
flags, not a taxonomy — confidentiality/indemnity/payment have none);
contract_type = CUAD's 25 + NDA/employment/purchase/lease/loan, keyword
mapping checked against the 510 CUAD titles. Both prompted AND mapped in
code; unknown → "Other". Gazetteer in `normalize.py` (US states + DC,
abbreviations, common foreign units; two different states → no guess).
Changed values ("n.y." → "new york") need a re-ingest.

**R8 — Real authorization.** API-key → tenant is a stopgap. Use an IdP (OIDC),
derive tenant *and* allowed matters from the token, and deny by default.
*Status:* interim **done** — matter-scoped API keys, deny by default
(`authorize_request`). **Open:** OIDC.

**R9 — Ingestion-side PII.** Run structured redaction (and optionally the LLM
PII pass or a self-hosted model) before any text leaves for Groq, or move
ingestion LLM calls to a self-hosted model for privileged matters
(`privilege_level=privileged` should never leave the boundary).
*Status:* **done** — invariant #12. **Open:** unstructured PII (names,
addresses) for non-privileged documents; the LLM PII pass would itself send
the text to Groq, so it needs the self-hosted model.

**R10 — Injection scanning at ingestion.** Flag clauses containing
instruction-like text ("ignore previous", "you are", role markers) at
ingestion, store the flag in payload, and surface it in the trace; consider a
dedicated guard model for the jailbreak classifier (spec §8 recommends one;
the prompted-LLM classifier shares fate with the generator's provider).
*Status:* scan **done** — `injection_scan.py`; 2 / 23,150 CUAD clauses
flagged (both benign), deliberately specific patterns (bare "you are" /
"act as" are ordinary contract English). Flag → payload, prompt warning,
`injection_flagged` trace stage, audit event; the clause is never dropped.
**Open:** guard model for the query-time jailbreak classifier.

**R11 — Version awareness (spec §1.4, unimplemented).** `supersession_links`
is always `[]`. Model amendments/versions: `effective_date`, `supersedes`,
and a comparison route that retrieves across versions of one agreement.

**R12 — Throughput.** Merge per-clause tag + enrich into one LLM call (halves
ingestion calls); batch clauses per call; async ingestion job instead of a
synchronous `POST /documents/text`; expose Docling upload.

**R13 — Clause parser residuals.** 183/510 CUAD docs (36%) still parse to
≤ 2 clauses; passage splitting keeps them retrievable but uncitable by
section. Sample those docs, bucket the causes (Docling/markdown headings,
`Section 1.01` forms, tabular exhibits), fix the largest bucket, re-run
`chunk_quality`.

**R14 — Verify Docker Compose end to end**, then add a smoke test
(ingest fixture → query → assert a validated citation).

**R15 — Architectural review items (2026-09-23).** Full rationale, comparison
against other repos and the research are in `docs/architecture-review.md`. In
order:
A5 fail-closed auth (refuse to start with empty `API_KEYS` unless
`DEV_MODE` + loopback); A6 `RERANK_ENABLED` default **off** until R3 shows a
gain; **H1/H2 human review workflow** (`flagged_for_review` currently ends
at `END`: add `query_id`, LangGraph checkpointer + `interrupt()`, `/reviews`
endpoints, evidence packet, per-claim approve/edit/reject, edits re-validated,
overrides audited with a reason); A3 version-stable citations
(`document_version` content hash, keep superseded versions); A2 + H3 + H5
unanswerable/injection release gates, sampled audit of `answered`, reviewer
decisions → golden set; H4 metadata review (`profile_status`); A4 matter-wide
search route; A7 dedicated guard model.
*Status 2026-09-24:* **A5, A6, H1, H2 done.** A5: `check_auth_configured`
at startup + per-request check in `authorize_tenant` (invariant #14). A6:
`rerank_enabled=False`. H1: `/query` returns `query_id`; flagged and
stripped outcomes that reached retrieval are queued as a `ReviewCase`
(`review_id` = `query_id`) with the evidence packet (question, route, trace,
shown clauses with injection flags, draft claims with quote + validator
`failure_reason`, failures first); `GET /reviews`, `GET /reviews/{id}`,
`POST /reviews/{id}/decision`, all matter-scoped (out-of-scope case = 404).
H2: per-claim decisions required, edits/approvals re-validated, overrides
need a reason and are audited, time from first view to decision recorded.
**Deviation:** no LangGraph `interrupt()`/checkpointer — the CRAG state
carries live clients that can't be checkpointed, and approve/edit/reject
don't need the model; the persisted `ReviewCase` is the state (rationale in
`query/review/workflow.py`). **Open:** reviewer identity is self-asserted
(needs R8); "re-retrieve with hint" is a new `/query`, not linked to the
case; seeded known-bad claims for catch rate (H2); no reviewer role separate
from API-key holders. Next in order: A3.

## Research basis

- LegalBench-RAG — Pipitone & Alami 2024, arXiv:2408.10343. Span-level legal
  retrieval benchmark; general-purpose reranker hurt; precise small spans
  beat large chunks.
- Summary-Augmented Chunking / Document-Level Retrieval Mismatch — Reuter et
  al. 2025, arXiv:2510.06999. DRM > 95% on NDAs; a ~150-char generic document
  summary per chunk roughly halves it; generic summaries beat expert-guided.
- Contextual Retrieval — Anthropic 2024. Context note generated with the whole
  document; prepend for both embeddings and BM25; −35% / −49% / −67% retrieval
  failures (embeddings / +BM25 / +rerank).
- Magesh et al., "Hallucination-Free?", J. Empirical Legal Studies 2025.
  Commercial legal RAG tools still hallucinate 17–33%.
- ALCE (Gao et al. 2023) and verbatim-evidence citation (FullCite-style) —
  claim-level attribution with NLI; the basis for the quote + entailment
  design of the validator.
- CUAD — Hendrycks et al. 2021, arXiv:2103.06268. 510 contracts, 41 categories,
  CC BY 4.0.
- Comparable open-source projects (for ideas, not code): ClauseLens-RAG,
  sangowu/lex-rag, A-Self-Correcting-Legal-Research-System, mazenDDr/Contract-rag
  (189-setup search ablation on CUAD — the kind of evidence R3 needs).
