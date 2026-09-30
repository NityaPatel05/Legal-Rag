# Legal Contract Clause Analysis RAG — Architecture & Best-Practices Doc (v3)

> **Changelog from v2** (changes marked with 🔶 throughout):
> 1. Split `jurisdiction` and `governing_law` into two distinct metadata fields (were previously conflated).
> 2. Added **CUAD (Contract Understanding Atticus Dataset)** as the standard external benchmark for validating clause-level chunking and retrieval precision, ahead of/alongside the lawyer-reviewed golden set.
> 3. Confirmed `clause_type` and `contract_type` as required metadata fields, mapped explicitly to CUAD's 41-category taxonomy where possible, for consistency with an established standard rather than an ad-hoc vocabulary.
> 4. Added a CUAD-based pre-flight validation step to the build order, before the private-corpus eval pass.

---

## Implementation status (as-built)

Everything below this line is the original design spec, written before
implementation and kept intact as the design rationale — it's still why
things are the way they are. This section is the delta: what actually got
built, where it lives, and where reality diverged from the plan. See
[`README.md`](../README.md) for the user-facing overview, diagrams, and API
reference; this section is the terse implementation index.

**All six build-order steps (§11) are implemented.** A 2026-09 design
review changed several of the decisions below (citation key, span check,
document-level metadata, passage splitting, retry semantics) — see
`CLAUDE.md` "Review findings" for what changed and why; where this spec and
CLAUDE.md disagree, CLAUDE.md is current. File mapping:

| Spec section | Lives in |
|---|---|
| §1.1 Clause-level chunking | `ingestion/parsing/clause_parser.py` — regex-based, CUAD-validated (99.9% boundary containment on all 510 contracts) |
| §1.2 Defined terms + cross-references | `ingestion/parsing/defined_terms.py`, `references.py` (regex, not LLM — see file docstrings for the tradeoff) |
| §1.3 Citation validation | `query/validation/citation_validator.py` — exists/in-context/span-supported, plus numeric-swap and negation-flip checks added after testing revealed pure word-overlap missed them |
| §1.7 jurisdiction/governing_law split | `ingestion/schemas.py` (`ClauseMetadata`), `query/filters/jurisdiction_governing_law.py` |
| §3 Ingestion pipeline (A1-A7) | `ingestion/pipeline.py` |
| §3 Query pipeline (B1-B7) | `query/pipeline.py` |
| §3 CRAG evaluator (B8-B15) | `query/generation/crag_graph.py` — LangGraph, as recommended in §8 |
| §5 Stage 1 (CUAD pre-flight) | `eval/cuad_preflight/` — real pass, results in `reports/boundary_report.json` |
| §5 Stage 2 (golden set) | `eval/golden_set/` — harness built, **lawyer review still required**, see its own README |
| §9 Guardrails | `query/filters/pii_sanitizer.py`, `audit_log.py`, `jailbreak_classifier.py`, `permission.py` |
| §11 Build order | All six steps done — see README's checklist |

**Not in the original spec, added during implementation:**

- **`app/` (FastAPI) and Docker Compose deployment.** The spec describes the
  pipeline logic but not a runnable service — nothing wired the library code
  to an actual HTTP entry point until this was added. `app/dependencies.py`
  wires real adapters from `ingestion/config.py` Settings; every optional
  provider (Cohere, LangSmith) degrades gracefully rather than hard-failing
  when unconfigured.
- **`embedding_server/`** — a standalone BGE-M3 service. §8 calls BGE-M3
  "self-hosted" but doesn't specify how; this repo owns both sides of the
  HTTP contract (`ingestion/embedding/embedder.py` client,
  `embedding_server/main.py` server) rather than depending on a third-party
  embedding-server image's exact API shape.
- **Qdrant point-ID fix.** Qdrant requires point IDs to be an unsigned
  integer or UUID — not an arbitrary string. `clause_id` (e.g.
  `"doc-1::clause-000"`) is neither. Every test passed anyway, since none of
  them exercise a real Qdrant server. Fixed via deterministic `uuid5(clause_id)`
  point IDs, with `clause_id` kept as its own payload field. See
  `ingestion/embedding/indexer.py`.
- **Reference-graph persistence.** The spec's A4 "section-reference graph"
  is computed at ingestion but the spec doesn't say where it's stored for
  query-time use across separate requests. Implemented as a second
  `record_type` ("reference" vs. "clause") of point in the *same* Qdrant
  collection, with a zero-vector — no second storage technology introduced.
  `ingestion/embedding/document_reader.py` reconstructs both types from one
  scroll at query time.
- **Groq as the LLM provider.** §8's recommended stack doesn't name a
  specific LLM provider for the prompted-LLM steps (metadata tagging,
  jailbreak classification, generation, etc.) — Groq (`openai/gpt-oss-120b`)
  was chosen for free-tier hosted inference; see `CLAUDE.md`'s "Tech stack"
  table for the current-scope decision.

**Known gaps** (also in README): RAGAS faithfulness scoring and the CUAD
classification-accuracy pass are built but not run live (need a real
corpus/API budget); ~36% of CUAD documents still under-segment (≤2 clauses;
down from 42% after adding ALL-CAPS and lettered-subsection heading
patterns — see `ingestion/parsing/clause_parser.py`'s docstring, residual
causes not yet re-diagnosed); Docling file upload isn't exposed over an HTTP
endpoint yet.

---

## 0. Why this use case is different from generic doc Q&A

A support-docs bot that's 85% right is useful. A contract-review bot that's 85% right is a liability — a wrong answer about a liability cap or termination clause has real financial/legal consequences. This changes almost every design decision downstream: chunking, retrieval, evaluation, and what "done" means.

The real-world stakes are documented, not hypothetical: courts have already sanctioned lawyers for AI-generated citations to non-existent cases, and even RAG-based legal tools — which are supposed to be more grounded than general chatbots — still fabricate citations in a meaningful share of responses. Independent peer-reviewed evaluation of commercial legal AI tools found hallucination rates in roughly the 17–33% range depending on the system, even with RAG in place. RAG measurably reduces this compared to a plain LLM, but does not eliminate it — vendor claims of "hallucination-free" citations have been directly challenged.

**Design principle for the whole project: assume every generated claim is wrong until proven grounded. Build the verification layer as a first-class citizen, not an afterthought.**

---

## 1. The problems unique to legal contracts (and why generic RAG breaks on them)

### 1.1 The clause is the unit of meaning — not the token count
Standard 512-token chunking cuts across legal reasoning units. The canonical failure mode: a retriever surfaces a chunk that straddles two clauses — e.g. the back half of a payment-terms section and the front half of a limitation-of-liability section — and the model grounds a confident answer in the wrong half.

**Fix: clause-level chunking, not token-count chunking.** Parse the contract so clauses stay intact as atomic units — retrieve at clause granularity, not arbitrary character windows. This is the single most important architectural decision for this use case, more important than any retrieval strategy layered on top.

🔶 **Validation:** benchmark your clause-level parser against **CUAD**'s 510 annotated contracts before trusting it on your private corpus. CUAD gives you ground-truth clause boundaries and category labels for free — use it as a pre-flight check, not just a paper reference (see §5 and §11).

### 1.2 Dense, cross-referential language
Legal text is full of forward/backward references ("as defined in Section 2.1," "notwithstanding Section 9.3") and defined terms used far from their definitions. A chunk in isolation is often meaningless without its definitions.

**Fix:** attach each clause chunk with its resolved defined terms (a "defined terms" lookup layer built at ingestion) and preserve section-reference metadata so retrieval can pull the referenced clause alongside the queried one.

### 1.3 Citation hallucination is the dominant failure mode, not vague answers
Because legal answers must cite the exact clause/section they rely on, the two ways this goes wrong are: (a) citing a real-sounding but non-existent clause/section, or (b) citing a real clause that doesn't actually say what the answer claims.

**Fix:** deterministic citation validation — after generation, programmatically check three things for every cited section/clause ID: (1) it actually exists in the source document, (2) the retrieved chunk backing that citation was actually part of what the model was shown, and (3) the cited clause's text actually supports the specific claim made — not just that it's topically the right clause. Document/chunk-level matching alone isn't enough; a chunk can be the right clause and still not say what the answer claims it says. This should be a hard, code-level check, not just an LLM self-critique.

### 1.4 Versioning, redlines, and multi-document comparison
Real contract work involves comparing versions (redlines), or cross-referencing an MSA against its amendments and exhibits.

**Fix:** version-aware metadata (effective date, supersession links) and a comparison-mode query path.

### 1.5 Confidentiality and privilege boundaries
Per-tenant/per-matter isolation is a hard requirement once you're handling anything resembling real contracts.

**Fix:** permission-first filtering applied before retrieval, not after.

### 1.6 Ambiguous clauses need a refusal path, not a guess
**Fix:** the CRAG-style evaluator's "insufficient" branch should route to an explicit ambiguity flag for legal-specific cases, not just a generic low-confidence message.

### 1.7 🔶 Jurisdiction and governing law are two different fields, not one
A contract can say "governed by Delaware law, exclusive jurisdiction in New York courts" — same clause, two different values. `governing_law` is which body of law interprets the contract; `jurisdiction` is which court/venue has authority to hear disputes. Storing these as a single field breaks filter logic the moment a contract splits them (which is common in real commercial agreements), silently producing false negatives on jurisdiction-scoped queries.

**Fix:** two separate metadata fields, both extracted at ingestion, both independently filterable.

---

## 2. Best practices synthesis

| Problem | Best practice | Where it lives |
|---|---|---|
| Clause split across chunks | Clause-level chunking at ingestion, not fixed-token splitting; validated against CUAD 🔶 | Ingestion pipeline |
| Dangling defined terms / cross-references | Defined-terms resolution layer + section-reference metadata | Ingestion + retrieval |
| Citation hallucination | Deterministic (code-level) citation validation, not LLM self-check | Post-generation |
| Contradictions between clauses/versions | LLM-based comparison between candidate clauses | Retrieval evaluator step |
| Privilege/confidentiality | Permission-first filtering before vector search | Pre-retrieval |
| Ambiguous clauses | Explicit refusal/flag-for-review path | Evaluator branch |
| **Governing law vs. jurisdiction conflation** 🔶 | **Separate `governing_law` and `jurisdiction` metadata fields, both filterable** | **Ingestion metadata + pre-retrieval filter** |
| General retrieval noise | Hybrid search (vector + BM25) + reranking + tight top-k (3-5) + reorder | Retrieval pipeline |
| Eval trustworthiness | CUAD benchmark pass 🔶 + lawyer-reviewed golden eval set; citation accuracy as a near-100% floor metric | Eval harness |
| Auditability | Full trace of what was retrieved and how the model used it | Logging/observability |

---

## 3. Final architecture

```mermaid
flowchart TD
    subgraph Ingestion [Ingestion Pipeline — offline, one-time per document]
        A1[Contract Upload] --> A2[Clause-Level Parsing<br/>preserve clause boundaries, not token counts]
        A2 --> A3[Defined-Terms Extraction<br/>prompted LLM, build per-document glossary]
        A2 --> A4[Section-Reference Graph<br/>map cross-references between clauses]
        A2 --> A5[Metadata Tagging<br/>contract_type, clause_type, party, effective date,<br/>jurisdiction, governing_law separately, matter/tenant ID, supersession links]
        A3 & A4 & A5 --> A6[Contextual Retrieval Enrichment<br/>prepend clause context before embedding]
        A6 --> A7[Embed + Index<br/>vector store + BM25 index, permission-tagged]
    end

    subgraph Query [Query-Time Pipeline]
        B1[User Query] --> B1a[Jailbreak / Injection Classifier]
        B1a --> B2[Permission-First Filter<br/>tenant/matter access check]
        B2 --> B2a[Jurisdiction / Governing-Law Filter<br/>applied as independent predicates]
        B2a --> B3[Query Router<br/>factual clause lookup / comparison / contradiction-check]
        B3 --> B4[Hybrid Retrieval<br/>vector + BM25, clause-level]
        B4 --> B5[Cross-Encoder Reranking<br/>tight top 3-5 clauses]
        B5 --> B6[Cross-Reference Expansion<br/>pull in referenced clauses + defined terms<br/>trust-label each chunk as data, not instruction]
        B6 --> B7[Long Context Reorder<br/>best clause first & last]
        B7 --> B8{CRAG-style Evaluator<br/>sufficient / ambiguous / insufficient}
        B8 -->|sufficient| B9[Generate Answer<br/>forced structured citation format]
        B8 -->|ambiguous| B10[Flag for Attorney Review<br/>explicit refusal, not a guess]
        B8 -->|insufficient| B11[Rewrite Query & Retry<br/>max 2 attempts, then escalate to B10]
        B9 --> B12[Deterministic Citation Validator<br/>code-level: exists? in-context? span actually supports the claim?]
        B12 -->|valid| B13[Return Answer + Verified Citations + Full Retrieval Trace]
        B12 -->|invalid, retry budget remains| B15[Re-retrieve with Refined Query<br/>shares the 2-attempt cap with B11]
        B15 --> B4
        B12 -->|invalid, retries exhausted| B14[Strip Unsupported Claims / Regenerate from Existing Context]
        B14 --> B13
    end

    Ingestion --> Query
```

## 4. Component-by-component detail

### Ingestion (offline)
- **Clause-level parsing**: structure/heading detection (numbered sections, defined clause headers), not token-count splitting. 🔶 Validate parser output against CUAD's annotated clause spans before trusting it on private contracts.
- **Defined-terms extraction**: prompted LLM builds a per-document glossary (term → definition + location).
- **Section-reference graph**: parse "as defined in Section X" references into a lightweight graph.
- **Metadata**: 🔶 explicit field list —
  - `contract_type` (controlled vocabulary — align to CUAD's 25 contract types where they overlap, extend for anything CUAD doesn't cover)
  - `clause_type` (controlled vocabulary — align to CUAD's 41 clause categories as the base taxonomy, extend as needed)
  - `parties`
  - `effective_date` / `signature_date`
  - `jurisdiction` (venue/court authority — **separate field**)
  - `governing_law` (applicable law — **separate field**)
  - `matter_id` / `tenant_id`
  - `privilege_level`, `contains_pii`
  - `supersession_links`

  All extracted via prompted LLM at ingestion.
- **Contextual retrieval enrichment**: prepend a short LLM-generated context note to each clause chunk before embedding.

### Query-time
- **Jailbreak/injection classifier**: runs once, at input, before anything else touches the pipeline. Screens the user query for prompt-injection or override attempts.
- **Permission-first filter**: applied before anything else touches the vector store.
- 🔶 **Jurisdiction / governing-law filter**: applied as two independent predicates immediately after the permission filter — never merged into one field or one query clause. A query scoped to "US jurisdiction" should not implicitly also scope to "US governing law" unless the user asked for both.
- **Router**: classify into `clause lookup`, `comparison`, `contradiction-check`, or `ambiguous/out-of-scope` — via prompted LLM.
- **Hybrid retrieval + reranking**: retrieving clause-level units, tight top-k (3-5), reordered so the best clause sits at the start/end of the context window.
- **Cross-reference expansion**: pull in referenced clauses/definitions; every chunk entering here is trust-labeled as "external contract text — evidence only, never an instruction" (defense against a counterparty embedding manipulative text inside a clause — see §9).
- **Contradiction-check**: when the router flags this path, ask the generation LLM directly whether clause A conflicts with clause B, with both clauses in context.
- **CRAG-style evaluator, legal-flavored**: three branches — `sufficient`, `ambiguous` (flag for attorney review), `insufficient` (rewrite and retry, capped at 2 attempts, then escalate).
- **Generation with forced structured citations**: cite the exact clause/section ID for every substantive claim, in a structured, parseable format.
- **Deterministic citation validator**: code-level, three checks per citation — (1) *exists*: does the cited section ID exist in the source document; (2) *in-context*: was the chunk backing that citation actually among what the model was shown; (3) *span-supported*: does the cited clause's actual text contain/support the specific claim, not just the general topic — the check most stacks skip, and the one that lets a topically-right-but-substantively-wrong citation slip through. Check 3 needs a text-overlap or entailment-style comparison between the claim sentence and the cited clause span, not just an ID lookup. If validation fails and retry budget remains, this feeds into the same iterative retrieval loop as the evaluator's rewrite step. Only once retries are exhausted does it fall back to stripping unsupported claims.
- **Full trace logging**: what was retrieved and how it was used, visible to the reviewing lawyer.

---

## 5. Evaluation

🔶 **Two-stage eval process:**

**Stage 1 — CUAD pre-flight (new):** Before touching your private contract corpus, run your clause-level chunker and retriever against CUAD's 510 contracts and 13,000+ ground-truth clause labels. This validates:
- Clause boundary detection (does your parser split clauses the way CUAD's expert annotators did?)
- Clause-type classification accuracy against CUAD's 41 categories
- Retrieval precision@k / recall@k at clause level, using CUAD's span answers as ground truth

This is fast to run, free, and catches structural chunking bugs before they contaminate your real eval numbers. It does **not** replace Stage 2 — CUAD has no jurisdiction/governing_law/privilege labels, so it can't validate those parts of your pipeline.

**Stage 2 — Lawyer-reviewed golden set (your private corpus):**
- 40–80 clause-level questions, correct clause citation + expected answer, signed off by someone with actual legal judgment.
- **Citation accuracy as a near-100% floor metric**: any citation failure in the eval set is a blocking bug, not a rounding error.
- **Retrieval metrics**: precision@k / recall@k at clause level — a chunk straddling two clauses counts as a miss.
- **Claim-level attribution accuracy** (distinct from retrieval precision): for each answer, does every cited span actually contain the claim it's attached to — scored separately from "was the right clause retrieved." A system can score well on clause-level retrieval and still fail here if it cites the right clause for the wrong sentence.
- **Faithfulness/groundedness scoring**: RAGAS-style LLM-as-judge, supplemented by the deterministic citation validator.
- **Ambiguity/refusal rate tracking**: how often the system correctly flags genuinely ambiguous clauses vs. forces an answer it shouldn't.
- **Latency logging**: expect legal users to tolerate 5–10s in exchange for verified citations.

---

## 6. What changes from the earlier general-purpose plan

| From the general plan | For legal contracts |
|---|---|
| Structure-aware chunking (headings) | **Clause-level chunking, validated against CUAD** 🔶 |
| Metadata pre-filter (date/category) | Same, plus **matter/tenant permission filter** and **separate jurisdiction/governing_law filters** 🔶, applied pre-retrieval |
| Groundedness critic (LLM-based) | **Add a deterministic citation validator alongside it** |
| CRAG evaluator: sufficient/partial/insufficient | **Add a third branch: ambiguous → flag for attorney review** |
| Golden eval set (self-built) | **CUAD pre-flight pass, then lawyer-reviewed golden set** 🔶 |
| GraphRAG | Not needed at this stage — parked |
| Self-Querying Retriever | Not needed at this stage — parked |

---

## 7. Non-negotiables checklist

- [ ] Clause-level chunking implemented **and validated against CUAD's annotated spans** 🔶 (no clause ever split across two chunks)
- [ ] Deterministic citation validator running on every generated answer, checking existence + in-context + span-level claim support (not existence alone)
- [ ] Permission/tenant filtering applied before retrieval, tested with at least two isolated tenants/matters
- [ ] 🔶 `jurisdiction` and `governing_law` stored and filtered as **two independent fields**, tested with a contract where they differ
- [ ] Explicit ambiguous-clause refusal path
- [ ] Lawyer-reviewed golden eval set, citation accuracy tracked as a blocking metric
- [ ] Full retrieval trace visible/exportable
- [ ] Jailbreak/injection classifier in front of the pipeline, logged as an auditable event

---

## 8. Recommended tech stack

| Layer | Recommendation | Why |
|---|---|---|
| Orchestration | **LangChain + LangGraph** | LangGraph adds the stateful branching/retry control the CRAG evaluator loop and the "flag for attorney review" gate need |
| Vector DB | **Qdrant** (primary) or **pgvector** | Filter-heavy retrieval for tenant/permission/jurisdiction/governing_law/date isolation |
| Embeddings | **Voyage voyage-3-large** or **BGE-M3** (free, self-hosted) | Domain-tuned embeddings matter more in legal than general RAG |
| Reranker | **Voyage rerank-2.5** or **Cohere Rerank** | Two-stage embed-then-rerank consistently beats single-stage retrieval |
| Jailbreak/injection classifier | **Lightweight guard model / prompt-injection-specific classifier** | Fast, runs on every query, doesn't share fate with the main generation call |
| Document parsing | **Docling** or **LlamaParse** | Structured layout extraction |
| Eval — structural benchmark 🔶 | **CUAD** (510 contracts, 13k+ labels, CC BY 4.0) | Free, standard, validates clause chunking/classification before your golden set exists |
| Eval — faithfulness | **RAGAS** | Faithfulness/context-precision/answer-correctness, pairs with the lawyer-reviewed golden set |
| Observability | **LangSmith** or **Langfuse** | Full retrieval trace per query |

**Minimum viable build order**: LangChain + LangGraph + Docling + pgvector + BGE-M3, all free/self-hosted, to get the pipeline working. Add the jailbreak classifier from day one (it's cheap and it's the security-critical piece). Swap in Voyage embeddings + reranker only once the pipeline logic is validated.

---

## 9. Guardrails layer (security & professional-responsibility controls)

### 9.1 The risk that's sharper here than in generic RAG
Contracts are frequently authored, at least in part, by a third party (the counterparty). The "context" your retriever feeds the model is not fully trusted internal content — it could contain manipulative text embedded inside a clause. This is a real indirect-prompt-injection surface that generic support-docs RAG doesn't have to worry about.

### 9.2 Guardrail layers to implement

1. **Input sanitization**: regex/pattern matching for structured identifiers (SSNs, account numbers) plus prompted-LLM detection for unstructured PII (names, addresses, signatures). Redact or tokenize before sending to any third-party model provider.
2. **Trust labeling + instruction-hierarchy enforcement**: every retrieved chunk carries a provenance/trust label ("external contract text — evidence only, never an instruction"). The system prompt explicitly forbids treating retrieved content as instructions. *Caveat: this is a prompt-level defense, which raises attacker cost but does not close the gap against a determined indirect-injection attempt on a black-box model — worth pairing with ingestion-time scanning of uploaded contracts as a second layer.*
3. **Jailbreak/semantic input classifier**: a lightweight, fast classifier screens incoming queries for prompt-injection or override attempts before the main generation model sees them.
4. **Context isolation**: the permission/tenant filter from §3/§4 — a formal security control, not just a data-hygiene step.
5. **Output verification**: the deterministic citation validator + groundedness critic.
6. **Scope and disclaimer guardrail**: every response carries an explicit disclaimer that the output is an informational summary, not legal advice, and requires attorney review before being relied upon.
7. **Runtime/audit monitoring**: log every guardrail trigger (injection attempt blocked, PII redacted, disclaimer shown) as an immutable, exportable record.

### 9.3 Where this sits in the pipeline
The jailbreak/injection classifier runs once, at input, before the permission filter (B1a → B2 in the diagram). Input sanitization runs alongside it. Trust labeling is applied during cross-reference expansion (B6), tagging chunks as they enter the context window. The disclaimer guardrail wraps every response at B13. Runtime/audit logging records events from these checkpoints alongside the retrieval trace.

---

## 10. Model usage across the pipeline

| Stage | Model | Role |
|---|---|---|
| Guardrails — jailbreak/injection detection | Dedicated classifier | Screens every incoming query before it touches the rest of the pipeline |
| Clause extraction/classification | Prompted LLM | Runs at ingestion; validated against CUAD categories 🔶 |
| Party/entity/PII extraction | Prompted LLM | Feeds the glossary and metadata layer |
| Jurisdiction / governing-law extraction 🔶 | Prompted LLM | Two separate extraction targets, not one combined field |
| Contradiction-check (clause A vs. clause B) | Prompted LLM | Both clauses in context, direct comparison |
| Router/evaluator | Prompted LLM | Classifies query type, branches CRAG evaluator |
| Embeddings + reranker | Voyage/BGE-M3, Voyage rerank/Cohere (§8) | Retrieval |

---

## 11. Suggested build order

1. **Ingestion first**: clause-level parser + LLM-based defined-terms extraction + metadata (including split jurisdiction/governing_law) + contextual retrieval enrichment.
2. 🔶 **CUAD pre-flight validation**: run the clause parser and clause-type classifier against CUAD's 510 contracts. Fix structural chunking/classification bugs here — it's cheap and catches issues before they show up in harder-to-diagnose private-corpus eval failures.
3. **Query-time baseline**: jailbreak classifier → permission filter → jurisdiction/governing-law filter → hybrid retrieval → rerank → cross-reference expansion → reorder. Verify on your golden set before adding agentic layers.
4. **Deterministic citation validator**: build and test in isolation — feed it known-good and known-bad citations and confirm it catches the bad ones, including the case where the citation exists and is in-context but the clause text doesn't actually say what the answer claims (this is the case existence-only checks miss).
5. **CRAG-style evaluator with the three-way branch** (sufficient/ambiguous/insufficient).
6. **Eval harness + lawyer review pass**, then iterate.

---

## 13. Optional module: recent case law / external legal authority (not core, opt-in)

This is a genuinely different task from clause review — referencing whether a recent judgment affects the enforceability of a clause type (e.g. a court just invalidated a common non-compete structure in a jurisdiction). Don't wire this in as a generic `web_search` tool on the main router; it breaks the closed-corpus, permission-filtered design of everything above and inherits none of its trust guarantees.

**If you want this capability:**
- **Separate query path, not the main router**: route "does recent case law affect this clause" queries to a dedicated external-authority tool, distinct from the clause-lookup/comparison/contradiction-check paths in §4.
- **Curated source, not open web search**: a case-law API/database (e.g. CourtListener, a licensed case-law feed, or your jurisdiction's official court records) — not general web search. Purpose-built legal tools consistently outperform general web search on citation reliability specifically because of curated, verified sources with citators, not because of RAG architecture alone.
- **"Good law" check is a separate validator from your clause validator**: existence + correct citation format isn't enough for case law — you additionally need to confirm the case hasn't been overturned, distinguished, or superseded since. This is a distinct check (commercial tools call this Shepardizing/KeyCite-style flagging) and doesn't reuse your clause-citation validator's logic.
- **Jurisdiction filtering is mandatory, not optional**: blending authority across jurisdictions is a well-documented hallucination pattern — a case that's binding in one state and merely persuasive (or inapplicable) in another gets treated as equally authoritative if you don't filter hard on jurisdiction before it enters context.
- **Separate disclaimer**: this output is legal research, not contract review — it carries its own professional-responsibility exposure and should say so explicitly, separately from the standard disclaimer in §9.2.
- **Keep it out of the citation accuracy floor metric in §5** until it has its own lawyer-reviewed golden set — folding a much harder, much less mature capability into the same "near-100%" bar as clause citation will either tank your existing metric or get quietly excluded from testing, both bad outcomes.

---

## Appendix: CUAD dataset access

- **Official site + download**: https://www.atticusprojectai.org/cuad
- **Hugging Face (SQuAD-style QA format)**: https://huggingface.co/datasets/theatticusproject/cuad-qa
- **GitHub (code + model checkpoints)**: https://github.com/TheAtticusProject/cuad
- **Paper**: https://arxiv.org/abs/2103.06268
- **License**: CC BY 4.0, free for commercial and non-commercial use
- **Contents**: 510 commercial contracts (from SEC EDGAR filings), 13,000+ expert annotations, 41 clause categories, 25 contract types
