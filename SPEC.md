# SPEC.md — Financial Analyst Agent

> Reverse-engineered from the current codebase as a baseline for spec-driven development. This document describes the system **as it actually exists today**. Future features should be proposed as diffs against this spec (update the relevant section, add a Changelog entry) before implementation begins.

## 1. Product Intent

From [docs/roadmap.md](docs/roadmap.md):

- Build an enterprise-grade **Automated Financial Investment Research Analyst**.
- Enforce strict runtime unit economics, explicit state recovery, and high data density.
- Architectural pillars called out in the roadmap:
  1. **Storage Engine** — relational SQLite parent/child schema + NumPy cosine similarity, standing in for a pgvector-style store.
  2. **Resilience** — LangGraph persistent checkpointers via a transactional SQLite store for mid-loop fault recovery. *(Not yet implemented — see §6.)*
  3. **FinOps Control** — an in-memory budget controller that kills runaway agent loops before they cause billing spikes.

## 2. Current Status

This is a working, end-to-end RAG pipeline: ingest PDFs → table-aware extraction + semantic chunking → embed → two-stage retrieval (vector search + rerank) → generate a cited answer → score it (Faithfulness/Context Precision) — with a Streamlit UI on top. **There is exactly one ingestion pipeline** ([src/ingestion/pipeline.py](src/ingestion/pipeline.py)) — earlier experimental branches (v2, v3) existed at various points during development and are gone or merged; §21 has the consolidation history if you're wondering why the debugging sections below (§12 onward) reference numbers that no longer match a fresh run.

| Layer | Status | Location |
|---|---|---|
| Ingestion (table-aware PDF extraction + semantic chunking + batched embeddings) | ✅ Implemented | [src/ingestion/](src/ingestion/) |
| FinOps budget controller | ✅ Implemented | [src/utils/billing.py](src/utils/billing.py) |
| CLI entrypoint (folder scan, all 7 source PDFs ingested) | ✅ Implemented | [scripts/run_ingestion.py](scripts/run_ingestion.py) |
| Retrieval — query expansion | ✅ Implemented | [src/retrieval/query_expansion.py](src/retrieval/query_expansion.py) |
| Retrieval — HyDE | ✅ Implemented | [src/retrieval/hyde.py](src/retrieval/hyde.py) |
| Retrieval — similarity search | ✅ Implemented | [src/retrieval/similarity_search.py](src/retrieval/similarity_search.py) |
| Retrieval — two-stage retrieval + Cohere rerank | ✅ Implemented | [src/retrieval/two_stage_retriever.py](src/retrieval/two_stage_retriever.py) |
| Generation — answer generator | ✅ Implemented | [src/generation/answer_generator.py](src/generation/answer_generator.py) |
| Evaluation — Faithfulness & Context Precision (DeepEval) | ✅ Implemented | [src/generation/faithfulness_eval.py](src/generation/faithfulness_eval.py) |
| Streamlit UI | ✅ Implemented, verified end-to-end | [app.py](app.py) |
| Agents / LangGraph orchestration | 🚧 Stub only (empty dir) | `src/agents/` |
| Config management | 🚧 Stub only (empty dir) | `config/` |
| Tests | ✅ Implemented — 73 tests, mocked APIs/DB, offline (§22) | [tests/](tests/) |
| LangGraph checkpointing / fault recovery | ❌ Not started | — |
| Ragas evaluation | ❌ Not attempted — confirmed non-functional in this environment (§17), DeepEval used instead | — |
| LangSmith tracing | ❌ Not started (`LANGCHAIN_API_KEY`/`LANGCHAIN_PROJECT` present, unused) | — |

## 3. Architecture (as-built)

```
scripts/run_ingestion.py  (folder scan over data/raw_documents/*.pdf)
  └─ StructuralIngestionPipeline (src/ingestion/pipeline.py)
       ├─ _extract_structured_text()        — dispatches by extension
       │    ├─ .pdf → extract_structured_text() (src/ingestion/pdf_table_extraction.py, pdfplumber)
       │    └─ .txt → plain read, narrative-only
       ├─ segment_text_semantically()       — sentence-level semantic chunking (narrative parents)
       │    └─ get_embedding_vectors()      — batched OpenAI embeddings
       │    └─ _cosine_distance()           — NumPy cosine distance
       ├─ TokenBudgetController (src/utils/billing.py) — cost tracking + hard budget cap
       └─ SQLiteVectorStore (src/ingestion/vector_store.py) — parent/child persistence
```

### 3.1 Ingestion pipeline — [src/ingestion/pipeline.py](src/ingestion/pipeline.py)

`StructuralIngestionPipeline.run_file_ingestion(file_path)` is the single public entrypoint. This is the result of consolidating an earlier experimental table-extraction branch into the pipeline directly — see §21 for why and what changed; this section describes only the current, single implementation.

1. **Structured extraction** (`_extract_structured_text`): dispatches on file extension.
   - `.pdf` → [extract_structured_text](src/ingestion/pdf_table_extraction.py) (pdfplumber-based, table-aware — see §18 for the algorithm and the data-loss bug found and fixed during consolidation). Returns `{"narrative_text": str, "table_blocks": list[str]}`: detected tables/stat-cards are converted to clean `Label: Value` chunks *before* they ever reach the sentence-splitter below; everything else (including any leftover text from a region that had a partial table match) becomes narrative text.
   - `.txt` → plain `open().read()`, treated entirely as narrative text (no table detection).
   - anything else → raises `ValueError`.
2. **Semantic chunking of narrative text** (`segment_text_semantically`): splits into sentences via a regex on sentence-ending punctuation (known limitation: doesn't guard against splitting mid-number, e.g. `"12.3%"` — not fixed, mitigated in practice by table content being pulled out before this step), embeds *all* sentences in batches (`get_embedding_vectors`, not one call per sentence — see below), and walks adjacent-sentence pairs. When cosine distance between neighbors exceeds `threshold` (default `0.35`), it cuts a new "parent" chunk; otherwise sentences are grouped together. This produces the **narrative parent blocks**.
3. **Narrative child chunking**: within each parent block, sentences are grouped into fixed non-overlapping windows of 3 (last window keeps whatever remains), embedded in one batched call per document rather than per window, and stored as child chunks.
4. **Table chunking**: each detected table/stat-card block from step 1 becomes its own single parent + child chunk (no further splitting — they're already compact), embedded in one batched call per document.
5. **Batched embeddings** (`get_embedding_vectors`): one OpenAI request per up to 250 texts, instead of one request per sentence/window/table — added because unbatched calls made the annual report's ingestion (~7,000+ sentences) take a very long time; see §21 for the measured before/after. `get_embedding_vector` (singular) still exists for single-text callers (query-side retrieval modules).
6. **Persistence**: each parent block (narrative or table) + its child tuples are written atomically via `SQLiteVectorStore.insert_document_pipeline`.
7. **Cost reporting**: prints cumulative FinOps cost at the end of the run.

Embedding model: `text-embedding-3-small` (OpenAI), hardcoded default.

### 3.2 Vector store — [src/ingestion/vector_store.py](src/ingestion/vector_store.py)

`SQLiteVectorStore` wraps a single SQLite file at `data/financial_intelligence.db`. Schema, created on first use:

```sql
parent_documents (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_file TEXT NOT NULL,
  section_title TEXT,
  full_content TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)

child_chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  parent_id INTEGER,
  chunk_content TEXT NOT NULL,
  embedding_json TEXT NOT NULL,   -- JSON-serialized embedding vector
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (parent_id) REFERENCES parent_documents(id) ON DELETE CASCADE
)
```

- `insert_document_pipeline(source_file, section_title, parent_text, child_tuples)` performs the parent insert + child batch insert inside one transaction, with rollback on failure.
- `fetch_all_child_chunks()` — read counterpart used by [SimilaritySearch](src/retrieval/similarity_search.py) (§10): returns every child chunk joined with its parent row (`source_file`, `section_title`, `full_content`). Embeddings are stored as JSON text (not a native vector type), so callers must `json.loads` each `embedding_json` themselves.
- No deduplication: re-running ingestion against the same file creates duplicate parent/child rows.

### 3.3 FinOps budget controller — [src/utils/billing.py](src/utils/billing.py)

`TokenBudgetController`:
- Tracks cumulative `prompt_tokens`, `completion_tokens`, `total_cost_usd` for the life of one pipeline instance (in-memory, not persisted).
- Pricing table (`PRICING_MATRIX`) hardcodes per-million-token rates for `gpt-4o-mini`, `gpt-4o`, `o3-mini` — **note: embedding calls are billed against the `gpt-4o-mini` output/input rates as a proxy**, not actual `text-embedding-3-small` pricing.
- `max_budget_usd` defaults to `$0.50`, overridable via `MAX_TOKEN_BUDGET_PER_RUN` env var.
- `update_usage_and_verify(...)` raises `PermissionError` once cumulative cost reaches the cap — this is the mechanism that stops runaway ingestion mid-run.
- `execute_safely(...)` is a chat-completion wrapper for future agent use; **currently unused** by the ingestion pipeline, and has a latent bug — `response.choices.message.content` should be `response.choices[0].message.content` (would raise `TypeError` if ever called, since `choices` is a list).

### 3.4 CLI entrypoint — [scripts/run_ingestion.py](scripts/run_ingestion.py)

- Loads `.env`, hard-fails if `OPENAI_API_KEY` is missing.
- Scans `data/raw_documents/*.pdf`, skips any file whose basename is already in `SQLiteVectorStore.get_ingested_source_files()`, and ingests the rest — each file gets a **fresh** `StructuralIngestionPipeline` (and therefore a fresh $0.50-default `TokenBudgetController` budget) so one large file hitting its cap doesn't block the others.
- Per-file `try/except`: a failure (including a budget breach) is caught, logged, and the run continues to the next file — same resilience principle as [faithfulness_eval.py](src/generation/faithfulness_eval.py) (§17).
- Prints a summary: skipped/succeeded/failed counts, per-file cost, total cost.
- No longer single-hardcoded-file — see §20 for the corpus expansion this enabled and what it revealed.

## 4. Data & Configuration

### 4.1 Source data — [data/](data/)

- `data/mock_company_data.txt` — original placeholder/test fixture, plaintext.
- `data/raw_documents/` — real source PDFs (ASX announcements, trading updates, dividend notices, half-year/full-year results, the annual report). Sizes range ~28KB–16MB. All 7 are ingested (§20).

### 4.2 Environment variables — [.env](.env)

| Var | Used today? | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | Embeddings (ingestion) + chat completions (future agents) |
| `MAX_TOKEN_BUDGET_PER_RUN` | ✅ | FinOps hard cap, defaults to `0.50` |
| `COHERE_API_KEY` | ✅ | Cohere Rerank API, [TwoStageRetriever](src/retrieval/two_stage_retriever.py) (§15) |
| `LANGCHAIN_API_KEY` | ❌ (reserved) | LangSmith tracing |
| `LANGCHAIN_PROJECT` | ❌ (reserved) | LangSmith project name |

### 4.3 Persistence

- `data/financial_intelligence.db` — SQLite file, created/opened by `SQLiteVectorStore`. Not currently in `.gitignore` scope discussion since the repo has no git history yet.

## 5. Dependencies — [requirements.txt](requirements.txt)

| Package | Used by current code | Implied future use |
|---|---|---|
| `openai` | ✅ embeddings, client init | chat completions for agents |
| `numpy` | ✅ cosine distance | vector math generally |
| `python-dotenv` | ✅ `.env` loading | — |
| `pdfplumber` | ✅ table-aware PDF extraction (§3.1, §18) | — |
| `cohere` | ✅ Rerank API, `TwoStageRetriever` (§15) | — |
| `deepeval` | ✅ Faithfulness & Context Precision judging (§17) | — |
| `streamlit` | ✅ UI (§19) | — |
| `tiktoken` | ✅ token counting in `TokenBudgetController` | — |
| `pydantic` | ✅ `TokenUsageSummary` model | agent/graph state schemas |
| `langchain` | ❌ unused | retrieval chains, tool wrappers |
| `langgraph` | ❌ unused | agent orchestration + checkpointing (§1 pillar 2) |
| `langsmith` | ❌ unused | tracing/observability |

`pypdf` and `ragas` were both removed after being tried and superseded — `pypdf` by `pdfplumber` (§3.1's table-aware extraction needed it), `ragas` by `deepeval` (§17 — `ragas` is non-functional in this environment, a broken `langchain-community` import unrelated to this project).

## 6. Known Gaps vs. Roadmap

These are explicitly called out in [docs/roadmap.md](docs/roadmap.md) but not yet built:

1. **Retrieval layer** (`src/retrieval/`) — query expansion, HyDE, similarity search, hit-rate evals, two-stage retrieval + rerank, generation, and Faithfulness/Context Precision evaluation all exist (§8-§10, §12-§13, §15-§17) — the pipeline runs end-to-end from question to scored answer. Still open: re-running §12/§13/§15/§17/§20's eval suites against the consolidated pipeline (§21) — the historical numbers in those sections predate it; combining HyDE + multi-query in one retrieval strategy; evaluating the reranker's effect on hit-rate the way §12/§13 evaluated HyDE/multi-query; the metadata-filtering / period-disambiguation work §20 identifies as the fix for cross-document confusion; and full agent orchestration tying retrieval into a research workflow.
2. **Agent orchestration** (`src/agents/`) — no LangGraph graph, nodes, or state machine exists yet.
3. **LangGraph persistent checkpointing** — the "mid-loop fault recovery" pillar from docs/roadmap.md has no implementation; there's no checkpointer, no graph to checkpoint.
4. **Config management** (`config/`) — currently all configuration is ad hoc env vars read directly in each module; no centralized config loader.
5. **Ragas evaluation**, **LangSmith tracing** — Ragas was tried and found non-functional in this environment (§17), DeepEval used instead; LangSmith integration not started.

Automated tests (`tests/`) — previously listed here as an open gap — now exist; see §22. `app.py` (Streamlit UI) remains untested by this suite; UI testing needs a different approach (`streamlit.testing.v1.AppTest` or Playwright, per §19's one-off precedent) and is still open.

## 7. Known Issues / Tech Debt

- `TokenBudgetController.execute_safely` indexes `response.choices` without `[0]` — will throw if ever invoked (currently dead code).
- Embedding API calls are cost-tracked using chat-completion pricing (`gpt-4o-mini`) rather than the actual embedding model's pricing — cost figures printed by the pipeline are an approximation, not an exact bill.
- No dedup/upsert on re-ingestion — running the same file twice doubles its rows in `financial_intelligence.db`.
- The sentence-splitting regex (`segment_text_semantically`) doesn't guard against splitting mid-number (e.g. `"12.3%"`) — a known limitation, not fixed; mitigated in practice since §18's table-aware extraction pulls numeric table content out before this step runs on it, but any decimal appearing in narrative prose is still at risk.
- `SimilaritySearch`/`SQLiteVectorStore` carry no structured metadata beyond `source_file`/`section_title` — no `reporting_period` or `fiscal_year` fields, which is the concrete gap §20 traces the cross-document confusion back to.

## 8. Retrieval — Query Expansion (Alternative Phrasing Generator)

**Status: ✅ Implemented.** First piece of the retrieval layer (§6 item 1). Given a user's research question, generates a small set of alternative phrasings so a future similarity-search step can query with multiple variants and improve recall against `child_chunks` — a standard multi-query RAG technique. This is generation only; it does not itself search the vector store (see §6 — similarity search is still unbuilt).

- **Location**: [src/retrieval/query_expansion.py](src/retrieval/query_expansion.py), class `QueryExpansionGenerator`.
- **Interface**: `generate_alternative_phrasings(query: str, num_variants: int = 3) -> list[str]` — returns up to `num_variants` reworded variants of the input query. The original query is **not** included in the returned list; a caller doing multi-query retrieval is expected to search the original plus these variants.
- **Model/prompting**: a single OpenAI chat completion (`gpt-4o-mini`). The prompt instructs the model to preserve intent, entities, and numbers from the source query while varying wording/structure, and to return one phrasing per line with no numbering or commentary. The response is parsed by splitting on newlines, stripping whitespace/leading list markers, and truncating to `num_variants`.
- **Cost control**: reuses [TokenBudgetController](src/utils/billing.py) exactly as `StructuralIngestionPipeline` does — instantiated in `__init__`, `update_usage_and_verify` called after the completion with the real `response.usage` prompt/completion token counts. Unlike the ingestion pipeline's embedding calls (§3.3, §7 — billed against `gpt-4o-mini` as a pricing proxy), this is an actual `gpt-4o-mini` chat completion, so the cost figure is exact, not approximated.
- **CLI entrypoint**: [scripts/run_query_expansion.py](scripts/run_query_expansion.py) (scripts/) mirrors [scripts/run_ingestion.py](scripts/run_ingestion.py) — loads `.env`, guards on `OPENAI_API_KEY`, instantiates `QueryExpansionGenerator`, runs a hardcoded sample query, prints the original plus generated phrasings, wrapped in a top-level `try/except`. Exists purely for manual verification, since no downstream retrieval step consumes this output yet.

## 9. Retrieval — HyDE (Hypothetical Document Embeddings)

**Status: ✅ Implemented.** Second piece of the retrieval layer (§6 item 1), alongside §8's query expansion. Instead of embedding the user's raw question, an LLM first drafts a short hypothetical answer passage — written in the style of the target documents (financial report/disclosure prose) — and that passage's embedding is used for similarity search instead of the question's embedding. Rationale: a plausible answer sits closer in embedding space to real document chunks than a short question does, improving retrieval recall. This is generation only; it does not itself search the vector store (see §6 — similarity search is still unbuilt).

- **Location**: [src/retrieval/hyde.py](src/retrieval/hyde.py), class `HyDEGenerator`.
- **Interface**: `generate_hypothetical_embedding(query: str) -> tuple[str, list]` — returns `(hypothetical_answer, embedding_vector)`.
- **Model/prompting**: two OpenAI calls per invocation —
  1. Chat completion (`gpt-4o-mini`) drafts a short (2-4 sentence) hypothetical passage that plausibly answers `query`, written in the voice of a financial report/ASX announcement — confident, factual-sounding prose, no hedging, no "I don't know," no meta-commentary.
  2. Embedding call (`text-embedding-3-small`) — matches the embedding model used at ingestion time ([pipeline.py](src/ingestion/pipeline.py) `get_embedding_vector`), since the resulting vector must live in the same space as the stored `child_chunks` vectors for cosine similarity to be meaningful.
- **Cost control**: reuses [TokenBudgetController](src/utils/billing.py), own instance in `__init__` (same pattern as `QueryExpansionGenerator` and `StructuralIngestionPipeline`). The chat completion is billed exactly (real `gpt-4o-mini` usage). The embedding call is billed as a `gpt-4o-mini`-rate proxy, the same known approximation already documented for ingestion embeddings (§3.3/§7).
- **CLI entrypoint**: [scripts/run_hyde.py](scripts/run_hyde.py) mirrors [scripts/run_query_expansion.py](scripts/run_query_expansion.py) — resolves `ROOT_DIR`, loads `.env` from it, guards `OPENAI_API_KEY`, runs a hardcoded sample query, prints the drafted hypothetical answer, the embedding vector's dimensionality, and total cost.

## 10. Retrieval — Similarity Search

**Status: ✅ Implemented.** The read-side counterpart to `SQLiteVectorStore.insert_document_pipeline` (§3.2). Given a query embedding — from anywhere: a raw query, `QueryExpansionGenerator` (§8), `HyDEGenerator` (§9) — returns the top-k nearest `child_chunks` ranked by cosine similarity, each joined with its parent row. This is the piece that makes §8 and §9 actually usable; before this, nothing could search with the embeddings they produced.

- **Location**: [src/retrieval/similarity_search.py](src/retrieval/similarity_search.py), class `SimilaritySearch`.
- **Interface**: `search(query_embedding: list, top_k: int = 5) -> list[dict]` — returns up to `top_k` results sorted by descending similarity, each dict with `child_id`, `parent_id`, `chunk_content`, `source_file`, `section_title`, `parent_content`, `similarity`.
- **Mechanics**: no vector index. Fetches every child chunk joined with its parent via `SQLiteVectorStore.fetch_all_child_chunks()`, `json.loads`'s each `embedding_json`, scores with NumPy cosine similarity (mirrors, but doesn't reuse, `StructuralIngestionPipeline._cosine_distance` — that method is private and instance-coupled to the ingestion pipeline), sorts, truncates. Full in-memory linear scan on every call — see §11 for the scaling implications.
- **Agnostic to embedding source**: this module does not generate embeddings itself; callers are responsible for producing the query embedding first (via §8, §9, or a raw embedding call) and passing it in.
- **CLI entrypoint**: [scripts/run_similarity_search.py](scripts/run_similarity_search.py) — exercises all three retrieval components together against the real OpenAI API (no mocking): embeds a raw sample query directly (small local helper mirroring `pipeline.py`'s `get_embedding_vector`, since nothing else in the codebase embeds a bare string) and searches with it; runs `QueryExpansionGenerator`, embeds each returned phrasing, and searches with each; runs `HyDEGenerator` (which returns an embedding directly) and searches with it. Prints ranked results for all three paths plus combined cost.

## 11. Scaling Considerations — Similarity Search

Kept separate from §7's general tech-debt list — these are specifically about what breaks as corpus size grows, not current-state bugs.

| Issue | Suggested Solution |
|---|---|
| `search()` does a full in-memory linear scan — fetches and scores every `child_chunk` on every call, O(n) per query with no index | Adopt an ANN index (FAISS, hnswlib) or migrate to a vector-native store (pgvector, Chroma, `sqlite-vec`) once corpus exceeds a few thousand chunks |
| Embeddings stored as `embedding_json` TEXT — every search `json.loads`'s every row, far slower than reading packed floats | Store embeddings as packed binary (BLOB via `numpy.tobytes()`) or move to a store with native vector columns |
| No caching — every `search()` call re-fetches and re-parses the entire dataset from SQLite, even for repeated queries in the same process | Load and cache the embedding matrix once per process (e.g. on `SimilaritySearch.__init__`), invalidate/refresh on new ingestion |
| Single SQLite file, single connection per call — no concurrent read/write scalability | Migrate to a server-based store (Postgres+pgvector) if concurrent ingestion + querying is ever needed |
| No metadata filtering pushed to SQL — `fetch_all_child_chunks()` always scans the whole store across all ingested documents, can't cheaply scope to one `source_file` | Push filters (e.g. `WHERE source_file = ?`) into the SQL query before Python-side similarity scoring, to shrink the candidate set |
| Full sort (`O(n log n)`) of all candidates then truncated to `top_k`, instead of a true top-k selection | Use `heapq.nlargest(top_k, ...)` (or a bounded min-heap) instead of sorting the entire candidate list |

## 12. Retrieval — HyDE vs. Baseline Hit-Rate Evaluation

**Status: ✅ Implemented.** Answers the original ask directly: quantifies whether HyDE (§9) improves retrieval recall over raw-query embedding, using real questions grounded in the ingested CBA disclosure.

> **Historical result** — measured against the pre-consolidation pypdf-based pipeline on a single document (91 parent rows). Not re-run against the consolidated table-aware pipeline (§21) or the full 7-document corpus (§20); the mechanism/methodology below is still accurate, the specific numbers are not current-state claims.

- **Location**: [src/retrieval/hyde_eval.py](src/retrieval/hyde_eval.py) (HyDE-specific comparison logic). Question set lives in [src/retrieval/eval_questions.py](src/retrieval/eval_questions.py) and the `embed_text` helper in [src/retrieval/embedding_utils.py](src/retrieval/embedding_utils.py) — both shared with §13's multi-query eval, extracted rather than duplicated once a second eval module needed them. CLI wrapper [scripts/run_hyde_eval.py](scripts/run_hyde_eval.py).
- **Question set**: 8 hand-picked questions (not LLM-generated), each written from actual `parent_documents` rows inspected directly in `data/financial_intelligence.db`, tagged with the known-correct `parent_id`(s). Reused verbatim by §13 so results are directly comparable:

  | # | Question | Expected parent_id(s) | Grounding (ASX Subsection) |
  |---|---|---|---|
  | 1 | What was CBA's statutory net profit after tax for the 2026 half year? | {29} | #17 — "Statutory NPAT $5,412m, 5% on 1H25, 8% on 2H25" |
  | 2 | What interim dividend per share did CBA declare for 1H26? | {23, 32, 46, 49} | #11/#20/#34/#37 — "$2.35 per share, fully franked" |
  | 3 | What was CBA's pre-provision profit for the half year? | {31} | #19 — "$8,131m, up 5% on 1H25, 5% on 2H25" |
  | 4 | What was CBA's net interest margin (NIM) for 1H26? | {35} | #23 — "2.04%, down 4bpts on 1H25 and 2H25" |
  | 5 | What was CBA's Common Equity Tier 1 (CET1) capital ratio? | {46} | #34 — "12.3%, vs APRA minimum 10.25%" |
  | 6 | What was CBA's return on equity (ROE) for the half? | {46, 48} | #34/#36 — "13.8%, up 10bpts" |
  | 7 | What was CBA's net stable funding ratio (NSFR)? | {50} | #38 — "117% (115% Jun 25)" |
  | 8 | What was CBA's loan impairment expense for the half? | {39} | #27 — "$319m, loan loss rate 6bpts, flat on 1H25" |

  Some questions have multiple valid `parent_id`s because the same fact legitimately appears in more than one block (e.g. the $2.35 dividend is stated in the headline metrics box and repeated in prose) — a hit against *any* of them counts.
- **Metric**: run at `top_k` of both 5 and 1 — for each question, embed it two ways (baseline: direct embedding of the raw question text; HyDE: `HyDEGenerator.generate_hypothetical_embedding`), run `SimilaritySearch.search(embedding, top_k)` for each, and count a hit if any result's `parent_id` is in the question's expected set. Aggregate hit rate = hits / 8 per mode. `evaluate_hit_rate(top_k=5)` is the default; `scripts/run_hyde_eval.py` runs hit@5 only (hit@1 below was run separately via the same function).
- **Not using `ragas`**: `requirements.txt` already declares `ragas` for future RAG evaluation (§5) but it's unused; this is one simple custom metric (hit@k) — pulling in the ragas framework for a single boolean check would be disproportionate. This does not fulfill that future integration.
- **Cost control**: reuses [TokenBudgetController](src/utils/billing.py) — one instance for baseline embedding calls, `HyDEGenerator`'s own instance for its calls — same pattern as every other retrieval component.
- **Output**: per-question hit/miss for both modes, a summary table comparing aggregate hit rates, and total cost.

**Results** (measured 2026-09-02):

| Metric | Baseline | HyDE |
|---|---|---|
| hit@5 | 100% (8/8) | 100% (8/8) |
| hit@1 | 88% (7/8) | 50% (4/8) |

At hit@5 both are perfect — the corpus (91 blocks, one document) and `top_k=5` are generous enough that neither approach is stressed. At the stricter hit@1, **HyDE underperforms baseline**: it misses on exactly the questions with precise numeric figures (NPAT, dividend, pre-provision profit, loan impairment). Cause: HyDE fabricates plausible-but-wrong specific numbers (observed inventing "$5.2 billion" against the real "$5,412m") — for dense numeric disclosures, that invented specificity pulls the embedding away from the chunk containing the real figure. HyDE wins or ties on the more qualitative/ratio questions (NIM, CET1, ROE, NSFR). This isn't evidence HyDE is broken generally — it's a poor fit for exact numeric-lookup queries already phrased in the document's own terminology, where the raw question already embeds close to the answer without help.

## 13. Retrieval — Multi-Query Expansion vs. Baseline Hit-Rate Evaluation

**Status: ✅ Implemented.** Same question as §12, for the other retrieval-augmentation technique already built (§8): does searching with multiple paraphrased variants find the right chunk more often than a single raw-query search?

> **Historical result** — same caveat as §12: measured pre-consolidation, single document. Not re-run since.

- **Location**: [src/retrieval/query_expansion_eval.py](src/retrieval/query_expansion_eval.py) (comparison logic), question set imported from [src/retrieval/eval_questions.py](src/retrieval/eval_questions.py) (same 8 questions as §12, for direct comparability), CLI wrapper [scripts/run_query_expansion_eval.py](scripts/run_query_expansion_eval.py).
- **Hit rule**: union of original + all 3 `QueryExpansionGenerator` variants. For each question, `SimilaritySearch.search(embedding, top_k)` is run 4 times — once for the raw question, once per generated phrasing — and their retrieved `parent_id`s are unioned; a hit is any expected `parent_id` appearing in that union. This matches §8's own documented intended usage ("search the original plus these variants"). Consequence: since the union always includes the baseline's own results as a subset, multi-query's hit rate is mathematically guaranteed to be **≥** baseline's, never below it — it can only match or improve on baseline, at the cost of ~4x the searches/embeddings.
- **Metric**: run at both `top_k=5` and `top_k=1`, same as §12. `scripts/run_query_expansion_eval.py` runs both in one invocation.
- **Cost control**: reuses [TokenBudgetController](src/utils/billing.py) — one instance for baseline embedding calls, `QueryExpansionGenerator`'s own instance for both its phrasing-generation chat call and its variants' embedding calls.
- **Output**: per-question hit/miss for both modes at both k values, summary hit rates, total cost.

**Results** (measured 2026-09-02):

| Metric | Baseline | Multi-Query |
|---|---|---|
| hit@5 | 100% (8/8) | 100% (8/8) |
| hit@1 | 88% (7/8) | 88% (7/8) |

Multi-query **tied** baseline exactly at both thresholds rather than beating it — the one baseline miss at hit@1 (NSFR) wasn't recovered by any of the 3 generated variants either, so the union added no incremental hit. Given the mathematical floor noted above (multi-query can't score below baseline by construction), a tie here means the technique added zero measurable value on this question set, for roughly 4x the retrieval cost of the baseline alone. Contrast with §12: HyDE actively *hurt* recall at hit@1, while multi-query expansion was merely inert.

## 14. Experimental — Fixed-Window Chunking (v2)

**Status: 🗑️ Tried and removed.** Was a fully separate, parallel branch — not a replacement of §3/§8-§13. Tested whether abandoning cosine-distance semantic chunking in favor of fixed-size sentence windows fixes the fragment-chunk problem diagnosed in §12/§13 (HyDE's hit@1 failures, and the more fundamental finding that `segment_text_semantically`'s breakpoint fires on almost every sentence for this document — a dense sequence of topically-distinct facts, not flowing prose — leaving all 91 course parent blocks with ≤3 sentences each, mostly 1, so the earlier child-window fix had nothing to group). **Findings kept below since the negative result is the valuable part; the code itself added no value once measured, so it was deleted** (tables `parent_documents_v2`/`child_chunks_v2` dropped, all `*_v2.py` files removed) rather than left as unused clutter.

- **Was isolated as**: new tables `parent_documents_v2` / `child_chunks_v2` in the same `data/financial_intelligence.db` (same schema shape as §3.2), plus `src/ingestion/pipeline_v2.py` (`IngestionPipelineV2`), `src/ingestion/vector_store_v2.py` (`SQLiteVectorStoreV2`), `src/retrieval/similarity_search_v2.py` (`SimilaritySearchV2`), `src/retrieval/eval_questions_v2.py`, `src/retrieval/hyde_eval_v2.py`, CLI `scripts/run_ingestion_v2.py` and `scripts/run_hyde_eval_v2.py` — none of §3/§8-§13's files were ever modified. All of the above no longer exist in the repo.
- **What's duplicated vs. reused**: storage/chunking/search layer (text extraction, embedding calls, DB schema, similarity ranking) is duplicated for full isolation. [HyDEGenerator](src/retrieval/hyde.py) and [embed_text](src/retrieval/embedding_utils.py) are reused as-is — pure generation utilities with no DB coupling, unaffected by how chunking works.
- **Chunking algorithm**: no embedding calls to detect boundaries at all. Split into sentences (same regex, same known decimal-splitting limitation as §3.1 — not fixed here either), group into fixed non-overlapping parent windows of **9 sentences**, then within each parent, fixed non-overlapping child windows of **3 sentences** (same child window size as §3.1, now actually able to group multiple sentences since parents are large enough). This also removes the parent-boundary embedding pass entirely — cheaper per run than the course pipeline, not just differently-chunked.
- **Ground truth**: same 8 questions as §12/§13, `expected_parent_ids` re-derived against v2's different partitioning (documented below).
- **Eval**: baseline vs. HyDE, hit@5 and hit@1 — same comparison as §12, run against v2 storage.

**Ground truth** (derived by searching `parent_documents_v2` for each fact's distinctive text after ingestion — 11 parent windows total, vs. the course pipeline's 91):

| # | Question | v2 parent_id(s) |
|---|---|---|
| 1 | Statutory NPAT | {2} |
| 2 | Interim dividend | {2, 3, 4, 5} |
| 3 | Pre-provision profit | {3} |
| 4 | NIM | {3} |
| 5 | CET1 ratio | {4} |
| 6 | ROE | {4} |
| 7 | NSFR | {5} |
| 8 | Loan impairment expense | {3} |

**Results** (measured 2026-09-02, compared against §12's course-pipeline numbers):

| Metric | Course Baseline | Course HyDE | v2 Baseline | v2 HyDE |
|---|---|---|---|---|
| hit@5 | 100% (8/8) | 100% (8/8) | 100% (8/8) | **88% (7/8)** |
| hit@1 | 88% (7/8) | 50% (4/8) | 75% (6/8) | 62% (5/8) |

**Fixed-window chunking did not fix HyDE's problem** — it's a more precise negative result, not a clean win. The baseline/HyDE gap at hit@1 did narrow (38pp → 13pp), but by both sides moving toward the middle: baseline got *worse* (88%→75%, coarser 9-sentence parents apparently cost baseline some precision), while HyDE improved only modestly (50%→62%) and actually got worse at hit@5, regressing where it was previously perfect.

Qualitative spot-check (the NPAT question, same one used throughout §12) shows why: HyDE's hypothetical answer still fabricates a wrong figure post-fix ("AUD 4.9 billion" vs. the real "$5,412m"), and its v2 top-1 match is now the document's header/masthead block (company name, ACN, address) — a *worse*, less complete match than before, not better. Baseline's v2 top-1 for the same question, by contrast, is a genuinely coherent chunk containing the real answer ("...Net profit after tax $5,412m $5,445m Statutory...") — chunking clearly did improve retrieval quality where the problem was structural (baseline benefits from real multi-sentence spans instead of isolated fragments).

**Revised conclusion**: the chunking fix helped baseline (a real structural improvement) but did nothing for HyDE's actual failure mode, which was never primarily about chunk shape — it's about HyDE fabricating specific numbers regardless of what the target chunks look like. That's a generation/prompting problem, not a retrieval-corpus problem, and would need a different intervention (e.g. constraining the hypothetical passage to avoid inventing figures, or restricting HyDE to the qualitative/ratio questions where it already performs comparably to baseline).

## 15. Retrieval — Two-Stage Retrieval + Cohere Rerank

**Status: ✅ Implemented.** Module 04. Precision-over-recall retrieval: a cheap, wide vector-similarity pass for recall, followed by an expensive, precise cross-encoder rerank pass to fix ordering and cut the window down to a small final set.

> **Historical result** — the reordering example and latency numbers below were measured pre-consolidation, single document. The mechanism (`TwoStageRetriever`) is unchanged and current; the specific example/numbers are not.

- **Location**: [src/retrieval/two_stage_retriever.py](src/retrieval/two_stage_retriever.py), class `TwoStageRetriever`. CLI [scripts/run_two_stage_retrieval.py](scripts/run_two_stage_retrieval.py).
- **Interface**:
  - `fetch_candidates(query: str, candidate_k: int = 20) -> list` — stage 1: embeds the query (`embed_text`, same helper as §12/§13) and calls [SimilaritySearch](src/retrieval/similarity_search.py).
  - `rerank(query: str, candidates: list, final_k: int = 5) -> list` — stage 2: calls Cohere's Rerank API (`rerank-english-v3.0`) with the raw query text and each candidate's `chunk_content`, returns the top `final_k` with `relevance_score` and the candidate's original `stage1_rank` attached (so reordering is directly visible, not just the final list).
  - `retrieve(query, candidate_k=20, final_k=5) -> dict` — runs both stages, also returns `stage1_latency_ms` / `stage2_latency_ms` measured via `time.perf_counter()` around each stage.
- **Why two stages, not one**: vector similarity (cosine distance between pre-computed embeddings) is cheap — a dot product — but approximate; it's good at broad topical recall, unreliable at fine-grained ordering (consistent with §12/§13's own hit@1 instability findings). A cross-encoder reranker reads the query and each candidate *together* and scores relevance directly, which is far more precise but requires real inference per (query, document) pair — cost and latency scale with however many documents you hand it. So: cast a wide net cheaply (`candidate_k=20`) to make sure the right chunk is *somewhere* in the candidate set, then spend the expensive precision pass only on that already-small set, not the whole corpus. Reranking everything would pay cross-encoder cost at full corpus scale for no benefit once the corpus grows past a handful of documents; asking the embedding search for `top_k=5` directly reintroduces the exact ranking unreliability this module exists to fix.
- **Cost control**: `TokenBudgetController` for the OpenAI embedding call only (negligible, ~$0.000003/query). Cohere's Rerank API is billed per search request (not OpenAI-token-based), so it is **not** tracked through `TokenBudgetController` — tracking it would require a different unit (requests, not tokens). This run's Cohere usage was within the free trial quota.
- **Cohere API key**: `COHERE_API_KEY` in `.env` was a placeholder (`"your_..."`) at first use — surfaced as a real `401` from Cohere, not a code bug. Resolved once a real trial key was set.

**Results** (measured 2026-09-03, one real run, query: *"What was CBA's statutory net profit after tax for the 2026 half year?"*, `candidate_k=20`, `final_k=5`):

| Rank | Stage 1 (vector similarity) | Stage 2 (Cohere rerank) | Moved from |
|---|---|---|---|
| 1 | parent_id=17 [0.5853] | parent_id=17 [relevance 0.9134] | rank 1 (unchanged) |
| 2 | parent_id=55 [0.5842] | parent_id=1 [relevance 0.7253] | **rank 6** |
| 3 | parent_id=19 [0.5492] | parent_id=23 [relevance 0.6299] | rank 4 |
| 4 | parent_id=23 [0.5437] | parent_id=38 [relevance 0.1588] | **rank 14** |
| 5 | parent_id=34 [0.5020] | parent_id=91 [relevance 0.0291] | **rank 15** |

- **Proof of reordering**: `parent_id=91` was **rank 15** by raw embedding similarity and lands in the **reranked top 5 (rank 5)** — the literal case the module asked to demonstrate. `parent_id=1` (rank 6→2) and `parent_id=38` (rank 14→4) are further examples.
- **Latency** (real, not estimated): stage 1 = **2105.5 ms** (dominated by the OpenAI embedding API round-trip, not the local SQLite scan over 92 rows, which is effectively instant), stage 2 = **300.3 ms** (one Cohere rerank call scoring 20 short documents). Stage 2 being faster than stage 1 here is a property of this specific run (small candidate set, network variance) — not a general claim; stage 2's cost scales with `candidate_k` while stage 1's fixed per-query overhead is the embedding call regardless of corpus size, with the scan portion scaling separately with corpus size.
- **Relevance-score cliff, worth understanding rather than glossing over**: scores drop steeply after rank 1 (0.91 → 0.73 → 0.63 → 0.16 → 0.03). Ranks 4-5 are the *least irrelevant of what's left*, not confidently relevant — in this run the reranker pushed `parent_id=19` (pre-provision profit, genuinely on-topic) out of the top 5 in favor of two document header/letterhead fragments that share surface vocabulary ("Commonwealth Bank of Australia") with the query but not real content overlap. This is the reranker behaving correctly given a candidate set where only 1-3 chunks are truly relevant — not a wiring bug.
- **Not yet done**: a hit-rate eval (§12/§13-style) comparing two-stage retrieval against baseline/HyDE across the full 8-question set — this section is a single real, verified example, not an aggregate measurement.

## 16. Generation — Minimal Answer Generator

**Status: ✅ Implemented.** Module 05. Nothing built in Modules 1-4 produces an actual answer — only relevant chunks. This is the first component that takes a question plus retrieved chunks and writes a real answer, grounded in that context, or honestly says the context doesn't contain it. It exists specifically because Faithfulness (hallucination detection) cannot be scored without a generated answer to check — that's the next step, not yet done.

- **Location**: [src/generation/answer_generator.py](src/generation/answer_generator.py), class `AnswerGenerator`. CLI [scripts/run_answer_generation.py](scripts/run_answer_generation.py).
- **Interface**: `generate_answer(question: str, chunks: list, model: str = "gpt-4o-mini") -> str`. Chunk-source-agnostic — works with output from [SimilaritySearch](src/retrieval/similarity_search.py) or [TwoStageRetriever](src/retrieval/two_stage_retriever.py) (§15), only requires each chunk dict to have `chunk_content` and `source_file`. The demo script feeds it `TwoStageRetriever.retrieve(...)["reranked"]` — the full top-5 post-rerank set, not a single chunk.
- **Prompting**: each chunk is rendered into the context as `[Chunk N | Source: <source_file>] <content>`, and the system prompt instructs the model to treat same-source chunks as one combined context (connecting a fact established in one chunk, e.g. reporting period, with a figure in another) while never merging facts across chunks from *different* sources. Answer using only the provided context; if the answer isn't present, say so explicitly rather than fabricating one.
- **Cost control**: `TokenBudgetController`, same pattern as every other generation/chat component (§8, §9).

**Debugging story worth keeping, not just the final prompt** — this went through three real iterations, each caught by actually running it, not by reasoning about the prompt in the abstract:

1. **v1 prompt** (context-only, honest-refusal instruction, no cross-chunk guidance): produced a **false refusal** on a genuinely answerable question — "What was CBA's statutory net profit after tax for the 2026 half year?" — even with the correct chunk (`parent_id=17`, rank 1, relevance 0.9134) right there in context. Reproduced 3x consecutively, not flaky. Isolated by testing with only that one chunk (still refused — not a multi-chunk noise problem) and by testing the same chunk without "for the 2026 half year" in the question (answered immediately, but with the **wrong figure** — $5,445m instead of the correct $5,412m for Statutory NPAT, a flattened-table column swap: the chunk lists `$5,412m $5,445m` above `Statutory NPAT² Cash NPAT²`, and the model paired the second number with the first label).
2. **First fix attempt** (added an Australian fiscal-half-year date-convention clarification to the prompt): tested, did **not** fix the refusal — wrong hypothesis, reverted.
3. **Correct diagnosis**: the chunk containing the NPAT figures never itself says "2026" or "half year ended" — that context lives in separate chunks (`parent_id=1`, `parent_id=23`, both containing "1H26 Results" / "11 February 2026"). The model was refusing rather than connecting a fact in one chunk with data in another, despite both being handed to it in the same call.
4. **Second fix**: told the model to read chunks as one combined context rather than atomistically. This fixed the refusal, and — as an **unexplained side effect**, verified across 4 consecutive runs — also fixed the column-swap: it now consistently answers $5,412m (correct).
5. **Generalization flaw caught before shipping**: the fix as first written hardcoded "the chunks are fragments of the same source document" — true only because exactly one document is currently ingested (§4.1 lists 6 more sitting unread in `data/raw_documents/`). Would become actively wrong once a second document is ingested — risk of merging facts across unrelated documents.
6. **Final version**: label each chunk with its real `source_file` in the context text, and instruct the model to only connect facts across chunks sharing a source, treating different-source chunks as independent. Re-verified: refusal still fixed, $5,412m still consistent across 4 runs, and the unanswerable-question test (a plausible-sounding question the document doesn't cover — CBA's marketing budget) still correctly refuses throughout every iteration above.

**Honest caveat**: the column-swap fix in step 4 is a confirmed outcome, not a fully understood mechanism — worth being precise that "verified fixed across repeated runs" and "understand exactly why" are different claims, and only the first one is being made here.

- **Verified behavior** (2026-09-07): answerable question → correct, cited answer ($5,412m); unanswerable question (marketing budget — not in the document) → honest refusal, both consistent across repeated runs.
- **Formal evaluation**: §17 scores this generator's output on Faithfulness and Context Precision across all 15 ground-truth questions, not just the 2 manually-checked examples above.

## 17. Evaluation — Faithfulness & Context Precision (DeepEval)

**Status: ✅ Implemented.** Module 06. Scores the generator (§16) on hallucination detection (Faithfulness) and the reranked retrieval set (§15) on ranking quality (Context Precision), across the full 15-question ground truth (`eval_questions_rerank.py`).

> **Historical result** — the debugging story and judge-model comparison below were measured pre-consolidation, single document. §20 re-ran this exact eval against the full 7-document corpus (different finding); neither has been re-run against the consolidated table-aware pipeline (§21). `eval_questions_rerank.py`'s `expected_parent_ids` have since been re-derived for the current DB — the IDs quoted in this section's examples (e.g. `parent_id=17`) no longer match a fresh query.

- **Location**: [src/generation/faithfulness_eval.py](src/generation/faithfulness_eval.py) (`evaluate_faithfulness_and_precision`), CLI [scripts/run_faithfulness_eval.py](scripts/run_faithfulness_eval.py).
- **Framework**: DeepEval, not Ragas. Ragas 0.4.3 (and 0.2.15, tried as a fallback) both hard-fail on import — `from langchain_community.chat_models.vertexai import ChatVertexAI`, an integration removed from current `langchain-community`, unconditionally imported at module load. Downgrading `langchain-community` far enough to fix it risks destabilizing `langchain`/`langgraph`, which this repo needs for real reasons. Removed `ragas` from `requirements.txt`, added `deepeval>=4.2.2`.
- **Interface**: for each question, runs the real pipeline — `TwoStageRetriever.retrieve()` → `AnswerGenerator.generate_answer()` — then builds a DeepEval `LLMTestCase(input, actual_output, expected_output, retrieval_context)` and scores it with `FaithfulnessMetric` and `ContextualPrecisionMetric`. Per-question failures are caught individually (not per-batch — see below) so one bad judge call doesn't drop the other 14 results.
- **Ground truth**: `expected_answer` fields added to the existing 15-question set in [eval_questions_rerank.py](src/retrieval/eval_questions_rerank.py) (kept separate from `eval_questions.py`'s 8, per §13's reasoning) — short reference answers, e.g. `"$5,412 million"`, verified the same way as `expected_parent_ids`: by direct inspection of `data/financial_intelligence.db`.
- **Judge models**: `ContextualPrecisionMetric` uses `gpt-4o-mini` throughout (zero failures, consistently sound reasoning — never needed changing). `FaithfulnessMetric` uses `gpt-4o` (upgraded from `gpt-4o-mini` — see Results). Both are wrapped in a `deepeval.models.llms.openai_model.OpenAIModel` instance (not a bare model-name string) specifically to set `generation_kwargs={"max_completion_tokens": 1000}` — DeepEval's judge calls are structured-JSON-output calls, and a runaway generation (model never converges on valid JSON) otherwise burns the model's full completion budget (16,384 tokens) before failing. Capping it makes the same failure fail in ~10s instead of ~90s+, without changing whether it fails.
- **Cost tracking gap**: DeepEval's own judge-model calls are not tracked through `TokenBudgetController` — the `total_cost_usd` this eval reports covers only the `TwoStageRetriever`/`AnswerGenerator` portion (~$0.0019/run). Real cost is higher, especially with the `gpt-4o` judge (~16x `gpt-4o-mini`'s per-token rate). Same category of gap as Cohere's cost in §15 — a different unit (per-request or a stronger model's rate) that the existing OpenAI-token-based controller doesn't model.

**Debugging story** (mirrors §16's format — the numbers alone don't explain what happened):

1. **First full run** (`gpt-4o-mini` judge, no token cap): 7 of 15 Faithfulness judge calls (47%) failed outright — `openai.LengthFinishReasonError`, `completion_tokens=16384`, the model never producing valid structured output. Before the per-question catch existed, this exception propagated out of the whole loop and killed the run — nothing printed for any of the 15 questions, since the CLI only prints after the full batch returns. Fixed by wrapping each `metric.measure()` call in its own try/except (§ "Interface" above).
2. **Verified the token cap works** on the exact case that had triggered the runaway: failed in 10.4s at the 1,000-token cap instead of ~90s+ at 16,384 — same underlying failure, contained instead of expensive.
3. **First full run with resilience** (`gpt-4o-mini` judge, capped): completed all 15, but avg Faithfulness = **0.44** with 7/15 still hitting the (now-fast) failure. Manually verified all 15 generated answers against ground truth — **every single one was factually correct.** The low scores were false negatives: the judge flagged correct temporal phrasing as contradictions — "2026" vs. "half year ended 31 December 2025" (Q1), "1H26" vs. a "1H25" comparison label it misread as the current period (Q2), "June 2025" vs. "Jun 25" (Q7), added-but-true context (Q12).
4. **Isolated the variable**: `ContextualPrecisionMetric`, same run, same messy chunk content, `gpt-4o-mini` judge — 0/15 failures, consistently sound reasoning. Ruled out "DeepEval is unreliable" and "our chunk content is unparseable" as the primary explanation, since the same judge model handled the same data fine for a different metric. Pointed at `FaithfulnessMetric`'s specific method (strict claim-by-claim structured entailment) combined with judge capability.
5. **Re-ran with `gpt-4o` as the Faithfulness judge only** (Context Precision left on `gpt-4o-mini` — it wasn't broken, no reason to spend more there). Confirms the diagnosis.

**Results** (measured 2026-09-09, 15 questions, `top_k=5`):

| Metric | `gpt-4o-mini` judge | `gpt-4o` judge |
|---|---|---|
| Avg Faithfulness | 0.44 | **0.93** |
| Faithfulness judge failures | 7/15 (47%) | **0/15** |
| Avg Context Precision | 0.80 (unchanged — same judge both runs) | 0.80 |

Upgrading the judge alone fixed the reliability collapse and every phrasing-pedantry false negative except one: **Q1 (the original statutory-NPAT question) still scores 0.00 even with `gpt-4o`** — *"the actual output incorrectly claims $5,412 million... not directly supported by the figures provided."* That's the one chunk with the genuinely ambiguous flattened table (`$5,412m $5,445m` above `Statutory NPAT² Cash NPAT²`) — the same layout that caused the generator's own original column-swap bug (§16). A stronger judge resolves capability-driven false negatives; it doesn't resolve a structural ambiguity in the source data. That distinction is the actual finding here: **judge model choice explains the reliability collapse and most of the phrasing false negatives; the flattened-table chunking problem (traced since §12) is a separate, still-unresolved cause that no judge upgrade fixes.**

## 18. Table-Aware Extraction (merged into the primary pipeline — history below)

**Status: ✅ Merged into [src/ingestion/pipeline.py](src/ingestion/pipeline.py) as of 2026-09-15 (§21).** This started as an isolated branch (`v3`, matching the earlier §14/`v2` precedent) to test whether the flattened-table problem traced since §12 — confirmed in §17 to survive even a `gpt-4o` judge — is fixable at its root: extraction, not chunking or judge capability. It was verified, then consolidated into the primary pipeline directly (§21). **The `_v3` files/tables described below no longer exist** — this section is kept as the historical record of how the algorithm was designed and debugged; the algorithm itself is now simply how `pipeline.py`'s extraction works (§3.1).

- **Original isolation** (now merged, described for history): separate `parent_documents_v3`/`child_chunks_v3` tables; `src/ingestion/pipeline_v3.py`, `vector_store_v3.py`, `src/retrieval/similarity_search_v3.py`, `scripts/run_ingestion_v3.py`. [TwoStageRetriever](src/retrieval/two_stage_retriever.py) (§15) gained an optional `search: SimilaritySearch = None` constructor parameter during this phase (defaults to current behavior for every caller) — that parameter remains, harmlessly, post-consolidation.
- **What was tried and rejected first, in order** (worth keeping — this wasn't the first idea, and the failures are informative): (1) `page.find_tables()` default strategy — swept entire two-column pages into one giant pseudo-table, jumbling the narrative column and the metric-box column together into a *worse* blob than the original flattened text. (2) `find_tables(table_settings={"vertical_strategy": "text", ...})` — fragmented individual words into fake columns based on kerning gaps ("Announcement" → "Annou"/"ncement"). Both confirmed: this document isn't a gridded data table, it's a magazine-style press release with floating metric callouts in flowing 2-column prose — a genuine layout mismatch with what `find_tables()` is built to detect, not a tuning problem.
- **What worked — a three-stage structural approach**, none of its thresholds tied to this document's specific content or coordinates:
  1. **Column-band detection**: find the widest gap in word x-positions near the page's horizontal middle; split into two bands if the gap is wide enough, else treat the page as one band.
  2. **Card segmentation**: within a band, `page.lines` gives real horizontal rule-line positions; they split the band into vertical card segments (confirmed via §18's earlier investigation: this is the actual mechanism connecting the NSFR chunk's original bug — pypdf fusing a stat card straight across a page boundary into the next page's marketing header, with nothing marking where one page's content ends).
  3. **Number/label pairing**: within a card, chars with font size in a *relative* range (1.4x-2.5x the page's most common/body size — bounded above so a one-off page masthead, ~4x body, doesn't get swept in as a "headline number") are treated as headline values; each is paired with the nearest smaller-font text within a proportional vertical gap and horizontal overlap below it.
- **Real bugs found and fixed during verification** (not a clean first pass): initial pairing only grabbed the single nearest label *word*, not the full label phrase (`"NPAT2: $5,412m"` instead of `"Statutory NPAT2: $5,412m"`) — fixed by grouping all label words on the same line. Initial headline-size threshold had no upper bound, so a page's 38pt masthead title got treated as a "headline number" and randomly paired with unrelated nearby text — fixed by bounding the ratio range.
- **Known remaining limitation** (as of the original verification below): not every detected card is clean — the page-0 masthead/section-header region still produces some misattributed fragments. Flagged then as unstarted broader quality auditing; turned out to be a real, more serious bug, caught during consolidation (§21): whenever a region had *any* successful label:value pairing — even a spurious one, like a contact name coincidentally styled larger than its title — the entire rest of that region's body text was silently discarded rather than falling through to narrative. On the CBA half-year document this dropped a whole footnotes page (281 regional branches, $190m CommBank Yello, $25bn business lending, $4.4bn shareholder returns — gone entirely from the corpus, not just misattributed). **Fixed** in `_extract_card_text`: only words actually consumed by a successful pairing are excluded from the region; everything else now flows into narrative text regardless of whether the same region also contained a real card. Re-verified: all previously-missing facts confirmed present after the fix, across all 7 documents (§21).

**Verification** (2026-09-15): ran the exact NPAT question that survived every previous fix (§14's chunking change, §17's judge upgrade) through v3 end-to-end — `TwoStageRetriever(search=SimilaritySearchV3())` → [AnswerGenerator](src/generation/answer_generator.py) (reused as-is) → `FaithfulnessMetric` with the `gpt-4o` judge, exactly matching §17's configuration.

- The v3 table chunk now reads `Statutory NPAT2: $5,412m` / `Cash NPAT2: $5,445m` — clean, correctly paired.
- Generated answer: *"CBA's statutory net profit after tax for the 2026 half year was $5,412 million."* — correct.
- **Faithfulness: 1.00** (was 0.00 in §17, even with `gpt-4o`). Reason: *"no contradictions... everything aligns perfectly."*

**The caveat found in the same test, not separately**: Cohere relevance scores for all 5 reranked candidates were extremely low (0.0398, 0.0021, 0.0015, 0.0009, 0.0003 — compare §15's 0.91 for a clean top-1 match), and the correct NPAT chunk ranked **4th of 5** at 0.0009, behind a generic letterhead fragment. Root cause: the old flattened chunk was bad for the *generator* (ambiguous column pairing) but accidentally good for the *retriever* — it still contained the narrative phrase "Net profit after tax" verbatim, giving real lexical overlap with the question. Stripping the table down to clean `Label: Value` pairs fixed the generation-side grounding problem by removing exactly the surface area retrieval was leaning on. **This didn't just fix a bug — it moved a version of it from generation to retrieval.** In reality, questions won't reliably echo a document's exact labels ("How much profit did CBA make?" shares almost no vocabulary with `Statutory NPAT2: $5,412m`), so this matters beyond the one test case.

- **Proposed next step, not yet built**: since the label:value pairs are structurally verified (not LLM-inferred), deterministically template a natural-language restatement alongside each one — e.g. `f"Net profit after tax was ${value} million ({label})."` — to give retrieval more lexical surface without reintroducing the fabrication risk §17 caught (nothing would be inferred, just verified data restated in prose shape via plain string formatting, not an LLM call).
- **Since done**: merging this extraction into the primary pipeline — §21. A full eval suite re-run against the consolidated pipeline (the way §17 built one originally) is still not done.

## 19. Streamlit UI

**Status: ✅ Implemented, verified end-to-end** (2026-09-15, via a headless-browser drive: real question typed → real answer rendered → 5 cited source chunks with rank/relevance/section shown → no console errors; re-verified after the §21 consolidation with a question targeting previously-missing content). Module 07. Wires up an interface to the pipeline that already answers questions (§15 `TwoStageRetriever` + §16 `AnswerGenerator`) — this module is UI plumbing, not new answering logic. Uses `SimilaritySearch` over `parent_documents`/`child_chunks` — the single primary pipeline (§3.1); no separate branch to choose between.

- **Location**: [app.py](app.py) (repo root, so `streamlit run app.py` matches the literal ask).
- **Flow**: text input → `TwoStageRetriever().retrieve(question)` (stage 1 top-20 vector search, stage 2 Cohere rerank top-5) → `AnswerGenerator().generate_answer(question, chunks)` → render the answer plus the cited source chunks (rank, relevance score, section, snippet) it was grounded in.
- **Real calls, not a mockup**: every question typed hits the real OpenAI/Cohere APIs through the same classes the CLI scripts use — no stubbed responses.
- **Cost control**: retriever/generator instances (and their `TokenBudgetController`s) are cached via `st.cache_resource`, so the existing $0.50 budget cap accumulates across a whole running session rather than resetting per question — same safety mechanism as every other component, just session-scoped instead of run-scoped.
- **Not yet done**: GitHub packaging (README, architecture diagram, `.gitignore`, push) — explicitly deferred by the user until the app is verified working locally first.

## 20. Multi-Document Corpus Expansion — Known RAG Limitation, Not a Bug

**Status: ✅ Ingested, degradation measured and understood.** `scripts/run_ingestion.py` (§3.4) was generalized from one hardcoded file to a folder scan, then run against the remaining 6 previously-unread PDFs in `data/raw_documents/`. `data/financial_intelligence.db` now holds **9,512 parent rows / 9,590 child chunks** across all 7 documents (was 91/91 for one document):

> **Historical row counts/costs** — measured on the pre-consolidation pypdf-based pipeline. §21's consolidation (table-aware extraction, applied to all 7 documents) produced different, generally lower row counts and different costs — see §21 for current numbers. The finding below (cross-document period confusion) is a property of having multiple overlapping documents in the corpus, independent of which extraction pipeline produced the chunks, so it's expected to still apply — just not re-measured yet against the consolidated pipeline.

| Document | Parent chunks | Ingestion cost |
|---|---|---|
| CBA-2026-Annual-Report.pdf | 6,748 | $0.120294 |
| CBA 2026 Half Year Results Profit Announcement.pdf | 1,344 | $0.030196 |
| CBA-2026-Full-Year-Results-Profit-Announcement.pdf | 1,102 | $0.027046 |
| 3Q26-Trading-Update.pdf | 116 | $0.001520 |
| CBA-2026-Full-Year-Results-ASX-Announcement.pdf | 93 | $0.000756 |
| Appendix 3A.1 (dividend notice) | 18 | $0.000781 |
| CBA 2026 Half Year Results ASX Announcement.pdf | 91 | *already ingested, skipped* |

Total: $0.180594. No file hit its budget cap. The annual report's ~6,700 blocks (vs. 91 for the half-year document alone) is exactly proportionate to its size, and is the concrete reason this run took noticeably longer — one unbatched embedding API call per sentence (§7), now at a scale where that cost is visible. Batching those calls is the planned next efficiency step, not yet built.

**Re-ran §17's exact 15-question Faithfulness/Context Precision eval against the expanded corpus, unchanged otherwise** (same questions, same `gpt-4o`/`gpt-4o-mini` judges):

| Metric | Single document (§17) | Full 7-document corpus |
|---|---|---|
| Avg Faithfulness | 0.93 | **0.72** |
| Avg Context Precision | 0.80 | **0.60** |
| Judge failures | 0/15 | 0/15 |

**What's actually happening, not just the score drop**: the corpus now contains the same metrics restated for multiple different reporting periods (1H26, FY26, prior-year comparisons), and retrieval has no mechanism to prefer the period the question actually asked about — vector similarity treats "CBA's NIM for 1H26" and "CBA's NIM for FY26" as nearly identical, since the period is a tiny fraction of the text's semantic content. Concrete cases:
- **Q1**: answered $5,499m (the full-year figure) for a question about "the 2026 half year"; the correct $5,412m sits in the same source table (`"Statutory net profit after tax ($M) 5,499 5,412"`) — wrong column picked. **Faithfulness still scored this 1.00**, since $5,499m is literally present in the retrieved context.
- **Q5**: CET1 answered as the FY26 figure (12.0%) instead of the 1H26 figure (12.3%), same pattern, same 1.00 Faithfulness score.
- **Q8**: fully garbled — four different loan-impairment figures from four different chunks/periods/currencies (NZD75m, $1m, $219m, $41m) dumped in one answer, none matching the expected $319m.
- **Q12**: the correct 1H26 answer (79,000 homes) was marked **0.00** by the judge because a different document now states 150,000 homes for a different (full-year) period — a right answer penalized for a conflict introduced by corpus expansion, not by the answer itself changing.

**The methodological point, not just this project's gap**: Faithfulness measures whether an answer is grounded in *some* retrieved context — not whether it's correct for the question's actual intent. On one document that distinction rarely mattered. On an overlapping multi-document corpus it's the whole story: Q1 and Q5 are both factually wrong answers that scored perfect Faithfulness, because the wrong number was still "in there somewhere."

**This is a known, named class of RAG problem, not a bug specific to this pipeline** — it's the standard failure mode for any corpus with recurring documents (quarterly/annual reports, amended filings, versioned specs). Production systems address it with techniques this project has none of yet:
1. **Metadata filtering** — tagging chunks with structured fields (`reporting_period`, `fiscal_year`) and filtering retrieval by them, not just ranking by vector similarity. `SimilaritySearch`/`SQLiteVectorStore` carry no such fields today — only `source_file` and `section_title`.
2. **Query understanding / routing before retrieval** — classifying which document/period a question is actually about before searching, rather than hoping vector similarity sorts it out. This is what `src/agents/` (still an empty stub, §6) and LangGraph (declared, unused, §5) are for.
3. **Explicit period markers inside chunk text**, not just metadata — [AnswerGenerator](src/generation/answer_generator.py) (§16) already labels each chunk's `source_file` in the prompt; it does not label which *period* a chunk covers, which is the more relevant disambiguator here.
4. **A broader metric suite** — pairing Faithfulness with Answer Correctness/Relevancy (against ground truth) and Context Recall, not relying on Faithfulness alone to catch wrong-but-grounded answers.

**Not yet done**: any of the four mitigations above. This section documents and explains the degradation; it does not fix it.

## 21. Consolidation — One Pipeline, Not Three

**Status: ✅ Done.** By §20, the repo had accumulated three ingestion-adjacent things: the original pypdf-based pipeline (§3.1 as it stood through §20), the fixed-window `v2` branch (§14, tried and already deleted), and the table-aware `v3` branch (§18, isolated and verified on one case). For a repo meant to be read by someone new, that's confusing — "which one is real?" This section merges `v3` into the primary pipeline, retiring the branch entirely, and fixes a real bug found in the process. It also batches ingestion's embedding calls, since that was the next planned step and touches the same file.

### What changed

- **`src/ingestion/pipeline.py`**: `StructuralIngestionPipeline` now uses table-aware extraction ([pdf_table_extraction.py](src/ingestion/pdf_table_extraction.py)) instead of `pypdf`. Class name, public interface (`run_file_ingestion`), and the semantic-chunking algorithm for narrative text are unchanged — only what feeds the chunker changed. Full current behavior described in §3.1, not duplicated here.
- **Batched embeddings**: new `get_embedding_vectors(texts, batch_size=250)` — one OpenAI request per batch instead of per sentence/window/table. `segment_text_semantically`, narrative child-chunking, and table-chunking all switched to it. `get_embedding_vector` (singular) kept for single-text callers elsewhere.
- **Deleted**: `pipeline_v3.py`, `vector_store_v3.py`, `similarity_search_v3.py`, `run_ingestion_v3.py` — logic now lives in the primary (non-suffixed) files. `SQLiteVectorStore`'s schema needed no change (v3's was identical, just different table names); `SimilaritySearch` needed no change either.
- **Dependency cleanup**: `pypdf[crypto]` removed from `requirements.txt` (no longer used anywhere — verified via grep before removal); `ragas` already removed in §17.

### A real bug, found during consolidation, not before

Verifying the merge (checking that previously-known facts — 281 regional branches, $190m CommBank Yello, etc. — were still retrievable) surfaced a genuine data-loss bug, not a cosmetic one: `_extract_card_text` treated a region as either "fully a table" or "fully narrative," all-or-nothing. Any region with **even one** successful label:value pairing — including a spurious one, like a contact name ("Melanie Kirk") coincidentally styled larger than its job title — caused the *entire rest of that region's body text* to be silently discarded, since only the paired lines were ever returned. On the CBA half-year document this dropped an entire footnotes page: 281 regional branches, $190m CommBank Yello value, $25bn business lending, $4.4bn shareholder returns — gone from the corpus entirely, not just misattributed.

**Fixed**: `_extract_card_text` now tracks exactly which label words were consumed by a successful pairing and returns the unconsumed remainder as separate `leftover_text`, which the caller merges into narrative content. A region can now correctly yield *both* a clean table block *and* its surrounding untouched prose, instead of one clobbering the other. Re-verified: all previously-missing facts confirmed present by direct SQL search after the fix, and the clean table blocks (NPAT, CET1, etc.) were unaffected.

### Data reset and ground truth re-derivation

`parent_documents`/`child_chunks` (pypdf-based) and `parent_documents_v3`/`child_chunks_v3` were dropped, then all 7 PDFs re-ingested through the unified, fixed, batched pipeline — twice (once to catch the bug via verification, once with the fix applied), since the DB is the only representation of "what got extracted."

**Final state**: 9,736 parent rows / 9,821 child chunks, $0.169343 total ingestion cost for all 7 documents.

`eval_questions.py` and `eval_questions_rerank.py`'s `expected_parent_ids` were re-derived against this final numbering by direct SQL search for each fact's distinctive text (same technique used every prior time IDs shifted) — **the specific IDs quoted inside §12/§13/§15/§17/§20's historical narrative do not match a fresh query against the current database.** The mechanisms and findings described in those sections remain accurate; only the row numbers are frozen at measurement time.

### An honest note on the "savings," and what batching actually bought

An intermediate (buggy) version of this consolidation cost only $0.037 for all 7 documents — a number that looked like a huge win but was actually an artifact of silently dropping real content; fewer tokens embedded because less text survived extraction. The correct, complete cost ($0.169343) is close to the original pypdf-based total (~$0.181, §20) — genuinely a bit lower, but not dramatically so, once nothing is being thrown away. **Batching's real benefit is wall-clock time** (far fewer sequential network round-trips for a large document like the annual report — thousands of individual embedding calls become tens of batched ones), not cost — OpenAI's embedding pricing is per-token, not per-request, so a correct implementation was never going to be much cheaper, only faster. The bulk of the token-count/row-count reduction that *is* real (9,512 → 9,736 is roughly flat; compare either to what an uncorrected, data-dropping run would show) comes from table-aware extraction no longer feeding the sentence-splitter thousands of garbled, over-fragmented table cells the way `pypdf`'s flattened text did.

### Not done as part of this consolidation

Re-running §12/§13/§15/§17/§20's eval suites against the consolidated pipeline (explicitly flagged in each section) — the debugging *methodology* in those sections is still valid and worth reading; the specific numbers are historical. Also not done: the templated natural-language restatement §18 proposed for the retrieval-confidence trade-off it found, and the metadata-filtering / period-disambiguation work §20 identified as the real fix for cross-document confusion.

## 22. Test Suite — pytest, Mocked APIs/DB

**Status: ✅ Done.** Everything called "tested"/"verified" through §21 was a one-off manual run against real APIs/real data, hand-judged, never repeatable. This section adds an actual automated suite: 73 tests across [tests/](tests/), `pytest tests/ -v`, no real OpenAI/Cohere network calls, no reads/writes against `data/financial_intelligence.db`. `app.py` (Streamlit) is explicitly out of scope — see §6.

### One small source change for testability

`SQLiteVectorStore.__init__` ([src/ingestion/vector_store.py](src/ingestion/vector_store.py)) gained an optional `db_path: str = None` parameter (defaults to the existing hardcoded production path — zero behavior change for every current caller), so tests can point it at a `tmp_path`-backed temp file. Same pattern already used for `TwoStageRetriever`'s optional `search` parameter (§18/§21).

### Structure and approach

```
tests/
├── conftest.py       — shared fixtures: fake OpenAI/Cohere response builders, mock clients, temp-db path
├── ingestion/         — test_pdf_table_extraction.py, test_pipeline.py, test_vector_store.py
├── retrieval/          — test_embedding_utils.py, test_hyde.py, test_query_expansion.py,
│                          test_similarity_search.py, test_two_stage_retriever.py
├── generation/         — test_answer_generator.py, test_faithfulness_eval.py
└── utils/              — test_billing.py
```

- `vector_store.py`'s own tests use **real, temporary SQLite** (`tmp_path`-backed `SQLiteVectorStore(db_path=...)`) — that module's job *is* the SQL logic, so a mock would test nothing real. Every other module's dependency on `SQLiteVectorStore`/`SimilaritySearch` is mocked or faked instead.
- `pdf_table_extraction.py`'s tests use small hand-built fake `page`/`region` objects (duck-typed to the exact pdfplumber surface the module calls — `.chars`, `.lines`, `.crop()`, `.filter()`, `.extract_words()`, `.extract_text()`) rather than rendering real PDFs — pdfplumber's own extraction is an external, already-tested library; the bugs this session actually found lived in the pairing/leftover logic layered on top of it, so that's what's under test. Two are explicit regressions for §21/earlier bugs: a multi-word label ("Statutory NPAT2") must not collapse to its last word, and a region with one successful pairing must still return its unconsumed remainder as `leftover_text`, not drop it (the footnotes-page data-loss bug).
- `faithfulness_eval.py`'s tests use a hand-written `FakeMetric` double (not a bare `Mock`) whose `.measure()` mutates its own `.score`/`.reason` or raises, per a pre-scripted sequence — this exercises the real per-question `try/except` resilience and aggregate-average-excludes-failures logic, matching how DeepEval's real metrics behave (constructed once, called repeatedly, state read off the same object).
- All OpenAI/Cohere clients are constructed with a dummy API key (set in `conftest.py`, harmless — no client library validates a key at construction time) and then have their `.client`/`.cohere_client` attribute swapped for a `MagicMock` before any test calls a method on it, so no test can accidentally reach the network.

### Verification

`pip install -r requirements-dev.txt && pytest tests/ -v` — 73 passed, ~4s, fully offline. Confirmed: `data/financial_intelligence.db`'s mtime unchanged across a full test run; `tests/` contains no real API keys and no reference to the production DB path (only a `tmp_path`-derived temp filename).

## Changelog

- 2026-09-24 — Added §22 Test Suite: 73 pytest tests across `tests/ingestion/`, `tests/retrieval/`, `tests/generation/`, `tests/utils/`, all mocked/faked APIs and a real temp-file SQLite for `vector_store.py`'s own tests. Added optional `db_path` parameter to `SQLiteVectorStore.__init__` for test isolation (zero behavior change for existing callers). Regression tests added for both bugs found during §21 consolidation (label-phrase truncation, leftover-text data loss). Removed the "Tests — empty" gap from §2/§6.
- 2026-09-15 — Added §21 Consolidation: merged the table-aware extraction branch (§18) into the primary ingestion pipeline, deleted all `_v3` files/tables, batched ingestion's embedding calls (`get_embedding_vectors`), removed `pypdf` from dependencies. Found and fixed a real data-loss bug during verification (regions with any successful label pairing were discarding the rest of their body text — lost an entire footnotes page's worth of real facts). Re-ingested all 7 documents (9,736 parent rows, $0.169343), re-derived both eval files' `expected_parent_ids`, re-verified the Streamlit app end-to-end. Added historical-result caveats to §12/§13/§15/§17/§20 and rewrote §2/§3/§5/§6/§18/§19 to describe the single consolidated pipeline instead of implying multiple live branches.
- 2026-09-15 — Added §20 Multi-Document Corpus Expansion: all 7 raw PDFs now ingested (9,512 parent rows, up from 91), §17's eval re-run and dropped (Faithfulness 0.93→0.72, Context Precision 0.80→0.60) due to cross-document/cross-period confusion — concrete examples (Q1, Q5, Q8, Q12) plus the general RAG lesson (Faithfulness measures grounding, not correctness) and named production mitigations (metadata filtering, query routing, period markers, broader metric suite), none yet built. Generalized `scripts/run_ingestion.py` from one hardcoded file to a resilient folder scan (§3.4).
- 2026-09-15 — Added §19 Streamlit UI (Module 07): wraps the course pipeline's `TwoStageRetriever` + `AnswerGenerator` in a real, verified-working app.py — headless-browser-driven test confirmed a real question produces a real answer with cited source chunks, no mockup data.
- 2026-09-15 — Added §18 Experimental — Table-Aware Extraction (v3): pdfplumber-based structural extraction (column bands → rule-line cards → font-size number/label pairing) fixes §17's one persisting Faithfulness failure (0.00→1.00 on the NPAT question) at its root cause. Also surfaced a new trade-off in the same test: clean table chunks have far less retrieval-side lexical surface than the messy originals (Cohere relevance dropped from ~0.91-range to ~0.001-range on the same question), a real concern for real-world question phrasing, not just this test case. Proposed fix (templated natural-language restatement of verified pairs) scoped but not built.
- 2026-09-09 — Added §17 Evaluation — Faithfulness & Context Precision (Module 06): DeepEval (Ragas confirmed non-functional — broken `langchain-community` import, unrelated to this project). Full debugging path from 47% judge-failure rate through a token-cap fix to isolating and confirming judge-capability as the dominant cause (0.44→0.93 avg Faithfulness switching `gpt-4o-mini`→`gpt-4o`), while identifying that one case (the flattened-table NPAT chunk) persists regardless of judge — a data problem, not a judge problem.
- 2026-09-07 — Added §16 Generation — Minimal Answer Generator (Module 05): first component producing real answers instead of just chunks. Documents the full debugging path (false refusal → wrong-hypothesis fix → correct diagnosis → cross-chunk fix → generalization bug caught → source-labeled final fix), not just the end state.
- 2026-09-03 — Added §15 Two-Stage Retrieval + Cohere Rerank (Module 04): wide top-20 vector search, then Cohere rerank down to top-5. Real measured latency (stage 1 2105.5ms, stage 2 300.3ms) and a verified reordering example (parent_id=91: rank 15 → rank 5).
- 2026-09-03 — Removed §14's v2 code (all `*_v2.py` files, `parent_documents_v2`/`child_chunks_v2` tables) — measured and confirmed not to have fixed HyDE's problem, so kept as dead weight with no further value. §14's writeup and Results kept as the historical record of the experiment.
- 2026-09-02 — Recorded §14 v2 Results: fixed-window chunking did not fix HyDE's numeric-lookup problem (hit@5 regressed 100%→88%, hit@1 only narrowed the gap by both sides moving worse/better rather than HyDE cleanly improving). Root cause confirmed generative (HyDE still fabricates wrong figures post-fix), not structural — chunking improved baseline's retrieval quality but couldn't fix a fabrication problem.
- 2026-09-02 — Added §14 Experimental — Fixed-Window Chunking (v2): fully separate parallel branch (new tables, new files) testing fixed-size sentence windows in place of cosine-distance semantic chunking, to isolate this experiment from the course pipeline (§3/§8-§13), which stays untouched.
- 2026-09-02 — Changed child chunking from one sentence per chunk to fixed 3-sentence windows (§3.1), motivated by §12's finding that HyDE's hit@1 failures traced back to fragment-level chunks whose "shape" matched neither a real answer nor a HyDE passage. Updated §7's double-embedding bullet to reflect the new ratio.
- 2026-09-02 — Added §13 Multi-Query Expansion vs. Baseline Hit-Rate Evaluation: same grounded question set as §12, union-of-variants hit rule, run at hit@5 and hit@1; extracted shared `eval_questions.py`/`embedding_utils.py` out of `hyde_eval.py` to avoid duplicating them a second time.
- 2026-09-02 — Recorded actual hit@1 results into §12 (previously only reported in chat): HyDE underperforms baseline (50% vs. 88%) on precise numeric-lookup questions due to fabricating plausible-but-wrong figures.
- 2026-09-02 — Added §12 HyDE vs. Baseline Hit-Rate Evaluation: hand-picked, grounded question set with hit@5 comparison between raw-query and HyDE retrieval.
- 2026-09-02 — Added §10 Similarity Search and §11 Scaling Considerations: the read-side query function over `child_chunks`, making query expansion and HyDE actually usable, plus a dedicated issue/solution table for scaling concerns.
- 2026-09-02 — Added §9 HyDE (Hypothetical Document Embeddings): second piece of the retrieval layer, drafts a hypothetical answer and embeds it in place of the raw query.
- 2026-09-02 — Added §8 Query Expansion (Alternative Phrasing Generator): first piece of the retrieval layer, LLM-based multi-query expansion.
- 2026-09-02 — Initial reverse-engineered spec, written after adding PDF ingestion support and pointing the entrypoint at a real source document.
