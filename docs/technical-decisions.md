# Technical Decisions

This document records the significant architecture and tooling decisions made while building FeliNet, along with the trade-offs each one carried. It's written as a lightweight set of **Architecture Decision Records (ADRs)**.

---

## 1. LLM provider: Groq -> OpenAI `gpt-4o-mini`

**Context:** I started generation on Groq's free tier (Llama 3.3 70B) to keep costs at zero. Over time I hit persistent billing/availability failures that were silently swallowed by a `try/except` in the pipeline, so instead of crashing loudly, roughly **34% of queries were quietly returning errors** - the worst kind of failure, because nothing looked broken.

**Decision:** Migrated generation to OpenAI `gpt-4o-mini`. This meant changing the model name, the env var (`OPENAI_API_KEY`), and the API URL - and cleaning up every stale Groq reference in comments and, most importantly, in log-facing error messages.

**Trade-off:** Gave up "$0 generation" for a few dollars a month. In return I got a reliable provider and a **0% error rate**. I prioritized fixing the log-facing messages first, because stale error text actively misleads debugging - a wrong log is worse than no log.

---

## 2. Hybrid retrieval: BM25 + dense, merged with RRF

**Context:** Dense (semantic) retrieval is great at matching meaning but misses exact terms - breed names, drug names, disease terminology - where the *literal* string matters. Pure keyword search (BM25) has the opposite problem: it nails exact terms but misses paraphrases.

**Decision:** Run both lanes in parallel and merge their ranked results with **Reciprocal Rank Fusion (RRF)**, which combines two ranked lists without needing to tune a weighting between incompatible score scales.

**Trade-off:** Two indices to build and keep in sync instead of one, and more moving parts at query time.

---

## 3. The "same-parquet" rule for both retrieval lanes

**Context:** Because I run two separate indices (Qdrant dense + a BM25 index), there's a subtle failure mode: if the two are built from *different* snapshots of the corpus, they silently drift apart and retrieval quality degrades in ways that are painful to diagnose.

**Decision:** Both lanes are built from the **same parquet file** (`BM25Index.from_parquet`). One source of truth, both indices derived from it.

**Trade-off:** A small amount of coupling - the parquet becomes a required build artifact. In exchange, an entire class of "why are dense and sparse disagreeing?" bugs becomes impossible by construction. Cheap insurance.

---

## 4. Cross-encoder reranking

**Context:** After hybrid retrieval returns a wide set of candidates, the ordering isn't perfect. A cross-encoder (`ms-marco-MiniLM-L-6-v2`) can re-score query–document pairs jointly, which is more accurate than the bi-encoder similarity used for first-pass retrieval.

**Decision:** Adopted the **retrieve-wide, rerank-narrow** pattern - retrieve a broad candidate set via hybrid search, then rerank and keep the top few.

**Trade-off:** This is a deliberate **latency cost**. Reranking roughly doubled end-to-end latency (~1.5 s -> ~3–5 s). On this particular 50-case corpus, source accuracy was already near-ceiling *before* reranking, so the measured accuracy gain here is marginal. I kept reranking in anyway because the benefit grows with corpus size and noise - on this small, clean corpus it's underused, but the pattern is the point, and I can articulate exactly when it pays off.

---

## 5. Fine-tuned `all-MiniLM-L6-v2` over a larger off-the-shelf model

**Context:** General-purpose embedding models are trained on broad web text and underperform on narrow domains with specialized vocabulary (feline veterinary terms, breed names, clinical phrasing).

**Decision:** Fine-tuned the small, fast `all-MiniLM-L6-v2` (384-dim) on ~5K synthetic feline query–passage pairs using `MultipleNegativesRankingLoss`, rather than reaching for a bigger pretrained model.

**Trade-off:** Fine-tuning cost real effort - synthetic data generation, a training run on Colab, and a full corpus re-embed step. But the payoff was concrete: **~10 points of improvement across every retrieval metric** (NDCG@10 0.680 -> 0.785, Accuracy@1 0.481 -> 0.581). A small, domain-tuned model beat larger general ones on-domain while staying CPU-friendly and cheap to run.

---

## 6. Index-time vs. query-time code separation

**Context:** RAG systems mix two kinds of work: offline work you do *once* (chunking, embedding, indexing) and online work you do *per query* (retrieval, reranking, generation). Tangling them makes the codebase hard to reason about.

**Decision:** Enforced a clean boundary - chunking and embedding live in `embeddings/` (index-time); retrieval and generation live in `rag/` (query-time).

**Trade-off:** None substantive. It keeps the system navigable and makes it obvious where any given piece of logic belongs.

---

## 7. Qdrant vector store, in-memory mode for deployment

**Context:** The deployment target is a free Hugging Face Space (CPU, limited resources). Running a persistent vector database server there would be heavy for a read-only demo.

**Decision:** Use Qdrant (chosen for its native hybrid-search support), but in **in-memory mode** for the deployed Space, loading the corpus from a bundled `qdrant_export.parquet` at startup.

**Trade-off:** No persistence across restarts - the index rebuilds from the parquet each time the Space wakes. For a read-only demo that never writes new data, that's perfectly fine, and it keeps the deployment lightweight. It would *not* be the choice for a system that ingests new documents live.

---

## 8. DeepEval for evaluation, run multiple times and averaged

**Context:** DeepEval's core metrics (faithfulness, relevancy) use an LLM as the judge. LLM judges are **non-deterministic** - the same pipeline on the same inputs scored 0.88 on one run and 0.90 on another.

**Decision:** Treat faithfulness as a *distribution*, not a point value - run the judge multiple times and report the average (~0.90, spread 0.88–0.90). Separately, I split cheap metrics from expensive ones: retrieval metrics (source accuracy, latency) require no LLM and run on every invocation, while the LLM-as-judge scoring runs on a schedule to control API cost.

**Trade-off:** Evaluation costs both money and time, so the expensive judge doesn't run per-invocation. I also learned a subtlety worth noting: the pipeline's "I don't know" fallback responses score a trivial 1.00 faithfulness (they cite nothing, so they can't contradict anything), which can *inflate* the mean and mask real differences - a reminder that a metric can be technically correct and still misleading.

---

## 9. DVC with a local remote

**Context:** I wanted proper data versioning - the ability to reproduce any prior state of the corpus - without paying for cloud storage on a portfolio project.

**Decision:** Use DVC with a **local remote** (a folder on my machine). All of `data/` is tracked as a single unit.

**Trade-off:** This is an honest, accepted limitation: because the remote is local-only, someone who clones the repo can't `dvc pull` the actual data files. The versioning discipline and workflow are fully demonstrated; the data just isn't distributed. For a portfolio project that's an acceptable scope cut - the deployed demo ships its own bundled parquet, so the live app works regardless.

---

## 10. Pragmatism over completeness (a recurring theme)

Several smaller decisions followed the same principle - for a solo portfolio project, a working, well-understood minimal implementation beats a heavyweight "complete" one:

- **Guardrails:** custom Pydantic validators + a topic check, rather than a full framework like NeMo Guardrails (steep learning curve for marginal added value here).
- **Git flow:** a simple two-branch `dev` -> `main` flow instead of complex sub-branching.
- **CI:** unit tests only in CI (no LLM calls), using `-p no:deepeval` to keep the pipeline fast, deterministic, and free - expensive eval runs happen deliberately, not on every push.

**Trade-off:** Each of these is "less than a large production team would build." That's the correct call for the context, and being able to name *why* I stopped where I did is itself the point.

---

## Decisions I'd revisit next

Being honest about what's unfinished is part of the record:

- **Confidence-gate calibration.** The retrieval confidence threshold (0.25) was designed for cosine similarity (0–1) and doesn't transfer cleanly to cross-encoder reranker logits (which run roughly −11 to +11). This is a known miscalibration to fix.
- **Latency.** P95 (~4.8 s) is dominated by reranking + generation. Caching and query-based model routing (cheap model for simple queries) are the obvious next optimizations.
- **Cloud DVC remote.** Would restore full reproducibility for anyone cloning the repo.
