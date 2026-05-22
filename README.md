# Autodesk Agentic RAG App

![Autodesk Agentic RAG application flow](figures/Infographic%20Logic%20Final.png)

This repository contains a completed retrieval-augmented generation application for answering Autodesk product and workflow questions with grounded evidence. The app cleans a raw Autodesk HTML corpus, builds local dense and lexical retrieval indexes, optionally adds web evidence, reranks candidate evidence, checks whether the evidence is sufficient, and then returns a concise sourced answer or a conservative no-answer response.

The production application is a Streamlit app in `app/streamlit_app.py`. The core RAG logic lives in `src/`, with reproducible corpus cleaning, indexing, evaluation, and monitoring artifacts included for reviewers.

## Reviewer Guide

| Area | Location | Why it matters |
|---|---|---|
| Streamlit app | `app/streamlit_app.py` | User-facing chat, search mode controls, evaluation dashboard, monitoring views |
| RAG agent | `src/agent.py` | Evidence-gated answer generation and runtime orchestration |
| Retrieval | `src/retriever.py`, `src/context_expansion.py`, `src/reranker.py` | Hybrid Chroma/BM25 retrieval, neighbor expansion, cross-encoder reranking |
| Corpus cleaning | `scripts/corpus_cleaning_pipeline.py`, `notebook_01_corpus_cleaning.ipynb` | HTML-to-Markdown cleaning and metadata enrichment |
| Index building | `scripts/build_retrieval_indexes.py`, `notebook_02_build_retrieval_indexes.ipynb` | Docling-aware chunking, Chroma embeddings, BM25 artifacts |
| Golden dataset | `eval_testset/autodesk_testset.csv` | Fixed 50-question evaluation set |
| Evaluation outputs | `eval_results/` | Saved metrics for all three search modes |
| Cleaning diagnostics | `cleaned_corpus_info/` | Corpus EDA, before/after stats, purge manifest, repeated-line candidates |
| Index manifests | `retrieval_indexes/manifests/` | Chunk and indexing reproducibility metadata |

## Completed System

The app supports three runtime search modes:

| Option | Search mode | Behavior | Best use |
|---|---|---|---|
| 1 | Local Document Search | Uses only local Autodesk corpus chunks from Chroma and BM25. | Fast, controlled local-corpus baseline. |
| 2 | Local Document Search + Autodesk.com | Combines local retrieval with official `autodesk.com` web evidence. | Preferred review/demo mode for current official Autodesk facts. |
| 3 | Local Document Search + Open Web Search | Combines local retrieval with capped open-web evidence. | Broader corroboration when official-only search may miss context. |

All modes use the same local retrieval backbone:

- Chroma vector search with OpenAI `text-embedding-3-small` embeddings.
- BM25 keyword search over chunk text plus enriched metadata.
- Weighted reciprocal rank fusion to combine dense and lexical rankings.
- Per-source caps to prevent one document from dominating the candidate set.
- Same-document neighbor expansion to restore chunk-boundary context.
- Cross-encoder reranking with `cross-encoder/ms-marco-MiniLM-L6-v2`.
- A strict adequacy gate before answer generation.

If the supplied evidence does not explicitly support an answer, the app returns:

```text
I could not find a reliable answer in the available documents or web sources.
```

## Run The App

Install dependencies, configure environment variables, then run:

```bash
streamlit run app/streamlit_app.py --server.port=8502
```

Open:

```text
http://localhost:8502
```

The interface has four user-facing areas:

- **Ask**: chat with the Autodesk RAG agent.
- **Settings & Eval**: choose the active search mode and launch evaluation.
- **Monitoring**: inspect logged interactions and latency/retrieval diagnostics when Supabase is configured.
- **About the App**: review architecture notes and guardrails.

## App UI

### Ask

![Autodesk RAG App Ask UI](figures/Autodesk%20RAG%20App%20UI%20Ask.png)

### Settings & Eval

![Autodesk RAG App Settings Eval UI](figures/Autodesk%20RAG%20App%20UI%20Settings%20Eval.png)

### Monitoring

![Autodesk RAG App Monitoring UI](figures/Autodesk%20RAG%20App%20UI%20Monitoring.png)

### About The App

![Autodesk RAG App About App UI](figures/Autodesk%20RAG%20App%20UI%20About%20App.png)

## Results

All three search modes were evaluated against the fixed 50-question golden dataset in `eval_testset/autodesk_testset.csv`. Metrics are saved in `eval_results/` and scored on a 0.00 to 1.00 scale, with latency in seconds.

| Option | Search mode | Faithfulness | Answer relevance | Context precision | Context recall | Avg latency | P50 latency | P99 latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Local Document Search | 0.91 | 0.66 | 0.72 | 0.58 | 6.31s | 6.55s | 15.11s |
| 2 | Local Document Search + Autodesk.com | 0.87 | 0.85 | 0.81 | 0.74 | 12.99s | 12.43s | 22.87s |
| 3 | Local Document Search + Open Web Search | 0.84 | 0.78 | 0.86 | 0.62 | 14.11s | 12.34s | 73.17s |

Option 2 is the strongest overall mode. It improves answer relevance and context recall by adding official Autodesk web evidence while keeping authority risk lower than open-web search. Option 1 remains the fastest and most faithful local-only baseline. Option 3 can retrieve useful broader evidence, but it has less predictable latency and authority.

## Corpus Cleaning And EDA

The raw corpus contains Autodesk HTML pages under `raw_corpus/`. The completed cleaning pipeline converts those pages into RAG-ready Markdown under `cleaned_corpus/` and writes diagnostics to `cleaned_corpus_info/`.

Key corpus processing results from `cleaned_corpus_info/cleaning_summary.md` and `cleaned_corpus_info/before_after_processing_stats.md`:

| Metric | Value |
|---|---:|
| Raw HTML files found | 1,218 |
| Initially cleaned Markdown files | 1,218 |
| Retained cleaned Markdown files after purge | 974 |
| Purged cleaned Markdown files | 244 |
| Cleaned file count reduction from purge | 20.03% |
| Raw corpus size | 303.27 MB |
| Cleaned Markdown corpus size | 4.37 MB |
| Raw character count | 317,446,072 |
| Cleaned character count | 3,574,578 |
| Approximate character reduction | 98.87% |

Cleaning removes scripts, styles, navigation, menus, cookie notices, page chrome, repeated layout text, very small documents, and known non-English documents. It preserves useful structures such as headings, lists, links, tables, and code-like blocks where possible.

The cleaning diagnostics folder is intentionally included for review:

- `cleaned_corpus_info/cleaning_summary.md`: overall cleaning results and largest reductions.
- `cleaned_corpus_info/before_after_processing_stats.md`: EDA-style before/after corpus size analysis.
- `cleaned_corpus_info/cleaning_manifest.csv`: per-file cleaning metadata.
- `cleaned_corpus_info/purged_cleaned_documents.csv`: files removed after enrichment.
- `cleaned_corpus_info/repeated_line_candidates.csv`: repeated boilerplate candidates found during analysis.

## Retrieval Indexes

The indexed corpus contains 974 cleaned Markdown documents and 19,611 chunks. Index metadata is saved in `retrieval_indexes/manifests/indexing_summary.md`.

| Metric | Value |
|---|---:|
| Cleaned files discovered | 974 |
| Files indexed | 974 |
| Files skipped | 0 |
| Files failed | 0 |
| Total chunks | 19,611 |
| Average chunks per document | 20.13 |
| Average chunk length | 187.1 characters |
| Longest chunk length | 4,002 characters |
| Embedding model | `text-embedding-3-small` |
| Embedding dimension | 1,536 |
| Chroma collection count | 19,611 |
| BM25 chunk count | 19,611 |
| Docling available | True |
| Docling HybridChunker enabled | False |

Generated retrieval artifacts:

```text
retrieval_indexes/
    chroma_autodesk_cleaned_corpus/
    bm25_autodesk_cleaned_corpus/
    manifests/
        indexing_manifest.csv
        chunk_manifest.csv
        indexing_summary.md
```

Docling is used for document-aware parsing where available, followed by the project heading-aware chunk-size guard. `DOCLING_USE_HYBRID_CHUNKER=false` is the default because the HybridChunker path can trigger tokenizer sequence-length warnings on long technical pages.

## Architecture

At runtime, the app follows this flow:

1. The user asks a natural-language Autodesk question in Streamlit.
2. The selected search mode determines whether retrieval is local-only, local plus Autodesk.com, or local plus capped open web.
3. Local retrieval searches both Chroma and BM25.
4. Results are merged with weighted reciprocal rank fusion.
5. Neighboring chunks from the same document are added for context continuity.
6. Web snippets, when enabled, are converted into evidence blocks.
7. Local and web evidence are reranked together with the cross-encoder.
8. The adequacy gate checks whether the retrieved evidence explicitly supports the answer.
9. The LLM generates a concise sourced answer, or the app returns the fixed no-answer message.
10. Optional monitoring records request, retrieval, latency, source, model, and outcome metadata.

Core modules:

```text
src/
    agent.py
    retriever.py
    reranker.py
    context_expansion.py
    evaluation.py
    evaluation_runner.py
    monitoring.py
    config.py
    tools.py
```

## Evaluation

The golden dataset is fixed at:

```text
eval_testset/autodesk_testset.csv
```

It contains 50 manually curated questions and ground truths:

| Tier | Questions | Purpose |
|---|---:|---|
| Required reviewer questions | 6 | Autodesk-specified product and support questions |
| Simple fact-based | 15 | Direct corpus lookup |
| Reasoning | 14 | Inference across one or two documents |
| Multi-context | 15 | Synthesis across three or more documents or product families |

The dataset source of truth is `eval_testset/generate_testset.py`, which writes the CSV without making LLM calls at runtime. Ground truths were authored from reviewed corpus evidence and include inline source titles. Claude was used during dataset authoring while the app uses OpenAI models, reducing the risk that the evaluated app simply matches its own model family.

Evaluation from the Streamlit **Settings & Eval** tab creates or reuses a LangSmith dataset keyed by the golden dataset hash, runs the selected search mode as a LangSmith experiment, scores faithfulness, answer relevance, context precision, and context recall, then caches results locally for the dashboard.

Saved evaluation files:

```text
eval_results/docling_chroma_bm25_hybrid_results.json
eval_results/docling_chroma_bm25_hybrid_autodesk_web_results.json
eval_results/docling_chroma_bm25_hybrid_open_web_results.json
eval_results/eval_results_log.csv
eval_status/docling_chroma_bm25_hybrid_status.json
eval_status/docling_chroma_bm25_hybrid_autodesk_web_status.json
eval_status/docling_chroma_bm25_hybrid_open_web_status.json
```

## Monitoring

Runtime monitoring is optional and backed by Supabase when configured. Each logged interaction can include the submitted question, generated response, anonymous session/request IDs, selected search policy, local/web usage, web fallback details, answerability outcome, source count, retrieved context size, top source names and scores, pipeline latency, model names, token/cost metadata when available, success/error status, and LangSmith metadata.

The Streamlit monitoring view includes:

- Summary metrics for total requests, latency, no-answer rate, web fallback rate, adequacy pass rate, source count, token use, and estimated cost.
- Backend and web-usage charts.
- Top source document frequency.
- Recent interaction debugging table.
- Stage-level latency diagnostics.

This is a portfolio proof of concept and may log full questions and generated responses. Avoid entering sensitive, private, or confidential information.

## Environment Configuration

Create a `.env` file with the required API keys and runtime settings. Important variables include:

```text
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

CHROMA_COLLECTION_NAME=autodesk-rag
STREAMLIT_HOST_PORT=8502

CHUNK_SIZE=3500
CHUNK_OVERLAP=500
RETRIEVER_K=10
HYBRID_CANDIDATE_K=30
HYBRID_VECTOR_WEIGHT=0.65
HYBRID_BM25_WEIGHT=0.35
HYBRID_MAX_PER_SOURCE=3
MIN_RELEVANCE_SCORE=0.30

CONTEXT_EXPANSION_ENABLED=true
CONTEXT_EXPANSION_MODE=neighbors
CONTEXT_NEIGHBOR_WINDOW=1
CONTEXT_MAX_EXPANDED_DOCS=8
CONTEXT_MAX_CHARS=18000

RERANKER_ENABLED=true
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2
RERANKER_TOP_N=5
RERANKER_BATCH_SIZE=16

DOCLING_ACCELERATOR_DEVICE=cuda
DOCLING_NUM_THREADS=4
DOCLING_DO_OCR=false
DOCLING_BATCH_SIZE=1
DOCLING_MAX_PAGES=250
DOCLING_PAGE_CHUNK_SIZE=30
DOCLING_PAGE_OVERLAP=5
DOCLING_USE_HYBRID_CHUNKER=false

SUPABASE_URL=
SUPABASE_KEY=
MONITORING_ADMIN_PASSWORD=
```

## Reproduce The Pipeline

Use the notebooks for a guided review workflow:

```bash
jupyter notebook notebook_01_corpus_cleaning.ipynb
jupyter notebook notebook_02_build_retrieval_indexes.ipynb
```

Or run the companion scripts:

```bash
python scripts/corpus_cleaning_pipeline.py
python scripts/build_retrieval_indexes.py
```

Before rebuilding indexes, confirm that `.env` contains a valid `OPENAI_API_KEY` and keep `DOCLING_USE_HYBRID_CHUNKER=false` unless intentionally testing Docling HybridChunker behavior.

## Repository Notes

Generated retrieval indexes can become large. Commit source code, notebooks, prompts, README, small manifests, and evaluation summaries normally. Keep large generated vector databases out of Git unless there is a specific reason to version them, or use Git LFS for large index artifacts.

Suggested `.gitignore` entries for large generated artifacts:

```text
retrieval_indexes/chroma_autodesk_cleaned_corpus/
retrieval_indexes/bm25_autodesk_cleaned_corpus/
*.pkl
*.sqlite3
*.sqlite
```

## Deployment

The app is designed for Streamlit Community Cloud with a private GitHub repository and optional password protection. Supabase can be used as durable monitoring storage for hosted demos.
