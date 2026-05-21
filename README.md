# Autodesk RAG App

## Project Overview

This project builds a retrieval-augmented generation (RAG) application over a corpus of Autodesk HTML webpages. The goal is to provide grounded, accurate answers to customer questions about Autodesk software products, ranging from general product questions to technical usage questions.

The project follows a step-by-step workflow inspired by lessons learned from the Cobb County RAG app:

1. Clean raw Autodesk HTML pages into RAG-ready Markdown.
2. Build retrieval indexes from the cleaned corpus.
3. Use hybrid retrieval with dense vector search and BM25 keyword search.
4. Generate grounded answers with strict evidence constraints.
5. Evaluate retrieval and answer quality using a golden dataset and RAGAS/LangSmith.

## Business Problem

Autodesk customers need fast, trustworthy answers about Autodesk software products and workflows. A successful RAG app should improve customer experience and user satisfaction by answering questions with clear supporting evidence from Autodesk source material.

Longer term, this type of tool could also reduce support burden by helping users resolve questions before creating support tickets.

## Available Data

- Raw Autodesk webpage corpus stored as HTML files in `raw_corpus/`.
- Cleaned Markdown corpus generated into `cleaned_corpus/`.
- Retrieval indexes generated into `retrieval_indexes/`.

## Current Pipeline Status

The project now includes the corpus pipeline, retrieval indexes, a Streamlit Agentic RAG app, and a fixed golden evaluation set.

### Run the Streamlit App

The app lives at:

```text
app/streamlit_app.py
```

Run it locally with:

```bash
streamlit run app/streamlit_app.py --server.port=8502
```

Open:

```text
http://localhost:8502
```

The interface has three tabs:

- **Ask** — chat with the Autodesk RAG agent.
- **Settings & Eval** — choose the active search mode and launch background evaluation.
- **About the App** — explain the architecture and guardrails.

All three search modes use the same local hybrid retrieval backend: the app searches the existing Chroma index at `retrieval_indexes/chroma_autodesk_cleaned_corpus`, searches the BM25 artifacts at `retrieval_indexes/bm25_autodesk_cleaned_corpus`, fuses rankings with Reciprocal Rank Fusion, and expands same-document neighboring chunks.

Before the adequacy gate runs, the app reranks the evidence candidate set with the open-source SentenceTransformers cross-encoder `cross-encoder/ms-marco-MiniLM-L6-v2`. In Option 1, that candidate set contains expanded local chunks only. In Options 2 and 3, individual web result snippets are converted into evidence blocks and reranked together with the expanded local chunks. The reranker keeps the best `RERANKER_TOP_N` evidence blocks for the strict adequacy gate and final answer generation.

The Settings & Eval radio button controls the web policy:

| Option | Label | Behavior |
|---|---|---|
| 1 | Local Document Search | Default. Uses only local Autodesk corpus documents. The lightweight router does not assess whether web search is suitable, and web search is disabled. |
| 2 | Local Document Search + Autodesk.com | Uses local documents and always incorporates SerpAPI Google results restricted to `autodesk.com/*` pages. This mode keeps web evidence official-source focused. |
| 3 | Local Document Search + Open Web Search | Uses local documents and always incorporates open web search results. Open-web retrieval is capped at 3 results to reduce latency and noise. |

For Option 2, web search uses up to 5 `autodesk.com` results. For Option 3, open web search uses up to 3 results. Local document retrieval remains the primary evidence source in every mode, but in web-enabled modes the cross-encoder can promote web snippets into the final evidence set when they are more relevant than local chunks.

Option 3 is useful when an answer may require broader web evidence, but it is less authoritative than Option 2 because open-web results can mix official Autodesk pages with third-party or stale pages. For current/latest questions, the adequacy gate checks the reranked web evidence separately; if the web evidence independently contains the exact current fact, it is allowed to override older local corpus evidence rather than letting stale local context cause a false refusal. Option 2 remains the preferred mode for official Autodesk product/version/pricing answers.

Evaluation uses the fixed dataset at `eval_testset/autodesk_testset.csv` and persists results under:

```text
eval_results/docling_chroma_bm25_hybrid_results.json
eval_results/docling_chroma_bm25_hybrid_autodesk_web_results.json
eval_results/docling_chroma_bm25_hybrid_open_web_results.json
eval_status/docling_chroma_bm25_hybrid_status.json
eval_status/docling_chroma_bm25_hybrid_autodesk_web_status.json
eval_status/docling_chroma_bm25_hybrid_open_web_status.json
eval_results/eval_results_log.csv
```

When evaluation is launched from Settings & Eval, the app now creates or reuses a LangSmith dataset named from the golden dataset hash, runs the selected search mode as a LangSmith experiment, applies LLM-as-judge evaluators for faithfulness, answer relevance, context precision, and context recall, and then caches the resulting scores locally for the Streamlit dashboard. The dashboard status file is updated during the background run so progress can be refreshed while the 50 questions are being answered and scored.

The `example_app/` folder is reference material only. The production app code for this project is under `app/` and `src/`.

### 1. Corpus Cleaning

The corpus cleaning pipeline converts raw Autodesk HTML files into cleaner Markdown documents.

Primary files:

```text
notebook_01_corpus_cleaning.ipynb
corpus_cleaning_pipeline.py
```

Main cleaning approach:

- BeautifulSoup removes deterministic boilerplate, including scripts, styles, navigation, headers, footers, hidden elements, cookie banners, and repeated page chrome.
- Trafilatura extracts main content as Markdown.
- BeautifulSoup fallback logic preserves tables, headings, lists, links, and code-like blocks when Trafilatura is too sparse.
- Cleaned Markdown front matter is enriched without an LLM using deterministic heading extraction, Lingua language detection, and corpus-level TF-IDF keywords.
- After enrichment, cleaned Markdown files under 600 bytes and known non-English documents are purged from `cleaned_corpus/`.
- Cleaned files are written to `cleaned_corpus/`.
- Cleaning diagnostics are written to `cleaned_corpus_info/`.

Important generated outputs:

```text
cleaned_corpus/
cleaned_corpus_info/cleaning_manifest.csv
cleaned_corpus_info/cleaning_summary.md
cleaned_corpus_info/before_after_processing_stats.md
cleaned_corpus_info/purged_cleaned_documents.csv
cleaned_corpus_info/repeated_line_candidates.csv
```

### 2. Retrieval Index Building

The retrieval indexing pipeline builds both dense and lexical retrieval indexes from the cleaned corpus.

Primary files:

```text
notebook_02_build_retrieval_indexes.ipynb
build_retrieval_indexes.py
```

Main indexing approach:

- Loads configuration from `.env`.
- Uses Docling-first document-aware parsing and chunking where available.
- Avoids Docling `HybridChunker` by default because it can trigger tokenizer sequence-length warnings on long technical pages.
- Prefers Docling structure-aware chunking/export followed by the project heading-aware chunk-size guard.
- Uses OpenAI embeddings for dense vector search.
- Stores dense embeddings in persistent ChromaDB.
- Builds a local BM25 keyword index for lexical search. BM25 uses chunk text plus enriched cleaned-corpus metadata such as subheadings, document headings, TF-IDF keywords, and language fields.
- Includes retrieval sanity-check helpers for vector, BM25, and hybrid search.
- Saves document-level and chunk-level manifests for reproducibility.

Important generated outputs:

```text
retrieval_indexes/
    chroma_autodesk_cleaned_corpus/
    bm25_autodesk_cleaned_corpus/
    manifests/
        indexing_manifest.csv
        chunk_manifest.csv
        indexing_summary.md
```

### 3. Golden Dataset Generation

The golden dataset is a 50-question evaluation test set used to measure retrieval and answer quality with RAGAS and LangSmith.

Primary files:

```text
eval_testset/generate_testset.py
eval_testset/autodesk_testset.csv
```

#### Production process

`generate_testset.py` encodes all 50 `(question, ground_truth)` pairs directly as a `TEST_CASES` list of Python tuples and writes them to `autodesk_testset.csv` (columns: `question`, `ground_truth`). The script is standalone — no LLM API calls at runtime — because ground truths were authored manually by reading the cleaned corpus documents and applying the production rules below.

**Why Claude, not OpenAI, authored the ground truths:** To reduce evaluator bias, the evaluation dataset was produced using a Claude model (Anthropic) rather than the OpenAI model used by the app. This means the evaluator and the app use different LLM families, reducing the risk that the app scores well simply because it mimics its own training signal.

#### Question difficulty tiers

The 50 questions are divided into four tiers of increasing complexity:

| Tier | Questions | Description |
|------|-----------|-------------|
| Required (first 6) | Q1–Q6 | Questions specified by Autodesk interviewers: Fusion 360, AutoCAD vs Revit, AutoCAD LT 3D, Maya release, Fusion on Mac, subscription plans |
| Simple fact-based | Q7–Q21 | Direct lookup questions: productivity gains, OS support, revenue figures, previous version counts, product descriptions |
| Reasoning | Q22–Q35 | Why/how questions requiring inference across one or two documents: trade-off analysis, design decisions, architectural choices |
| Multi-context | Q36–Q50 | Multi-hop questions requiring synthesis across three or more corpus documents or product families |

#### Production rules

All ground truths follow four strict rules:

1. **Corpus-first grounding** — the primary source for every answer is a specific local corpus document, cited by its full document title inline within the answer text (e.g., `"According to the document 'Autodesk AutoCAD LT 2024 | Get Prices & Subscribe To AutoCAD LT'..."`). This allows RAGAS faithfulness metrics to verify that the RAG app retrieves and cites the same evidence.

2. **Inline sourcing** — document titles are embedded directly in the answer body rather than in footnotes. This format is compatible with RAGAS answer relevance and faithfulness scoring, which compare retrieved context against the ground truth answer.

3. **Answer format** — each ground truth is 2–3 short paragraphs. The first paragraph cites the primary corpus document; subsequent paragraphs add supporting corpus evidence or acknowledge gaps.

4. **Negative constraint** — when a specific fact was not found in the reviewed corpus documents, the answer explicitly states this (e.g., `"Detailed capability descriptions were not found in the reviewed corpus documents."`) and may cite autodesk.com as a secondary source. This prevents inflated recall scores from answers that assume the corpus is complete.

#### Source abbreviations used in the script

The script header defines short source aliases used during authoring for traceability:

```text
[LT-Product]  = "Autodesk AutoCAD LT 2024 | Get Prices & Subscribe To AutoCAD LT"
[PrevVer]     = "Autodesk Account Basics | Previous Product Versions | Available Versions"
[Fusion-Comp] = "Compare Fusion 360 vs Fusion 360 for Personal Use | Autodesk"
[Fusion-Mfg]  = "Autodesk Fusion Manufacturing Cloud | Autodesk Fusion"
[ArchProd]    = "Benefits of the Architecture Toolset | AutoCAD | Autodesk"
[ElecCase]    = "Martz Technologies, Inc.| AutoCAD Electrical Toolset| Autodesk"
[Q3FY24]      = "AUTODESK, INC. ANNOUNCES FISCAL 2024 THIRD QUARTER RESULTS"
[RevitLT]     = "Autodesk Revit LT Software | Get Prices & Buy Official Revit LT 2023"
[ThomasH]     = "Thomas & Hutton | Site Development Drives the Future of Building Design"
[TradeIn]     = "Trade in Your Perpetual License | Global Promotions | Autodesk"
[ECAD]        = "Autodesk Fusion 360 | ECAD and MCAD | Software Collaboration Tools"
[BIMCollab]   = "BIM Coordination & Collaboration | Autodesk BIM Collaborate"
```

#### Generated output

```text
eval_testset/
    generate_testset.py      — source of truth; re-run to regenerate the CSV
    autodesk_testset.csv     — 50 rows, columns: question, ground_truth
```

## Environment Configuration

The project uses a `.env` file for model, embedding, Chroma, Docling, chunking, and retrieval settings.

Current important `.env` variables include:

```text
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai

OPENAI_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

EMBEDDING_BATCH_SIZE=32
EMBEDDING_BATCH_DELAY_SECONDS=3.0
EMBEDDING_MAX_RETRIES=8

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
```

The indexing workflow should respect these values before falling back to defaults.

## Embedding And Retrieval Design

The current retrieval system uses a hybrid search architecture:

### Dense Semantic Retrieval

- Embedding provider: OpenAI
- Embedding model: `text-embedding-3-small`
- Vector database: ChromaDB
- Chroma collection: `autodesk-rag`

Dense retrieval is useful for semantic matches where the user query and source text use different wording.

### Lexical Retrieval

- BM25 keyword index
- Local persisted artifacts
- Technical-token-aware tokenizer

BM25 is useful for exact Autodesk terminology, product names, API names, error codes, parameters, and short technical phrases.

During index building, BM25 uses a dedicated lexical text field, `bm25_text`, rather than the dense embedding text alone. This field includes the normal chunk text plus selected enriched front-matter fields from cleaning: `subheadings`, `headings`, `tfidf_keywords`, `document_language`, and `document_language_name`. The Chroma/vector embedding text remains lighter, with only title, section, source, and passage text, so metadata keywords help lexical recall without over-steering semantic embeddings.

### Hybrid Retrieval

Hybrid retrieval combines dense vector results and BM25 results using weighted reciprocal rank fusion (RRF). By default, the app retrieves a deeper candidate pool of 30 dense semantic chunks and 30 BM25 keyword chunks, then fuses them with a 0.65 vector weight and 0.35 BM25 weight. It also caps the final fused local list to 3 chunks per source document before context expansion. This keeps BM25's exact-term strength while reducing cases where title keyword matches or one very high-scoring document overwhelm more semantically relevant body-text evidence.

### Cross-Encoder Reranking

After local retrieval and deterministic context expansion, the app reranks evidence with `cross-encoder/ms-marco-MiniLM-L6-v2` from the `sentence-transformers` library. The reranker scores `(query, passage)` pairs and sorts the candidate evidence blocks by cross-encoder relevance before the adequacy gate and final generation.

This is a second-stage reranker, not a retriever. Chroma and BM25 still find the initial local candidate set; Reciprocal Rank Fusion still combines dense and lexical rankings; neighbor expansion still protects chunk-boundary context. In Option 1, the cross-encoder selects the strongest local evidence blocks. In Options 2 and 3, SerpAPI web results are converted into individual evidence blocks and reranked together with local chunks, so web snippets can compete directly with local evidence before the strict adequacy gate runs.

### Runtime Search Modes

The Streamlit app exposes three runtime search modes:

- **Option 1: Local Document Search** — local Chroma + BM25 only. This is the default and fastest mode.
- **Option 2: Local Document Search + Autodesk.com** — local Chroma + BM25 plus official Autodesk web results on every query. Local and web evidence are reranked together before adequacy checking.
- **Option 3: Local Document Search + Open Web Search** — local Chroma + BM25 plus capped open-web results on every query. Local and web evidence are reranked together before adequacy checking.

The same strict answer-generation rules apply in all modes: every factual claim must be supported by supplied local excerpts, supplied web snippets, or runtime context. If the supplied evidence is insufficient, the app returns the fixed no-answer response.

## Docling Chunking

Chunking is intended to be context-aware rather than blind fixed-size splitting.

The latest indexing update avoids Docling `HybridChunker` by default because it can emit warnings such as:

```text
Token indices sequence length is longer than the specified maximum sequence length for this model (580 > 512)
```

That warning is caused by Docling's chunker/tokenizer path, not by OpenAI embeddings. The current approach keeps Docling in the pipeline but avoids the warning-prone hybrid chunker path unless explicitly enabled.

Current Docling strategy:

1. Use Docling to convert/read each cleaned Markdown or text document.
2. Prefer structure-aware Docling chunking or Docling export where supported by the installed Docling version.
3. Apply the project heading-aware splitter and chunk-size guard after Docling export.
4. Fall back to the raw Markdown heading-aware splitter only when Docling conversion fails for a specific file.
5. Record the chunking method in both the document and chunk manifests.

The default behavior is controlled by:

```text
DOCLING_USE_HYBRID_CHUNKER=false
```

Keep this set to `false` unless there is a specific reason to test HybridChunker again.

The chunking workflow preserves:

- Source file path
- Relative source path
- Document title
- Heading hierarchy
- Section context
- Chunk index
- Character count
- Approximate token count
- Source metadata
- Source URL if available
- Chunking method used

Chunk text should include lightweight context before embedding, for example:

```text
Title: <document title>
Section: <heading path>
Source: <relative source path>

<chunk text>
```

This improves retrieval quality because the embedding includes both the local passage and its document/section context.

BM25 receives a slightly richer lexical context:

```text
Title: <document title>
Section: <heading path>
Source: <relative source path>
Subheadings: <front-matter subheadings>
Document headings: <front-matter headings>
Document keywords: <front-matter TF-IDF keywords>
Document language: <language code>
Document language name: <language name>

<chunk text>
```

This keeps BM25 aware of corpus-level and document-level lexical signals while preserving a cleaner vector embedding input.

## Exploratory Data Analysis

Initial EDA should answer the following questions:

- How many webpages are in the corpus?
- What is the average file size?
- What is the total corpus size?
- What are the general characteristics of the webpages?
- Are the pages text heavy, image heavy, or mixed?
- Is the content clearly sectioned with headings, subheadings, lists, tables, or other structural cues?
- How strong are logical or topic linkages between webpages?
- Is cleaning required, and how much?
- How prevalent is technical Autodesk-specific jargon in the corpus?
- Is useful metadata available?
- How much character/file-size reduction was achieved after cleaning?
- How many chunks were generated after indexing?
- Which files produce unusually short or unusually long chunks?

## Success Criteria

### Technical Metrics

RAG quality will be evaluated with RAGAS metrics:

- Faithfulness: target 0.90+
- Answer relevance: target 0.70+
- Context precision: target 0.70+
- Context recall: target 0.70+

Additional engineering metrics:

- Retrieval latency
- Answer latency
- Index build time
- Average chunk length
- Chroma collection size
- BM25 index size
- Percentage of queries with sufficient retrieved evidence above `MIN_RELEVANCE_SCORE`

Latency is important, though deep latency optimization may be beyond the scope of this take-home project.

### Evaluation Dataset

A 50-question golden dataset has been produced and is available at `eval_testset/autodesk_testset.csv`. See [Golden Dataset Generation](#3-golden-dataset-generation) for full production details, logic, and design decisions.

Summary of what is in the dataset:

- 6 required questions specified by Autodesk interviewers.
- 15 simple fact-based questions (direct corpus lookups).
- 14 reasoning questions (inference across one or two documents).
- 15 multi-context/multi-hop questions (synthesis across three or more documents).

Evaluator bias is controlled: ground truths were authored using a Claude model (Anthropic) while the app uses OpenAI models. LangSmith should be used to track evaluation runs.

### Business Metrics

These are beyond the scope of the take-home project but are useful future metrics:

- User feedback on answers, such as thumbs up or thumbs down.
- Whether use of the RAG app reduces the likelihood of creating a support ticket.

## Prior Work

This project is similar to the Cobb County RAG app previously developed to answer user queries from Cobb County Building and Fire code documents. The target performance metrics and system design are influenced by lessons learned from that project.

Relevant lessons carried forward:

- Clean, structured chunks matter as much as the model.
- Hybrid retrieval is stronger than vector-only retrieval for technical corpora.
- Strict answer prompts reduce hallucination.
- Evidence sufficiency checks help avoid unsupported answers.
- Deterministic neighbor/section-aware chunk expansion is safer than dumping large irrelevant context windows.
- Manifests and diagnostics are essential for debugging retrieval failures.

## Proposed Model Architecture

Based on lessons from the Cobb County RAG app, the proposed system includes:

- HTML boilerplate removal with BeautifulSoup and Trafilatura.
- Docling-first document-aware parsing over cleaned Markdown, with HybridChunker disabled by default and heading-aware chunk-size guards.
- OpenAI embeddings using `text-embedding-3-small`.
- Chroma as the persistent vector database.
- Local BM25 keyword index.
- Hybrid retrieval:
  - BM25 keyword search.
  - Dense semantic vector search.
  - Reciprocal rank fusion or similar score/rank fusion.
- Configurable retrieval depth using `RETRIEVER_K`.
- Minimum relevance filtering using `MIN_RELEVANCE_SCORE`.
- Cross-encoder re-ranking with `cross-encoder/ms-marco-MiniLM-L6-v2`.
- Strict evidence checking and guardrails.
- Grounded answer generation using OpenAI model settings from `.env`.
- Three runtime search modes: local-only, local plus `autodesk.com`, and local plus capped open-web retrieval.

## Evaluation And Monitoring

Evaluation should include:

- RAGAS faithfulness.
- RAGAS answer relevance.
- RAGAS context precision.
- RAGAS context recall.
- Latency.
- Retrieval hit inspection.
- Failed-query analysis.
- No-answer rate.
- Evidence sufficiency rate.

Monitoring should support rerunning golden-dataset tests when:

- The model changes.
- The embedding model changes.
- The corpus is updated.
- HTML cleaning rules change.
- Docling/chunking logic changes.
- `DOCLING_USE_HYBRID_CHUNKER` behavior changes.
- Retrieval fusion logic changes.
- Generation prompts or guardrails change.

## Local Development Workflow

Recommended order:

```bash
# 1. Clean raw Autodesk HTML files.
jupyter notebook notebook_01_corpus_cleaning.ipynb

# 2. Build Chroma and BM25 retrieval indexes.
jupyter notebook notebook_02_build_retrieval_indexes.ipynb
```

Or run the companion scripts if available:

```bash
python corpus_cleaning_pipeline.py
python build_retrieval_indexes.py
```

Before indexing, confirm that `.env` contains the required OpenAI and retrieval settings. Keep `DOCLING_USE_HYBRID_CHUNKER=false` unless intentionally testing HybridChunker, because the default structure-aware/export path avoids the Docling 512-token warning seen during chunking.

## Troubleshooting

### Docling 512-token warning during chunking

If the console shows warnings like this during `Chunking documents`:

```text
Token indices sequence length is longer than the specified maximum sequence length for this model (580 > 512)
```

the warning is coming from Docling's tokenizer/chunker path, not from OpenAI embeddings.

Recommended fix:

```text
DOCLING_USE_HYBRID_CHUNKER=false
```

Then rerun the updated `notebook_02_build_retrieval_indexes.ipynb` or `build_retrieval_indexes.py`.

The updated indexing pipeline avoids the warning-prone HybridChunker path by default and instead uses Docling conversion/export plus the project's heading-aware chunk-size guard. This keeps the benefits of Docling document parsing while avoiding tokenizer warnings from a 512-token chunker model.

### OpenAI embedding requirements

OpenAI embeddings require an API key available in the environment or `.env`:

```text
OPENAI_API_KEY=...
EMBEDDING_PROVIDER=openai
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Embedding throughput is controlled by:

```text
EMBEDDING_BATCH_SIZE=32
EMBEDDING_BATCH_DELAY_SECONDS=3.0
EMBEDDING_MAX_RETRIES=8
```

## Git And Git LFS Notes

Generated retrieval indexes can become large. Recommended practice:

- Commit source code, notebooks, prompts, README, and small manifests normally.
- Do not commit large generated vector databases unless there is a specific reason.
- If large generated index files must be uploaded to GitHub, use Git LFS.
- Keep reproducible generated index folders in `.gitignore` when practical.

Suggested `.gitignore` entries:

```text
retrieval_indexes/chroma_autodesk_cleaned_corpus/
retrieval_indexes/bm25_autodesk_cleaned_corpus/
*.pkl
*.sqlite3
*.sqlite
```

Suggested Git LFS commands if large index artifacts must be versioned:

```bash
git lfs install
git lfs track "*.pkl"
git lfs track "*.sqlite3"
git lfs track "*.sqlite"
git lfs track "retrieval_indexes/**"
git add .gitattributes
```

## Deployment

The application is expected to be deployed with:

- Streamlit Community Cloud.
- A private GitHub repository.
- Password protection for app access.

The current `.env` includes:

```text
STREAMLIT_HOST_PORT=8502
```

This is useful for local Docker/Streamlit port configuration.

## Next Steps

Recommended next steps:

1. Run the updated indexing notebook/script and inspect `retrieval_indexes/manifests/indexing_summary.md`.
2. Test hybrid retrieval with representative Autodesk questions.
3. Build or adapt the Streamlit RAG interface.
4. Add strict grounded-answer prompts and evidence sufficiency checks.
5. ~~Generate a 50-question golden dataset.~~ **Done** — see `eval_testset/autodesk_testset.csv`.
6. Evaluate with RAGAS and LangSmith.
7. Iterate on cleaning, chunking, retrieval depth, and prompts based on failure analysis.
