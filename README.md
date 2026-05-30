---
title: Autodesk RAG Assistant
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Evidence-grounded Autodesk answers with hybrid RAG
---
# Autodesk Agentic RAG App

**Live demo:** [Autodesk RAG Assistant on Hugging Face Spaces](https://huggingface.co/spaces/helsharif/autodesk-rag-assistant)

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
- Compare/contrast query planning that extracts mentioned products, generates focused retrieval subqueries, deduplicates chunks, and balances context across the compared entities.
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

## Sample Answers

The examples below show generated answers to sample questions as of May 21, 2026, alongside the reference answers from the evaluation test set. These examples are static and are not automatically updated when evaluations are re-run.

### 1. What does Fusion 360 do?

**Reference Answer**

According to the document 'Autodesk Fusion Manufacturing Cloud | Autodesk Fusion', Autodesk Fusion (formerly known as Autodesk Fusion 360) is design, engineering, electronics, and manufacturing software all-in-one. It connects the entire product development process into one cloud-based platform with integrated 3D CAD, CAM, CAE, and PCB.

The document 'Compare Fusion 360 vs Fusion 360 for Personal Use | Autodesk' further describes it as professional cloud CAD, CAM, CAE, and PCB software that includes all design and 3D modeling tools, plus a fully featured CAM, CAE, and PCB development platform. The 'Autodesk Fusion 360 | ECAD and MCAD | Software Collaboration Tools' page adds that Fusion 360 enables ECAD and MCAD designers to work natively and independently on the same design project within the same design management platform.

In summary, Fusion 360 is a unified cloud-based platform covering product concept, detailed design, simulation, electronics, and manufacturing — used by product designers, mechanical engineers, and manufacturing professionals throughout the product lifecycle.

**Option 1: Local Document Search**

Fusion 360 is a comprehensive design and manufacturing software that offers a range of features including advanced 4- and 5-axis toolpaths for machining, turning strategies for programming 2D turned parts, and stock simulation to verify toolpaths and avoid collisions. It supports safe, smooth, and predictable machine motion with automatic collision avoidance, requiring a subscription for some advanced machining functionalities like the Machining Extension.

Additionally, Fusion 360 provides simulation capabilities to test for multiple failure criteria and explore design changes efficiently, including cloud solving to handle complex design outcomes simultaneously. This allows users to cut solve time and focus on other tasks while the software processes simulations in the background (Local 1, 2, 3, 4, 6, 7, 8).

**Option 2: Local Document Search + Autodesk.com**

Fusion 360 is an integrated platform by Autodesk that combines 3D CAD, CAM, CAE, PCB, data management, and collaboration tools. It enables users to create 3D designs, perform simulations, generate toolpaths for manufacturing, and manage product development workflows all within one software environment. This includes specialized machining strategies such as 4- and 5-axis toolpaths, turning strategies, and stock simulation to verify toolpaths and avoid collisions (Local 1, 2, 3; Web 1, 3).

Additionally, Fusion 360 supports simulation capabilities to test designs against various failure criteria and optimize design changes, with cloud-based solving to speed up computations. It also facilitates manufacturing processes with cutting strategies for different methods like water jet cutting and supports documentation of flat patterns for sheet metal parts (Local 4, 6, 7, 8; Web 4). Overall, Fusion 360 serves as a comprehensive product development platform covering design, engineering, and manufacturing needs.

**Option 3: Local Document Search + Open Web Search**

Fusion 360 is an integrated platform by Autodesk that combines 3D CAD, CAM, CAE, PCB, and data management tools. It supports modeling, manufacturing, industrial design, electronics, and mechanical engineering workflows. The software offers specialized toolpaths for machining, including 4- and 5-axis toolpaths, turning strategies, and stock simulation to verify toolpaths and avoid collisions. It also provides simulation capabilities to test for various failure criteria and explore design changes efficiently, including cloud-based solving to speed up computations and allow multitasking (Local 1, 2, 3, 4, 6, 7, 8; Web 1, 2, 3).

In summary, Fusion 360 enables users to design, simulate, and manufacture products within a single platform, streamlining the product development process from concept through production.

### 2. What's the difference between AutoCAD and Revit?

**Reference Answer**

Based on the document 'Thomas & Hutton | Site Development Drives the Future of Building Design | Autodesk', AutoCAD is described as Autodesk's 'software to design anything,' creating precise 2D drawings and 3D models. In the same case study, external architects used Revit for BIM (Building Information Modeling), which 'seamlessly interfaced with AutoCAD' — illustrating that AutoCAD is a general-purpose design and drafting tool while Revit is purpose-built for BIM.

The document 'Autodesk Revit LT Software | Get Prices & Buy Official Revit LT 2023' clarifies the distinction further: Revit is used to 'design, document, visualize, and deliver architecture and engineering projects' in a BIM context, with tools for 3D modeling, scheduling, annotating, and document production. The 'BIM Coordination & Collaboration | Autodesk BIM Collaborate' page notes that Revit supports cloud work-sharing for design teams, automatic clash detection, and issue assignment across multi-disciplinary models.

In short, AutoCAD is a flexible 2D/3D drafting tool used across many industries, while Revit is a BIM-focused platform for architecture, structure, and MEP design that produces an intelligent, coordinated building model rather than standalone drawings.

**Option 1: Local Document Search**

I could not find a reliable answer in the available documents or web sources.

**Option 2: Local Document Search + Autodesk.com**

The main difference between AutoCAD and Revit is their core purpose and functionality. AutoCAD is a general-purpose CAD software used primarily for creating precise 2D drawings and 3D models across various industries. It is a broad drawing tool suitable for detailed drafting and schematics. In contrast, Revit is specialized BIM (Building Information Modeling) software designed specifically for architecture, engineering, and construction. Revit enables users to create intelligent 3D models that include detailed constructability information and support collaboration among project stakeholders (Web 1, Web 2, Web 5).

Additionally, Revit can import and export AutoCAD files, allowing integration between the two tools. Together, they offer efficient workflows from design to documentation, with Revit focusing on building models and AutoCAD on detailed drafting (Local 2, Local 5, Local 8). Thus, AutoCAD is more about drafting and design flexibility, while Revit emphasizes building modeling and information management.

**Option 3: Local Document Search + Open Web Search**

The main difference between AutoCAD and Revit is their core purpose and functionality. AutoCAD is a general CAD (Computer-Aided Design) software used for creating 2D and 3D drawings across various industries, serving as a broad drawing tool. In contrast, Revit is specialized BIM (Building Information Modeling) software designed specifically for the AEC (Architecture, Engineering, and Construction) industry to create intelligent 3D models that contain real information about the building components (Web 2, Web 3).

Revit enables faster project creation with integrated data, making it more efficient for building design and documentation workflows. AutoCAD, while versatile, is more focused on drafting and detailing, whereas Revit supports a more comprehensive design-to-documentation process with real-time information embedded in the model (Web 1, Local 1, Local 2). Together, they complement each other by offering effective workflows from design to documentation.

### 3. Does AutoCAD LT do 3d?

**Reference Answer**

No. According to the document 'Autodesk AutoCAD LT 2024 | Get Prices & Subscribe To AutoCAD LT', AutoCAD LT is CAD software used for precise 2D drafting, design, and documentation. It explicitly states: 'One of the biggest differences between AutoCAD and AutoCAD LT is that AutoCAD LT provides precision 2D drafting and documentation capabilities, while AutoCAD also provides 3D functionalities.'

The same document also notes that AutoCAD LT does not include third-party connections, development tools, or the seven specialized toolsets available in full AutoCAD. AutoCAD LT does support some customization through AutoLISP, and it can run on Microsoft Windows, Apple macOS, and Linux.

Therefore, AutoCAD LT is strictly a 2D drafting tool. Users who need 3D modeling capabilities must use full AutoCAD or another product such as Fusion 360 or Inventor.

**Option 1: Local Document Search**

AutoCAD LT does not support 3D functionalities. It provides precision 2D drafting and documentation capabilities only. In contrast, the full version of AutoCAD includes 3D functionalities such as 3D graphics rotation, advanced geometric views, mesh modeling, surface modeling, and parametric modeling, which are not available in AutoCAD LT [Local 1], [Local 2], [Local 3].

Therefore, if you need to work with 3D models, AutoCAD LT would not be suitable, and you would need the full AutoCAD version.

**Option 2: Local Document Search + Autodesk.com**

AutoCAD LT does not have 3D modeling or 3D workspace capabilities. It is designed specifically for precision 2D drafting and documentation. While AutoCAD LT can open 3D models created in other software, it does not provide tools for 3D viewing, navigation, or editing within the program itself. These 3D functionalities are available only in the full AutoCAD version, not in AutoCAD LT ([Local 2], [Local 5], [Web 1], [Web 5]).

In summary, AutoCAD LT supports 2D drafting only and lacks the 3D features found in AutoCAD. If you need 3D modeling and editing, you would need to use the full AutoCAD product.

**Option 3: Local Document Search + Open Web Search**

AutoCAD LT does not support 3D modeling or 3D functionalities. It is designed primarily for precision 2D drafting and documentation. While AutoCAD LT can open 3D models, it does not provide 3D view or navigation tools, as it is meant for 2D work only. In contrast, the full version of AutoCAD includes 3D capabilities along with 2D drafting (Local 1, Local 2, Web 1).

Therefore, if you need to work with 3D models or perform 3D design tasks, AutoCAD LT would not be suitable, and you would need the full AutoCAD software instead.

### 4. What's the latest release for Maya?

**Reference Answer**

According to the document 'Autodesk Account Basics | Previous Product Versions | Available Versions', the current (latest) release of Autodesk Maya is Maya 2024. The document lists Maya 2024 in the table of products alongside its eligible previous versions.

The same document confirms that subscribers to Maya 2024 are eligible to download and use the following three previous releases: Maya 2023, Maya 2022, and Maya 2020. For subscribers who switched from maintenance plan seats of Autodesk Maya with Softimage, previous versions with Softimage integration (Maya 2023, 2022, and 2020 with Softimage) are also available.

**Option 1: Local Document Search**

The latest release for Maya is the 2024 version. This release includes features such as the Maya USD plug-in available as an open-source project for studios to customize, accelerated performance with interactive workflows, and improvements like cached playback in Viewport 2.0 to review animations faster and produce fewer playblasts (Local 1, Local 2, Local 3, Local 4, Local 5).

**Option 2: Local Document Search + Autodesk.com**

The latest release for Maya is Maya 2026, which you can access from your Autodesk Account. This release includes new features and updated workflows as detailed in the official Autodesk Maya 2026 documentation [Web 2].

While the local documents highlight features for Maya 2024, such as integrated Arnold renderer, fast playback, and open-source customizable USD plug-in, the most current version according to the web sources is Maya 2026 [Local 1-8], [Web 2].

**Option 3: Local Document Search + Open Web Search**

The latest release for Maya is Maya 2026. This version includes new features and updated workflows, and it is accessible through your Autodesk Account (Web 2). Additionally, the 2024 features highlight performance acceleration with interactive workflows and the availability of the Maya USD plug-in as an open-source project (Local 1, Local 2).

### 5. Can I use fusion 360 on a Mac?

**Reference Answer**

Yes. Autodesk's support documentation (autodesk.com) confirms that Fusion 360 (Autodesk Fusion) is compatible with macOS. A specific Autodesk support article titled 'End of life for Autodesk Fusion on macOS 11 (Big Sur) — Effective March 2024' confirms that macOS support exists and that support for macOS 11 Big Sur ended in March 2024, meaning users must be on macOS 12 (Monterey) or a newer Apple-supported version to continue using the software.

Within the cleaned corpus documents, the document 'Autodesk Account Basics | Previous Product Versions | Available Versions' also lists 'AutoCAD for Mac 2024' and 'AutoCAD LT for Mac 2024' as separate product entries, confirming Autodesk's broader cross-platform macOS support across its portfolio. For specific current macOS hardware and version requirements for Fusion, users should consult the official Autodesk system requirements page.

**Option 1: Local Document Search**

I could not find a reliable answer in the available documents or web sources.

**Option 2: Local Document Search + Autodesk.com**

Yes, you can use Fusion 360 on a Mac. Autodesk provides support for Fusion 360 on Mac systems, and many Mac users have been running Fusion 360 successfully. However, some users report that it can feel somewhat clunky, indicating it may not be as fully optimized as on Windows. Additionally, Autodesk Fusion 360 will require an Apple-supported OS and hardware configuration, especially after updates from March 2025 onward (Web 3, Web 5).

Furthermore, Autodesk also supports running some of its products on Mac through virtualized environments like Parallels Desktop and VMware Fusion, which can be an alternative if you encounter issues with native Mac support (Web 1, Web 4).

**Option 3: Local Document Search + Open Web Search**

Yes, you can use Fusion 360 on a Mac. Fusion 360 is a versatile CAM software that supports various machining operations and is available as an integrated engineering and design solution. While the local documents do not explicitly state Mac compatibility, user reports from web sources indicate that Fusion 360 runs well on Mac devices, including newer models like the M3 MacBook (Web 3). Additionally, Autodesk supports running some of its products on Mac through virtualized environments such as Parallels Desktop and VMware Fusion (Web 1).

Therefore, whether natively or via virtualization, Fusion 360 is accessible for Mac users. For more detailed assistance, you can also contact a Fusion 360 expert at Autodesk (Local 3).

### 6. What are the differences between Standard, Premium, and Enterprise subscription plans?

**Reference Answer**

The document 'Trade in Your Perpetual License | Global Promotions | Autodesk' states: 'Our Premium plan is designed to help businesses that manage 50 or more subscriptions operate more efficiently. From single sign-on to 24/7 support, save time and increase security.' This identifies the core Premium differentiators: SSO and around-the-clock support, for organisations managing 50+ subscriptions.

According to Autodesk's plans page (autodesk.com/asean/plans), Standard is the default plan included with all product subscriptions at no additional cost, providing 8-hour business-hour support and self-help resources. Premium adds 24/7 live support, Single Sign-On (SSO), and Usage Reporting for informed licence management. Enterprise is available via an Enterprise Business Agreement (EBA) and includes all Premium features plus personalised assistance from dedicated Autodesk experts.

The document 'AUTODESK, INC. ANNOUNCES FISCAL 2024 THIRD QUARTER RESULTS' references Autodesk signing its 'largest-ever EBA' in Q3 FY2024, driven by record Enterprise-tier performance in construction and water verticals — indicating the strategic importance of the Enterprise tier for large organisations.

**Option 1: Local Document Search**

I could not find a reliable answer in the available documents or web sources.

**Option 2: Local Document Search + Autodesk.com**

The differences between Autodesk's Standard, Premium, and Enterprise subscription plans primarily relate to features for managing large teams, advanced reporting, user management, and support capabilities.

- The Standard plan provides basic subscription and Flex token access, user assignment, two-step verification, single sign-on (SSO), and basic usage reporting with user details. It supports adding and assigning users, bulk import, groups, and automating Flex user assignments.
- The Premium plan is designed for extra-large teams and includes all Standard features plus enhanced capabilities such as directory sync, automated user adding upon first sign-in, API access to connect usage data to third-party tools, and expert coaching for end users with 24x7 live support. It also offers better reporting tools to view and track product/seat usage.
- The Enterprise plan includes all Premium features plus additional controls such as the ability to add your own metadata to enrich reports, controlled access to downloads (restricting downloads to admins), a Managed Success Plan, and more advanced usage reporting with detailed user data. It also supports Flex token access and usage reporting with user details, directory sync, and automated user management at scale.

In summary, Standard covers essential subscription management, Premium adds advanced user management and support for large teams, and Enterprise offers the most comprehensive controls, reporting, and support for very large organizations with complex needs (Local 1, 2, 3, 4, 5, 6, 7, 8; Web 2, 3).

**Option 3: Local Document Search + Open Web Search**

The differences between the Standard, Premium, and Enterprise Autodesk subscription plans primarily lie in advanced user management, reporting capabilities, and integration features.

All three plans provide access to products through dedicated single-user subscriptions (Standard and Premium) or Flex tokens (Enterprise), and all support subscription usage reporting and Flex usage reporting. However, Enterprise uniquely allows adding your own metadata to enrich reports and offers API access to connect data to third-party tools, which Premium also supports but Standard does not. Enterprise also supports control over user access to downloads and directory synchronization, which Premium supports but Standard does not. Additionally, Enterprise and Premium allow automating user additions and directory sync, while Standard does not. Enterprise includes all Premium features plus these enhanced administrative and integration capabilities (Locals 1, 2, 3, 4, 5).

In summary, Standard covers basic subscription and usage reporting with essential user management, Premium adds automation and directory sync, and Enterprise includes all Premium features plus advanced reporting metadata, API integrations, and stricter user access controls.

More example answers can be reviewed in the app under the **Settings and Eval** tab.

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
3. The agent applies its router. Non-Autodesk questions can abstain, current/latest/pricing questions can request web in web-enabled modes, and compare/contrast questions trigger local compare retrieval planning.
4. Local retrieval searches both Chroma and BM25. For compare/contrast questions, the agent also retrieves separately over focused subqueries for each mentioned product and direct comparison dimensions.
5. Results are merged with weighted reciprocal rank fusion, deduplicated, and capped. Compare/contrast retrieval additionally prefers balanced local context so one product does not crowd out the other.
6. Neighboring chunks from the same document are added for context continuity.
7. Web snippets, when enabled, are converted into evidence blocks.
8. Local and web evidence are reranked together with the cross-encoder.
9. The adequacy gate checks whether the retrieved evidence explicitly supports the answer.
10. The LLM generates a concise sourced answer, or the app returns the fixed no-answer message.
11. Optional monitoring records request, retrieval, latency, source, model, and outcome metadata.

The compare/contrast branch improves what local evidence is retrieved and uses a compare-aware adequacy check. A direct comparison passage is not required when the retrieved evidence separately provides substantive facts about each compared product. It does not inject product claims, hardcode product pairs, or pre-write final answers; answer generation remains grounded in retrieved local and optional web evidence.

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
RERANKER_TOP_N=8
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

The live demo is deployed on Hugging Face Spaces:

[https://huggingface.co/spaces/helsharif/autodesk-rag-assistant](https://huggingface.co/spaces/helsharif/autodesk-rag-assistant)

The app is configured for Hugging Face Spaces using Docker. The root `Dockerfile` installs `requirements.txt`, copies the application into `/app`, exposes port `7860`, and starts:

```bash
streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=7860
```

For Hugging Face deployment, create a Docker Space and configure required values such as `OPENAI_API_KEY`, `SERPAPI_API_KEY`, `LANGSMITH_API_KEY`, `SUPABASE_URL`, and `SUPABASE_KEY` as Space secrets. Do not commit `.env` or local secret files. Supabase can be used as durable monitoring storage for hosted demos.

The `.dockerignore` file excludes local secrets, raw corpus HTML, notebooks checkpoints, temporary files, and other local-only artifacts from the Docker build context.
