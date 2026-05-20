# Autodesk RAG App

## Project Overview

This project will build a retrieval augmented generation (RAG) application over a corpus of Autodesk HTML webpages. The goal is to provide grounded, accurate answers to customer questions about Autodesk software products, ranging from general product questions to technical usage questions.

The app will be developed step by step, beginning with corpus exploration and moving toward parsing, retrieval, generation, evaluation, and deployment.

## Business Problem

Autodesk customers need fast, trustworthy answers about Autodesk software products and workflows. A successful RAG app should improve customer experience and user satisfaction by answering questions with clear supporting evidence from Autodesk source material.

Longer term, this type of tool could also reduce support burden by helping users resolve questions before creating support tickets.

## Available Data

- Corpus of Autodesk webpages stored as HTML files.

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

## Success Criteria

### Technical Metrics

RAG quality will be evaluated with RAGAS metrics:

- Faithfulness: target 0.90+
- Answer relevance: target 0.70+
- Context precision: target 0.70+
- Context recall: target 0.70+

Latency is also important, though deep latency optimization may be beyond the scope of this take-home project.

### Evaluation Dataset

The project should use an advanced LLM to mine the corpus and produce a 50-question golden dataset with varying levels of difficulty and complexity.

The golden dataset should include:

- General product and usage questions.
- Technical questions.
- Multi-hop or cross-page questions where appropriate.
- The specific questions requested by Autodesk or interviewers.

To reduce evaluator bias, the LLM used to generate or evaluate the golden dataset should differ from the LLM used in the app. For example:

- App model: OpenAI model.
- Evaluation or dataset generation model: Claude Sonnet or Claude Opus.

LangSmith should be used to track evaluation runs.

### Business Metrics

These are beyond the scope of the take-home project but are useful future metrics:

- User feedback on answers, such as thumbs up or thumbs down.
- Whether use of the RAG app reduces the likelihood of creating a support ticket.

## Prior Work

This project is similar to the Cobb County RAG app previously developed to answer user queries from Cobb County Building and Fire code documents. The target performance metrics and system design are influenced by lessons learned from that project.

## Proposed Model Architecture

Based on lessons from the Cobb County RAG app, the proposed system includes:

- Docling or context-aware parsing for HTML content.
- Context-aware chunking.
- Hybrid retrieval:
  - BM25 keyword search.
  - Dense semantic vector search.
- Chroma as the vector database.
- Deterministic nearest-neighbor chunk retrieval.
- Re-ranking.
- Strict evidence checking and guardrails.
- Grounded answer generation.
- A separate web-search-only option limited to Autodesk domain websites.

## Evaluation And Monitoring

Evaluation should include:

- RAGAS faithfulness.
- RAGAS answer relevance.
- RAGAS context precision.
- RAGAS context recall.
- Latency.

Monitoring should support rerunning golden-dataset tests when:

- The model changes.
- The corpus is updated.
- Parsing, chunking, retrieval, or generation logic changes.

## Deployment

The application is expected to be deployed with:

- Streamlit Community Cloud.
- A private GitHub repository.
- Password protection for app access.

