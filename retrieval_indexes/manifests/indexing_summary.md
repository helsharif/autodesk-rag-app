# Retrieval Indexing Summary

- Timestamp UTC: `2026-05-21T00:38:41.583038+00:00`
- Cleaned files discovered: `1,218`
- Files indexed: `1,218`
- Files skipped: `0`
- Files failed: `0`
- Total chunks: `29,846`
- Average chunks per indexed document: `24.50`
- Average chunk length: `162.6`
- Shortest chunk length: `1`
- Longest chunk length: `4002`
- Embedding provider: `openai`
- OpenAI embedding model used: `text-embedding-3-small`
- Embedding dimension: `1536`
- Chroma collection name: `autodesk-rag`
- Chroma persistence directory: `retrieval_indexes/chroma_autodesk_cleaned_corpus`
- Chroma collection count: `29,846`
- BM25 persistence directory: `retrieval_indexes/bm25_autodesk_cleaned_corpus`
- BM25 chunk count: `29,846`
- Docling available: `True`
- Docling accelerator device: `cuda`
- Docling num threads: `4`
- Docling OCR enabled: `False`
- Docling HybridChunker enabled: `False`
- Minimum relevance score: `0.3`

## Notes

- Chunking is routed through Docling first. The notebook records the chunking method per document and per chunk.
- If Docling conversion/chunking fails for a specific cleaned Markdown file, the raw Markdown heading-aware fallback is used for that file and recorded in the manifests.
- Embeddings are created with OpenAI, so an `OPENAI_API_KEY` must be available in the environment or `.env`.
- GPU settings apply to Docling where supported. BM25 and tokenization remain CPU-bound.