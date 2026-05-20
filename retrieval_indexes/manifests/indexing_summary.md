# Retrieval Indexing Summary

- Timestamp UTC: `2026-05-20T22:51:22.525656+00:00`
- Cleaned files discovered: `1,218`
- Files indexed: `1,218`
- Files skipped: `0`
- Files failed: `0`
- Total chunks: `3,806`
- Average chunks per indexed document: `3.12`
- Average chunk length: `1218.6`
- Shortest chunk length: `6`
- Longest chunk length: `4002`
- Embedding model used: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding dimension: `384`
- Chroma collection name: `autodesk-rag`
- Chroma persistence directory: `retrieval_indexes/chroma_autodesk_cleaned_corpus`
- Chroma collection count: `3,806`
- BM25 persistence directory: `retrieval_indexes/bm25_autodesk_cleaned_corpus`
- BM25 chunk count: `3,806`
- Docling available: `True`

## Notes

- The corpus is already cleaned Markdown/text, so Markdown-aware parsing is the primary chunking strategy.
- Docling is available for future document-aware ingestion experiments, but this notebook avoids unnecessary PDF-style conversion.
- GPU can improve embedding throughput when SentenceTransformers can use CUDA. BM25 and parsing remain CPU-bound.