"""Cross-encoder reranking for retrieval results."""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.documents import Document

from src.config import Settings, get_settings
from src.retriever import RetrievedSource


logger = logging.getLogger(__name__)


def rerank_documents(
    query: str,
    docs: list[Document],
    sources: list[RetrievedSource],
    settings: Settings | None = None,
) -> tuple[list[Document], list[RetrievedSource]]:
    """Rerank retrieved context blocks with a SentenceTransformers CrossEncoder."""

    settings = settings or get_settings()
    if not settings.reranker_enabled or len(docs) <= 1:
        return docs, sources

    limit = max(1, min(settings.reranker_top_n, len(docs)))
    try:
        model = _get_cross_encoder(settings.reranker_model)
        pairs = [(query, _passage_text(doc)) for doc in docs]
        scores = model.predict(pairs, batch_size=settings.reranker_batch_size)
    except Exception as exc:
        logger.warning("Cross-encoder reranking failed; preserving retrieval order: %s", exc)
        return docs, sources

    scored: list[tuple[float, Document, RetrievedSource]] = []
    for index, (doc, source, score) in enumerate(zip(docs, sources, scores)):
        metadata = dict(doc.metadata or {})
        metadata["cross_encoder_model"] = settings.reranker_model
        metadata["cross_encoder_score"] = float(score)
        metadata["pre_rerank_position"] = index + 1
        scored.append(
            (
                float(score),
                Document(page_content=doc.page_content, metadata=metadata),
                RetrievedSource(source.source, source.page, float(score), source.snippet),
            )
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc, _ in scored[:limit]], [source for _, _, source in scored[:limit]]


@lru_cache(maxsize=2)
def _get_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def _passage_text(doc: Document) -> str:
    metadata = doc.metadata or {}
    title = metadata.get("title") or metadata.get("source") or metadata.get("source_file") or ""
    section = metadata.get("heading_path") or metadata.get("section") or ""
    return f"{title}\n{section}\n{doc.page_content}".strip()
