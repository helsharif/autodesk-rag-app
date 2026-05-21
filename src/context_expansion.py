"""Deterministic same-document neighbor expansion for retrieved Autodesk chunks."""

from __future__ import annotations

import csv
from functools import lru_cache

from langchain_core.documents import Document

from src.config import Settings, get_settings
from src.retriever import RetrievedSource, _fetch_chroma_by_ids


def expand_retrieved_docs(
    retrieved_docs: list[Document],
    retrieved_sources: list[RetrievedSource],
    collection_name: str,
    settings: Settings | None = None,
) -> tuple[list[Document], list[RetrievedSource]]:
    settings = settings or get_settings()
    if not settings.context_expansion_enabled or settings.context_expansion_mode == "off":
        return retrieved_docs, retrieved_sources
    manifest = _load_manifest(str(settings.chunk_manifest_path))
    by_doc_index = {(row.get("source_file"), _as_int(row.get("chunk_index"))): row for row in manifest}
    expanded_docs: list[Document] = []
    expanded_sources: list[RetrievedSource] = []
    seen: set[str] = set()
    total_chars = 0
    for rank, (doc, source) in enumerate(zip(retrieved_docs, retrieved_sources), start=1):
        metadata = dict(doc.metadata or {})
        source_file = metadata.get("source_file")
        chunk_index = _as_int(metadata.get("chunk_index"))
        if source_file is None or chunk_index is None:
            _append_with_budget(doc, source, expanded_docs, expanded_sources, seen, settings, total_chars)
            continue
        wanted_ids = []
        for neighbor_index in range(chunk_index - 1, chunk_index + 2):
            row = by_doc_index.get((source_file, neighbor_index))
            if row and row.get("chunk_id"):
                wanted_ids.append(row["chunk_id"])
        docs_by_id = _fetch_chroma_by_ids(wanted_ids, settings)
        for chunk_id in wanted_ids:
            candidate = docs_by_id.get(chunk_id)
            if candidate is None:
                continue
            candidate.metadata["retrieval_rank"] = rank
            candidate.metadata["expansion_type"] = "retrieved_chunk" if _as_int(candidate.metadata.get("chunk_index")) == chunk_index else "neighbor_chunk"
            candidate_source = RetrievedSource(
                source=str(candidate.metadata.get("title") or candidate.metadata.get("source_file") or source.source),
                page=None,
                score=source.score,
                snippet=candidate.page_content[:350].replace("\n", " ").strip(),
            )
            key = str(candidate.metadata.get("chunk_id") or candidate.page_content[:120])
            if key in seen:
                continue
            seen.add(key)
            if len(expanded_docs) >= settings.context_max_expanded_docs:
                return expanded_docs, expanded_sources
            remaining = settings.context_max_chars - total_chars
            if remaining <= 0:
                return expanded_docs, expanded_sources
            text = candidate.page_content[:remaining]
            total_chars += len(text)
            expanded_docs.append(Document(page_content=text, metadata=candidate.metadata))
            expanded_sources.append(candidate_source)
    return (expanded_docs, expanded_sources) if expanded_docs else (retrieved_docs, retrieved_sources)


def _append_with_budget(doc, source, docs, sources, seen, settings, total_chars) -> None:
    key = str(doc.metadata.get("chunk_id") or doc.page_content[:120])
    if key in seen or len(docs) >= settings.context_max_expanded_docs:
        return
    seen.add(key)
    remaining = settings.context_max_chars - total_chars
    if remaining <= 0:
        return
    docs.append(Document(page_content=doc.page_content[:remaining], metadata=doc.metadata))
    sources.append(source)


@lru_cache(maxsize=1)
def _load_manifest(path: str) -> list[dict[str, str]]:
    try:
        with open(path, "r", encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))
    except FileNotFoundError:
        return []


def _as_int(value) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except Exception:
        return None
