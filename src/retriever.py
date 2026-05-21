"""Hybrid BM25 plus Chroma retrieval for the Autodesk corpus."""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document

from src.config import Settings, get_embeddings, get_settings


@dataclass
class RetrievedSource:
    source: str
    page: int | None
    score: float
    snippet: str


def vectorstore_exists(collection_name: str | None = None) -> bool:
    settings = get_settings()
    if not settings.chroma_dir.exists() or not (settings.chroma_dir / "chroma.sqlite3").exists():
        return False
    try:
        client = _chroma_client(str(settings.chroma_dir))
        names: set[str] = set()
        for collection in client.list_collections():
            if isinstance(collection, str):
                names.add(collection)
            else:
                try:
                    names.add(collection.name)
                except Exception:
                    names.add(str(collection))
        return (collection_name or settings.collection_name) in names
    except Exception:
        return False


def bm25_index_exists(settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    return (
        (settings.bm25_dir / "bm25_index.pkl").exists()
        and (settings.bm25_dir / "bm25_chunk_ids.json").exists()
        and (settings.bm25_dir / "bm25_chunk_metadata.json").exists()
    )


def search_documents(query: str, k: int | None = None, collection_name: str | None = None) -> tuple[list[Document], list[RetrievedSource]]:
    settings = get_settings()
    top_k = k or settings.retriever_k
    candidate_k = max(top_k, settings.hybrid_candidate_k)
    dense_ranked = _search_dense(query, candidate_k, settings)
    bm25_ranked = _search_bm25(query, candidate_k, settings)
    return _fuse_ranked_results(
        [
            (dense_ranked, settings.hybrid_vector_weight),
            (bm25_ranked, settings.hybrid_bm25_weight),
        ],
        top_k,
        max_per_source=settings.hybrid_max_per_source,
    )


def has_sufficient_retrieval(sources: list[RetrievedSource]) -> bool:
    if not sources:
        return False
    return max(source.score for source in sources) >= get_settings().min_relevance_score


def get_chroma_collection(settings: Settings | None = None):
    settings = settings or get_settings()
    client = _chroma_client(str(settings.chroma_dir))
    return client.get_collection(settings.collection_name)


@lru_cache(maxsize=4)
def _chroma_client(path: str):
    return chromadb.PersistentClient(path=path, settings=ChromaSettings(anonymized_telemetry=False))


def _search_dense(query: str, k: int, settings: Settings) -> list[tuple[str, Document, RetrievedSource]]:
    if not vectorstore_exists(settings.collection_name):
        return []
    collection = get_chroma_collection(settings)
    query_embedding = get_embeddings(settings).embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=k, include=["documents", "metadatas", "distances"])
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ranked: list[tuple[str, Document, RetrievedSource]] = []
    for index, chunk_id in enumerate(ids):
        distance = float(distances[index]) if index < len(distances) else 1.0
        score = 1.0 / (1.0 + max(distance, 0.0))
        metadata = dict(metadatas[index] or {}) if index < len(metadatas) else {}
        metadata.setdefault("chunk_id", chunk_id)
        doc = Document(page_content=documents[index] or "", metadata=metadata)
        if _is_low_value_metadata_chunk(doc):
            continue
        source = _source_from_document(doc, score)
        ranked.append((_document_identity(doc, source), doc, source))
    return ranked


def _search_bm25(query: str, k: int, settings: Settings) -> list[tuple[str, Document, RetrievedSource]]:
    if not bm25_index_exists(settings):
        return []
    with (settings.bm25_dir / "bm25_index.pkl").open("rb") as file:
        bm25_index = pickle.load(file)
    chunk_ids = json.loads((settings.bm25_dir / "bm25_chunk_ids.json").read_text(encoding="utf-8"))
    metadata_by_id = json.loads((settings.bm25_dir / "bm25_chunk_metadata.json").read_text(encoding="utf-8"))
    scores = bm25_index.get_scores(_tokenize(query))
    if len(scores) == 0:
        return []
    ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:k]
    selected_ids = [chunk_ids[idx] for idx in ranked_indices if idx < len(chunk_ids)]
    docs_by_id = _fetch_chroma_by_ids(selected_ids, settings)
    max_score = float(max(scores)) or 1.0
    ranked: list[tuple[str, Document, RetrievedSource]] = []
    for idx in ranked_indices:
        if idx >= len(chunk_ids):
            continue
        chunk_id = chunk_ids[idx]
        doc = docs_by_id.get(chunk_id)
        metadata = dict(metadata_by_id.get(chunk_id) or {})
        if doc is None:
            doc = Document(page_content=str(metadata.get("preview") or ""), metadata=metadata)
        doc.metadata.setdefault("chunk_id", chunk_id)
        if _is_low_value_metadata_chunk(doc):
            continue
        source = _source_from_document(doc, float(scores[idx]) / max_score if max_score else 0.0)
        ranked.append((_document_identity(doc, source), doc, source))
    return ranked


def _fetch_chroma_by_ids(ids: list[str], settings: Settings) -> dict[str, Document]:
    if not ids or not vectorstore_exists(settings.collection_name):
        return {}
    collection = get_chroma_collection(settings)
    payload = collection.get(ids=ids, include=["documents", "metadatas"])
    found: dict[str, Document] = {}
    for chunk_id, text, metadata in zip(payload.get("ids", []), payload.get("documents", []), payload.get("metadatas", [])):
        meta = dict(metadata or {})
        meta.setdefault("chunk_id", chunk_id)
        found[chunk_id] = Document(page_content=text or "", metadata=meta)
    return found


def _fuse_ranked_results(ranked_sets: list[tuple[list[tuple[str, Document, RetrievedSource]], float]], top_k: int, rrf_k: int = 60, max_per_source: int = 0) -> tuple[list[Document], list[RetrievedSource]]:
    scores: dict[str, float] = {}
    best_docs: dict[str, Document] = {}
    best_sources: dict[str, RetrievedSource] = {}
    for ranked, weight in ranked_sets:
        seen: set[str] = set()
        for rank, (doc_id, doc, source) in enumerate(ranked, start=1):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + weight * (1.0 / (rrf_k + rank))
            if doc_id not in best_sources or source.score > best_sources[doc_id].score:
                best_docs[doc_id] = doc
                best_sources[doc_id] = source
    if not scores:
        return [], []
    max_score = max(scores.values()) or 1.0
    ranked_ids = _select_diverse_ranked_ids(
        sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True),
        best_docs,
        best_sources,
        top_k,
        max_per_source,
    )
    docs = [best_docs[doc_id] for doc_id in ranked_ids]
    sources = []
    for doc_id in ranked_ids:
        source = best_sources[doc_id]
        sources.append(RetrievedSource(source.source, source.page, float(scores[doc_id] / max_score), source.snippet))
    return docs, sources


def _source_from_document(doc: Document, score: float) -> RetrievedSource:
    metadata = doc.metadata or {}
    source = metadata.get("title") or metadata.get("source_file") or metadata.get("relative_source_path") or "Autodesk corpus"
    return RetrievedSource(
        source=str(source),
        page=None,
        score=score,
        snippet=(doc.page_content or "")[:350].replace("\n", " ").strip(),
    )


def _document_identity(doc: Document, source: RetrievedSource) -> str:
    metadata = doc.metadata or {}
    return "::".join(str(part) for part in (metadata.get("chunk_id") or "", source.source, metadata.get("chunk_index") or "0"))


def _select_diverse_ranked_ids(
    ranked_ids: list[str],
    docs_by_id: dict[str, Document],
    sources_by_id: dict[str, RetrievedSource],
    top_k: int,
    max_per_source: int,
) -> list[str]:
    if max_per_source <= 0:
        return ranked_ids[:top_k]
    selected: list[str] = []
    counts: dict[str, int] = {}
    deferred: list[str] = []
    for doc_id in ranked_ids:
        source_key = _source_key(docs_by_id[doc_id], sources_by_id[doc_id])
        if counts.get(source_key, 0) < max_per_source:
            selected.append(doc_id)
            counts[source_key] = counts.get(source_key, 0) + 1
        else:
            deferred.append(doc_id)
        if len(selected) == top_k:
            return selected
    for doc_id in deferred:
        if len(selected) == top_k:
            break
        selected.append(doc_id)
    return selected


def _source_key(doc: Document, source: RetrievedSource) -> str:
    metadata = doc.metadata or {}
    return str(metadata.get("source_file") or metadata.get("relative_source_path") or source.source)


def _is_low_value_metadata_chunk(doc: Document) -> bool:
    body = _body_text(doc.page_content or "").strip().lower()
    if not body:
        return True
    metadata_prefixes = (
        "source_file:",
        "relative_source_path:",
        "title:",
        "cleaned_format:",
        "extraction_method:",
        "raw_char_count:",
        "cleaned_char_count:",
    )
    return body.startswith(metadata_prefixes)


def _body_text(text: str) -> str:
    parts = text.split("\n\n", 1)
    return parts[1] if len(parts) > 1 else text


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())
