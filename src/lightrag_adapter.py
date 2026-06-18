"""LightRAG SDK integration for Option 4 retrieval."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import threading
import time
from typing import Any

import numpy as np
from langchain_core.documents import Document

from src.config import Settings, get_settings
from src.retriever import RetrievedSource


logger = logging.getLogger(__name__)
_ATOMIC_WRITE_PATCHED = False
_ATOMIC_WRITE_PATCH_LOCK = threading.Lock()
_ATOMIC_WRITE_FILE_LOCKS: dict[str, threading.Lock] = {}


def lightrag_index_exists(settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    path = settings.lightrag_working_dir
    return path.exists() and any(
        (path / name).exists()
        for name in (
            "kv_store_doc_status.json",
            "kv_store_full_docs.json",
            "vdb_chunks.json",
            "graph_chunk_entity_relation.graphml",
        )
    )


def search_lightrag_mixed(query: str, settings: Settings | None = None) -> tuple[list[Document], list[RetrievedSource]]:
    settings = settings or get_settings()
    if not lightrag_index_exists(settings):
        logger.warning("LightRAG index not found at %s", settings.lightrag_working_dir)
        return [], []
    return _run_async(_asearch_lightrag_mixed(query, settings))


async def _asearch_lightrag_mixed(query: str, settings: Settings) -> tuple[list[Document], list[RetrievedSource]]:
    rag = None
    try:
        rag = await create_lightrag(settings)
        from lightrag import QueryParam

        context = await rag.aquery(
            query,
            param=QueryParam(
                mode=_lightrag_mode(settings.lightrag_retrieval_mode),
                only_need_context=True,
                top_k=settings.lightrag_top_k,
                chunk_top_k=settings.lightrag_chunk_top_k,
                enable_rerank=False,
            ),
        )
    finally:
        if rag is not None:
            await rag.finalize_storages()

    text = str(context or "").strip()
    if not text:
        return [], []

    doc = Document(
        page_content=text,
        metadata={
            "evidence_type": "local",
            "retrieval_mode": "LightRAG mixed",
            "source_file": "LightRAG Autodesk mixed index",
            "relative_source_path": str(settings.lightrag_working_dir),
            "title": "LightRAG mixed local corpus evidence",
        },
    )
    source = RetrievedSource(
        source="LightRAG mixed local corpus evidence",
        page=None,
        score=1.0,
        snippet=_snippet_from_lightrag_context(text),
    )
    return [doc], [source]


async def create_lightrag(settings: Settings | None = None):
    settings = settings or get_settings()
    _ensure_openai_key(settings)
    settings.lightrag_working_dir.mkdir(parents=True, exist_ok=True)
    _patch_lightrag_atomic_write_for_windows()

    from lightrag import LightRAG

    return await _initialized_rag(
        LightRAG(
            working_dir=str(settings.lightrag_working_dir),
            embedding_func=_embedding_func(settings),
            llm_model_func=_llm_model_func(settings),
            llm_model_name=settings.lightrag_llm_model,
            max_parallel_insert=max(1, min(int(settings.lightrag_ingest_concurrency or 2), 3)),
        )
    )


async def _initialized_rag(rag: Any):
    await rag.initialize_storages()
    return rag


def _llm_model_func(settings: Settings):
    async def llm_model_func(prompt, system_prompt=None, history_messages=None, keyword_extraction=False, **kwargs) -> str:
        from lightrag.llm.openai import openai_complete_if_cache

        return await openai_complete_if_cache(
            settings.lightrag_llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=settings.openai_api_key,
            **kwargs,
        )

    return llm_model_func


def _embedding_func(settings: Settings):
    from lightrag.llm.openai import openai_embed
    from lightrag.utils import wrap_embedding_func_with_attrs

    @wrap_embedding_func_with_attrs(
        embedding_dim=_openai_embedding_dim(settings.lightrag_embedding_model),
        max_token_size=8192,
        model_name=settings.lightrag_embedding_model,
    )
    async def embedding_func(texts: list[str]) -> np.ndarray:
        return await openai_embed.func(
            texts,
            model=settings.lightrag_embedding_model,
            api_key=settings.openai_api_key,
        )

    return embedding_func


def _ensure_openai_key(settings: Settings) -> None:
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY or OPEN_API_KEY is required for LightRAG.")
    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)


def _patch_lightrag_atomic_write_for_windows() -> None:
    """Retry and serialize LightRAG JSON/VDB replaces on Windows filesystems.

    LightRAG's JSON/NanoVectorDB storages use tmp-file writes followed by
    ``os.replace``. On Windows, concurrent async flushes or brief AV/indexer
    handles can raise WinError 5 even though the next replace would succeed.
    This keeps LightRAG's atomic-write semantics but adds a per-destination
    process lock and short exponential retry window.
    """

    global _ATOMIC_WRITE_PATCHED
    if _ATOMIC_WRITE_PATCHED or os.name != "nt":
        return

    with _ATOMIC_WRITE_PATCH_LOCK:
        if _ATOMIC_WRITE_PATCHED:
            return

        import lightrag.file_atomic as file_atomic

        original_atomic_write = file_atomic.atomic_write

        def atomic_write_with_retry(file_name: str, write_fn, workspace: str = "_") -> None:
            lock = _file_lock(file_name)
            with lock:
                last_exc: PermissionError | None = None
                for attempt in range(8):
                    try:
                        return original_atomic_write(file_name, write_fn, workspace)
                    except PermissionError as exc:
                        last_exc = exc
                        if getattr(exc, "winerror", None) != 5:
                            raise
                        delay = min(0.25 * (2**attempt), 4.0)
                        logger.warning(
                            "LightRAG storage replace was temporarily locked; retrying in %.2fs: %s",
                            delay,
                            file_name,
                        )
                        time.sleep(delay)
                raise last_exc or PermissionError(f"Could not replace LightRAG storage file: {file_name}")

        file_atomic.atomic_write = atomic_write_with_retry
        _patch_loaded_atomic_write_references(atomic_write_with_retry)
        _ATOMIC_WRITE_PATCHED = True


def _file_lock(file_name: str) -> threading.Lock:
    key = os.path.abspath(file_name).lower()
    with _ATOMIC_WRITE_PATCH_LOCK:
        lock = _ATOMIC_WRITE_FILE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
        _ATOMIC_WRITE_FILE_LOCKS[key] = lock
        return lock


def _patch_loaded_atomic_write_references(atomic_write_func) -> None:
    for module_name in (
        "lightrag.kg.nano_vector_db_impl",
        "lightrag.kg.networkx_impl",
        "lightrag.kg.faiss_impl",
    ):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "atomic_write"):
            setattr(module, "atomic_write", atomic_write_func)


def _lightrag_mode(value: str) -> str:
    normalized = (value or "mix").strip().lower()
    if normalized in {"mixed", "mix"}:
        return "mix"
    return normalized


def _openai_embedding_dim(model: str) -> int:
    model = (model or "").lower()
    if model == "text-embedding-3-small":
        return 1536
    if model in {"text-embedding-3-large", "text-embedding-ada-002"}:
        return 3072 if model == "text-embedding-3-large" else 1536
    return 1536


def _snippet_from_lightrag_context(text: str, limit: int = 350) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact[:limit]


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
