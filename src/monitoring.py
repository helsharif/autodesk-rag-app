"""Supabase-backed runtime monitoring helpers for the RAG app."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date, datetime
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any

import streamlit as st


TABLE_NAME = "rag_interactions"


def _secrets_file_exists() -> bool:
    candidates = [
        Path.home() / ".streamlit" / "secrets.toml",
        Path.cwd() / ".streamlit" / "secrets.toml",
    ]
    return any(path.exists() for path in candidates)


def _secret_or_env(name: str) -> str | None:
    if _secrets_file_exists():
        try:
            value = st.secrets.get(name)
            if value:
                return str(value)
        except Exception:
            pass
    return os.getenv(name) or None


@lru_cache(maxsize=1)
def get_supabase_client():
    """Return a cached Supabase client, or None when monitoring is not configured."""

    url = _secret_or_env("SUPABASE_URL")
    key = _secret_or_env("SUPABASE_KEY")
    if not url or not key:
        return None
    try:
        from supabase import create_client

        return create_client(url, key)
    except Exception:
        return None


def supabase_monitoring_enabled() -> bool:
    return get_supabase_client() is not None


def hash_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def safe_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def safe_int(value):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return str(value)


def extract_source_fields(sources) -> dict[str, list]:
    names: list[str] = []
    ids: list[str] = []
    scores: list[float] = []
    metadata: list[dict] = []

    for index, source in enumerate(sources or [], start=1):
        if isinstance(source, dict):
            name = source.get("source") or source.get("name") or source.get("title") or source.get("url")
            source_id = source.get("id") or source.get("source_id") or source.get("chunk_id") or source.get("url")
            score = safe_float(source.get("score") or source.get("relevance"))
            raw_metadata = source
        else:
            text = str(source)
            name = text.split(" (score ", 1)[0].strip() or f"Source {index}"
            source_id = name
            score = None
            if " (score " in text:
                score_text = text.rsplit(" (score ", 1)[-1].rstrip(")")
                score = safe_float(score_text)
            raw_metadata = {"label": text}

        names.append(str(name or f"Source {index}")[:500])
        ids.append(str(source_id or name or f"source-{index}")[:500])
        if score is not None:
            scores.append(score)
        metadata.append(_json_safe(raw_metadata))

    return {
        "top_source_names": names[:5],
        "top_source_ids": ids[:5],
        "top_source_scores": scores[:5],
        "source_metadata": metadata,
    }


def _clean_payload(event: dict) -> dict:
    payload = dict(event or {})
    question = str(payload.get("question") or "")
    payload["question_hash"] = payload.get("question_hash") or hash_text(question)

    source_fields = extract_source_fields(payload.get("sources") or payload.get("source_metadata") or [])
    for key, value in source_fields.items():
        payload.setdefault(key, value)

    numeric_float_fields = {
        "latency_total_sec",
        "latency_router_sec",
        "latency_retrieval_sec",
        "latency_expansion_sec",
        "latency_adequacy_sec",
        "latency_web_sec",
        "latency_generation_sec",
        "estimated_cost_usd",
    }
    numeric_int_fields = {
        "source_count",
        "retrieved_chunk_count",
        "expanded_context_chars",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    }
    for field in numeric_float_fields:
        if field in payload:
            payload[field] = safe_float(payload.get(field))
    for field in numeric_int_fields:
        if field in payload:
            payload[field] = safe_int(payload.get(field))

    allowed_fields = {
        "app_name",
        "app_environment",
        "session_id",
        "request_id",
        "question",
        "question_hash",
        "final_response",
        "retrieval_backend",
        "router_decision",
        "used_local",
        "used_web",
        "web_fallback_reason",
        "adequacy_answerable",
        "no_answer",
        "source_count",
        "retrieved_chunk_count",
        "expanded_context_chars",
        "top_source_names",
        "top_source_ids",
        "top_source_scores",
        "source_metadata",
        "latency_total_sec",
        "latency_router_sec",
        "latency_retrieval_sec",
        "latency_expansion_sec",
        "latency_adequacy_sec",
        "latency_web_sec",
        "latency_generation_sec",
        "model_name",
        "embedding_model_name",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "estimated_cost_usd",
        "success",
        "error",
        "error_type",
        "extra",
    }
    return {key: _json_safe(value) for key, value in payload.items() if key in allowed_fields}


def log_rag_interaction(event: dict) -> bool:
    client = get_supabase_client()
    if client is None:
        return False
    try:
        client.table(TABLE_NAME).insert(_clean_payload(event)).execute()
        return True
    except Exception:
        return False


def fetch_recent_interactions(limit: int = 1000) -> list[dict]:
    client = get_supabase_client()
    if client is None:
        return []
    try:
        response = (
            client.table(TABLE_NAME)
            .select("*")
            .order("created_at", desc=True)
            .limit(max(1, int(limit)))
            .execute()
        )
        return list(getattr(response, "data", None) or [])
    except Exception:
        return []


def clear_monitoring_logs() -> bool:
    client = get_supabase_client()
    if client is None:
        return False
    try:
        client.table(TABLE_NAME).delete().gt("id", 0).execute()
        return True
    except Exception:
        return False


def monitoring_admin_password() -> str | None:
    return _secret_or_env("MONITORING_ADMIN_PASSWORD")
