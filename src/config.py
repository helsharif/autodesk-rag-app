"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
load_dotenv(ROOT_DIR / ".env")

AUTODESK_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "autodesk-rag")
HYBRID_BACKEND_NAME = "docling_chroma_bm25_hybrid"
LOCAL_ONLY_MODE = "local_only"
AUTODESK_WEB_MODE = "autodesk_web"
OPEN_WEB_MODE = "open_web"
OPTION_1_LABEL = "Option 1: Local Document Search"
OPTION_2_LABEL = "Option 2: Local Document Search + Autodesk.com"
OPTION_3_LABEL = "Option 3: Local Document Search + Open Web Search"
SEARCH_MODE_OPTIONS = {
    OPTION_1_LABEL: LOCAL_ONLY_MODE,
    OPTION_2_LABEL: AUTODESK_WEB_MODE,
    OPTION_3_LABEL: OPEN_WEB_MODE,
}
COLLECTION_OPTIONS = {label: HYBRID_BACKEND_NAME for label in SEARCH_MODE_OPTIONS}
COLLECTION_SLUGS = {HYBRID_BACKEND_NAME: "docling_chroma_bm25_hybrid"}


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: str) -> int:
    return int(os.getenv(name, default))


def _env_float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


def _openai_api_key() -> str | None:
    return os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")


@dataclass(frozen=True)
class Settings:
    root_dir: Path = ROOT_DIR
    retrieval_indexes_dir: Path = ROOT_DIR / "retrieval_indexes"
    chroma_dir: Path = ROOT_DIR / "retrieval_indexes" / "chroma_autodesk_cleaned_corpus"
    bm25_dir: Path = ROOT_DIR / "retrieval_indexes" / "bm25_autodesk_cleaned_corpus"
    chunk_manifest_path: Path = ROOT_DIR / "retrieval_indexes" / "manifests" / "chunk_manifest.csv"
    eval_results_dir: Path = ROOT_DIR / "eval_results"
    eval_status_dir: Path = ROOT_DIR / "eval_status"
    eval_testset_path: Path = ROOT_DIR / "eval_testset" / "autodesk_testset.csv"
    collection_name: str = field(default_factory=lambda: _env_str("CHROMA_COLLECTION_NAME", "autodesk-rag"))
    openai_api_key: str | None = field(default_factory=_openai_api_key)
    openai_model: str = field(default_factory=lambda: _env_str("OPENAI_MODEL", "gpt-4.1-mini"))
    openai_embedding_model: str = field(default_factory=lambda: _env_str("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    retriever_k: int = field(default_factory=lambda: _env_int("RETRIEVER_K", "10"))
    hybrid_candidate_k: int = field(default_factory=lambda: _env_int("HYBRID_CANDIDATE_K", "30"))
    hybrid_vector_weight: float = field(default_factory=lambda: _env_float("HYBRID_VECTOR_WEIGHT", "0.65"))
    hybrid_bm25_weight: float = field(default_factory=lambda: _env_float("HYBRID_BM25_WEIGHT", "0.35"))
    hybrid_max_per_source: int = field(default_factory=lambda: _env_int("HYBRID_MAX_PER_SOURCE", "3"))
    compare_retrieval_max_workers: int = field(default_factory=lambda: _env_int("COMPARE_RETRIEVAL_MAX_WORKERS", "2"))
    min_relevance_score: float = field(default_factory=lambda: _env_float("MIN_RELEVANCE_SCORE", "0.30"))
    context_expansion_enabled: bool = field(default_factory=lambda: _env_bool("CONTEXT_EXPANSION_ENABLED", "true"))
    context_expansion_mode: str = field(default_factory=lambda: _env_str("CONTEXT_EXPANSION_MODE", "neighbors").lower())
    context_neighbor_window: int = field(default_factory=lambda: _env_int("CONTEXT_NEIGHBOR_WINDOW", "1"))
    context_max_expanded_docs: int = field(default_factory=lambda: _env_int("CONTEXT_MAX_EXPANDED_DOCS", "8"))
    context_max_chars: int = field(default_factory=lambda: _env_int("CONTEXT_MAX_CHARS", "18000"))
    reranker_enabled: bool = field(default_factory=lambda: _env_bool("RERANKER_ENABLED", "true"))
    reranker_model: str = field(default_factory=lambda: _env_str("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2"))
    reranker_top_n: int = field(default_factory=lambda: _env_int("RERANKER_TOP_N", "8"))
    reranker_batch_size: int = field(default_factory=lambda: _env_int("RERANKER_BATCH_SIZE", "16"))
    serpapi_api_key: str | None = field(default_factory=lambda: os.getenv("SERPAPI_API_KEY") or None)
    langsmith_api_key: str | None = field(default_factory=lambda: os.getenv("LANGSMITH_API_KEY") or None)
    langsmith_tracing: str | None = field(default_factory=lambda: os.getenv("LANGSMITH_TRACING") or None)
    langsmith_project: str | None = field(default_factory=lambda: os.getenv("LANGSMITH_PROJECT") or None)
    langsmith_endpoint: str | None = field(default_factory=lambda: os.getenv("LANGSMITH_ENDPOINT") or None)
    eval_judge_model: str = field(default_factory=lambda: _env_str("EVAL_JUDGE_MODEL", "gpt-5.1"))
    eval_judge_delay_seconds: float = field(default_factory=lambda: _env_float("EVAL_JUDGE_DELAY_SECONDS", "1.0"))
    eval_judge_max_retries: int = field(default_factory=lambda: _env_int("EVAL_JUDGE_MAX_RETRIES", "8"))


def get_settings() -> Settings:
    settings = Settings()
    if settings.openai_api_key:
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
    if settings.serpapi_api_key:
        os.environ.setdefault("SERPAPI_API_KEY", settings.serpapi_api_key)
    if settings.langsmith_api_key:
        os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
    if settings.langsmith_tracing:
        os.environ.setdefault("LANGSMITH_TRACING", settings.langsmith_tracing)
    if settings.langsmith_project:
        os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
    if settings.langsmith_endpoint:
        os.environ.setdefault("LANGSMITH_ENDPOINT", settings.langsmith_endpoint)
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "False")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return settings


def get_embeddings(settings: Settings | None = None):
    settings = settings or get_settings()
    from langchain_openai import OpenAIEmbeddings

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY or OPEN_API_KEY is required for OpenAI embeddings.")
    return OpenAIEmbeddings(model=settings.openai_embedding_model)


def get_chat_model(settings: Settings | None = None, temperature: float = 0.0, model: str | None = None):
    settings = settings or get_settings()
    from langchain_openai import ChatOpenAI

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY or OPEN_API_KEY is required for OpenAI chat models.")
    return ChatOpenAI(model=model or settings.openai_model, temperature=temperature)
