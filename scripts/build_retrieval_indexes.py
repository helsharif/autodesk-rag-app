# %% [markdown]
# # 02 Build Retrieval Indexes For Autodesk RAG
#
# This notebook builds the local retrieval assets used by the Autodesk RAG app:
#
# 1. A persistent ChromaDB vector database for semantic retrieval.
# 2. A local BM25 keyword index for lexical retrieval.
# 3. Document and chunk manifests for reproducibility and inspection.
# 4. Small search helpers for vector, BM25, and hybrid RRF sanity checks.
#
# It should be run after `notebook_01_corpus_cleaning.ipynb`, which writes the
# cleaned Markdown corpus to `cleaned_corpus/`.

# %%
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import pickle
import re
import shutil
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm

try:
    from IPython.display import display
except Exception:
    def display(obj: Any) -> None:
        print(obj)

# %% [markdown]
# ## 1. Setup And Configuration
#
# The project `.env` is loaded first. Existing environment variables are used
# where they match this project. The notebook falls back to safe local defaults
# and does not print secrets or overwrite `.env`.
#
# The cleaned corpus is already Markdown/text, so this workflow uses
# Markdown-aware parsing and chunking as the primary approach. Docling is checked
# and recorded as an available document representation option, but it is not
# forced into a PDF-style parsing workflow where it would add little value.

# %%
load_dotenv()


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except ValueError:
        print(f"Warning: {name}={value!r} is not an integer; using {default}.")
        return default


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        print(f"Warning: {name}={value!r} is not numeric; using {default}.")
        return default


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


CLEANED_CORPUS_DIR = Path(env_str("CLEANED_CORPUS_DIR", "cleaned_corpus"))
INDEX_ROOT_DIR = Path(env_str("INDEX_ROOT_DIR", "retrieval_indexes"))

CHROMA_DIR = Path(
    env_str("CHROMA_DIR", str(INDEX_ROOT_DIR / "chroma_autodesk_cleaned_corpus"))
)
BM25_DIR = Path(env_str("BM25_DIR", str(INDEX_ROOT_DIR / "bm25_autodesk_cleaned_corpus")))
MANIFEST_DIR = Path(env_str("MANIFEST_DIR", str(INDEX_ROOT_DIR / "manifests")))

CHUNK_SIZE_CHARS = env_int("CHUNK_SIZE_CHARS", env_int("CHUNK_SIZE", 2500))
CHUNK_OVERLAP_CHARS = env_int("CHUNK_OVERLAP_CHARS", env_int("CHUNK_OVERLAP", 300))
MIN_CHUNK_CHARS = env_int("MIN_CHUNK_CHARS", 300)
MAX_CHUNK_CHARS = env_int("MAX_CHUNK_CHARS", 4000)

# Keep indexing fully local by default. Existing OpenAI/Gemini embedding
# provider variables are app-level settings and are intentionally not used here.
EMBEDDING_MODEL_NAME = env_str(
    "EMBEDDING_MODEL_NAME",
    env_str("LOCAL_EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
)
CHROMA_COLLECTION_NAME = env_str("CHROMA_COLLECTION_NAME", "autodesk_cleaned_corpus")

BATCH_SIZE = env_int("BATCH_SIZE", env_int("EMBEDDING_BATCH_SIZE", 64))
N_WORKERS = env_int("N_WORKERS", max(1, (os.cpu_count() or 2) - 1))
OVERWRITE_EXISTING_INDEXES = env_bool("OVERWRITE_EXISTING_INDEXES", False)

FILE_EXTENSIONS = (".md", ".txt")
IGNORE_FILENAMES = {
    "cleaning_manifest.csv",
    "cleaning_summary.md",
    "before_after_processing_stats.md",
    "repeated_line_candidates.csv",
    "sample_cleaning_quality_review.md",
    "sample_cleaning_quality_review.csv",
}

VECTOR_QUERY_K = env_int("VECTOR_QUERY_K", env_int("RETRIEVER_K", 10))
BM25_QUERY_K = env_int("BM25_QUERY_K", env_int("RETRIEVER_K", 10))
HYBRID_RRF_K = env_int("HYBRID_RRF_K", 60)

CONFIG = {
    "CLEANED_CORPUS_DIR": str(CLEANED_CORPUS_DIR),
    "INDEX_ROOT_DIR": str(INDEX_ROOT_DIR),
    "CHROMA_DIR": str(CHROMA_DIR),
    "BM25_DIR": str(BM25_DIR),
    "MANIFEST_DIR": str(MANIFEST_DIR),
    "CHUNK_SIZE_CHARS": CHUNK_SIZE_CHARS,
    "CHUNK_OVERLAP_CHARS": CHUNK_OVERLAP_CHARS,
    "MIN_CHUNK_CHARS": MIN_CHUNK_CHARS,
    "MAX_CHUNK_CHARS": MAX_CHUNK_CHARS,
    "EMBEDDING_MODEL_NAME": EMBEDDING_MODEL_NAME,
    "CHROMA_COLLECTION_NAME": CHROMA_COLLECTION_NAME,
    "BATCH_SIZE": BATCH_SIZE,
    "N_WORKERS": N_WORKERS,
    "OVERWRITE_EXISTING_INDEXES": OVERWRITE_EXISTING_INDEXES,
}

pd.DataFrame([CONFIG]).T.rename(columns={0: "value"})

# %% [markdown]
# ## 2. Environment And Dependency Check
#
# Required packages are imported from the current virtual environment. If a
# package is missing, install it in the existing environment rather than
# creating a new environment inside this notebook.

# %%
REQUIRED_PACKAGES = {
    "python-dotenv": "dotenv",
    "pandas": "pandas",
    "tqdm": "tqdm",
    "chromadb": "chromadb",
    "rank-bm25": "rank_bm25",
    "sentence-transformers": "sentence_transformers",
}

OPTIONAL_PACKAGES = {
    "docling": "docling",
    "torch": "torch",
}


def dependency_report() -> pd.DataFrame:
    rows = []
    for package_name, import_name in REQUIRED_PACKAGES.items():
        available = importlib.util.find_spec(import_name) is not None
        rows.append(
            {
                "package": package_name,
                "import_name": import_name,
                "required": True,
                "available": available,
            }
        )
    for package_name, import_name in OPTIONAL_PACKAGES.items():
        available = importlib.util.find_spec(import_name) is not None
        rows.append(
            {
                "package": package_name,
                "import_name": import_name,
                "required": False,
                "available": available,
            }
        )
    report = pd.DataFrame(rows)
    missing_required = report[report["required"] & ~report["available"]]
    if not missing_required.empty:
        missing = " ".join(missing_required["package"].tolist())
        print("Missing required packages. Install with:")
        print(f"pip install {missing}")
    return report


deps = dependency_report()
deps

# %%
if not deps[deps["required"] & ~deps["available"]].empty:
    raise RuntimeError("Install missing required packages before building indexes.")

import chromadb
from sentence_transformers import SentenceTransformer

try:
    import torch
except Exception:
    torch = None

DOCLING_AVAILABLE = importlib.util.find_spec("docling") is not None
DEVICE_NOTE = "auto"
if torch is not None:
    DEVICE_NOTE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Docling available: {DOCLING_AVAILABLE}")
print(f"SentenceTransformers device availability: {DEVICE_NOTE}")
print(
    "GPU can speed up embedding generation when available; BM25 and Markdown "
    "parsing are CPU/string-processing workloads."
)

# %% [markdown]
# ## 3. File Discovery
#
# Discover cleaned `.md` and `.txt` documents under `cleaned_corpus/`. Generated
# reports are ignored so diagnostics do not become part of the RAG corpus.

# %%
def discover_documents(root: Path) -> pd.DataFrame:
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in FILE_EXTENSIONS:
            continue
        if path.name in IGNORE_FILENAMES:
            continue
        rel = path.relative_to(root).as_posix()
        records.append(
            {
                "source_file": str(path),
                "relative_source_path": rel,
                "file_size_bytes": path.stat().st_size,
                "suffix": path.suffix.lower(),
            }
        )
    return pd.DataFrame(records)


docs_df = discover_documents(CLEANED_CORPUS_DIR)
print(f"Discovered {len(docs_df):,} cleaned corpus documents.")
docs_df.head()

# %% [markdown]
# ## 4. Markdown Metadata And Structure Parsing
#
# Cleaned documents start with YAML-style provenance metadata. The parser below
# preserves that metadata, extracts a title, and splits content primarily by
# Markdown headings. It is deliberately conservative: short technical statements,
# tables, list items, and code blocks are not discarded just because they are
# compact.

# %%
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.S)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./#:+-]*|\d+(?:\.\d+)*")


@dataclass
class Section:
    heading_path: str
    text: str


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    metadata: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"').strip("'")
    return metadata, text[match.end() :]


def markdown_title(body: str, metadata: dict[str, str], fallback: str) -> str:
    if metadata.get("title"):
        return metadata["title"]
    for line in body.splitlines():
        match = HEADING_RE.match(line.strip())
        if match and len(match.group(1)) == 1:
            return match.group(2).strip()
    return Path(fallback).stem


def normalize_heading_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip("#").strip()
    return text or "Untitled Section"


def split_markdown_sections(body: str, title: str) -> list[Section]:
    sections: list[Section] = []
    heading_stack: dict[int, str] = {}
    current_lines: list[str] = []
    current_path = title
    in_code_block = False

    def flush() -> None:
        nonlocal current_lines, current_path
        text = "\n".join(current_lines).strip()
        if text:
            sections.append(Section(heading_path=current_path, text=text))
        current_lines = []

    for line in body.splitlines():
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            current_lines.append(line)
            continue

        match = HEADING_RE.match(line.strip()) if not in_code_block else None
        if match:
            flush()
            level = len(match.group(1))
            heading = normalize_heading_text(match.group(2))
            heading_stack[level] = heading
            for deeper in list(heading_stack):
                if deeper > level:
                    heading_stack.pop(deeper, None)
            ordered = [heading_stack[i] for i in sorted(heading_stack) if i <= level]
            current_path = " > ".join(ordered) if ordered else title
            current_lines.append(line)
        else:
            current_lines.append(line)

    flush()
    if not sections and body.strip():
        sections.append(Section(heading_path=title, text=body.strip()))
    return sections


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def clean_chunk_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def split_large_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    text = clean_chunk_text(text)
    if len(text) <= max_chars:
        return [text] if text else []

    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) > max_chars:
            if current.strip():
                chunks.append(current.strip())
                current = ""
            start = 0
            while start < len(para):
                end = min(len(para), start + max_chars)
                chunks.append(para[start:end].strip())
                if end == len(para):
                    break
                start = max(0, end - overlap_chars)
            continue
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current.strip():
                chunks.append(current.strip())
            current = para

    if current.strip():
        chunks.append(current.strip())

    if overlap_chars <= 0 or len(chunks) <= 1:
        return chunks

    overlapped = []
    previous_tail = ""
    for chunk in chunks:
        merged = f"{previous_tail}\n\n{chunk}".strip() if previous_tail else chunk
        overlapped.append(merged)
        previous_tail = chunk[-overlap_chars:].strip()
    return overlapped


def merge_small_sections(sections: list[Section], min_chars: int) -> list[Section]:
    merged: list[Section] = []
    buffer: Section | None = None

    for section in sections:
        text = clean_chunk_text(section.text)
        if not text:
            continue
        section = Section(section.heading_path, text)
        if buffer is None:
            buffer = section
            continue
        if len(buffer.text) < min_chars:
            combined_path = buffer.heading_path or section.heading_path
            combined_text = f"{buffer.text}\n\n{section.text}".strip()
            buffer = Section(combined_path, combined_text)
        else:
            merged.append(buffer)
            buffer = section

    if buffer is not None:
        if merged and len(buffer.text) < min_chars:
            prior = merged.pop()
            merged.append(
                Section(prior.heading_path, f"{prior.text}\n\n{buffer.text}".strip())
            )
        else:
            merged.append(buffer)
    return merged


def chunk_document(
    text: str, relative_path: str, source_file: str
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    metadata, body = parse_frontmatter(text)
    body = body.strip()
    title = markdown_title(body, metadata, relative_path)
    sections = merge_small_sections(split_markdown_sections(body, title), MIN_CHUNK_CHARS)

    chunks: list[dict[str, Any]] = []
    for section in sections:
        for piece in split_large_text(
            section.text,
            max_chars=min(MAX_CHUNK_CHARS, CHUNK_SIZE_CHARS),
            overlap_chars=CHUNK_OVERLAP_CHARS,
        ):
            piece = clean_chunk_text(piece)
            if not piece:
                continue
            context = (
                f"Title: {title}\n"
                f"Section: {section.heading_path}\n"
                f"Source: {relative_path}\n\n"
            )
            embedding_text = f"{context}{piece}"
            chunk_index = len(chunks)
            chunk_hash = hashlib.sha1(
                f"{relative_path}|{chunk_index}|{piece}".encode("utf-8")
            ).hexdigest()[:16]
            chunk_id = f"autodesk_{chunk_hash}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source_file": source_file,
                    "relative_source_path": relative_path,
                    "title": title,
                    "heading_path": section.heading_path,
                    "chunk_index": chunk_index,
                    "chunk_text": piece,
                    "embedding_text": embedding_text,
                    "chunk_char_count": len(piece),
                    "embedding_char_count": len(embedding_text),
                    "approx_token_count": estimate_tokens(embedding_text),
                    "cleaned_format": metadata.get("cleaned_format", "markdown"),
                    "source_metadata_json": json.dumps(metadata, ensure_ascii=False),
                    "source_url": metadata.get("source_url", ""),
                }
            )

    return metadata, chunks

# %% [markdown]
# ## 5. Build Chunk Manifest
#
# Each document is parsed independently. Failures are recorded at document level
# and indexing continues for the rest of the corpus.

# %%
def process_document(row: pd.Series) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_file = row["source_file"]
    relative_path = row["relative_source_path"]
    file_size = row["file_size_bytes"]
    try:
        text = Path(source_file).read_text(encoding="utf-8", errors="replace")
        metadata, chunks = chunk_document(text, relative_path, source_file)
        return (
            {
                "source_file": source_file,
                "relative_source_path": relative_path,
                "file_size_bytes": file_size,
                "document_title": metadata.get("title")
                or (chunks[0]["title"] if chunks else Path(relative_path).stem),
                "num_chunks_generated": len(chunks),
                "total_characters": len(text),
                "status": "indexed" if chunks else "skipped",
                "warning_or_error": "" if chunks else "no_nonempty_chunks_generated",
            },
            chunks,
        )
    except Exception as exc:
        return (
            {
                "source_file": source_file,
                "relative_source_path": relative_path,
                "file_size_bytes": file_size,
                "document_title": Path(relative_path).stem,
                "num_chunks_generated": 0,
                "total_characters": 0,
                "status": "failed",
                "warning_or_error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            },
            [],
        )


document_records: list[dict[str, Any]] = []
all_chunks: list[dict[str, Any]] = []

for _, row in tqdm(docs_df.iterrows(), total=len(docs_df), desc="Chunking documents"):
    doc_record, chunks = process_document(row)
    document_records.append(doc_record)
    all_chunks.extend(chunks)

indexing_manifest_df = pd.DataFrame(document_records)
chunk_manifest_df = pd.DataFrame(all_chunks)

print(f"Generated {len(chunk_manifest_df):,} chunks from {len(indexing_manifest_df):,} documents.")
indexing_manifest_df["status"].value_counts(dropna=False)

# %%
if not chunk_manifest_df.empty:
    duplicate_ids = chunk_manifest_df["chunk_id"].duplicated().sum()
    if duplicate_ids:
        raise RuntimeError(f"Duplicate chunk IDs detected: {duplicate_ids}")
    chunk_manifest_df[
        [
            "chunk_id",
            "relative_source_path",
            "title",
            "heading_path",
            "chunk_index",
            "chunk_char_count",
        ]
    ].head()

# %% [markdown]
# ## 6. Create Output Folders
#
# Generated index directories are reproducible artifacts. If
# `OVERWRITE_EXISTING_INDEXES=True`, existing Chroma and BM25 folders are
# removed first. Otherwise, Chroma is upserted by stable chunk ID and BM25
# artifacts are rewritten from the current chunk manifest.

# %%
INDEX_ROOT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

if OVERWRITE_EXISTING_INDEXES:
    for folder in [CHROMA_DIR, BM25_DIR]:
        if folder.exists():
            shutil.rmtree(folder)

CHROMA_DIR.mkdir(parents=True, exist_ok=True)
BM25_DIR.mkdir(parents=True, exist_ok=True)

print(f"Chroma directory: {CHROMA_DIR}")
print(f"BM25 directory: {BM25_DIR}")
print(f"Manifest directory: {MANIFEST_DIR}")

# %% [markdown]
# ## 7. Build ChromaDB Vector Index
#
# Chroma stores the context-prefixed chunk text and scalar metadata. The
# embedding model is a local SentenceTransformers model by default. The model can
# be swapped later by setting `EMBEDDING_MODEL_NAME` in `.env`.

# %%
def chroma_scalar(value: Any) -> str | int | float | bool:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False)


def chunk_to_chroma_metadata(chunk: dict[str, Any]) -> dict[str, str | int | float | bool]:
    return {
        "source_file": chroma_scalar(chunk["source_file"]),
        "relative_source_path": chroma_scalar(chunk["relative_source_path"]),
        "title": chroma_scalar(chunk["title"]),
        "heading_path": chroma_scalar(chunk["heading_path"]),
        "chunk_index": int(chunk["chunk_index"]),
        "chunk_char_count": int(chunk["chunk_char_count"]),
        "approx_token_count": int(chunk["approx_token_count"]),
        "cleaned_format": chroma_scalar(chunk["cleaned_format"]),
        "source_url": chroma_scalar(chunk.get("source_url", "")),
    }


embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
embedding_dimension = int(embedding_model.get_sentence_embedding_dimension())
print(f"Embedding model: {EMBEDDING_MODEL_NAME}")
print(f"Embedding dimension: {embedding_dimension}")

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
if OVERWRITE_EXISTING_INDEXES:
    try:
        chroma_client.delete_collection(CHROMA_COLLECTION_NAME)
    except Exception:
        pass

collection = chroma_client.get_or_create_collection(
    name=CHROMA_COLLECTION_NAME,
    metadata={
        "description": "Autodesk cleaned corpus chunks",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimension": embedding_dimension,
    },
)

# %%
embedded_success: dict[str, bool] = {}

ids = chunk_manifest_df["chunk_id"].tolist()
documents = chunk_manifest_df["embedding_text"].tolist()
metadatas = [chunk_to_chroma_metadata(row) for row in all_chunks]

for start in tqdm(range(0, len(ids), BATCH_SIZE), desc="Embedding and upserting Chroma"):
    end = min(start + BATCH_SIZE, len(ids))
    batch_ids = ids[start:end]
    batch_docs = documents[start:end]
    batch_metas = metadatas[start:end]
    try:
        embeddings = embedding_model.encode(
            batch_docs,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()
        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=embeddings,
            metadatas=batch_metas,
        )
        for chunk_id in batch_ids:
            embedded_success[chunk_id] = True
    except Exception as exc:
        print(f"Embedding/upsert failed for batch {start}:{end}: {exc}")
        for chunk_id in batch_ids:
            embedded_success[chunk_id] = False

print(f"Chroma collection count: {collection.count():,}")

# %% [markdown]
# ## 8. Build BM25 Keyword Index
#
# The tokenizer keeps technical tokens such as API names, model identifiers,
# numbers, file-like strings, and slash/dash/hash terms. Stopword removal is
# intentionally minimal to avoid discarding domain-specific short tokens.

# %%
def bm25_tokenize(text: str) -> list[str]:
    tokens = [token.lower() for token in TOKEN_RE.findall(text)]
    return [token for token in tokens if len(token) > 1 or token in {"c", "r"}]


bm25_texts = chunk_manifest_df["embedding_text"].tolist()
bm25_chunk_ids = chunk_manifest_df["chunk_id"].tolist()
bm25_tokenized_corpus = [
    bm25_tokenize(text) for text in tqdm(bm25_texts, desc="Tokenizing BM25 corpus")
]
bm25_index = BM25Okapi(bm25_tokenized_corpus)

bm25_metadata = {
    row["chunk_id"]: {
        "source_file": row["source_file"],
        "relative_source_path": row["relative_source_path"],
        "title": row["title"],
        "heading_path": row["heading_path"],
        "chunk_index": int(row["chunk_index"]),
        "chunk_char_count": int(row["chunk_char_count"]),
        "approx_token_count": int(row["approx_token_count"]),
        "preview": row["chunk_text"][:500],
    }
    for _, row in chunk_manifest_df.iterrows()
}

with (BM25_DIR / "bm25_index.pkl").open("wb") as f:
    pickle.dump(bm25_index, f)
with (BM25_DIR / "bm25_tokenized_corpus.pkl").open("wb") as f:
    pickle.dump(bm25_tokenized_corpus, f)
(BM25_DIR / "bm25_chunk_ids.json").write_text(
    json.dumps(bm25_chunk_ids, indent=2), encoding="utf-8"
)
(BM25_DIR / "bm25_chunk_metadata.json").write_text(
    json.dumps(bm25_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
)

bm25_success = {chunk_id: True for chunk_id in bm25_chunk_ids}
print(f"Saved BM25 index with {len(bm25_chunk_ids):,} chunks.")

# %% [markdown]
# ## 9. Save Manifests And Summary
#
# The document manifest tracks source-level status. The chunk manifest tracks
# retrieval units and whether each chunk was added successfully to each index.

# %%
if not chunk_manifest_df.empty:
    chunk_manifest_df["embedded_successfully"] = chunk_manifest_df["chunk_id"].map(
        embedded_success
    ).fillna(False)
    chunk_manifest_df["added_to_bm25_successfully"] = chunk_manifest_df["chunk_id"].map(
        bm25_success
    ).fillna(False)
    chunk_manifest_df["first_300_characters"] = chunk_manifest_df["chunk_text"].str[:300]

chunk_manifest_save_cols = [
    "chunk_id",
    "source_file",
    "relative_source_path",
    "title",
    "heading_path",
    "chunk_index",
    "chunk_char_count",
    "approx_token_count",
    "first_300_characters",
    "embedded_successfully",
    "added_to_bm25_successfully",
]

indexing_manifest_path = MANIFEST_DIR / "indexing_manifest.csv"
chunk_manifest_path = MANIFEST_DIR / "chunk_manifest.csv"
summary_path = MANIFEST_DIR / "indexing_summary.md"

indexing_manifest_df.to_csv(indexing_manifest_path, index=False)
chunk_manifest_df[chunk_manifest_save_cols].to_csv(chunk_manifest_path, index=False)

run_timestamp = datetime.now(timezone.utc).isoformat()
status_counts = indexing_manifest_df["status"].value_counts().to_dict()

summary_lines = [
    "# Retrieval Indexing Summary",
    "",
    f"- Timestamp UTC: `{run_timestamp}`",
    f"- Cleaned files discovered: `{len(docs_df):,}`",
    f"- Files indexed: `{status_counts.get('indexed', 0):,}`",
    f"- Files skipped: `{status_counts.get('skipped', 0):,}`",
    f"- Files failed: `{status_counts.get('failed', 0):,}`",
    f"- Total chunks: `{len(chunk_manifest_df):,}`",
    f"- Average chunks per indexed document: `{len(chunk_manifest_df) / max(1, status_counts.get('indexed', 0)):.2f}`",
    f"- Average chunk length: `{chunk_manifest_df['chunk_char_count'].mean():.1f}`",
    f"- Shortest chunk length: `{int(chunk_manifest_df['chunk_char_count'].min())}`",
    f"- Longest chunk length: `{int(chunk_manifest_df['chunk_char_count'].max())}`",
    f"- Embedding model used: `{EMBEDDING_MODEL_NAME}`",
    f"- Embedding dimension: `{embedding_dimension}`",
    f"- Chroma collection name: `{CHROMA_COLLECTION_NAME}`",
    f"- Chroma persistence directory: `{CHROMA_DIR.as_posix()}`",
    f"- Chroma collection count: `{collection.count():,}`",
    f"- BM25 persistence directory: `{BM25_DIR.as_posix()}`",
    f"- BM25 chunk count: `{len(bm25_chunk_ids):,}`",
    f"- Docling available: `{DOCLING_AVAILABLE}`",
    "",
    "## Notes",
    "",
    "- The corpus is already cleaned Markdown/text, so Markdown-aware parsing is the primary chunking strategy.",
    "- Docling is available for future document-aware ingestion experiments, but this notebook avoids unnecessary PDF-style conversion.",
    "- GPU can improve embedding throughput when SentenceTransformers can use CUDA. BM25 and parsing remain CPU-bound.",
]

summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

print(f"Saved {indexing_manifest_path}")
print(f"Saved {chunk_manifest_path}")
print(f"Saved {summary_path}")

# %% [markdown]
# ## 10. Load Indexes For Retrieval Tests
#
# These helpers can be reused in later notebooks or copied into the Streamlit
# app. Results include source, title, heading path, chunk ID, score, and a short
# preview.

# %%
chunk_lookup = {
    row["chunk_id"]: row.to_dict() for _, row in chunk_manifest_df.iterrows()
}


def preview(text: str, max_chars: int = 500) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def result_from_chunk(
    chunk_id: str,
    rank: int,
    score: float,
    score_name: str,
) -> dict[str, Any]:
    row = chunk_lookup.get(chunk_id, {})
    return {
        "rank": rank,
        score_name: float(score),
        "chunk_id": chunk_id,
        "source_file": row.get("source_file", ""),
        "relative_source_path": row.get("relative_source_path", ""),
        "title": row.get("title", ""),
        "heading_path": row.get("heading_path", ""),
        "preview": preview(row.get("chunk_text", "")),
    }


def search_vector(query: str, k: int = 10) -> list[dict[str, Any]]:
    query_embedding = embedding_model.encode(
        [query], show_progress_bar=False, normalize_embeddings=True
    ).tolist()[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    output = []
    for i, chunk_id in enumerate(results.get("ids", [[]])[0]):
        distance = results.get("distances", [[]])[0][i]
        score = 1.0 / (1.0 + float(distance))
        row = result_from_chunk(chunk_id, i + 1, score, "vector_score")
        if not row["preview"]:
            row["preview"] = preview(results.get("documents", [[]])[0][i])
        output.append(row)
    return output


def search_bm25(query: str, k: int = 10) -> list[dict[str, Any]]:
    query_tokens = bm25_tokenize(query)
    scores = bm25_index.get_scores(query_tokens)
    if len(scores) == 0:
        return []
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    output = []
    for rank, idx in enumerate(top_indices, start=1):
        chunk_id = bm25_chunk_ids[idx]
        output.append(result_from_chunk(chunk_id, rank, float(scores[idx]), "bm25_score"))
    return output


def search_hybrid_rrf(
    query: str,
    k: int = 10,
    vector_k: int = 20,
    bm25_k: int = 20,
    rrf_k: int = HYBRID_RRF_K,
) -> list[dict[str, Any]]:
    vector_results = search_vector(query, k=vector_k)
    bm25_results = search_bm25(query, k=bm25_k)

    fused_scores: defaultdict[str, float] = defaultdict(float)
    sources: dict[str, dict[str, Any]] = {}

    for result in vector_results:
        chunk_id = result["chunk_id"]
        fused_scores[chunk_id] += 1.0 / (rrf_k + result["rank"])
        sources.setdefault(chunk_id, result)

    for result in bm25_results:
        chunk_id = result["chunk_id"]
        fused_scores[chunk_id] += 1.0 / (rrf_k + result["rank"])
        sources.setdefault(chunk_id, result)

    ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:k]
    output = []
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        base = sources[chunk_id].copy()
        base["rank"] = rank
        base["rrf_score"] = float(fused_scores[chunk_id])
        output.append(base)
    return output


def show_results(results: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(results)[
        [
            "rank",
            "chunk_id",
            "title",
            "heading_path",
            "relative_source_path",
            "preview",
        ]
    ]

# %% [markdown]
# ## 11. Retrieval Sanity Checks
#
# These example Autodesk-oriented queries are intended for a quick quality
# check. They are not an evaluation set.

# %%
EXAMPLE_QUERIES = [
    "How do I authenticate with Autodesk APIs?",
    "What permissions are required to access model data?",
    "How do I upload a file to a bucket?",
    "How do I handle derivative translation errors?",
    "What are the differences between Standard, Premium, and Enterprise subscription plans?",
]

for query in EXAMPLE_QUERIES:
    print("\n" + "=" * 100)
    print(f"Query: {query}")
    display(show_results(search_hybrid_rrf(query, k=5, vector_k=15, bm25_k=15)))

# %% [markdown]
# ## 12. Git And Git LFS Notes
#
# Generated retrieval indexes can become large and are reproducible from the
# cleaned corpus plus this notebook.
#
# Recommended practice:
#
# - Commit source code, notebooks, prompts, and small manifests normally.
# - Do not commit large generated vector databases unless there is a specific
#   reason.
# - If large generated index files must be uploaded to GitHub, use Git LFS.
# - Keep generated index folders in `.gitignore` when they are reproducible.
#
# Suggested `.gitignore` entries:
#
# ```text
# retrieval_indexes/chroma_autodesk_cleaned_corpus/
# retrieval_indexes/bm25_autodesk_cleaned_corpus/
# *.pkl
# *.sqlite3
# *.sqlite
# ```
#
# Suggested Git LFS commands if large index artifacts must be versioned:
#
# ```bash
# git lfs install
# git lfs track "*.pkl"
# git lfs track "*.sqlite3"
# git lfs track "*.sqlite"
# git lfs track "retrieval_indexes/**"
# git add .gitattributes
# ```
#
# Do not run these Git commands from the notebook unless you explicitly want to
# change repository tracking behavior.
