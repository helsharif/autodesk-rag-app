"""Ingest cleaned Autodesk Markdown corpus into a dedicated LightRAG index."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
import sys

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import get_settings
from src.lightrag_adapter import create_lightrag


LOGGER = logging.getLogger("ingest_lightrag_autodesk")
MANIFEST_NAME = "ingestion_manifest.json"


@dataclass(frozen=True)
class MarkdownDocument:
    path: Path
    relative_path: str
    checksum: str
    title: str
    source_url: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest cleaned_corpus/*.md into the Option 4 LightRAG index.")
    parser.add_argument("--corpus-dir", type=Path, default=ROOT_DIR / "cleaned_corpus")
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--rebuild", action="store_true", help="Delete and rebuild only the configured LightRAG index folder.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for smoke-testing ingestion.")
    parser.add_argument("--concurrency", type=int, default=None, help="Files to submit per LightRAG insert batch. Clamped to 1-3.")
    return parser.parse_args()


def main() -> None:
    load_dotenv(ROOT_DIR / ".env")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.perf_counter()
    args = parse_args()
    settings = get_settings()
    index_dir = args.index_dir or settings.lightrag_working_dir
    index_dir = index_dir.resolve()
    corpus_dir = args.corpus_dir.resolve()

    if args.rebuild:
        _delete_index_dir(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(corpus_dir.rglob("*.md"))
    if args.limit and args.limit > 0:
        files = files[: args.limit]
    LOGGER.info("Markdown files found: %s", len(files))
    LOGGER.info("Target LightRAG index path: %s", index_dir)

    manifest_path = index_dir / MANIFEST_NAME
    manifest = _load_manifest(manifest_path)
    docs = [_read_markdown(path, corpus_dir) for path in files]
    pending = [doc for doc in docs if manifest.get("files", {}).get(doc.relative_path, {}).get("checksum") != doc.checksum]
    skipped = len(docs) - len(pending)
    LOGGER.info("Already ingested/skipped files: %s", skipped)
    LOGGER.info("Documents pending insert: %s", len(pending))

    concurrency = _ingest_concurrency(args.concurrency, settings.lightrag_ingest_concurrency)
    LOGGER.info("LightRAG ingestion concurrency: %s files", concurrency)
    inserted = asyncio.run(_insert_documents(pending, index_dir, concurrency))
    for doc in pending[:inserted]:
        manifest.setdefault("files", {})[doc.relative_path] = {
            "checksum": doc.checksum,
            "title": doc.title,
            "source_url": doc.source_url,
        }
    manifest["index_dir"] = str(index_dir)
    manifest["corpus_dir"] = str(corpus_dir)
    manifest["total_markdown_files"] = len(files)
    manifest["inserted_files"] = len(manifest.get("files", {}))
    _write_manifest(manifest_path, manifest)

    elapsed = time.perf_counter() - started
    LOGGER.info("Documents inserted: %s", inserted)
    LOGGER.info("Skipped files: %s", skipped)
    LOGGER.info("Total ingestion time: %.2f seconds", elapsed)


async def _insert_documents(docs: list[MarkdownDocument], index_dir: Path, concurrency: int) -> int:
    if not docs:
        return 0
    settings = get_settings()
    object.__setattr__(settings, "lightrag_working_dir", index_dir)
    rag = None
    inserted = 0
    try:
        rag = await create_lightrag(settings)
        for batch in _batches(docs, concurrency):
            await rag.ainsert(
                [_document_payload(doc) for doc in batch],
                ids=[doc.checksum for doc in batch],
                file_paths=[doc.relative_path for doc in batch],
            )
            inserted += len(batch)
            if inserted % 10 == 0 or inserted == len(docs):
                LOGGER.info("Inserted %s/%s documents", inserted, len(docs))
    finally:
        if rag is not None:
            await rag.finalize_storages()
    return inserted


def _read_markdown(path: Path, corpus_dir: Path) -> MarkdownDocument:
    text = path.read_text(encoding="utf-8", errors="replace")
    relative_path = path.relative_to(corpus_dir).as_posix()
    return MarkdownDocument(
        path=path,
        relative_path=relative_path,
        checksum=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        title=_extract_title(text, path),
        source_url=_extract_source_url(text),
        text=text,
    )


def _document_payload(doc: MarkdownDocument) -> str:
    metadata_lines = [
        "---",
        f"source_filename: {doc.path.name}",
        f"relative_source_path: {doc.relative_path}",
        f"title: {doc.title}",
        f"source_url: {doc.source_url}" if doc.source_url else "source_url:",
        "retrieval_mode: LightRAG mixed",
        "---",
        "",
    ]
    return "\n".join(metadata_lines) + doc.text


def _extract_title(text: str, path: Path) -> str:
    title_match = re.search(r"(?m)^#\s+(.+?)\s*$", text)
    if title_match:
        return title_match.group(1).strip()
    metadata_match = re.search(r"(?im)^title:\s*(.+?)\s*$", text)
    if metadata_match:
        return metadata_match.group(1).strip()
    return path.stem


def _extract_source_url(text: str) -> str:
    metadata_match = re.search(r"(?im)^(?:source_url|url):\s*(https?://\S+)\s*$", text)
    if metadata_match:
        return metadata_match.group(1).strip()
    url_match = re.search(r"https?://(?:www\.)?autodesk\.com/\S+", text)
    return url_match.group(0).rstrip(").,]") if url_match else ""


def _ingest_concurrency(cli_value: int | None, settings_value: int) -> int:
    value = cli_value if cli_value is not None else settings_value
    return max(1, min(int(value or 2), 3))


def _batches(docs: list[MarkdownDocument], batch_size: int):
    for index in range(0, len(docs), batch_size):
        yield docs[index : index + batch_size]


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"files": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(path: Path, payload: dict) -> None:
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(path)


def _delete_index_dir(index_dir: Path) -> None:
    resolved = index_dir.resolve()
    allowed_parent = (ROOT_DIR / "retrieval_indexes").resolve()
    if allowed_parent not in resolved.parents:
        raise ValueError(f"Refusing to delete index outside retrieval_indexes: {resolved}")
    if resolved.exists():
        LOGGER.warning("Rebuild requested; deleting LightRAG index folder: %s", resolved)
        shutil.rmtree(resolved)


if __name__ == "__main__":
    main()
