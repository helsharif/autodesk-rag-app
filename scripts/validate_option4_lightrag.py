"""Smoke-test Option 4 LightRAG + Autodesk.com retrieval."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agent import AutodeskRAGAgent
from src.config import HYBRID_BACKEND_NAME, LIGHTRAG_AUTODESK_WEB_MODE, get_settings
from src.lightrag_adapter import lightrag_index_exists


EXAMPLE_QUERIES = (
    "What is AutoCAD used for?",
    "What is the difference between AutoCAD and Revit?",
    "What Autodesk products support BIM workflows?",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Option 4 LightRAG mixed mode + Autodesk.com.")
    parser.add_argument("--query", action="append", help="Query to test. Can be supplied more than once.")
    parser.add_argument("--skip-answer", action="store_true", help="Only check index/config readiness.")
    return parser.parse_args()


def main() -> int:
    load_dotenv(ROOT_DIR / ".env")
    args = parse_args()
    settings = get_settings()

    print(f"LightRAG index path: {settings.lightrag_working_dir}")
    print(f"LightRAG index exists: {lightrag_index_exists(settings)}")
    print(f"LightRAG retrieval mode: {settings.lightrag_retrieval_mode}")
    print(f"Autodesk.com web search configured: {bool(settings.serpapi_api_key)}")

    if not lightrag_index_exists(settings):
        print("Missing LightRAG index. Run: python scripts/ingest_lightrag_autodesk.py")
        return 1
    if args.skip_answer:
        return 0

    agent = AutodeskRAGAgent(collection_name=HYBRID_BACKEND_NAME, search_mode=LIGHTRAG_AUTODESK_WEB_MODE)
    for query in args.query or EXAMPLE_QUERIES:
        started = time.perf_counter()
        result = agent.answer(query)
        elapsed = time.perf_counter() - started
        print("\n" + "=" * 80)
        print(f"Query: {query}")
        print(f"Elapsed: {elapsed:.2f}s")
        print(f"Used local LightRAG evidence: {result.used_local}")
        print(f"Used Autodesk.com web evidence: {result.used_web}")
        print(f"Web query: {result.web_query}")
        if result.web_search_error:
            print(f"Web search error: {result.web_search_error}")
        print("Sources:")
        for source in result.sources:
            print(f"- {source}")
        print("\nAnswer:")
        print(result.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
