"""Background evaluation runner launched by Streamlit."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import get_settings
from src.evaluation import run_evaluation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-name", default="docling_chroma_bm25_hybrid")
    args = parser.parse_args()
    settings = get_settings()
    status_path = settings.eval_status_dir / "docling_chroma_bm25_hybrid_status.json"
    started = _now()
    _write_status(status_path, {"status": "running", "phase": "starting", "message": "Starting evaluation.", "current": 0, "total": 50, "started_at_utc": started, "updated_at_utc": started})

    def progress(update: dict[str, Any]) -> None:
        _write_status(status_path, {"status": "running", "started_at_utc": started, "updated_at_utc": _now(), **update})

    try:
        results = run_evaluation(args.collection_name, progress_callback=progress)
        _write_status(status_path, {"status": "complete", "phase": "complete", "message": "Evaluation complete.", "current": results.get("question_count"), "total": results.get("question_count"), "started_at_utc": started, "finished_at_utc": _now()})
        return 0
    except Exception as exc:
        _write_status(status_path, {"status": "error", "phase": "error", "message": "Evaluation failed.", "error": str(exc), "traceback": traceback.format_exc(limit=8), "finished_at_utc": _now()})
        return 1


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
