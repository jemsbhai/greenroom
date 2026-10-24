"""Append-only store of gold labels.

One JSONL file per annotator under <data>/gold/. Each save appends a record;
the latest record per window wins. Nothing is ever deleted, so the labeling
history is preserved for the write-up. Gold labels are participant-level
annotations and stay under the data directory, never in the repository.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .config import gold_dir
from .taxonomy import GUIDELINE_VERSION


def annotator_slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    if not slug:
        raise ValueError("annotator name is empty")
    return slug


def gold_path(annotator: str) -> Path:
    return gold_dir() / f"{annotator_slug(annotator)}.jsonl"


def make_record(
    window: dict,
    annotator: str,
    primary: str,
    secondary: list[str],
    facilitation: bool,
    uncertain: bool,
    note: str = "",
    context_used: bool = False,
    seconds_spent: float | None = None,
) -> dict:
    return {
        "window_id": window["window_id"],
        "video_id": window["video_id"],
        "start_s": window["start_s"],
        "end_s": window["end_s"],
        "annotator": annotator_slug(annotator),
        "primary": primary,
        "secondary": sorted(set(secondary) - {primary}),
        "facilitation": bool(facilitation),
        "uncertain": bool(uncertain),
        "note": note.strip(),
        "context_used": bool(context_used),
        "seconds_spent": None if seconds_spent is None else round(float(seconds_spent), 1),
        "guideline_version": GUIDELINE_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def append_record(record: dict) -> Path:
    path = gold_path(record["annotator"])
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def load_records(annotator: str) -> list[dict]:
    path = gold_path(annotator)
    if not path.is_file():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_latest(annotator: str) -> dict[str, dict]:
    """Latest record per window_id for one annotator."""
    latest: dict[str, dict] = {}
    for record in load_records(annotator):
        latest[record["window_id"]] = record
    return latest


def load_all_latest() -> dict[str, dict[str, dict]]:
    """{annotator_slug: {window_id: record}} across every gold file."""
    result: dict[str, dict[str, dict]] = {}
    for path in sorted(gold_dir().glob("*.jsonl")):
        result[path.stem] = load_latest(path.stem)
    return result


def progress(latest: dict[str, dict], windows: list[dict]) -> tuple[int, int]:
    done = sum(1 for w in windows if w["window_id"] in latest)
    return done, len(windows)
