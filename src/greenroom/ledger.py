"""Append-only, hash-chained ledger of staff review decisions.

Every decision is one JSON line in <data>/records/<video_id>/reviews.jsonl.
Each line carries the SHA-256 of the previous line's hash plus its own
content, so any edit, deletion or reordering of past decisions breaks the
chain and `greenroom verify` reports exactly where. Nothing is ever deleted:
a changed decision is a new line, and the latest line per window wins.

pollard (Phase 3) seals these chains into its execution tree; the file format
stays the same so records written now remain valid.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .taxonomy import GUIDELINE_VERSION

GENESIS = "0" * 64
DECISIONS = ("accepted", "edited", "rejected", "deferred")


def reviews_path(video_id: str) -> Path:
    folder = config.records_dir() / video_id
    folder.mkdir(parents=True, exist_ok=True)
    return folder / "reviews.jsonl"


def _canonical(entry: dict) -> str:
    return json.dumps({k: v for k, v in entry.items() if k != "hash"}, sort_keys=True, ensure_ascii=False)


def _digest(entry: dict) -> str:
    return hashlib.sha256(_canonical(entry).encode("utf-8")).hexdigest()


def read_chain(video_id: str) -> list[dict]:
    path = reviews_path(video_id)
    if not path.is_file():
        return []
    entries: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def append_decision(
    video_id: str,
    window: dict,
    reviewer: str,
    decision: str,
    ai_record: dict,
    final_primary: str,
    final_secondary: list[str],
    facilitation_flag: bool | None,
    staff_contextual_observation: str | None,
    expression_label: str | None,
    note: str,
    seconds_spent: float | None,
) -> dict:
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}, got {decision!r}")
    chain = read_chain(video_id)
    prev_hash = chain[-1]["hash"] if chain else GENESIS
    entry = {
        "seq": len(chain),
        "prev_hash": prev_hash,
        "video_id": video_id,
        "window_id": window["window_id"],
        "start_s": window["start_s"],
        "end_s": window["end_s"],
        "reviewer": reviewer.strip(),
        "decision": decision,
        "ai_primary": ai_record.get("primary_activity"),
        "ai_secondary": ai_record.get("secondary_activities", []),
        "ai_review_status": ai_record.get("review_status"),
        "ai_confidence": ai_record.get("confidence"),
        "final_primary": final_primary,
        "final_secondary": sorted(set(final_secondary) - {final_primary}),
        "facilitation_flag": facilitation_flag,
        "staff_contextual_observation": (staff_contextual_observation or "").strip() or None,
        "expression_label": expression_label or None,
        "note": note.strip(),
        "seconds_spent": None if seconds_spent is None else round(float(seconds_spent), 1),
        "guideline_version": GUIDELINE_VERSION,
        "pipeline_git_sha": (ai_record.get("provenance") or {}).get("git_sha"),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    entry["hash"] = _digest(entry)
    with reviews_path(video_id).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def latest_decisions(video_id: str) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for entry in read_chain(video_id):
        latest[entry["window_id"]] = entry
    return latest


def verify_chain(video_id: str) -> dict:
    """Recompute every hash and link; return a report rather than raising."""
    chain = read_chain(video_id)
    problems: list[str] = []
    prev = GENESIS
    for i, entry in enumerate(chain):
        if entry.get("seq") != i:
            problems.append(f"line {i}: seq {entry.get('seq')} out of order")
        if entry.get("prev_hash") != prev:
            problems.append(f"line {i}: prev_hash does not match line {i - 1}")
        if _digest(entry) != entry.get("hash"):
            problems.append(f"line {i}: content does not match its hash")
        prev = entry.get("hash", "")
    return {"video_id": video_id, "entries": len(chain), "ok": not problems, "problems": problems}


def verify_all() -> list[dict]:
    reports = []
    for folder in sorted(config.records_dir().iterdir()):
        if folder.is_dir() and (folder / "reviews.jsonl").is_file():
            reports.append(verify_chain(folder.name))
    return reports
