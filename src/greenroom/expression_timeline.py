"""Read-only, timestamped expression contract for review and integration.

AI suggestions and explicit staff observations are separate fields. Reviewing
an activity never turns a model expression into a staff observation. This module
uses only the standard library and does not load models or alter source records.
"""
from __future__ import annotations

import csv
import io
import math
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP

SCHEMA_VERSION = "greenroom.expression-timeline.v1"
AI_STATUS = "ai_suggested_unverified"
CSV_COLUMNS = [
    "schema_version", "video_id", "source_file", "time_unit", "interval_semantics",
    "window_id", "start_s", "end_s", "start_time", "end_time",
    "ai_label", "ai_share", "ai_positive_share", "ai_neutral_share", "ai_negative_share",
    "ai_upset_share", "ai_face_readings", "ai_frames_with_faces", "ai_frames_checked",
    "ai_model_id", "ai_status", "staff_label", "staff_reviewer", "staff_reviewed_at",
    "staff_ledger_hash", "review_decision",
]


def format_timestamp(seconds: float) -> str:
    """Format elapsed video time to milliseconds, including durations over 24h."""
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Timestamps must be finite, nonnegative seconds")
    milliseconds = int((Decimal(str(seconds)) * 1000).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    hours, remaining = divmod(milliseconds, 3_600_000)
    minutes, remaining = divmod(remaining, 60_000)
    whole_seconds, fraction = divmod(remaining, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{fraction:03d}"


def build_expression_timeline(
    video_id: str,
    records: list[dict],
    decisions: dict | None = None,
    source_file: str | None = None,
) -> dict:
    """Build all recorded windows, ordered by source-video time.

    ``decisions`` is the window-id mapping returned by ``latest_decisions``.
    Only that latest entry's explicit ``expression_label`` is a staff label;
    clearing a later label does not resurrect an earlier observation. A null AI
    field means no suggestion was available, not a Neutral classification.
    Numeric seconds remain authoritative; display timestamps round to 1 ms.
    """
    decisions = decisions or {}
    windows = []
    seen = set()
    for record in records:
        window_id = record["window_id"]
        if window_id in seen:
            raise ValueError(f"Duplicate window_id: {window_id}")
        if record.get("video_id", video_id) != video_id:
            raise ValueError(f"Window {window_id} belongs to a different video")
        seen.add(window_id)
        start, end = float(record["start_s"]), float(record["end_s"])
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError(f"Window {window_id} needs finite timestamps with 0 <= start_s < end_s")
        suggestion = deepcopy(record.get("expression_suggestion"))
        if suggestion is not None:
            # This contract never represents a model's output as human-verified.
            suggestion["status"] = AI_STATUS
        latest = decisions.get(window_id) or {}
        staff = None
        if latest.get("expression_label"):
            staff = {
                "label": latest["expression_label"],
                "reviewer": latest.get("reviewer"),
                "reviewed_at": latest.get("timestamp"),
                "ledger_hash": latest.get("hash"),
            }
        windows.append({
            "window_id": window_id,
            "start_s": start,
            "end_s": end,
            "start_time": format_timestamp(start),
            "end_time": format_timestamp(end),
            "ai_expression": suggestion,
            "staff_expression": staff,
            "review_decision": latest.get("decision") or "pending",
        })
    windows.sort(key=lambda window: (window["start_s"], window["end_s"], window["window_id"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "video_id": video_id,
        "source_file": source_file,
        "time_unit": "seconds",
        "interval_semantics": "start_inclusive_end_exclusive",
        "windows": windows,
    }


def expression_timeline_csv(document: dict) -> str:
    """Flatten a timeline to UTF-8-ready CSV; null fields become empty cells."""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for window in document["windows"]:
        ai = window.get("ai_expression") or {}
        mixture = ai.get("mixture") or {}
        staff = window.get("staff_expression") or {}
        row = {key: document.get(key) for key in CSV_COLUMNS[:5]}
        row.update({key: window.get(key) for key in ("window_id", "start_s", "end_s", "start_time", "end_time")})
        row.update({
            "ai_label": ai.get("label"),
            "ai_share": ai.get("share"),
            "ai_positive_share": mixture.get("positive"),
            "ai_neutral_share": mixture.get("neutral"),
            "ai_negative_share": mixture.get("negative"),
            "ai_upset_share": ai.get("upset_share"),
            "ai_face_readings": ai.get("face_readings"),
            "ai_frames_with_faces": ai.get("frames_with_faces"),
            "ai_frames_checked": ai.get("frames_checked"),
            "ai_model_id": ai.get("model_id"),
            "ai_status": ai.get("status"),
            "staff_label": staff.get("label"),
            "staff_reviewer": staff.get("reviewer"),
            "staff_reviewed_at": staff.get("reviewed_at"),
            "staff_ledger_hash": staff.get("ledger_hash"),
            "review_decision": window["review_decision"],
        })
        writer.writerow(row)
    return stream.getvalue()
