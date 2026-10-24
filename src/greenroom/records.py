"""Annotation records in the schema the brief asks for, plus provenance.

One record per window. Field order follows the brief, then our additions
(opinions, conflict, review status, provenance). Written as JSON-LD (a
context plus a graph of records), as a flat CSV, and as merged segments.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from . import taxonomy as T
from .config import repo_root
from .fuse import Decision
from .opinions import Opinion

NAMESPACE = "https://github.com/jemsbhai/greenroom/ns#"

CONTEXT = {
    "@vocab": NAMESPACE,
    "sl": "https://jemsbhai.github.io/jsonld-ex/sl#",
    "video_id": "@id",
    "opinion_primary": "sl:opinion",
    "opinions_by_label": "sl:opinionsByLabel",
    "opinions_by_modality": "sl:opinionsByModality",
    "start_s": {"@type": "http://www.w3.org/2001/XMLSchema#decimal"},
    "end_s": {"@type": "http://www.w3.org/2001/XMLSchema#decimal"},
    "review_date": {"@type": "http://www.w3.org/2001/XMLSchema#dateTime"},
}

CSV_COLUMNS = [
    "video_id",
    "window_id",
    "start_s",
    "end_s",
    "primary_activity",
    "secondary_activities",
    "facilitation_flag",
    "observable_indicators",
    "staff_contextual_observation",
    "expression_suggestion",
    "confidence",
    "belief",
    "disbelief",
    "uncertainty",
    "conflict_score",
    "review_status",
    "origin",
    "reviewer_decision",
    "guideline_version",
    "reviewer_note",
    "review_date",
    "profile",
    "git_sha",
]


def git_sha(root: Path | None = None) -> str | None:
    """Read the current commit from .git without the git CLI."""
    root = root or repo_root()
    head = root / ".git" / "HEAD"
    if not head.is_file():
        return None
    content = head.read_text(encoding="utf-8").strip()
    if content.startswith("ref:"):
        ref = root / ".git" / content.split(" ", 1)[1].strip()
        if ref.is_file():
            return ref.read_text(encoding="utf-8").strip()
        packed = root / ".git" / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1] == content.split(" ", 1)[1].strip():
                    return parts[0]
        return None
    return content


def provenance(profile: dict, model_ids: list[str], stages: list[str]) -> dict:
    return {
        "pipeline_version": __version__,
        "git_sha": git_sha(),
        "profile": profile.get("name"),
        "stages": list(stages),
        "model_ids": list(model_ids),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def build_record(
    window: dict,
    fused: dict[str, Opinion],
    conflicts: dict[str, float],
    decision: Decision,
    by_modality: dict[str, dict[str, Opinion]],
    prov: dict,
    facilitation_flag: bool | None = None,
    observable_indicators: str = "",
    expression_suggestion: dict | None = None,
) -> dict:
    primary = decision.primary
    return {
        # fields named in the brief, in its order
        "video_id": window["video_id"],
        "window_id": window["window_id"],
        "start_s": window["start_s"],
        "end_s": window["end_s"],
        "primary_activity": primary,
        "secondary_activities": decision.secondary,
        "facilitation_flag": facilitation_flag,
        "observable_indicators": observable_indicators,
        "staff_contextual_observation": None,
        "expression_suggestion": expression_suggestion,
        "confidence": decision.confidence,
        "origin": "ai_generated",
        "reviewer_decision": "pending",
        "guideline_version": T.GUIDELINE_VERSION,
        "reviewer_note": None,
        "review_date": None,
        # additions
        "opinion_primary": fused[primary].to_dict(),
        "opinions_by_label": {lid: fused[lid].to_dict() for lid in T.ALL_LABEL_IDS},
        "opinions_by_modality": {
            mod: {lid: op.to_dict() for lid, op in ops.items() if not op.is_vacuous} for mod, ops in by_modality.items()
        },
        "conflict_score": round(conflicts.get(primary, 0.0), 6),
        "review_status": decision.review_status,
        "review_reasons": decision.reasons,
        "candidate": decision.candidate,
        "margin": decision.margin,
        "provenance": prov,
    }


def flatten(record: dict) -> dict:
    op = record["opinion_primary"]
    expr = record.get("expression_suggestion") or {}
    return {
        "video_id": record["video_id"],
        "window_id": record["window_id"],
        "start_s": record["start_s"],
        "end_s": record["end_s"],
        "primary_activity": record["primary_activity"],
        "secondary_activities": "|".join(record["secondary_activities"]),
        "facilitation_flag": record["facilitation_flag"],
        "observable_indicators": record["observable_indicators"],
        "staff_contextual_observation": record["staff_contextual_observation"],
        "expression_suggestion": expr.get("label", ""),
        "confidence": record["confidence"],
        "belief": op["b"],
        "disbelief": op["d"],
        "uncertainty": op["u"],
        "conflict_score": record["conflict_score"],
        "review_status": record["review_status"],
        "origin": record["origin"],
        "reviewer_decision": record["reviewer_decision"],
        "guideline_version": record["guideline_version"],
        "reviewer_note": record["reviewer_note"],
        "review_date": record["review_date"],
        "profile": record["provenance"].get("profile"),
        "git_sha": record["provenance"].get("git_sha"),
    }


def write_outputs(out_dir: Path, records: list[dict], segments: list[dict], run_meta: dict) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "records_jsonld": out_dir / "records.jsonld",
        "records_csv": out_dir / "records.csv",
        "segments_json": out_dir / "segments.json",
        "run_meta_json": out_dir / "run_meta.json",
    }
    paths["records_jsonld"].write_text(
        json.dumps({"@context": CONTEXT, "@graph": records}, indent=2), encoding="utf-8"
    )
    with paths["records_csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(flatten(record))
    paths["segments_json"].write_text(json.dumps(segments, indent=2), encoding="utf-8")
    paths["run_meta_json"].write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    return paths


def load_records(out_dir: Path) -> list[dict]:
    path = out_dir / "records.jsonld"
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("@graph", [])
