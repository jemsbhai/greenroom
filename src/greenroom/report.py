"""Results report: everything measurable without hand labels, plus whatever labels exist.

Sources, each reported for what it is (PLAN.md 8.1):
  1. file-name weak labels for the single-activity clips (promos, songwriting)
  2. staff review decisions from the hash-chained ledgers
  3. inter-channel diagnostics (agreement, conflict, uncertainty, single-source share)
  4. throughput and GPU energy per video from the latest run
Writes <data>/records/report.md and report.json and prints the markdown.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from . import config
from . import ledger
from . import taxonomy as T
from .records import load_records
from .windows import load_index

# File-name patterns that name a clip's dominant activity. Songwriting sessions
# contain the song being sung or rapped, so those count as consistent too.
WEAK_LABELS: list[tuple[str, set[str]]] = [
    ("promo-rap", {"lyricism_spoken_word_rap"}),
    ("promo-dance", {"dance"}),
    ("promo-singing", {"singing_vocal_performance"}),
    ("promo-painting", {"visual_arts"}),
    ("songwriting", {"songwriting", "singing_vocal_performance", "lyricism_spoken_word_rap"}),
]


def _name(lid: str | None) -> str:
    return T.DISPLAY_NAME.get(lid, lid or "")


def _load_run_meta(video_id: str) -> dict:
    path = config.records_dir() / video_id / "run_meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _channel_primary(record: dict, channel: str) -> str | None:
    ops = (record.get("opinions_by_modality") or {}).get(channel) or {}
    if not ops:
        return None
    return max(ops, key=lambda lid: ops[lid]["P"])


def build_report() -> dict:
    index = load_index(config.windows_index_path())
    videos = [v for v in index["videos"] if (config.records_dir() / v["video_id"] / "records.jsonld").is_file()]
    report: dict = {"videos": [], "overall": {}}
    all_records: list[tuple[dict, dict]] = []
    status_total: Counter = Counter()
    label_total: Counter = Counter()
    decisions_all: list[dict] = []
    stage_seconds: Counter = Counter()
    total_duration = 0.0
    total_processing = 0.0
    total_joules = 0.0
    joules_measured = True

    for video in videos:
        vid = video["video_id"]
        records = load_records(config.records_dir() / vid)
        meta = _load_run_meta(vid)
        decided = ledger.latest_decisions(vid)
        win_s = video["duration_s"] / max(1, video["n_windows"])
        labels = Counter(r["primary_activity"] for r in records)
        statuses = Counter(r["review_status"] for r in records)
        status_total.update(statuses)
        label_total.update(labels)
        total_duration += video["duration_s"]
        total_processing += float(meta.get("processing_wall_seconds") or 0.0)
        if meta.get("gpu_joules_total") is None:
            joules_measured = False
        else:
            total_joules += float(meta["gpu_joules_total"])
        for stage, sm in (meta.get("stage_meta") or {}).items():
            stage_seconds[stage] += float(sm.get("wall_seconds") or 0.0)

        weak = None
        for pattern, expected in WEAK_LABELS:
            if pattern in vid:
                dominant = labels.most_common(1)[0][0] if labels else None
                primary_hits = sum(1 for r in records if r["primary_activity"] in expected)
                any_hits = sum(
                    1 for r in records if r["primary_activity"] in expected or set(r["secondary_activities"]) & expected
                )
                weak = {
                    "expected": sorted(expected),
                    "dominant_primary": dominant,
                    "dominant_agrees": dominant in expected,
                    "windows_primary_in_expected": f"{primary_hits}/{len(records)}",
                    "windows_any_label_in_expected": f"{any_hits}/{len(records)}",
                }
                break

        reviewed = [decided[r["window_id"]] for r in records if r["window_id"] in decided]
        decisions_all.extend(reviewed)
        for r in records:
            all_records.append((r, decided.get(r["window_id"]) or {}))

        report["videos"].append(
            {
                "video_id": vid,
                "duration_min": round(video["duration_s"] / 60.0, 2),
                "windows": len(records),
                "segments": meta.get("n_segments"),
                "channels": meta.get("channels"),
                "profile": meta.get("profile"),
                "processing_seconds": meta.get("processing_wall_seconds"),
                "video_min_per_wall_min": meta.get("video_minutes_per_wall_minute"),
                "gpu_joules_per_video_hour": meta.get("gpu_joules_per_video_hour"),
                "label_minutes": {lid: round(c * win_s / 60.0, 2) for lid, c in labels.most_common()},
                "status_counts": dict(statuses),
                "weak_label_check": weak,
                "reviewed_windows": len(reviewed),
            }
        )

    # Review-based metrics
    judged = [d for d in decisions_all if d["decision"] in ("accepted", "edited", "rejected")]
    agree = [d for d in judged if d["decision"] == "accepted" or (d["decision"] == "edited" and d["final_primary"] == d["ai_primary"])]
    per_label: dict[str, Counter] = defaultdict(Counter)
    for d in judged:
        per_label[d["ai_primary"]]["judged"] += 1
        if d in agree:
            per_label[d["ai_primary"]]["agreed"] += 1
    spent = [d["seconds_spent"] for d in decisions_all if d.get("seconds_spent") is not None]
    by_status: dict[str, Counter] = defaultdict(Counter)
    status_of = {r["window_id"]: r["review_status"] for r, _ in all_records}
    for d in judged:
        s = status_of.get(d["window_id"], "unknown")
        by_status[s]["judged"] += 1
        if d in agree:
            by_status[s]["agreed"] += 1
    review = {
        "decisions": len(decisions_all),
        "decision_counts": dict(Counter(d["decision"] for d in decisions_all)),
        "judged": len(judged),
        "primary_agreed": len(agree),
        "primary_agreement_rate": round(len(agree) / len(judged), 3) if judged else None,
        "agreement_by_ai_status": {
            s: {"judged": c["judged"], "agreed": c["agreed"], "rate": round(c["agreed"] / c["judged"], 3) if c["judged"] else None}
            for s, c in by_status.items()
        },
        "agreement_by_label": {
            lid: {"judged": c["judged"], "agreed": c["agreed"], "rate": round(c["agreed"] / c["judged"], 3) if c["judged"] else None}
            for lid, c in sorted(per_label.items())
        },
        "median_seconds_per_decision": round(statistics.median(spent), 1) if spent else None,
        "reviewers": sorted({d["reviewer"] for d in decisions_all}),
    }

    # Inter-channel diagnostics
    channels = sorted({ch for r, _ in all_records for ch in (r.get("opinions_by_modality") or {})})
    chan_agree: Counter = Counter()
    chan_scored: Counter = Counter()
    pair_agree: Counter = Counter()
    pair_both: Counter = Counter()
    support_counts: Counter = Counter()
    uncertainties: list[float] = []
    conflicts: list[float] = []
    for r, _ in all_records:
        fused = r["primary_activity"]
        uncertainties.append(r["opinion_primary"]["u"])
        conflicts.append(r["conflict_score"])
        prim = {ch: _channel_primary(r, ch) for ch in channels}
        support = sum(1 for ch in channels if fused in ((r.get("opinions_by_modality") or {}).get(ch) or {}))
        support_counts[support] += 1
        for ch in channels:
            if prim[ch] is not None:
                chan_scored[ch] += 1
                if prim[ch] == fused:
                    chan_agree[ch] += 1
        for i, a in enumerate(channels):
            for b in channels[i + 1 :]:
                if prim[a] is not None and prim[b] is not None:
                    pair_both[(a, b)] += 1
                    if prim[a] == prim[b]:
                        pair_agree[(a, b)] += 1
    diagnostics = {
        "channels": channels,
        "channel_primary_matches_fused": {
            ch: {"scored_windows": chan_scored[ch], "rate": round(chan_agree[ch] / chan_scored[ch], 3) if chan_scored[ch] else None}
            for ch in channels
        },
        "pairwise_channel_agreement": {
            f"{a}+{b}": {"windows": pair_both[(a, b)], "rate": round(pair_agree[(a, b)] / pair_both[(a, b)], 3) if pair_both[(a, b)] else None}
            for (a, b) in sorted(pair_both)
        },
        "windows_by_channels_supporting_primary": dict(sorted(support_counts.items())),
        "mean_primary_uncertainty": round(statistics.mean(uncertainties), 3) if uncertainties else None,
        "mean_conflict": round(statistics.mean(conflicts), 3) if conflicts else None,
        "share_conflict_at_least_0_4": round(sum(1 for c in conflicts if c >= 0.4) / len(conflicts), 3) if conflicts else None,
    }

    n = sum(status_total.values())
    report["overall"] = {
        "videos": len(videos),
        "duration_min": round(total_duration / 60.0, 1),
        "windows": n,
        "status_counts": dict(status_total),
        "status_shares": {s: round(c / n, 3) for s, c in status_total.items()} if n else {},
        "label_minutes": {lid: round(c * (total_duration / n) / 60.0, 1) for lid, c in label_total.most_common()} if n else {},
        "processing_seconds_total": round(total_processing, 1),
        "video_min_per_wall_min": round((total_duration / 60.0) / (total_processing / 60.0), 2) if total_processing else None,
        "gpu_joules_per_video_hour": round(total_joules / (total_duration / 3600.0), 0) if joules_measured and total_duration else None,
        "stage_seconds": {k: round(v, 1) for k, v in stage_seconds.most_common()},
        "weak_label_checks": [v["weak_label_check"] | {"video_id": v["video_id"]} for v in report["videos"] if v["weak_label_check"]],
        "review": review,
        "diagnostics": diagnostics,
    }
    return report


def to_markdown(report: dict) -> str:
    o = report["overall"]
    lines = ["# greenroom results", ""]
    lines.append(f"Videos: {o['videos']}, {o['duration_min']} minutes, {o['windows']} windows of 5 s.")
    lines.append(
        f"Processing: {o['processing_seconds_total']} s total, {o['video_min_per_wall_min']} video-minutes per wall-minute"
        + (f", {o['gpu_joules_per_video_hour']:.0f} J per video hour (NVML)." if o.get("gpu_joules_per_video_hour") else ".")
    )
    lines.append("")
    lines.append("## Triage (review status, all windows)")
    for s, c in o["status_counts"].items():
        lines.append(f"- {s}: {c} ({100 * o['status_shares'][s]:.0f}%)")
    lines.append("")
    lines.append("## Minutes per primary label (AI, before review)")
    for lid, m in o["label_minutes"].items():
        lines.append(f"- {_name(lid)}: {m} min")
    lines.append("")
    lines.append("## File-name weak labels (single-activity clips)")
    for w in o["weak_label_checks"]:
        lines.append(
            f"- {w['video_id']}: expected {', '.join(_name(x) for x in w['expected'])}; dominant {_name(w['dominant_primary'])} "
            f"({'agrees' if w['dominant_agrees'] else 'differs'}); primary in expected {w['windows_primary_in_expected']}; "
            f"any label in expected {w['windows_any_label_in_expected']}"
        )
    lines.append("")
    r = o["review"]
    lines.append("## Staff review decisions")
    if r["decisions"]:
        lines.append(f"- Decisions: {r['decisions']} ({r['decision_counts']}), reviewers: {', '.join(r['reviewers'])}")
        lines.append(f"- Primary label agreed in {r['primary_agreed']} of {r['judged']} judged windows ({r['primary_agreement_rate']})")
        for s, c in r["agreement_by_ai_status"].items():
            lines.append(f"  - by AI status {s}: {c['agreed']}/{c['judged']} ({c['rate']})")
        for lid, c in r["agreement_by_label"].items():
            lines.append(f"  - {_name(lid)}: {c['agreed']}/{c['judged']} ({c['rate']})")
        lines.append(f"- Median seconds per decision: {r['median_seconds_per_decision']}")
    else:
        lines.append("- No decisions recorded yet.")
    lines.append("")
    d = o["diagnostics"]
    lines.append("## Inter-channel diagnostics (not accuracy)")
    for ch, c in d["channel_primary_matches_fused"].items():
        lines.append(f"- {ch} top label matches fused primary: {c['rate']} over {c['scored_windows']} windows")
    for pair, c in d["pairwise_channel_agreement"].items():
        lines.append(f"- {pair} agree on top label: {c['rate']} over {c['windows']} windows")
    lines.append(f"- Windows by number of channels supporting the fused primary: {d['windows_by_channels_supporting_primary']}")
    lines.append(
        f"- Mean uncertainty of primary: {d['mean_primary_uncertainty']}; mean conflict: {d['mean_conflict']}; "
        f"share with conflict >= 0.4: {d['share_conflict_at_least_0_4']}"
    )
    lines.append("")
    lines.append("## Per video")
    for v in report["videos"]:
        lines.append(
            f"- {v['video_id']}: {v['duration_min']} min, {v['windows']} windows, {v['segments']} segments, "
            f"{v['video_min_per_wall_min']} video-min/wall-min, status {v['status_counts']}, reviewed {v['reviewed_windows']}"
        )
    lines.append("")
    lines.append(f"Stage seconds across all videos: {o['stage_seconds']}")
    return "\n".join(lines)


def write_report() -> tuple[Path, Path, str]:
    report = build_report()
    md = to_markdown(report)
    out_md = config.records_dir() / "report.md"
    out_json = config.records_dir() / "report.json"
    out_md.write_text(md, encoding="utf-8")
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out_md, out_json, md
