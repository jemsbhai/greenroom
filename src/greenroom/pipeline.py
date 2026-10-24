"""Pipeline orchestration: windows -> evidence channels -> opinions -> fusion -> records.

Phase 1 ships the audio channel. Each later channel (transcript, vision,
transcript cues) plugs in as another entry in `by_modality` with the same
shape: {label_id: Opinion} per window. Stages run one after another and the
GPU is released between them.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from . import config
from . import taxonomy as T
from .audio_events import AudioEventScorer
from .energy import StageMeter
from .fuse import Thresholds, decide, fuse_labels
from .media import extract_audio
from .opinions import Opinion, from_score, vacuous
from .records import build_record, provenance, write_outputs
from .windows import load_index, merge_runs, windows_for_video

STAGES = ("audio",)


def base_rates(profile: dict) -> dict[str, float]:
    fusion = profile.get("fusion") or {}
    default = float(fusion.get("base_rate_default", 0.1))
    per_label = fusion.get("base_rates") or {}
    return {lid: float(per_label.get(lid, default)) for lid in T.ALL_LABEL_IDS}


def evidence_weights(profile: dict) -> dict[str, float]:
    return {k: float(v) for k, v in ((profile.get("fusion") or {}).get("evidence_weight") or {}).items()}


def audio_opinions(score, weight: float, rates: dict[str, float], prior_weight: float) -> dict[str, Opinion]:
    """Opinions from one window's audio scores: scored labels get evidence, the rest stay vacuous."""
    ops: dict[str, Opinion] = {}
    for lid in T.ALL_LABEL_IDS:
        if lid in score.label_scores:
            ops[lid] = from_score(score.label_scores[lid], weight, score.quality, rates[lid], prior_weight)
        else:
            ops[lid] = vacuous(rates[lid])
    return ops


def _release_gpu() -> None:
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def run_videos(
    video_ids: list[str],
    profile: dict,
    stages: tuple[str, ...] = STAGES,
    fusion_method: str = "cumulative",
    index: dict | None = None,
    log=print,
) -> list[dict]:
    index = index or load_index(config.windows_index_path())
    by_id = {v["video_id"]: v for v in index["videos"]}
    unknown = [v for v in video_ids if v not in by_id]
    if unknown:
        raise ValueError(f"unknown video id(s): {unknown}; known: {sorted(by_id)}")

    rates = base_rates(profile)
    weights = evidence_weights(profile)
    prior_weight = float((profile.get("fusion") or {}).get("prior_weight", 2.0))
    thresholds = Thresholds.from_profile(profile)
    want_gpu = bool((profile.get("energy") or {}).get("nvml", True))
    audio_cfg = profile.get("audio_events") or {}

    scorer: AudioEventScorer | None = None
    if "audio" in stages:
        with StageMeter("load_audio_model", gpu=want_gpu) as m_load:
            scorer = AudioEventScorer(
                audio_cfg.get("clap_model", "laion/larger_clap_music_and_speech"),
                audio_cfg.get("clap_fallback_model"),
                device=profile.get("device"),
            )
        log(f"audio model {scorer.model_id} on {scorer.device} ({m_load.wall_seconds:.1f} s)")

    summaries: list[dict] = []
    for video_id in video_ids:
        video = by_id[video_id]
        windows = windows_for_video(index, video_id)
        source = config.videos_dir() / video["source_file"]
        out_dir = config.records_dir() / video_id
        stage_meta: dict[str, dict] = {}
        model_ids: list[str] = []
        by_modality_per_window: dict[str, dict[str, dict[str, Opinion]]] = {w["window_id"]: {} for w in windows}
        channel_outputs: dict[str, list[dict]] = {}

        if "audio" in stages and scorer is not None:
            with StageMeter("extract_audio", gpu=False) as m_wav:
                wav = extract_audio(source, config.audio_dir() / f"{video_id}.wav")
            stage_meta["extract_audio"] = m_wav.to_dict()
            with StageMeter("audio_events", gpu=want_gpu) as m_audio:
                scores, audio_meta = scorer.score_windows(
                    wav, windows, analysis_s=float(audio_cfg.get("analysis_window_seconds", 10.0))
                )
            stage_meta["audio_events"] = {**m_audio.to_dict(), **audio_meta}
            model_ids.append(scorer.model_id)
            weight = weights.get("audio", 10.0)
            for score in scores:
                by_modality_per_window[score.window_id]["audio"] = audio_opinions(score, weight, rates, prior_weight)
            channel_outputs["audio"] = [s.to_dict() for s in scores]
            _release_gpu()

        prov = provenance(profile, model_ids, list(stages))
        records: list[dict] = []
        with StageMeter("fusion_and_records", gpu=False) as m_fuse:
            for w in windows:
                mods = by_modality_per_window[w["window_id"]]
                fused, conflicts = fuse_labels(mods, rates, method=fusion_method)
                decision = decide(fused, conflicts, thresholds)
                records.append(build_record(w, fused, conflicts, decision, mods, prov))
            segments = merge_runs([r["primary_activity"] for r in records], windows)
        stage_meta["fusion_and_records"] = m_fuse.to_dict()

        duration_s = float(video["duration_s"])
        processing_s = sum(m["wall_seconds"] for m in stage_meta.values())
        joules = [m["gpu_joules"] for m in stage_meta.values() if m.get("gpu_joules") is not None]
        counts: dict[str, int] = {}
        for r in records:
            counts[r["primary_activity"]] = counts.get(r["primary_activity"], 0) + 1
        statuses: dict[str, int] = {}
        for r in records:
            statuses[r["review_status"]] = statuses.get(r["review_status"], 0) + 1
        run_meta = {
            "video_id": video_id,
            "source_file": video["source_file"],
            "duration_s": duration_s,
            "n_windows": len(windows),
            "stages": list(stages),
            "fusion_method": fusion_method,
            "profile": profile.get("name"),
            "thresholds": thresholds.__dict__,
            "evidence_weights": weights,
            "stage_meta": stage_meta,
            "processing_wall_seconds": round(processing_s, 3),
            "video_minutes_per_wall_minute": round(duration_s / processing_s, 3) if processing_s > 0 else None,
            "gpu_joules_total": round(sum(joules), 2) if joules else None,
            "gpu_joules_per_video_hour": round(sum(joules) / (duration_s / 3600.0), 1) if joules and duration_s else None,
            "primary_label_counts": counts,
            "review_status_counts": statuses,
            "n_segments": len(segments),
            "provenance": prov,
            "channel_outputs": channel_outputs,
        }
        paths = write_outputs(out_dir, records, segments, run_meta)
        with (config.records_dir() / "runs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {k: v for k, v in run_meta.items() if k != "channel_outputs"}
                    | {"written_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                )
                + "\n"
            )
        log(
            f"{video_id}: {len(windows)} windows, {len(segments)} segments, "
            f"{processing_s:.1f} s processing for {duration_s / 60:.1f} min of video "
            f"({run_meta['video_minutes_per_wall_minute']} video-min per wall-min)"
        )
        log(f"  primary labels: {counts}")
        log(f"  review status:  {statuses}")
        log(f"  outputs: {paths['records_jsonld'].parent}")
        summaries.append({k: v for k, v in run_meta.items() if k != "channel_outputs"})
    return summaries
