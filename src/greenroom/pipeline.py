"""Pipeline orchestration: windows -> evidence channels -> opinions -> fusion -> records.

Channels in this build: audio (CLAP), xclip (zero-shot video-text) and
videomae (Kinetics-400 supervised, mapped). Each channel yields, per window,
{label_id: score} plus an input-quality factor, and becomes its own opinion in
the fusion. Later channels (transcript cues) plug in with the same shape.
Stages run one after another and the GPU is released between them.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from . import config
from . import taxonomy as T
from .energy import StageMeter
from .fuse import Thresholds, decide, fuse_labels
from .media import extract_audio, sample_frames
from .opinions import Opinion, from_score, relative_support, vacuous
from .records import build_record, provenance, write_outputs
from .windows import load_index, merge_runs, windows_for_video

STAGES = ("audio", "vision", "transcript")


def base_rates(profile: dict) -> dict[str, float]:
    fusion = profile.get("fusion") or {}
    default = float(fusion.get("base_rate_default", 0.1))
    per_label = fusion.get("base_rates") or {}
    return {lid: float(per_label.get(lid, default)) for lid in T.ALL_LABEL_IDS}


DEFAULT_CHANNEL = {"pos": 10.0, "neg": 10.0, "score_mode": "raw", "competitors": 2}


def channel_settings(profile: dict) -> dict[str, dict]:
    """Per-channel evidence settings from fusion.channels (a bare number means symmetric weights)."""
    fusion = profile.get("fusion") or {}
    raw = fusion.get("channels") or fusion.get("evidence_weight") or {}
    settings: dict[str, dict] = {}
    for name, value in raw.items():
        if isinstance(value, dict):
            merged = {**DEFAULT_CHANNEL, **value}
        else:
            merged = {**DEFAULT_CHANNEL, "pos": float(value), "neg": float(value)}
        merged["pos"] = float(merged["pos"])
        merged["neg"] = float(merged["neg"])
        merged["competitors"] = int(merged["competitors"])
        settings[name] = merged
    return settings


def scores_to_opinions(
    label_scores: dict[str, float],
    quality: float,
    setting: dict,
    rates: dict[str, float],
    prior_weight: float,
    top_score: float | None = None,
) -> dict[str, Opinion]:
    """Scored labels get evidence; labels the channel did not score stay vacuous.

    score_mode:
      raw           use the channel's score as the probability
      relative      chance-normalized support of each label's softmax share
      relative_top  the channel's leader gets its chance-normalized support;
                    every other label gets that times its share of the leader's
                    score. A flat channel then contributes little for any label,
                    and runners-up stay below the presence floor unless they are
                    close to the leader. top_score is the leader's raw score over
                    everything the channel scored (for audio this includes the
                    context classes such as speech and music).
    """
    mode = setting.get("score_mode", "raw")
    competitors = setting.get("competitors", 2)
    if top_score is None:
        top_score = max(label_scores.values()) if label_scores else 0.0
    peak = relative_support(top_score, competitors) if top_score > 0 else 0.0
    ops: dict[str, Opinion] = {}
    for lid in T.ALL_LABEL_IDS:
        if lid in label_scores:
            p = label_scores[lid]
            if mode == "relative":
                p = relative_support(p, competitors)
            elif mode == "relative_top":
                p = (p / top_score) * peak if top_score > 0 else 0.0
            ops[lid] = from_score(p, setting["pos"], quality, rates[lid], prior_weight, negative_weight=setting["neg"])
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


def _load_models(profile: dict, stages: tuple[str, ...], want_gpu: bool, log) -> dict:
    """Load every model the requested stages need, once per run."""
    models: dict = {}
    device = profile.get("device")
    if "audio" in stages:
        from .audio_events import AudioEventScorer

        cfg = profile.get("audio_events") or {}
        with StageMeter("load_audio_model", gpu=want_gpu) as m:
            models["audio"] = AudioEventScorer(
                cfg.get("clap_model", "laion/larger_clap_music_and_speech"), cfg.get("clap_fallback_model"), device=device
            )
        log(f"audio model {models['audio'].model_id} on {models['audio'].device} ({m.wall_seconds:.1f} s)")
        if cfg.get("ast_model"):
            from .audio_events import AudioSetScorer

            with StageMeter("load_ast", gpu=want_gpu) as m:
                models["ast"] = AudioSetScorer(cfg["ast_model"], device=device)
            log(
                f"ast {models['ast'].model_id}, {len(models['ast'].class_map)} AudioSet classes mapped "
                f"to taxonomy labels ({m.wall_seconds:.1f} s)"
            )
    if "vision" in stages:
        from .vision_tags import VideoMaeScorer, XClipScorer

        cfg = profile.get("vision") or {}
        if cfg.get("xclip_model"):
            with StageMeter("load_xclip", gpu=want_gpu) as m:
                models["xclip"] = XClipScorer(cfg["xclip_model"], device=device)
            log(f"xclip {models['xclip'].model_id}, {models['xclip'].num_frames} frames per window ({m.wall_seconds:.1f} s)")
        if cfg.get("videomae_model"):
            with StageMeter("load_videomae", gpu=want_gpu) as m:
                models["videomae"] = VideoMaeScorer(cfg["videomae_model"], device=device)
            mapped = models["videomae"].resolved_mapping()
            log(
                f"videomae {models['videomae'].model_id}, {models['videomae'].num_frames} frames per window, "
                f"{len(mapped)} Kinetics classes mapped to taxonomy labels ({m.wall_seconds:.1f} s)"
            )
    if "transcript" in stages:
        from .transcribe import Transcriber
        from .transcript_cues import OllamaCueModel

        asr = profile.get("asr") or {}
        with StageMeter("load_asr", gpu=want_gpu) as m:
            models["asr"] = Transcriber(
                asr.get("model", "large-v3"), asr.get("compute_type", "float16"), device="cuda" if device == "cuda" else device
            )
        log(f"asr {models['asr'].model_id} ({m.wall_seconds:.1f} s)")
        llm = profile.get("transcript_llm") or {}
        with StageMeter("load_cue_model", gpu=False) as m:
            models["cues"] = OllamaCueModel(
                llm.get("model", "qwen2.5:7b-instruct"),
                host=llm.get("host", "http://127.0.0.1:11434"),
                temperature=float(llm.get("temperature", 0.0)),
            )
        log(f"transcript cues via {models['cues'].model_id}, local only ({m.wall_seconds:.1f} s)")
    return models


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
    settings = channel_settings(profile)
    prior_weight = float((profile.get("fusion") or {}).get("prior_weight", 2.0))
    thresholds = Thresholds.from_profile(profile)
    want_gpu = bool((profile.get("energy") or {}).get("nvml", True))
    audio_cfg = profile.get("audio_events") or {}
    vision_cfg = profile.get("vision") or {}

    models = _load_models(profile, stages, want_gpu, log)
    summaries: list[dict] = []

    for video_id in video_ids:
        video = by_id[video_id]
        windows = windows_for_video(index, video_id)
        source = config.videos_dir() / video["source_file"]
        out_dir = config.records_dir() / video_id
        stage_meta: dict[str, dict] = {}
        model_ids: list[str] = []
        per_window: dict[str, dict[str, dict[str, Opinion]]] = {w["window_id"]: {} for w in windows}
        channel_outputs: dict[str, list[dict]] = {}
        extra_meta: dict = {}

        if "audio" in models:
            with StageMeter("extract_audio", gpu=False) as m_wav:
                wav = extract_audio(source, config.audio_dir() / f"{video_id}.wav")
            stage_meta["extract_audio"] = m_wav.to_dict()
            with StageMeter("audio_events", gpu=want_gpu) as m_audio:
                scores, meta = models["audio"].score_windows(
                    wav, windows, analysis_s=float(audio_cfg.get("analysis_window_seconds", 10.0))
                )
            stage_meta["audio_events"] = {**m_audio.to_dict(), **meta}
            model_ids.append(models["audio"].model_id)
            s_audio = settings.get("audio", DEFAULT_CHANNEL)
            for s in scores:
                per_window[s.window_id]["audio"] = scores_to_opinions(
                    s.label_scores, s.quality, s_audio, rates, prior_weight, top_score=max(s.class_probs.values(), default=0.0)
                )
            channel_outputs["audio"] = [s.to_dict() for s in scores]
            _release_gpu()

        if "ast" in models:
            with StageMeter("ast", gpu=want_gpu) as m_ast:
                scores, meta = models["ast"].score_windows(
                    wav, windows, analysis_s=float(audio_cfg.get("analysis_window_seconds", 10.0))
                )
            stage_meta["ast"] = {**m_ast.to_dict(), **meta}
            model_ids.append(models["ast"].model_id)
            s_ast = settings.get("ast", {**DEFAULT_CHANNEL, "neg": 1.5})
            for s in scores:
                per_window[s.window_id]["ast"] = scores_to_opinions(s.label_scores, s.quality, s_ast, rates, prior_weight)
            channel_outputs["ast"] = [s.to_dict() for s in scores]
            extra_meta["audioset_to_taxonomy"] = models["ast"].resolved_mapping()
            _release_gpu()

        if "xclip" in models or "videomae" in models:
            sample_fps = float(vision_cfg.get("sample_fps", 4.0))
            with StageMeter("sample_frames", gpu=False) as m_frames:
                frames_dir = config.frames_dir() / video_id
                frame_paths = sample_frames(source, frames_dir, fps=sample_fps, height=int(vision_cfg.get("frame_height", 256)))
            stage_meta["sample_frames"] = {**m_frames.to_dict(), "n_frames": len(frame_paths), "sample_fps": sample_fps}
            for channel in ("xclip", "videomae"):
                if channel not in models:
                    continue
                scorer = models[channel]
                with StageMeter(channel, gpu=want_gpu) as m_vis:
                    scores, meta = scorer.score_windows(
                        frames_dir,
                        sample_fps,
                        windows,
                        batch_size=int(vision_cfg.get("batch_size", 8)),
                        dark_luma=float(vision_cfg.get("dark_luminance", 12.0)),
                        bright_luma=float(vision_cfg.get("full_quality_luminance", 50.0)),
                    )
                stage_meta[channel] = {**m_vis.to_dict(), **meta}
                model_ids.append(scorer.model_id)
                s_ch = settings.get(channel, DEFAULT_CHANNEL)
                for s in scores:
                    per_window[s.window_id][channel] = scores_to_opinions(s.label_scores, s.quality, s_ch, rates, prior_weight)
                channel_outputs[channel] = [s.to_dict() for s in scores]
                if channel == "videomae":
                    extra_meta["kinetics_to_taxonomy"] = scorer.resolved_mapping()
                _release_gpu()

        cue_by_window: dict[str, dict] = {}
        if "asr" in models and "cues" in models:
            from .transcribe import load_or_transcribe
            from .transcript_cues import score_windows as score_cues

            asr = profile.get("asr") or {}
            llm = profile.get("transcript_llm") or {}
            wav = extract_audio(source, config.audio_dir() / f"{video_id}.wav")
            with StageMeter("transcribe", gpu=want_gpu) as m_asr:
                transcript, from_cache = load_or_transcribe(
                    models["asr"], video_id, wav, asr.get("language", "en"), bool(asr.get("vad_filter", True))
                )
            stage_meta["transcribe"] = {
                **m_asr.to_dict(),
                "model_id": models["asr"].model_id,
                "from_cache": from_cache,
                "segments": len(transcript.get("segments", [])),
                "words": sum(len(s.get("words", [])) for s in transcript.get("segments", [])),
            }
            model_ids.append(models["asr"].model_id)
            _release_gpu()
            with StageMeter("transcript_cues", gpu=want_gpu) as m_cue:
                cues, meta = score_cues(
                    models["cues"],
                    transcript,
                    windows,
                    context_pad_s=float(llm.get("context_pad_seconds", 20.0)),
                    no_speech_threshold=float(asr.get("no_speech_threshold", 0.6)),
                    log=log,
                )
            stage_meta["transcript_cues"] = {**m_cue.to_dict(), **meta}
            model_ids.append(models["cues"].model_id)
            s_tr = settings.get("transcript", {**DEFAULT_CHANNEL, "pos": 5.0, "neg": 2.0})
            for c in cues:
                if c.label_scores:
                    per_window[c.window_id]["transcript"] = scores_to_opinions(c.label_scores, c.quality, s_tr, rates, prior_weight)
                cue_by_window[c.window_id] = c.to_dict()
            channel_outputs["transcript"] = [c.to_dict() for c in cues]

        prov = provenance(profile, model_ids, list(stages))
        records: list[dict] = []
        with StageMeter("fusion_and_records", gpu=False) as m_fuse:
            for w in windows:
                mods = per_window[w["window_id"]]
                fused, conflicts = fuse_labels(mods, rates, method=fusion_method)
                decision = decide(fused, conflicts, thresholds)
                cue = cue_by_window.get(w["window_id"]) or {}
                facilitation = None
                if cue.get("facilitation") is not None and cue.get("facilitation_confidence", 0.0) >= 0.5:
                    facilitation = bool(cue["facilitation"])
                record = build_record(w, fused, conflicts, decision, mods, prov, facilitation_flag=facilitation)
                if cue:
                    record["speech"] = {
                        "word_count": cue.get("word_count", 0),
                        "speech_kind": cue.get("speech_kind", "unclear"),
                        "facilitation_confidence": cue.get("facilitation_confidence", 0.0),
                        "rationale": cue.get("rationale", ""),
                    }
                records.append(record)
            segments = merge_runs([r["primary_activity"] for r in records], windows)
        stage_meta["fusion_and_records"] = m_fuse.to_dict()

        duration_s = float(video["duration_s"])
        processing_s = sum(m["wall_seconds"] for m in stage_meta.values())
        joules = [m["gpu_joules"] for m in stage_meta.values() if m.get("gpu_joules") is not None]
        counts: dict[str, int] = {}
        statuses: dict[str, int] = {}
        for r in records:
            counts[r["primary_activity"]] = counts.get(r["primary_activity"], 0) + 1
            statuses[r["review_status"]] = statuses.get(r["review_status"], 0) + 1
        run_meta = {
            "video_id": video_id,
            "source_file": video["source_file"],
            "duration_s": duration_s,
            "n_windows": len(windows),
            "stages": list(stages),
            "channels": sorted({ch for mods in per_window.values() for ch in mods}),
            "fusion_method": fusion_method,
            "profile": profile.get("name"),
            "thresholds": thresholds.__dict__,
            "channel_settings": settings,
            "stage_meta": stage_meta,
            "processing_wall_seconds": round(processing_s, 3),
            "video_minutes_per_wall_minute": round(duration_s / processing_s, 3) if processing_s > 0 else None,
            "gpu_joules_total": round(sum(joules), 2) if joules else None,
            "gpu_joules_per_video_hour": round(sum(joules) / (duration_s / 3600.0), 1) if joules and duration_s else None,
            "primary_label_counts": counts,
            "review_status_counts": statuses,
            "n_segments": len(segments),
            "provenance": prov,
            **extra_meta,
            "channel_outputs": channel_outputs,
        }
        paths = write_outputs(out_dir, records, segments, run_meta)
        with (config.records_dir() / "runs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {k: v for k, v in run_meta.items() if k not in ("channel_outputs", "kinetics_to_taxonomy", "audioset_to_taxonomy")}
                    | {"written_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                )
                + "\n"
            )
        log(
            f"{video_id}: {len(windows)} windows, {len(segments)} segments, "
            f"{processing_s:.1f} s processing for {duration_s / 60:.1f} min of video "
            f"({run_meta['video_minutes_per_wall_minute']} video-min per wall-min)"
        )
        log(f"  channels:       {run_meta['channels']}")
        log(f"  primary labels: {counts}")
        log(f"  review status:  {statuses}")
        log(f"  outputs: {paths['records_jsonld'].parent}")
        summaries.append({k: v for k, v in run_meta.items() if k not in ("channel_outputs",)})
    return summaries
