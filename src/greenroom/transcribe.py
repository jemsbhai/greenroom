"""Local speech transcription with faster-whisper, cached per video.

Transcripts are the participants' own words and the most sensitive derived
artifact in the system. They are written only under <data>/derived/transcripts/,
are never copied into records or run metadata (those carry word counts and
cues, not text), and are never sent to any external model. No speaker
diarization and no speaker embeddings are computed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import config


def transcript_path(video_id: str) -> Path:
    folder = config.derived_dir() / "transcripts"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{video_id}.json"


class Transcriber:
    def __init__(self, model_size: str = "large-v3", compute_type: str = "float16", device: str | None = None,
                 cpu_threads: int | None = None):
        from faster_whisper import WhisperModel

        self.model_size = model_size
        self.device = device or "cuda"
        self.compute_type = compute_type
        options = {"cpu_threads": max(1, int(cpu_threads))} if cpu_threads is not None else {}
        self.model = WhisperModel(model_size, device=self.device, compute_type=compute_type, **options)

    @property
    def model_id(self) -> str:
        return f"faster-whisper:{self.model_size}:{self.compute_type}"

    def transcribe(self, wav_path: str | Path, language: str | None = "en", vad_filter: bool = True) -> dict:
        t0 = time.perf_counter()
        vad_used = vad_filter
        try:
            segments, info = self.model.transcribe(
                str(wav_path),
                language=language,
                vad_filter=vad_filter,
                word_timestamps=True,
                beam_size=5,
                condition_on_previous_text=False,
            )
            segments = list(segments)
        except RuntimeError as exc:
            if not vad_filter or "onnxruntime" not in str(exc):
                raise
            # The VAD model needs onnxruntime; without it, fall back to Whisper's own
            # no-speech probabilities (windows.py filters segments on them downstream).
            vad_used = False
            segments, info = self.model.transcribe(
                str(wav_path),
                language=language,
                vad_filter=False,
                word_timestamps=True,
                beam_size=5,
                condition_on_previous_text=False,
            )
            segments = list(segments)
        out_segments = []
        for seg in segments:
            words = [
                {"w": w.word.strip(), "s": round(float(w.start), 3), "e": round(float(w.end), 3), "p": round(float(w.probability), 3)}
                for w in (seg.words or [])
                if w.word and w.word.strip()
            ]
            out_segments.append(
                {
                    "start": round(float(seg.start), 3),
                    "end": round(float(seg.end), 3),
                    "text": seg.text.strip(),
                    "no_speech_prob": round(float(seg.no_speech_prob), 4),
                    "avg_logprob": round(float(seg.avg_logprob), 4),
                    "words": words,
                }
            )
        return {
            "model_id": self.model_id,
            "vad_filter": vad_used,
            "language": getattr(info, "language", language),
            "language_probability": round(float(getattr(info, "language_probability", 0.0) or 0.0), 3),
            "duration_s": round(float(getattr(info, "duration", 0.0) or 0.0), 3),
            "segments": out_segments,
            "wall_seconds": round(time.perf_counter() - t0, 3),
        }


def load_or_transcribe(transcriber: Transcriber, video_id: str, wav_path: Path, language: str | None, vad_filter: bool) -> tuple[dict, bool]:
    """Return (transcript, from_cache). The cache is keyed by video id and ASR model id."""
    path = transcript_path(video_id)
    if path.is_file():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("model_id") == transcriber.model_id:
            return cached, True
    transcript = transcriber.transcribe(wav_path, language=language, vad_filter=vad_filter)
    path.write_text(json.dumps(transcript, ensure_ascii=False, indent=1), encoding="utf-8")
    return transcript, False


def words_in(
    transcript: dict,
    start_s: float,
    end_s: float,
    no_speech_threshold: float = 0.6,
    logprob_threshold: float = -1.0,
) -> list[dict]:
    """Words whose midpoints fall inside [start_s, end_s).

    A segment is skipped only when Whisper's no-speech probability is high AND
    its decoding log-probability is poor (Whisper's own rule). Sung or rapped
    lyrics often carry a high no-speech probability while being decoded
    confidently; those stay, which is what the cue model needs to tell a
    performance from a conversation.
    """
    out: list[dict] = []
    for seg in transcript.get("segments", []):
        if seg.get("no_speech_prob", 0.0) > no_speech_threshold and seg.get("avg_logprob", -9.0) < logprob_threshold:
            continue
        if seg["end"] < start_s or seg["start"] >= end_s:
            continue
        for w in seg.get("words", []):
            mid = 0.5 * (w["s"] + w["e"])
            if start_s <= mid < end_s:
                out.append(w)
    return out


def text_of(words: list[dict]) -> str:
    return " ".join(w["w"] for w in words).strip()
