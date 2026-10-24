"""Audio evidence channel: zero-shot audio-event scoring with CLAP.

For every 5 s window we score a 10 s analysis segment centered on the window
against text prompts grouped into audio classes, map the classes to taxonomy
labels, and attach an input-quality factor q (how much of the segment carries
signal above the file's noise floor). Labels that audio cannot observe
(painting, fashion design, ...) get no score here and end up vacuous in the
fusion, which is the intended behavior: audio says nothing about them.

The only audio artifact read is the mono 16 kHz WAV written by media.extract_audio;
it is resampled to 48 kHz in memory for CLAP. Nothing identity-related is
computed: no speaker embeddings, no diarization.
"""
from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CLAP_SAMPLE_RATE = 48_000
RMS_FRAME_S = 0.1

# class_id -> prompts (scored jointly under one softmax) and the taxonomy labels
# the class supports. Classes with an empty label list are context only.
AUDIO_CLASSES: dict[str, dict] = {
    "singing": {
        "prompts": [
            "a person singing",
            "a singer performing a song with a melody",
            "someone singing into a microphone",
        ],
        "labels": ["singing_vocal_performance"],
    },
    "rap": {
        "prompts": [
            "rap vocals over a beat",
            "a person rapping",
            "hip hop vocals with rhythmic rhymes",
        ],
        "labels": ["lyricism_spoken_word_rap"],
    },
    "spoken_word": {
        "prompts": [
            "spoken word poetry performed into a microphone",
            "a person reciting a poem to an audience",
        ],
        "labels": ["lyricism_spoken_word_rap"],
    },
    "speech": {
        "prompts": [
            "people talking in a room",
            "a conversation between people",
            "a person explaining something to a group",
        ],
        "labels": [],
    },
    "applause": {
        "prompts": [
            "applause from an audience",
            "a crowd clapping and cheering",
        ],
        "labels": ["audience_applause_transition"],
    },
    "music": {
        "prompts": [
            "instrumental music without vocals",
            "a beat playing from a speaker",
        ],
        "labels": [],
    },
    "silence": {
        "prompts": [
            "silence or quiet room noise",
            "background noise with no activity",
        ],
        "labels": ["setup_idle_empty"],
    },
}

AUDIO_OBSERVABLE_LABELS: list[str] = sorted({lbl for spec in AUDIO_CLASSES.values() for lbl in spec["labels"]})

# AudioSet class-name patterns -> taxonomy labels, for the second audio channel
# (AST). Resolved against the model's own id2label at load time and written to
# run_meta. Speech, music and instrument classes stay context only: a song bed
# under a montage must not count as recording arts. The parent class "Singing"
# is also context only: in the AudioSet ontology "Rapping" is its child, so the
# parent fires on rap and would erase the rap label; only the specific singing
# classes count as singing.
AUDIOSET_RULES: list[tuple[str, tuple[str, ...]]] = [
    (
        r"\b(male|female|child|synthetic) singing\b|\bchoir\b|\bchant\b|\bhumming\b|yodeling|a capella|\bopera\b",
        ("singing_vocal_performance",),
    ),
    (r"\brapping\b|beatboxing", ("lyricism_spoken_word_rap",)),
    (r"applause|\bclapping\b|cheering|\bcrowd\b|hubbub|speech noise", ("audience_applause_transition",)),
    (r"^silence$", ("setup_idle_empty",)),
]


def map_audioset_classes(id2label: dict[int, str]) -> dict[int, tuple[str, ...]]:
    import re

    mapping: dict[int, tuple[str, ...]] = {}
    for idx, name in id2label.items():
        lowered = name.lower()
        labels: list[str] = []
        for pattern, targets in AUDIOSET_RULES:
            if re.search(pattern, lowered):
                labels.extend(t for t in targets if t not in labels)
        if labels:
            mapping[int(idx)] = tuple(labels)
    return mapping


@dataclass
class WindowAudioScore:
    window_id: str
    quality: float
    class_probs: dict[str, float]
    label_scores: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "window_id": self.window_id,
            "quality": round(self.quality, 4),
            "class_probs": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.class_probs.items()},
            "label_scores": {k: round(v, 4) for k, v in self.label_scores.items()},
        }


def _frame_rms(y: np.ndarray, sr: int, frame_s: float = RMS_FRAME_S) -> np.ndarray:
    n = max(1, int(sr * frame_s))
    usable = (len(y) // n) * n
    if usable == 0:
        return np.zeros(0, dtype=np.float32)
    frames = y[:usable].reshape(-1, n)
    return np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1))


def signal_quality(segment: np.ndarray, sr: int, floor: float) -> float:
    """Fraction of 100 ms frames whose RMS exceeds the noise floor."""
    rms = _frame_rms(segment, sr)
    if rms.size == 0:
        return 0.0
    return float(np.mean(rms > floor))


def noise_floor(y: np.ndarray, sr: int) -> float:
    """A file-relative floor: a tenth of the 95th percentile frame RMS, at least 1e-3."""
    rms = _frame_rms(y, sr)
    if rms.size == 0:
        return 1e-3
    return float(max(1e-3, 0.1 * np.percentile(rms, 95)))


class AudioEventScorer:
    """CLAP zero-shot scorer. Loads lazily; falls back to a second checkpoint if the first fails."""

    def __init__(self, model_id: str, fallback_id: str | None = None, device: str | None = None):
        import torch
        from transformers import ClapModel, ClapProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        last_error: Exception | None = None
        for candidate in [model_id] + ([fallback_id] if fallback_id else []):
            try:
                self.processor = ClapProcessor.from_pretrained(candidate)
                self.model = ClapModel.from_pretrained(candidate).eval().to(self.device)
                self.model_id = candidate
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
        else:
            raise RuntimeError(f"Could not load a CLAP checkpoint: {last_error}")
        self.prompts: list[str] = [p for spec in AUDIO_CLASSES.values() for p in spec["prompts"]]
        self.prompt_class: list[str] = [cid for cid, spec in AUDIO_CLASSES.items() for _ in spec["prompts"]]
        # transformers renamed the processor keyword from `audios` to `audio`; pick whichever this version has.
        try:
            params = inspect.signature(self.processor.__call__).parameters
        except (TypeError, ValueError):
            params = {}
        self._audio_kw = "audios" if ("audios" in params and "audio" not in params) else "audio"

    def _batch_probs(self, segments: list[np.ndarray]) -> np.ndarray:
        import torch

        kwargs = dict(text=self.prompts, return_tensors="pt", padding=True, sampling_rate=CLAP_SAMPLE_RATE)
        inputs = self.processor(**{self._audio_kw: segments}, **kwargs)
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.model(**inputs)
            probs = out.logits_per_audio.softmax(dim=-1)
        return probs.float().cpu().numpy()

    def score_windows(
        self,
        wav_path: str | Path,
        windows: list[dict],
        analysis_s: float = 10.0,
        batch_size: int = 16,
        min_segment_s: float = 1.0,
    ) -> tuple[list[WindowAudioScore], dict]:
        import librosa

        t0 = time.perf_counter()
        y, sr = librosa.load(str(wav_path), sr=CLAP_SAMPLE_RATE, mono=True)
        floor = noise_floor(y, sr)
        results: list[WindowAudioScore] = []
        pending: list[tuple[dict, np.ndarray, float]] = []

        def flush() -> None:
            if not pending:
                return
            probs = self._batch_probs([seg for _, seg, _ in pending])
            for (window, _, quality), row in zip(pending, probs):
                class_probs = {cid: 0.0 for cid in AUDIO_CLASSES}
                for cid, p in zip(self.prompt_class, row.tolist()):
                    class_probs[cid] += p
                # Classes are mutually exclusive under the softmax, so the mass of every
                # class that supports a label adds up (rap + spoken word both mean lyricism).
                label_scores: dict[str, float] = {}
                for cid, spec in AUDIO_CLASSES.items():
                    for lbl in spec["labels"]:
                        label_scores[lbl] = min(1.0, label_scores.get(lbl, 0.0) + class_probs[cid])
                results.append(WindowAudioScore(window["window_id"], quality, class_probs, label_scores))
            pending.clear()

        for window in windows:
            center = 0.5 * (window["start_s"] + window["end_s"])
            a = max(0.0, center - analysis_s / 2.0)
            b = min(len(y) / sr, a + analysis_s)
            seg = y[int(a * sr) : int(b * sr)]
            if len(seg) < int(min_segment_s * sr):
                results.append(WindowAudioScore(window["window_id"], 0.0, {cid: 0.0 for cid in AUDIO_CLASSES}, {}))
                continue
            quality = signal_quality(seg, sr, floor)
            pending.append((window, seg, quality))
            if len(pending) >= batch_size:
                flush()
        flush()

        order = {w["window_id"]: i for i, w in enumerate(windows)}
        results.sort(key=lambda r: order[r.window_id])
        meta = {
            "model_id": self.model_id,
            "device": self.device,
            "analysis_seconds": analysis_s,
            "noise_floor_rms": round(floor, 6),
            "audio_seconds": round(len(y) / sr, 3),
            "wall_seconds": round(time.perf_counter() - t0, 3),
        }
        return results, meta


AST_SAMPLE_RATE = 16_000


class AudioSetScorer:
    """Second audio channel: AST fine-tuned on AudioSet (multi-label, sigmoid over 527 classes).

    Scores are the sigmoid probabilities of the mapped classes, summed per
    taxonomy label and clipped to 1. Unmapped classes (speech, music,
    instruments) are context and never become evidence. Reads the 16 kHz WAV
    directly; nothing identity-related is computed.
    """

    def __init__(self, model_id: str, device: str | None = None):
        import torch
        from transformers import ASTForAudioClassification, AutoFeatureExtractor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = ASTForAudioClassification.from_pretrained(model_id).eval().to(self.device)
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.class_map = map_audioset_classes(self.id2label)

    def resolved_mapping(self) -> dict[str, list[str]]:
        return {self.id2label[idx]: list(labels) for idx, labels in sorted(self.class_map.items())}

    def _batch_probs(self, segments: list[np.ndarray]) -> np.ndarray:
        import torch

        inputs = self.extractor(segments, sampling_rate=AST_SAMPLE_RATE, return_tensors="pt")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits
            probs = torch.sigmoid(logits)
        return probs.float().cpu().numpy()

    def score_windows(
        self,
        wav_path: str | Path,
        windows: list[dict],
        analysis_s: float = 10.0,
        batch_size: int = 16,
        min_segment_s: float = 1.0,
    ) -> tuple[list[WindowAudioScore], dict]:
        import soundfile as sf

        t0 = time.perf_counter()
        y, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if getattr(y, "ndim", 1) > 1:
            y = y.mean(axis=1)
        if sr != AST_SAMPLE_RATE:
            import librosa

            y = librosa.resample(y, orig_sr=sr, target_sr=AST_SAMPLE_RATE)
            sr = AST_SAMPLE_RATE
        floor = noise_floor(y, sr)
        results: list[WindowAudioScore] = []
        pending: list[tuple[dict, np.ndarray, float]] = []

        def flush() -> None:
            if not pending:
                return
            probs = self._batch_probs([seg for _, seg, _ in pending])
            for (window, _, quality), row in zip(pending, probs):
                label_scores: dict[str, float] = {}
                for idx, labels in self.class_map.items():
                    p = float(row[idx])
                    for lbl in labels:
                        label_scores[lbl] = min(1.0, label_scores.get(lbl, 0.0) + p)
                top = int(np.argmax(row))
                class_probs = {"top_audioset_class": self.id2label[top], "top_audioset_prob": round(float(row[top]), 4)}
                results.append(WindowAudioScore(window["window_id"], quality, class_probs, label_scores))
            pending.clear()

        for window in windows:
            center = 0.5 * (window["start_s"] + window["end_s"])
            a = max(0.0, center - analysis_s / 2.0)
            b = min(len(y) / sr, a + analysis_s)
            seg = y[int(a * sr) : int(b * sr)]
            if len(seg) < int(min_segment_s * sr):
                results.append(WindowAudioScore(window["window_id"], 0.0, {}, {}))
                continue
            pending.append((window, seg, signal_quality(seg, sr, floor)))
            if len(pending) >= batch_size:
                flush()
        flush()
        order = {w["window_id"]: i for i, w in enumerate(windows)}
        results.sort(key=lambda r: order[r.window_id])
        meta = {
            "model_id": self.model_id,
            "device": self.device,
            "analysis_seconds": analysis_s,
            "noise_floor_rms": round(floor, 6),
            "mapped_classes": len(self.class_map),
            "wall_seconds": round(time.perf_counter() - t0, 3),
        }
        return results, meta
