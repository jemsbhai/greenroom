"""Vision evidence channels (D18): X-CLIP zero-shot and VideoMAE Kinetics-400.

Both read the low-resolution frames sampled by media.sample_frames for the
whole video and pick, per 5 s window, exactly the number of frames the model
was trained with (read from the model config). Each channel returns its own
per-window label scores so it becomes its own opinion in the fusion.

- X-CLIP scores the window's frames against a prompt ensemble per taxonomy
  label; the softmax over all prompts is summed within each label.
- VideoMAE (fine-tuned on Kinetics-400) gives probabilities over its own 400
  classes; a fixed keyword mapping sends mass to the taxonomy labels those
  classes correspond to; labels without a mapped class stay vacuous.

Input quality q is a luminance check for now: near-black windows carry no
visual evidence. Person counting (a detector) replaces it in a later phase.
No crops, no faces, no identity; frames are read, scored and dropped.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import taxonomy as T
from .windows import frame_indices_for_window

XCLIP_PROMPTS: dict[str, list[str]] = {
    "singing_vocal_performance": [
        "a person singing into a microphone",
        "a singer performing a song on stage",
        "teenagers singing together in a room",
    ],
    "songwriting": [
        "a person writing song lyrics in a notebook",
        "a songwriter working out a song at a keyboard",
        "young people writing lyrics and trying out a melody",
    ],
    "lyricism_spoken_word_rap": [
        "a person rapping into a microphone",
        "a spoken word poet performing on stage",
        "a rapper performing over a beat",
    ],
    "dance": [
        "people dancing in a studio",
        "a group dance performance on stage",
        "a dancer rehearsing choreography",
    ],
    "acting_theater": [
        "actors performing a scene on stage",
        "a theater rehearsal with dialogue",
        "a person delivering a dramatic monologue",
    ],
    "film_media_production": [
        "a person operating a video camera",
        "a film crew setting up a shot",
        "an interview being recorded on camera",
    ],
    "recording_arts_audio_production": [
        "a person recording vocals in a studio booth",
        "a producer working at a mixing desk wearing headphones",
        "making a beat on a laptop with studio monitors",
    ],
    "podcast_production": [
        "people recording a podcast at a table with microphones",
        "a podcast conversation with headphones and microphones",
    ],
    "visual_arts": [
        "a person painting on a canvas",
        "a person drawing in a sketchbook",
        "young people working on an art project with paint",
    ],
    "fashion_design": [
        "a person sewing fabric for a garment",
        "a fashion designer draping fabric on a dress form",
        "a person sketching clothing designs",
    ],
    "modeling_runway": [
        "a model walking down a runway",
        "a fashion show runway walk",
        "a person posing for a photo shoot",
    ],
    "group_performance_or_showcase": [
        "a group of performers on stage together",
        "an ensemble performance in front of an audience",
        "a showcase with many performers on stage",
    ],
    "instruction_facilitation": [
        "a teacher instructing a group of students",
        "an instructor giving feedback to a performer",
        "a workshop leader explaining something to a class",
    ],
    "audience_applause_transition": [
        "an audience clapping and cheering",
        "performers walking on and off a stage",
        "a host speaking between acts",
    ],
    "setup_idle_empty": [
        "an empty room with chairs",
        "people setting up equipment in a room",
        "people waiting around and chatting",
    ],
}

# Labels a camera cannot tell apart. A vision channel that sees a person at a
# microphone has no information to separate singing from rapping, so both
# members of a group receive the group's summed support and the audio channels
# break the tie. Applied to every vision channel's label scores.
VISUAL_EQUIVALENCE: list[tuple[str, ...]] = [
    ("singing_vocal_performance", "lyricism_spoken_word_rap"),
]


def apply_visual_equivalence(scores: dict[str, float]) -> dict[str, float]:
    out = dict(scores)
    for group in VISUAL_EQUIVALENCE:
        present = [lid for lid in group if lid in out]
        if len(present) >= 1:
            total = min(1.0, sum(out[lid] for lid in present))
            for lid in group:
                out[lid] = total
    return out


# Kinetics-400 class name patterns -> taxonomy labels. Applied to the model's
# own id2label names at load time; the resolved mapping is written to run_meta
# so it can be audited. Mass from several classes is summed per label.
KINETICS_RULES: list[tuple[str, tuple[str, ...]]] = [
    (r"danc|zumba|krumping|cheerleading", ("dance",)),
    (r"\bsinging\b", ("singing_vocal_performance",)),
    (r"recording music", ("recording_arts_audio_production",)),
    (r"applaud|clapping", ("audience_applause_transition",)),
    (r"drawing|painting|sketch", ("visual_arts",)),
    (r"\bwriting\b", ("songwriting", "visual_arts")),
    (
        r"playing (drums|guitar|piano|keyboard|bass guitar|ukulele|violin|saxophone|trumpet|cello|harmonica|"
        r"xylophone|accordion|flute|clarinet|trombone|organ|recorder|didgeridoo|bagpipes|harp)|beatboxing|\bdj\b|\bdjing\b",
        ("recording_arts_audio_production",),
    ),
    (r"\bsewing\b|\bknitting\b", ("fashion_design",)),
]


def map_kinetics_classes(id2label: dict[int, str]) -> dict[int, tuple[str, ...]]:
    """Return {class_index: taxonomy labels} for the classes any rule matches."""
    mapping: dict[int, tuple[str, ...]] = {}
    for idx, name in id2label.items():
        lowered = name.lower()
        labels: list[str] = []
        for pattern, targets in KINETICS_RULES:
            if re.search(pattern, lowered):
                labels.extend(t for t in targets if t not in labels)
        if labels:
            mapping[int(idx)] = tuple(labels)
    return mapping


@dataclass
class WindowVisionScore:
    window_id: str
    quality: float
    label_scores: dict[str, float]
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "window_id": self.window_id,
            "quality": round(self.quality, 4),
            "label_scores": {k: round(v, 4) for k, v in self.label_scores.items()},
            **({"extra": self.extra} if self.extra else {}),
        }


def _load_frames(frames_dir: Path, indices: list[int]):
    from PIL import Image

    frames = []
    for k in indices:
        path = frames_dir / f"{k:06d}.jpg"
        if path.is_file():
            with Image.open(path) as im:
                frames.append(im.convert("RGB").copy())
    return frames


def _pick_exactly(indices: list[int], n: int) -> list[int]:
    """Spread n picks over the available indices, repeating when there are too few."""
    if not indices or n <= 0:
        return []
    positions = np.linspace(0, len(indices) - 1, n)
    return [indices[int(round(p))] for p in positions]


def _pixel_values(processed) -> "object":
    """Pull the video tensor out of a processor result whatever this transformers version calls it."""
    for key in ("pixel_values", "pixel_values_videos"):
        value = processed.get(key) if hasattr(processed, "get") else getattr(processed, key, None)
        if value is not None:
            if value.dim() == 4:  # a single video (T, C, H, W): add the batch axis
                value = value.unsqueeze(0)
            return value
    raise RuntimeError(f"frame processor returned no pixel values; keys: {list(processed.keys())}")


def _frame_processor_of(processor, model_id: str):
    """The frame-handling component of a composite processor, or a standalone one."""
    for attr in ("image_processor", "video_processor"):
        component = getattr(processor, attr, None)
        if component is not None:
            return component
    from transformers import AutoImageProcessor

    return AutoImageProcessor.from_pretrained(model_id)


def luminance_quality(frames, dark: float, bright: float) -> float:
    if not frames:
        return 0.0
    means = []
    for im in frames:
        arr = np.asarray(im.convert("L").resize((32, 32)), dtype=np.float32)
        means.append(float(arr.mean()))
    luma = float(np.mean(means))
    if bright <= dark:
        return 1.0 if luma > dark else 0.0
    return float(min(1.0, max(0.0, (luma - dark) / (bright - dark))))


class _VisionScorerBase:
    channel: str = "vision"

    def __init__(self, device: str | None = None):
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_frames = 8
        self.model_id = ""

    def _probs(self, batch_frames: list[list]) -> np.ndarray:  # pragma: no cover - model specific
        raise NotImplementedError

    def _label_scores(self, row: np.ndarray) -> tuple[dict[str, float], dict]:  # pragma: no cover
        raise NotImplementedError

    def score_windows(
        self,
        frames_dir: str | Path,
        sample_fps: float,
        windows: list[dict],
        batch_size: int = 8,
        dark_luma: float = 12.0,
        bright_luma: float = 50.0,
    ) -> tuple[list[WindowVisionScore], dict]:
        t0 = time.perf_counter()
        frames_dir = Path(frames_dir)
        results: list[WindowVisionScore] = []
        pending: list[tuple[dict, list, float]] = []

        def flush() -> None:
            if not pending:
                return
            probs = self._probs([fr for _, fr, _ in pending])
            for (window, _, quality), row in zip(pending, probs):
                label_scores, extra = self._label_scores(row)
                results.append(WindowVisionScore(window["window_id"], quality, label_scores, extra))
            pending.clear()

        for window in windows:
            available = frame_indices_for_window(window["start_s"], window["end_s"], sample_fps, 10_000)
            picked = _pick_exactly(available, self.num_frames)
            frames = _load_frames(frames_dir, picked)
            if len(frames) < self.num_frames:
                # fill by repeating the last frame; if there are no frames at all the window is vacuous
                if frames:
                    frames = frames + [frames[-1]] * (self.num_frames - len(frames))
                else:
                    results.append(WindowVisionScore(window["window_id"], 0.0, {}, {"note": "no frames"}))
                    continue
            quality = luminance_quality(frames, dark_luma, bright_luma)
            pending.append((window, frames, quality))
            if len(pending) >= batch_size:
                flush()
        flush()
        order = {w["window_id"]: i for i, w in enumerate(windows)}
        results.sort(key=lambda r: order[r.window_id])
        meta = {
            "model_id": self.model_id,
            "device": self.device,
            "num_frames_per_window": self.num_frames,
            "sample_fps": sample_fps,
            "wall_seconds": round(time.perf_counter() - t0, 3),
        }
        return results, meta


class XClipScorer(_VisionScorerBase):
    channel = "xclip"

    def __init__(self, model_id: str, device: str | None = None):
        super().__init__(device)
        from transformers import AutoProcessor, XCLIPModel

        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.frame_processor = _frame_processor_of(self.processor, model_id)
        self.model = XCLIPModel.from_pretrained(model_id).eval().to(self.device)
        self.num_frames = int(getattr(self.model.config.vision_config, "num_frames", 8))
        self.prompts: list[str] = [p for lid in T.ALL_LABEL_IDS for p in XCLIP_PROMPTS[lid]]
        self.prompt_label: list[str] = [lid for lid in T.ALL_LABEL_IDS for _ in XCLIP_PROMPTS[lid]]

    def _probs(self, batch_frames: list[list]) -> np.ndarray:
        import torch

        text = self.tokenizer(self.prompts, padding=True, return_tensors="pt")
        pixel_values = _pixel_values(self.frame_processor(batch_frames, return_tensors="pt"))
        with torch.inference_mode():
            out = self.model(
                input_ids=text["input_ids"].to(self.device),
                attention_mask=text["attention_mask"].to(self.device),
                pixel_values=pixel_values.to(self.device),
            )
            probs = out.logits_per_video.softmax(dim=-1)
        return probs.float().cpu().numpy()

    def _label_scores(self, row: np.ndarray) -> tuple[dict[str, float], dict]:
        scores = {lid: 0.0 for lid in T.ALL_LABEL_IDS}
        for lid, p in zip(self.prompt_label, row.tolist()):
            scores[lid] += p
        return apply_visual_equivalence(scores), {}


class VideoMaeScorer(_VisionScorerBase):
    channel = "videomae"

    def __init__(self, model_id: str, device: str | None = None):
        super().__init__(device)
        from transformers import AutoImageProcessor, VideoMAEForVideoClassification

        self.model_id = model_id
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = VideoMAEForVideoClassification.from_pretrained(model_id).eval().to(self.device)
        self.num_frames = int(getattr(self.model.config, "num_frames", 16))
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.class_map = map_kinetics_classes(self.id2label)

    def resolved_mapping(self) -> dict[str, list[str]]:
        return {self.id2label[idx]: list(labels) for idx, labels in sorted(self.class_map.items())}

    def _probs(self, batch_frames: list[list]) -> np.ndarray:
        import torch

        pixel_values = _pixel_values(self.processor(batch_frames, return_tensors="pt"))
        with torch.inference_mode():
            logits = self.model(pixel_values=pixel_values.to(self.device)).logits
            probs = logits.softmax(dim=-1)
        return probs.float().cpu().numpy()

    def _label_scores(self, row: np.ndarray) -> tuple[dict[str, float], dict]:
        scores: dict[str, float] = {}
        for idx, labels in self.class_map.items():
            p = float(row[idx])
            for lid in labels:
                scores[lid] = min(1.0, scores.get(lid, 0.0) + p)
        top = int(np.argmax(row))
        return apply_visual_equivalence(scores), {
            "top_kinetics_class": self.id2label[top],
            "top_kinetics_prob": round(float(row[top]), 4),
        }
