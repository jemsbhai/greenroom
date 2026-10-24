"""Expression suggestions from a dedicated face-expression model (D19).

What this stage does and deliberately does not do:

- Faces are detected with OpenCV's bundled Haar cascade (frontal faces, no
  learned identity features) on frames sampled at 1 fps, and each face region is
  passed in memory to an image classifier fine-tuned for facial expressions.
- Nothing per face is kept: no crop, no box, no embedding, no track across
  frames. The only output per window is an aggregated mixture (what share of
  the detected face readings fell in each expression class), the number of
  readings, and the model id.
- The suggestion is one of Positive, Neutral, Negative or Upset, with status
  ai_suggested_unverified. "Confident" is never produced: a face model cannot
  observe it. Staff accept, edit or discard the suggestion in the review app;
  it never enters any aggregate on its own.
- Known limits (system card): expression classifiers are trained on posed adult
  faces, perform unevenly across skin tones, ages and lighting, and an
  expression is not an emotion. Performers act. These are reasons the output is
  a suggestion to a human who knows the context, not a claim.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .windows import frame_indices_for_window

# Expression-model class names (lowercased) -> the brief's generalized words.
POSITIVE = {"happy", "happiness", "surprise", "surprised", "joy"}
NEGATIVE = {"sad", "sadness", "angry", "anger", "fear", "fearful", "disgust", "disgusted", "contempt"}
UPSET = {"sad", "sadness", "angry", "anger"}
NEUTRAL = {"neutral", "calm"}


@dataclass
class WindowExpression:
    window_id: str
    readings: int
    frames_with_faces: int
    frames_checked: int
    mixture: dict[str, float] = field(default_factory=dict)  # positive / neutral / negative shares
    upset_share: float = 0.0

    def suggestion(self, model_id: str, min_readings: int = 2) -> dict | None:
        if self.readings < min_readings or not self.mixture:
            return None
        label, share = max(self.mixture.items(), key=lambda kv: kv[1])
        word = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}[label]
        if word == "Negative" and self.upset_share >= 0.5 * share:
            word = "Upset"
        return {
            "label": word,
            "share": round(share, 3),
            "mixture": {k: round(v, 3) for k, v in self.mixture.items()},
            "upset_share": round(self.upset_share, 3),
            "face_readings": self.readings,
            "frames_with_faces": self.frames_with_faces,
            "frames_checked": self.frames_checked,
            "status": "ai_suggested_unverified",
            "source": "face_expression_model",
            "model_id": model_id,
            "note": "Generalized expression read from faces by a model; not a statement about how anyone feels.",
        }

    def to_dict(self) -> dict:
        return {
            "window_id": self.window_id,
            "readings": self.readings,
            "frames_with_faces": self.frames_with_faces,
            "frames_checked": self.frames_checked,
            "mixture": {k: round(v, 4) for k, v in self.mixture.items()},
            "upset_share": round(self.upset_share, 4),
        }


def bucket_probs(class_probs: dict[str, float]) -> tuple[dict[str, float], float]:
    """Collapse an expression model's classes into positive/neutral/negative shares plus an upset share."""
    pos = sum(p for c, p in class_probs.items() if c in POSITIVE)
    neg = sum(p for c, p in class_probs.items() if c in NEGATIVE)
    neu = sum(p for c, p in class_probs.items() if c in NEUTRAL)
    upset = sum(p for c, p in class_probs.items() if c in UPSET)
    total = pos + neg + neu
    if total <= 0:
        return {}, 0.0
    return {"positive": pos / total, "neutral": neu / total, "negative": neg / total}, upset / total


class ExpressionTagger:
    def __init__(self, model_id: str, fallback_id: str | None = None, device: str | None = None, min_face_px: int = 40):
        import cv2
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(str(cascade_path))
        if self.detector.empty():
            raise RuntimeError(f"Could not load the Haar cascade at {cascade_path}")
        last_error: Exception | None = None
        for candidate in [model_id] + ([fallback_id] if fallback_id else []):
            try:
                self.processor = AutoImageProcessor.from_pretrained(candidate)
                self.model = AutoModelForImageClassification.from_pretrained(candidate).eval().to(self.device)
                self.model_id = candidate
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        else:
            raise RuntimeError(f"Could not load an expression classifier: {last_error}")
        self.id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.min_face_px = int(min_face_px)

    def _faces(self, image) -> list:
        """Face regions of one PIL image, returned as in-memory PIL crops with a margin. Never stored."""
        import cv2
        from PIL import Image

        arr = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        boxes = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(self.min_face_px, self.min_face_px))
        crops = []
        h, w = gray.shape[:2]
        for x, y, bw, bh in boxes:
            m = int(0.2 * max(bw, bh))
            x0, y0 = max(0, x - m), max(0, y - m)
            x1, y1 = min(w, x + bw + m), min(h, y + bh + m)
            crops.append(Image.fromarray(arr[y0:y1, x0:x1]))
        return crops

    def _classify(self, crops: list) -> np.ndarray:
        import torch

        inputs = self.processor(images=crops, return_tensors="pt")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits
            probs = logits.softmax(dim=-1)
        return probs.float().cpu().numpy()

    def score_windows(
        self,
        frames_dir: str | Path,
        sample_fps: float,
        windows: list[dict],
        max_frames_per_window: int = 5,
        batch_faces: int = 32,
    ) -> tuple[list[WindowExpression], dict]:
        from PIL import Image

        t0 = time.perf_counter()
        frames_dir = Path(frames_dir)
        results: list[WindowExpression] = []
        total_readings = 0
        for window in windows:
            indices = frame_indices_for_window(window["start_s"], window["end_s"], sample_fps, max_frames_per_window)
            crops: list = []
            frames_with_faces = 0
            frames_checked = 0
            for k in indices:
                path = frames_dir / f"{k:06d}.jpg"
                if not path.is_file():
                    continue
                frames_checked += 1
                with Image.open(path) as im:
                    found = self._faces(im)
                if found:
                    frames_with_faces += 1
                    crops.extend(found)
            mixture_sum = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
            upset_sum = 0.0
            readings = 0
            for start in range(0, len(crops), batch_faces):
                probs = self._classify(crops[start : start + batch_faces])
                for row in probs:
                    class_probs = {self.id2label[i]: float(p) for i, p in enumerate(row.tolist())}
                    mixture, upset = bucket_probs(class_probs)
                    if not mixture:
                        continue
                    for key, value in mixture.items():
                        mixture_sum[key] += value
                    upset_sum += upset
                    readings += 1
            crops.clear()  # nothing per face survives this loop
            if readings:
                mixture = {k: v / readings for k, v in mixture_sum.items()}
                results.append(WindowExpression(window["window_id"], readings, frames_with_faces, frames_checked, mixture, upset_sum / readings))
            else:
                results.append(WindowExpression(window["window_id"], 0, frames_with_faces, frames_checked))
            total_readings += readings
        meta = {
            "model_id": self.model_id,
            "detector": "opencv_haar_frontalface_default",
            "device": self.device,
            "sample_fps": sample_fps,
            "max_frames_per_window": max_frames_per_window,
            "face_readings_total": total_readings,
            "windows_with_suggestion": sum(1 for r in results if r.readings >= 2),
            "wall_seconds": round(time.perf_counter() - t0, 3),
        }
        return results, meta
