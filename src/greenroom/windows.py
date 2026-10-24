"""Fixed-length windows are the unit of labeling, scoring and evaluation (D12).

Every video is cut into consecutive non-overlapping windows of WINDOW_SECONDS.
Gold labels are collected per window, the pipeline scores per window, and
metrics are computed per window. Segments shown on the timeline are merged
runs of windows and are derived later; they are never the evaluation unit.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

from .config import VIDEO_EXTENSIONS

WINDOW_SECONDS = 5.0
# A trailing remainder shorter than this is merged into the previous window
# instead of becoming a window of its own.
MIN_TAIL_SECONDS = 2.0


def slugify(name: str) -> str:
    """Stable, filesystem- and URL-safe id derived from a file name."""
    stem = Path(name).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    return slug or "video"


@dataclass
class Window:
    window_id: str
    video_id: str
    source_file: str
    index: int
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def make_windows(
    duration_s: float,
    video_id: str,
    source_file: str,
    window_s: float = WINDOW_SECONDS,
    min_tail_s: float = MIN_TAIL_SECONDS,
) -> list[Window]:
    if duration_s <= 0 or window_s <= 0:
        return []
    count = int(duration_s // window_s)
    tail = duration_s - count * window_s
    windows: list[Window] = []
    for i in range(count):
        windows.append(
            Window(
                window_id=f"{video_id}:{i:05d}",
                video_id=video_id,
                source_file=source_file,
                index=i,
                start_s=round(i * window_s, 3),
                end_s=round((i + 1) * window_s, 3),
            )
        )
    if tail > 1e-6:
        if tail >= min_tail_s or not windows:
            windows.append(
                Window(
                    window_id=f"{video_id}:{count:05d}",
                    video_id=video_id,
                    source_file=source_file,
                    index=count,
                    start_s=round(count * window_s, 3),
                    end_s=round(duration_s, 3),
                )
            )
        else:
            windows[-1].end_s = round(duration_s, 3)
    return windows


def unique_slugs(files: Iterable[Path]) -> dict[str, str]:
    """Map source file name to a unique slug (collisions get a numeric suffix)."""
    mapping: dict[str, str] = {}
    seen: set[str] = set()
    for f in files:
        base = slugify(f.name)
        slug, k = base, 2
        while slug in seen:
            slug = f"{base}-{k}"
            k += 1
        seen.add(slug)
        mapping[f.name] = slug
    return mapping


def build_index(videos_dir: str | Path, probe_fn: Callable[[Path], dict], window_s: float = WINDOW_SECONDS) -> dict:
    """Probe every video and produce the window index (a plain dict, JSON-serializable)."""
    videos_dir = Path(videos_dir)
    files = sorted(p for p in videos_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    slugs = unique_slugs(files)
    videos: list[dict] = []
    windows: list[dict] = []
    for f in files:
        meta = probe_fn(f)
        ws = make_windows(float(meta["duration_s"]), slugs[f.name], f.name, window_s)
        videos.append(
            {
                "video_id": slugs[f.name],
                "source_file": f.name,
                "duration_s": round(float(meta["duration_s"]), 3),
                "fps": meta.get("fps"),
                "width": meta.get("width"),
                "height": meta.get("height"),
                "has_audio": meta.get("has_audio"),
                "n_windows": len(ws),
            }
        )
        windows.extend(asdict(w) for w in ws)
    return {"window_seconds": window_s, "videos": videos, "windows": windows}


def save_index(index: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return path


def load_index(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Window index not found at {path}. Run: python -m greenroom windows")
    return json.loads(path.read_text(encoding="utf-8"))


def windows_for_video(index: dict, video_id: str) -> list[dict]:
    return [w for w in index["windows"] if w["video_id"] == video_id]


def frame_indices_for_window(start_s: float, end_s: float, fps: float, n_frames: int) -> list[int]:
    """Indices (1-based, as written by media.sample_frames) of the frames whose
    centers fall inside [start_s, end_s); at most n_frames, evenly spread."""
    if fps <= 0 or n_frames <= 0:
        return []
    first = int(start_s * fps) + 1  # frame k covers [(k-1)/fps, k/fps)
    last = max(first, int(end_s * fps))
    available = list(range(first, last + 1))
    if len(available) <= n_frames:
        return available
    step = len(available) / n_frames
    return [available[int(i * step + step / 2)] for i in range(n_frames)]


def merge_runs(labels: list[str], windows: list[dict]) -> list[dict]:
    """Merge consecutive windows with the same label into segments for display."""
    segments: list[dict] = []
    for label, w in zip(labels, windows):
        if segments and segments[-1]["label"] == label and segments[-1]["video_id"] == w["video_id"]:
            segments[-1]["end_s"] = w["end_s"]
            segments[-1]["window_ids"].append(w["window_id"])
        else:
            segments.append(
                {
                    "video_id": w["video_id"],
                    "label": label,
                    "start_s": w["start_s"],
                    "end_s": w["end_s"],
                    "window_ids": [w["window_id"]],
                }
            )
    return segments
