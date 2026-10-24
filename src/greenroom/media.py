"""ffmpeg and ffprobe helpers.

The only derived media this module writes are a mono 16 kHz audio track,
low-resolution subclips for review, and low-resolution sampled frames. No
crops of any kind are produced anywhere in the pipeline.

ffmpeg is taken from PATH when present, otherwise from the binary bundled with
the imageio-ffmpeg package. ffprobe is optional: when it is absent, duration
and frame rate are parsed from the ffmpeg banner.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path


def ffmpeg_path() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg  # bundled static binary

    return imageio_ffmpeg.get_ffmpeg_exe()


def ffprobe_path() -> str | None:
    return shutil.which("ffprobe")


def _parse_rate(value: str | None) -> float | None:
    if not value:
        return None
    value = value.strip()
    if "/" in value:
        num, den = value.split("/", 1)
        try:
            den_f = float(den)
            return float(num) / den_f if den_f else None
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def probe(path: str | Path) -> dict:
    """Return {duration_s, fps, width, height, has_audio} for a media file."""
    path = Path(path)
    ffprobe = ffprobe_path()
    if ffprobe:
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,width,height,avg_frame_rate,r_frame_rate:format=duration",
            "-of",
            "json",
            str(path),
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        info = json.loads(out)
        streams = info.get("streams") or []
        video = next((s for s in streams if s.get("codec_type") == "video"), {})
        has_audio = any(s.get("codec_type") == "audio" for s in streams)
        fps = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(video.get("r_frame_rate"))
        duration = float(info.get("format", {}).get("duration") or 0.0)
        return {
            "duration_s": duration,
            "fps": fps,
            "width": video.get("width"),
            "height": video.get("height"),
            "has_audio": has_audio,
        }

    # Fallback: ffmpeg prints stream metadata on stderr and exits non-zero without an output.
    cmd = [ffmpeg_path(), "-hide_banner", "-i", str(path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    text = res.stderr or ""
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if not match:
        raise RuntimeError(f"Could not read duration of {path}: {text[-500:]}")
    duration = int(match.group(1)) * 3600 + int(match.group(2)) * 60 + float(match.group(3))
    fps_match = re.search(r"(\d+(?:\.\d+)?)\s*fps", text)
    size_match = re.search(r",\s*(\d{2,5})x(\d{2,5})[,\s\[]", text)
    return {
        "duration_s": duration,
        "fps": float(fps_match.group(1)) if fps_match else None,
        "width": int(size_match.group(1)) if size_match else None,
        "height": int(size_match.group(2)) if size_match else None,
        "has_audio": "Audio:" in text,
    }


def cut_subclip(
    src: str | Path,
    dst: str | Path,
    start_s: float,
    duration_s: float,
    height: int = 480,
    overwrite: bool = False,
) -> Path:
    """Cut a low-resolution subclip for review. Cached: returns early when dst exists."""
    dst = Path(dst)
    if dst.exists() and not overwrite:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    start_s = max(0.0, float(start_s))
    cmd = [
        ffmpeg_path(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(src),
        "-t",
        f"{max(0.1, float(duration_s)):.3f}",
        "-vf",
        f"scale=-2:{int(height)}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "28",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-movflags",
        "+faststart",
        "-avoid_negative_ts",
        "make_zero",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    return dst


def extract_audio(src: str | Path, dst: str | Path, sample_rate: int = 16000, overwrite: bool = False) -> Path:
    """Extract a mono 16 kHz PCM WAV track (input for ASR and audio-event models)."""
    dst = Path(dst)
    if dst.exists() and not overwrite:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_path(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate)),
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    return dst


def sample_frames(
    src: str | Path,
    out_dir: str | Path,
    fps: float = 1.0,
    height: int = 384,
    overwrite: bool = False,
) -> list[Path]:
    """Sample low-resolution frames at a fixed rate into out_dir/000001.jpg, ...

    Frame k (1-based) sits at time (k - 0.5) / fps seconds, the center of its
    sampling interval, which is how windows.py maps frames to windows.
    """
    out_dir = Path(out_dir)
    existing = sorted(out_dir.glob("*.jpg"))
    if existing and not overwrite:
        return existing
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_path(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-vf",
        f"fps={fps},scale=-2:{int(height)}",
        "-q:v",
        "4",
        str(out_dir / "%06d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    return sorted(out_dir.glob("*.jpg"))
