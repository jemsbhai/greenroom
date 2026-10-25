"""Local video intake and a durable, sequential background processing queue.

Media, jobs and logs stay under GREENROOM_DATA_DIR. OS locks serialize index
changes and GPU workers, and are released by the OS when a process exits.
"""
from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

from . import config
from .media import ffmpeg_path, probe
from .windows import WINDOW_SECONDS, load_index, make_windows, slugify

MAX_UPLOAD_BYTES = 2048 * 1024 * 1024
_RESERVED = {"con", "prn", "aux", "nul"} | {f"{p}{n}" for p in ("com", "lpt") for n in range(1, 10)}


class _FileLock:
    """Small cross-platform advisory lock; no stale PID or lock-file deletion."""

    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def acquire(self, blocking: bool = True) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            handle.write(b"\0")
            handle.flush()
        while True:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.handle = handle
                return True
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    handle.close()
                    raise
                if not blocking:
                    handle.close()
                    return False
                time.sleep(0.05)

    def release(self) -> None:
        if self.handle is None:
            return
        try:
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_):
        self.release()


def _queue_dir() -> Path:
    data = config.data_dir()
    if data.is_relative_to(config.repo_root()):
        raise ValueError("Uploaded videos must be stored outside the repository")
    folder = data / "derived" / "uploads"
    (folder / "jobs").mkdir(parents=True, exist_ok=True)
    (folder / "logs").mkdir(exist_ok=True)
    return folder


def _state_lock() -> _FileLock:
    return _FileLock(_queue_dir() / "state.lock")


def _worker_lock() -> _FileLock:
    return _FileLock(_queue_dir() / "worker.lock")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, suffix=".tmp", delete=False) as handle:
        temp = Path(handle.name)
        try:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            handle.close()
            temp.unlink(missing_ok=True)
            raise
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict) -> None:
    _atomic_bytes(path, json.dumps(value, indent=2, allow_nan=False).encode("utf-8"))


def _job_path(job_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise ValueError("Invalid upload job ID")
    return _queue_dir() / "jobs" / f"{job_id}.json"


def _save_job(job: dict) -> None:
    job["updated_at"] = _now()
    _atomic_json(_job_path(job["job_id"]), job)


def _read_jobs() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((_queue_dir() / "jobs").glob("*.json"))]


def _fail_interrupted(jobs: list[dict]) -> None:
    for job in jobs:
        if job["status"] == "running":
            job.update(status="failed", finished_at=_now(), error="Processing was interrupted. Retry this upload to resume.", message="Processing interrupted")
            _save_job(job)


def list_jobs() -> list[dict]:
    """List newest jobs first; interrupted processing becomes explicitly retryable."""
    with _state_lock():
        jobs = _read_jobs()
        worker = _worker_lock()
        if worker.acquire(blocking=False):
            try:
                _fail_interrupted(jobs)
            finally:
                worker.release()
        return sorted(jobs, key=lambda j: (j["created_at"], j["job_id"]), reverse=True)


def _probe_video(path: Path) -> dict:
    """Check container metadata and decode a real video frame (audio-only is invalid)."""
    try:
        meta = probe(path)
        duration = float(meta.get("duration_s") or 0)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Video duration must be finite and greater than zero")
        for dimension in ("width", "height"):
            if meta.get(dimension) is not None and int(meta[dimension]) <= 0:
                raise ValueError("The video stream has invalid dimensions")
        result = subprocess.run(
            [ffmpeg_path(), "-v", "error", "-nostdin", "-i", str(path), "-map", "0:v:0", "-frames:v", "1",
             "-vf", "scale=2:2", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"],
            capture_output=True, timeout=60, check=False,
        )
        if result.returncode != 0 or len(result.stdout) != 12:
            raise ValueError("The file does not contain a readable video stream")
        return meta
    except (OSError, subprocess.SubprocessError, TypeError, ValueError) as exc:
        raise ValueError(f"Cannot read this video: {exc}") from exc


def submit_upload(upload, profile_name: str = "rtx4090", max_bytes: int = MAX_UPLOAD_BYTES) -> dict:
    """Validate and enqueue a file-like upload without rebuilding existing windows.

    Repeated submission of identical bytes with the same profile returns the
    original job. Retrying a failed job is an explicit, separate operation.
    """
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", profile_name):
        raise ValueError("Invalid processing profile")
    config.load_profile(profile_name)
    original_name = PureWindowsPath(str(upload.name).replace("/", "\\")).name
    extension = Path(original_name).suffix.lower()
    if extension not in config.VIDEO_EXTENSIONS:
        raise ValueError("Choose an MP4, MPEG4, MOV, MKV, AVI, WEBM or M4V video")
    if max_bytes <= 0:
        raise ValueError("Upload limit must be greater than zero")
    folder = _queue_dir()
    digest = hashlib.sha256()
    size = 0
    with tempfile.NamedTemporaryFile(dir=folder, suffix=".part", delete=False) as handle:
        temp = Path(handle.name)
        try:
            upload.seek(0)
            while chunk := upload.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise ValueError(f"Video exceeds the {max_bytes / (1024 * 1024):g} MB upload limit")
                digest.update(chunk)
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            handle.close()
            temp.unlink(missing_ok=True)
            raise
    try:
        if not size:
            raise ValueError("The uploaded video is empty")
        with _state_lock():
            duplicate = next((j for j in _read_jobs() if j.get("sha256") == digest.hexdigest() and j["profile_name"] == profile_name), None)
            if duplicate:
                return duplicate
        meta = _probe_video(temp)
        duration = float(meta["duration_s"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Video duration must be finite and greater than zero")
        with _state_lock():
            # Recheck after decoding because another session may submit the same bytes.
            duplicate = next((j for j in _read_jobs() if j.get("sha256") == digest.hexdigest() and j["profile_name"] == profile_name), None)
            if duplicate:
                return duplicate
            index_path = config.windows_index_path()
            original_index = index_path.read_bytes() if index_path.exists() else None
            index = json.loads(original_index) if original_index is not None else {"window_seconds": WINDOW_SECONDS, "videos": [], "windows": []}
            base = slugify(original_name)[:72]
            if base in _RESERVED:
                base = f"video-{base}"
            used = {str(v["video_id"]).casefold() for v in index["videos"]}
            used.update(str(w["video_id"]).casefold() for w in index["windows"])
            filenames = {p.name.casefold() for p in config.videos_dir().iterdir()}
            while True:
                job_id = uuid.uuid4().hex
                video_id = f"{base}-upload-{job_id[:12]}"
                source_file = video_id + extension
                if (video_id.casefold() not in used and source_file.casefold() not in filenames
                        and not (config.records_dir() / video_id).exists() and not _job_path(job_id).exists()):
                    break
            windows = [asdict(w) for w in make_windows(duration, video_id, source_file, float(index["window_seconds"]))]
            if not windows or any(w["end_s"] <= w["start_s"] for w in windows):
                raise ValueError("The video is too short to create a review window")
            video = {"video_id": video_id, "source_file": source_file, "upload_job_id": job_id, "duration_s": round(duration, 3),
                     **{k: meta.get(k) for k in ("fps", "width", "height", "has_audio")}, "n_windows": len(windows)}
            index["videos"].append(video)
            index["windows"].extend(windows)
            job = {"job_id": job_id, "video_id": video_id, "source_file": source_file,
                   "original_name": original_name[:255], "profile_name": profile_name, "sha256": digest.hexdigest(),
                   "size_bytes": size, "status": "queued", "created_at": _now(), "updated_at": _now(),
                   "started_at": None, "finished_at": None, "attempts": 0, "error": None,
                   "message": "Waiting for processing", "n_windows": len(windows),
                   "log_path": str(folder / "logs" / f"{job_id}.log")}
            destination = config.videos_dir() / source_file
            # Each replacement is atomic. The worker only reads under this same lock.
            temp.replace(destination)
            try:
                _atomic_json(index_path, index)
                _save_job(job)
            except BaseException:
                # Roll back only this intake; retain the original index bytes exactly.
                if original_index is None:
                    index_path.unlink(missing_ok=True)
                else:
                    _atomic_bytes(index_path, original_index)
                _job_path(job_id).unlink(missing_ok=True)
                destination.unlink(missing_ok=True)
                raise
            return job
    finally:
        temp.unlink(missing_ok=True)


def retry_job(job_id: str) -> dict:
    """Requeue a failed upload, preserving its video and timestamp IDs."""
    list_jobs()  # Converts a crashed worker's running job into a retryable failure.
    with _state_lock():
        job = json.loads(_job_path(job_id).read_text(encoding="utf-8"))
        if job["status"] != "failed":
            raise ValueError("Only failed uploads can be retried")
        job.update(status="queued", error=None, message="Waiting to retry", started_at=None, finished_at=None)
        _save_job(job)
        return job


def start_worker() -> bool:
    """Start a detached local worker if jobs need it; no shell or visible console."""
    with _state_lock():
        if not any(j["status"] in ("queued", "running") for j in _read_jobs()):
            return False
        lock = _worker_lock()
        if not lock.acquire(blocking=False):
            return False
        lock.release()
        environment = os.environ.copy()
        environment[config.ENV_DATA_DIR] = str(config.data_dir())
        kwargs = {"start_new_session": True} if os.name != "nt" else {
            "creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
        }
        with (_queue_dir() / "worker.log").open("ab") as output:
            subprocess.Popen(
                [sys.executable, "-m", "greenroom.uploads"], cwd=str(config.repo_root()), env=environment,
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, close_fds=True, **kwargs,
            )
        return True


def _run_job(job: dict) -> None:
    from . import pipeline

    index = load_index(config.windows_index_path())
    video = next((v for v in index["videos"] if v["video_id"] == job["video_id"]), None)
    if not video or video["source_file"] != job["source_file"]:
        raise ValueError("The uploaded video no longer matches the window index")
    reviews = config.records_dir() / job["video_id"] / "reviews.jsonl"
    if reviews.exists() and reviews.stat().st_size:
        raise ValueError("This upload already has staff reviews; processing will not overwrite its records")
    profile = config.load_profile(job["profile_name"])
    stages = tuple(stage for stage in pipeline.STAGES if video.get("has_audio") or stage not in ("audio", "transcript"))

    def log(message):
        text = str(message)
        with Path(job["log_path"]).open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")
        with _state_lock():
            job["message"] = text[:300]
            _save_job(job)

    log(f"Processing {job['original_name']} ({', '.join(stages)})")
    summaries = pipeline.run_videos([job["video_id"]], profile, stages=stages, index=index, log=log)
    if summaries:
        job["windows_with_suggestion"] = summaries[0].get("stage_meta", {}).get("expression", {}).get("windows_with_suggestion", 0)


def process_queue() -> int:
    """Drain queued videos in order with one GPU worker across app sessions."""
    worker = _worker_lock()
    if not worker.acquire(blocking=False):
        return 0
    processed = 0
    try:
        with _state_lock():
            _fail_interrupted(_read_jobs())
        while True:
            with _state_lock():
                queued = sorted((j for j in _read_jobs() if j["status"] == "queued"), key=lambda j: (j["created_at"], j["job_id"]))
                if not queued:
                    # Release before unlocking state so a simultaneous submit/start
                    # cannot mistake an exiting worker for one that will pick it up.
                    worker.release()
                    return processed
                job = queued[0]
                job.update(status="running", started_at=_now(), finished_at=None, error=None,
                           attempts=job["attempts"] + 1, message="Loading local models")
                _save_job(job)
            try:
                _run_job(job)
            except Exception as exc:  # One failed upload must not block the rest.
                with Path(job["log_path"]).open("a", encoding="utf-8") as handle:
                    traceback.print_exc(file=handle)
                job.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:1500], message="Processing failed")
            else:
                job.update(status="completed", error=None, message="Ready for review")
            with _state_lock():
                job["finished_at"] = _now()
                _save_job(job)
            processed += 1
    finally:
        worker.release()


if __name__ == "__main__":
    process_queue()
