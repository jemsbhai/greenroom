"""Local uploads preserve existing reviews and run only new work. No real media/models."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from greenroom import config, pipeline, uploads


class Upload(io.BytesIO):
    def __init__(self, name="new clip.mp4", content=b"synthetic video bytes"):
        super().__init__(content)
        self.name = name
        self.size = len(content)


@pytest.fixture
def upload_env(tmp_path, monkeypatch):
    (tmp_path / "videos").mkdir()
    monkeypatch.setenv("GREENROOM_DATA_DIR", str(tmp_path))
    profile = {
        "name": "test", "window_seconds": 5, "device": "cpu",
        "audio_events": {"clap_model": "test/audio"},
        "vision": {"xclip_model": "test/vision"},
        "asr": {"model": "test/asr"},
        "transcript_llm": {"model": "test/local"},
        "expression": {"enabled": True, "model": "test/expression"},
    }
    monkeypatch.setattr(config, "load_profile", lambda name: {**profile, "name": name})
    monkeypatch.setattr(uploads, "_probe_video", lambda path: {
        "duration_s": 12.0, "width": 640, "height": 360, "fps": 25.0, "has_audio": True,
    })
    return tmp_path


def read_index():
    return json.loads(config.windows_index_path().read_text(encoding="utf-8"))


def saved_job(job_id):
    return next(job for job in uploads.list_jobs() if job["job_id"] == job_id)


def test_upload_stages_bytes_and_timestamped_windows_without_running_models(upload_env, monkeypatch):
    monkeypatch.setattr(pipeline, "run_videos", lambda *a, **kw: pytest.fail("submission must not run models"))
    content = b"a synthetic uploaded video"
    upload = Upload("My Clip.MP4", content)
    upload.seek(5)  # Rerendered file upload objects may have already been read.

    job = uploads.submit_upload(upload, profile_name="test")

    assert job["status"] == "queued" and job["profile_name"] == "test"
    assert job["original_name"] == "My Clip.MP4"
    assert job["created_at"] and job["updated_at"] and job["job_id"]
    assert job["sha256"] == hashlib.sha256(content).hexdigest()
    source = upload_env / "videos" / job["source_file"]
    assert source.read_bytes() == content
    index = read_index()
    assert [video["video_id"] for video in index["videos"]] == [job["video_id"]]
    assert [(w["start_s"], w["end_s"]) for w in index["windows"]] == [(0, 5), (5, 10), (10, 12)]
    assert all(w["video_id"] == job["video_id"] and w["source_file"] == source.name for w in index["windows"])
    assert job["n_windows"] == 3
    assert saved_job(job["job_id"])["status"] == "queued"


@pytest.mark.parametrize("name,content,max_bytes", [
    ("clip.txt", b"not video", 100),
    ("clip.mp4.exe", b"not video", 100),
    ("clip.mp4", b"", 100),
    ("clip.mp4", b"123456", 5),
])
def test_rejected_upload_does_not_publish_source_index_or_job(upload_env, name, content, max_bytes):
    with pytest.raises(ValueError):
        uploads.submit_upload(Upload(name, content), max_bytes=max_bytes)

    assert list((upload_env / "videos").iterdir()) == []
    assert not config.windows_index_path().exists()
    assert uploads.list_jobs() == []


def test_actual_stream_length_enforces_limit_even_when_reported_size_is_small(upload_env):
    upload = Upload(content=b"123456")
    upload.size = 1
    with pytest.raises(ValueError):
        uploads.submit_upload(upload, max_bytes=5)
    assert list((upload_env / "videos").iterdir()) == []
    assert uploads.list_jobs() == []


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_invalid_duration_is_rejected_before_indexing(upload_env, monkeypatch, duration):
    monkeypatch.setattr(uploads, "_probe_video", lambda path: {
        "duration_s": duration, "width": 640, "height": 360, "fps": 25, "has_audio": False,
    })
    with pytest.raises(ValueError):
        uploads.submit_upload(Upload())
    assert list((upload_env / "videos").iterdir()) == []
    assert not config.windows_index_path().exists()
    assert uploads.list_jobs() == []


def test_decode_failure_leaves_no_pending_upload(upload_env, monkeypatch):
    def cannot_decode(path):
        assert path.is_file()
        raise ValueError("No decodable video frame")

    monkeypatch.setattr(uploads, "_probe_video", cannot_decode)
    with pytest.raises(ValueError, match="decodable"):
        uploads.submit_upload(Upload())
    assert list((upload_env / "videos").iterdir()) == []
    assert not config.windows_index_path().exists()
    assert uploads.list_jobs() == []


@pytest.mark.parametrize("name", ["../../outside.mp4", r"C:\\outside\\nested\\clip.mp4", "CON.mp4", "NUL.mp4", "LPT1.mp4"])
def test_upload_names_cannot_escape_videos_or_use_reserved_windows_names(upload_env, name):
    job = uploads.submit_upload(Upload(name))
    source_name = job["source_file"]
    assert "/" not in source_name and "\\" not in source_name and ":" not in source_name
    assert Path(source_name).stem.upper() not in {"CON", "NUL", "LPT1"}
    assert (upload_env / "videos" / source_name).resolve().parent == (upload_env / "videos").resolve()
    assert (upload_env / "videos" / source_name).read_bytes() == b"synthetic video bytes"
    assert not (upload_env.parent / "outside.mp4").exists()


def test_filename_and_slug_collisions_preserve_index_media_and_human_reviews(upload_env):
    source = upload_env / "videos" / "Existing Clip.mp4"
    source.write_bytes(b"original source")
    protected = upload_env / "records" / "existing-clip"
    protected.mkdir(parents=True)
    (protected / "records.jsonld").write_text('{"staff": "keep me"}', encoding="utf-8")
    (protected / "reviews.jsonl").write_text('{"hash": "keep original ledger"}\n', encoding="utf-8")
    originals = {path: path.read_bytes() for path in (source, *protected.iterdir())}
    original_video = {"video_id": "existing-clip", "source_file": source.name, "duration_s": 5, "extension": "keep"}
    original_window = {"window_id": "existing-clip:00000", "video_id": "existing-clip", "start_s": 0, "end_s": 5,
                       "source_file": source.name, "index": 0, "extension": "keep"}
    config.windows_index_path().write_text(json.dumps({
        "window_seconds": 5, "videos": [original_video], "windows": [original_window], "extension": "keep",
    }), encoding="utf-8")

    first = uploads.submit_upload(Upload(source.name, b"new bytes"))
    second = uploads.submit_upload(Upload("existing_clip.mov", b"other new bytes"))

    assert len({"existing-clip", first["video_id"], second["video_id"]}) == 3
    assert len({source.name.casefold(), first["source_file"].casefold(), second["source_file"].casefold()}) == 3
    index = read_index()
    assert index["extension"] == "keep"
    assert index["videos"][0] == original_video and index["windows"][0] == original_window
    assert all(path.read_bytes() == original for path, original in originals.items())


@pytest.mark.parametrize("has_audio,expected", [
    (True, ("audio", "vision", "transcript", "expression")),
    (False, ("vision", "expression")),
])
def test_worker_runs_full_pipeline_only_for_new_video_and_skips_audio_on_silent_video(upload_env, monkeypatch, has_audio, expected):
    monkeypatch.setattr(uploads, "_probe_video", lambda path: {
        "duration_s": 12, "width": 640, "height": 360, "fps": 25, "has_audio": has_audio,
    })
    job = uploads.submit_upload(Upload(), profile_name="test")
    calls = []

    def run(video_ids, profile, *, stages, index, log):
        calls.append((video_ids, profile, stages, index))
        # Status must already be durable while the model runs.
        assert saved_job(job["job_id"])["status"] == "running"
        log("synthetic expression stage finished")
        return [{"video_id": video_ids[0], "n_windows": 3}]

    monkeypatch.setattr(pipeline, "run_videos", run)
    assert uploads.process_queue() == 1
    assert len(calls) == 1
    assert calls[0][0] == [job["video_id"]] and calls[0][1]["name"] == "test"
    assert calls[0][2] == expected
    assert calls[0][3] == read_index()
    completed = saved_job(job["job_id"])
    assert completed["status"] == "completed" and completed["started_at"] and completed["finished_at"]
    assert completed["attempts"] == 1 and not completed["error"]
    assert "synthetic expression stage finished" in Path(completed["log_path"]).read_text(encoding="utf-8")
    assert uploads.process_queue() == 0
    assert len(calls) == 1  # Refreshing/restarting the worker never reruns completed jobs.
    with pytest.raises(ValueError):
        uploads.retry_job(job["job_id"])


def test_failed_job_is_durable_and_retry_does_not_duplicate_index_or_source(upload_env, monkeypatch):
    job = uploads.submit_upload(Upload())
    original_index = read_index()
    original_sources = sorted(path.name for path in (upload_env / "videos").iterdir())
    calls = []

    def run(video_ids, profile, **kwargs):
        calls.append(video_ids)
        if len(calls) == 1:
            raise RuntimeError("Synthetic model failure")
        return [{"video_id": video_ids[0], "n_windows": 3}]

    monkeypatch.setattr(pipeline, "run_videos", run)
    uploads.process_queue()
    failed = saved_job(job["job_id"])
    assert failed["status"] == "failed" and "Synthetic model failure" in failed["error"]
    assert failed["finished_at"]
    assert uploads.process_queue() == 0
    retry = uploads.retry_job(job["job_id"])
    assert retry["status"] == "queued" and retry["video_id"] == job["video_id"]
    uploads.process_queue()
    completed = saved_job(job["job_id"])
    assert completed["status"] == "completed" and completed["attempts"] == 2
    assert calls == [[job["video_id"]], [job["video_id"]]]
    assert read_index() == original_index
    assert sorted(path.name for path in (upload_env / "videos").iterdir()) == original_sources


def test_a_busy_worker_prevents_a_second_model_run(upload_env, monkeypatch):
    job = uploads.submit_upload(Upload())
    monkeypatch.setattr(pipeline, "run_videos", lambda *a, **kw: pytest.fail("another worker owns the GPU"))
    lock = uploads._worker_lock()
    assert lock.acquire(blocking=False)
    try:
        assert uploads.process_queue() == 0
        assert saved_job(job["job_id"])["status"] == "queued"
    finally:
        lock.release()


def test_jobs_are_newest_first_and_queued_jobs_cannot_be_retried(upload_env):
    first = uploads.submit_upload(Upload("first.mp4", b"first video"))
    second = uploads.submit_upload(Upload("second.mp4", b"second video"))
    assert [job["job_id"] for job in uploads.list_jobs()] == [second["job_id"], first["job_id"]]
    with pytest.raises(ValueError):
        uploads.retry_job(first["job_id"])


def test_duplicate_content_returns_existing_job_without_index_or_media_duplication(upload_env):
    first = uploads.submit_upload(Upload("first.mp4", b"same bytes"))
    original_index = read_index()
    repeated = uploads.submit_upload(Upload("renamed.mov", b"same bytes"))
    assert repeated["job_id"] == first["job_id"]
    assert len(uploads.list_jobs()) == 1
    assert read_index() == original_index
    assert len(list((upload_env / "videos").iterdir())) == 1


@pytest.mark.parametrize("partial_job_written", [False, True])
def test_job_persistence_failure_restores_original_index_and_removes_new_media(upload_env, monkeypatch, partial_job_written):
    existing = uploads.submit_upload(Upload("existing.mp4", b"existing video"))
    index_path = config.windows_index_path()
    # Preserve the actual original bytes, including formatting, during rollback.
    index_path.write_text(json.dumps(read_index(), indent=4) + "\n", encoding="utf-8")
    original_index = index_path.read_bytes()
    original_source = (upload_env / "videos" / existing["source_file"]).read_bytes()
    real_save = uploads._save_job

    def fail_to_save(job):
        if partial_job_written:
            real_save(job)
        raise OSError("Synthetic queue persistence failure")

    monkeypatch.setattr(uploads, "_save_job", fail_to_save)
    with pytest.raises(OSError, match="persistence failure"):
        uploads.submit_upload(Upload("new.mp4", b"new video"))

    assert index_path.read_bytes() == original_index
    assert [(path.name, path.read_bytes()) for path in (upload_env / "videos").iterdir()] == [
        (existing["source_file"], original_source),
    ]
    assert [job["job_id"] for job in uploads.list_jobs()] == [existing["job_id"]]
    assert list(uploads._queue_dir().glob("*.part")) == []


def test_crashed_running_job_is_failed_and_waits_for_explicit_retry(upload_env, monkeypatch):
    job = uploads.submit_upload(Upload())
    # This is the durable state left after a worker dies, releasing its OS lock.
    job.update(status="running", attempts=1, started_at=job["created_at"])
    uploads._job_path(job["job_id"]).write_text(json.dumps(job), encoding="utf-8")
    monkeypatch.setattr(pipeline, "run_videos", lambda *a, **kw: pytest.fail("interrupted work needs an explicit retry"))

    failed = saved_job(job["job_id"])
    assert failed["status"] == "failed" and "interrupt" in failed["error"].lower()
    assert failed["finished_at"]
    assert uploads.process_queue() == 0
    assert uploads.retry_job(job["job_id"])["status"] == "queued"


def test_worker_continues_with_later_upload_after_failure(upload_env, monkeypatch):
    first = uploads.submit_upload(Upload("first.mp4", b"first video"))
    second = uploads.submit_upload(Upload("second.mp4", b"second video"))
    calls = []

    def run(video_ids, profile, **kwargs):
        calls.extend(video_ids)
        if video_ids == [first["video_id"]]:
            raise RuntimeError("Synthetic first-video failure")
        return [{"video_id": video_ids[0]}]

    monkeypatch.setattr(pipeline, "run_videos", run)
    assert uploads.process_queue() == 2
    assert calls == [first["video_id"], second["video_id"]]
    assert saved_job(first["job_id"])["status"] == "failed"
    assert saved_job(second["job_id"])["status"] == "completed"


def test_worker_refuses_to_overwrite_upload_that_already_has_staff_reviews(upload_env, monkeypatch):
    job = uploads.submit_upload(Upload())
    out = config.records_dir() / job["video_id"]
    out.mkdir()
    ledger = out / "reviews.jsonl"
    original = b'{"staff":"synthetic existing review"}\n'
    ledger.write_bytes(original)
    monkeypatch.setattr(pipeline, "run_videos", lambda *a, **kw: pytest.fail("staff-reviewed exports must not be regenerated"))
    uploads.process_queue()
    assert saved_job(job["job_id"])["status"] == "failed"
    assert ledger.read_bytes() == original


@pytest.mark.parametrize("returncode,frame", [(1, b""), (0, b""), (0, b"short")])
def test_probe_rejects_audio_only_or_undecodable_input_even_with_duration(tmp_path, monkeypatch, returncode, frame):
    monkeypatch.setattr(uploads, "probe", lambda path: {"duration_s": 12, "width": None, "height": None})
    monkeypatch.setattr(uploads, "ffmpeg_path", lambda: "fake-ffmpeg")
    monkeypatch.setattr(uploads.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=returncode, stdout=frame))
    with pytest.raises(ValueError, match="video stream"):
        uploads._probe_video(tmp_path / "pretend.mp4")


def test_probe_accepts_decodable_video_when_metadata_fallback_omits_dimensions(tmp_path, monkeypatch):
    metadata = {"duration_s": 12, "width": None, "height": None, "has_audio": False}
    monkeypatch.setattr(uploads, "probe", lambda path: metadata)
    monkeypatch.setattr(uploads, "ffmpeg_path", lambda: "fake-ffmpeg")
    calls = []

    def decode(args, **kwargs):
        calls.append(args)
        assert "0:v:0" in args and kwargs["timeout"] <= 60
        return SimpleNamespace(returncode=0, stdout=bytes(12))

    monkeypatch.setattr(uploads.subprocess, "run", decode)
    assert uploads._probe_video(tmp_path / "pretend.mp4") == metadata
    assert len(calls) == 1
