"""Upload queue interactions use synthetic data and never start a real worker."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from greenroom import ledger, media, taxonomy, uploads
from greenroom.fuse import Decision
from greenroom.opinions import vacuous
from greenroom.records import build_record


def _element(elements, label):
    return next(element for element in elements if element.label == label)


def _healthy(app):
    assert not app.exception, [item.message for item in app.exception]


@pytest.fixture
def upload_ui(tmp_path, monkeypatch):
    monkeypatch.setenv("GREENROOM_DATA_DIR", str(tmp_path))
    (tmp_path / "videos").mkdir()
    (tmp_path / "derived").mkdir()
    placeholder = tmp_path / "videos" / "synthetic.mp4"
    placeholder.write_bytes(b"Synthetic fixture; media decoding is intercepted")
    monkeypatch.setattr(media, "cut_subclip", lambda *args, **kwargs: placeholder)

    # Preserve an actual, valid existing staff ledger while exercising uploads.
    ledger.append_decision(
        "previous-review", {"window_id": "previous-review:00000", "start_s": 0, "end_s": 5},
        "Fixture Reviewer", "accepted", {"primary_activity": "dance"}, "dance", [],
        None, None, None, "Existing staff decision", 1,
    )
    ledger_path = ledger.reviews_path("previous-review")
    original_ledger = ledger_path.read_bytes()
    append_decision = Mock(return_value={})
    monkeypatch.setattr(ledger, "append_decision", append_decision)

    state = SimpleNamespace(jobs=[], selected=[], events=[])
    monkeypatch.setattr(
        st, "file_uploader",
        lambda *args, **kwargs: state.selected if kwargs["key"] == "video_uploads_0" else [],
    )
    monkeypatch.setattr(uploads, "list_jobs", lambda: [dict(job) for job in state.jobs])

    def submit(upload):
        state.events.append(("submit", upload.name))
        if upload.name == "broken.mp4":
            raise ValueError("Video has no decodable frames")
        job = {
            "job_id": f"job-{len(state.jobs)}", "video_id": Path(upload.name).stem,
            "original_name": upload.name, "status": "queued", "n_windows": 0, "error": None,
        }
        state.jobs.append(job)
        return job

    def start():
        state.events.append(("start", None))
        return True

    def retry(job_id):
        state.events.append(("retry", job_id))
        next(job for job in state.jobs if job["job_id"] == job_id)["status"] = "queued"

    state.submit = Mock(side_effect=submit)
    state.start = Mock(side_effect=start)
    state.retry = Mock(side_effect=retry)
    monkeypatch.setattr(uploads, "submit_upload", state.submit)
    monkeypatch.setattr(uploads, "start_worker", state.start)
    monkeypatch.setattr(uploads, "retry_job", state.retry)

    def write_index(video_ids, *, records=True):
        index = {"videos": [], "windows": []}
        for video_id in video_ids:
            window = {
                "window_id": f"{video_id}:00000", "video_id": video_id,
                "index": 0, "start_s": 0.0, "end_s": 5.0,
            }
            index["windows"].append(window)
            index["videos"].append({
                "video_id": video_id, "source_file": placeholder.name,
                "duration_s": 5.0, "n_windows": 1,
            })
            if records:
                record = build_record(
                    window, {label: vacuous(0.1) for label in taxonomy.ALL_LABEL_IDS},
                    {label: 0.0 for label in taxonomy.ALL_LABEL_IDS},
                    Decision("dance", [], "ai_suggested", False, 0.1, 0.0, []),
                    {}, {"model_ids": ["synthetic/model"]},
                )
                path = tmp_path / "records" / video_id / "records.jsonld"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"@graph": [record]}), encoding="utf-8")
        path = tmp_path / "derived" / "windows.json"
        previous_mtime = path.stat().st_mtime_ns if path.exists() else 0
        path.write_text(json.dumps(index), encoding="utf-8")
        # Ensure a real mtime change even on filesystems with coarse resolution.
        info = path.stat()
        os.utime(path, ns=(info.st_atime_ns, max(info.st_mtime_ns, previous_mtime + 1_000_000)))
        return path

    state.write_index = write_index
    state.app = lambda: AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "app" / "review_app.py"), default_timeout=20,
    ).run()
    yield state
    append_decision.assert_not_called()
    assert ledger_path.read_bytes() == original_ledger
    assert list(tmp_path.rglob("reviews.jsonl")) == [ledger_path]


@pytest.mark.parametrize("initial_state", ["no-index", "no-records", "no-reviewer"])
def test_upload_controls_available_before_review_can_start(upload_ui, initial_state):
    if initial_state != "no-index":
        upload_ui.write_index(["existing"], records=initial_state != "no-records")
    app = upload_ui.app()
    _healthy(app)
    assert "Add videos" in [item.label for item in app.sidebar.expander]
    assert _element(app.sidebar.button, "Add to processing queue").disabled
    assert _element(app.sidebar.text_input, "Reviewer name").value == ""
    assert not app.sidebar.selectbox
    upload_ui.start.assert_not_called()


def test_multiple_uploads_enqueue_successes_despite_one_invalid_file(upload_ui):
    files = [SimpleNamespace(name=name) for name in ("first.mp4", "broken.mp4", "second.mov")]
    upload_ui.selected = files
    app = upload_ui.app()
    _healthy(app)
    _element(app.sidebar.button, "Add to processing queue").click().run()
    _healthy(app)
    assert [call.args[0] for call in upload_ui.submit.call_args_list] == files
    assert [job["original_name"] for job in upload_ui.jobs] == ["first.mp4", "second.mov"]
    assert any("first.mp4: added to the processing queue" in item.value for item in app.success)
    assert any("second.mov: added to the processing queue" in item.value for item in app.success)
    assert any("broken.mp4: Video has no decodable frames" in item.value for item in app.error)
    assert len([item for item in app.caption if item.value == "Queued for processing"]) == 2
    upload_ui.start.assert_called_once()
    assert upload_ui.events[-1] == ("start", None)
    # AppTest dispatches widget changes as full runs; advance the next panel refresh.
    app.run()
    _healthy(app)
    assert _element(app.sidebar.button, "Add to processing queue").disabled


def test_completed_upload_refreshes_cached_index_and_video_choices(upload_ui):
    index_path = upload_ui.write_index(["existing-a", "existing-b"])
    previous_mtime = index_path.stat().st_mtime_ns
    upload_ui.jobs.append({
        "job_id": "new-job", "video_id": "new-video", "original_name": "new-video.mp4",
        "status": "running", "n_windows": 0, "error": None,
    })
    app = upload_ui.app()
    _element(app.text_input, "Reviewer name").set_value("Synthetic Reviewer").run()
    _healthy(app)
    assert len(_element(app.sidebar.selectbox, "Video").options) == 2
    _element(app.sidebar.selectbox, "Video").select("existing-b").run()
    _healthy(app)
    upload_ui.write_index(["existing-a", "existing-b", "new-video"])
    assert index_path.stat().st_mtime_ns > previous_mtime
    upload_ui.jobs[0].update(status="completed", n_windows=1)
    app.run()
    _healthy(app)
    choices = _element(app.sidebar.selectbox, "Video")
    assert len(choices.options) == 3
    assert choices.value == "existing-b"  # Completion must not interrupt the current review.
    choices.select("new-video").run()
    _healthy(app)
    assert "new-video  |  0 s to 5 s" in app.subheader[0].value
    assert any("Ready to review" in item.value for item in app.caption)


@pytest.mark.parametrize("status", ["queued", "running", "failed"])
def test_unfinished_upload_records_are_hidden_from_review(upload_ui, status):
    upload_ui.write_index(["existing", "unfinished"])
    upload_ui.jobs.append({
        "job_id": "pending-job", "video_id": "unfinished", "original_name": "unfinished.mp4",
        "status": status, "n_windows": 0, "error": "Synthetic interrupted processing",
    })
    app = upload_ui.app()
    _element(app.text_input, "Reviewer name").set_value("Synthetic Reviewer").run()
    _healthy(app)
    choices = _element(app.sidebar.selectbox, "Video")
    assert choices.value == "existing"
    assert len(choices.options) == 1
    assert "unfinished" not in choices.options[0]


def test_retry_requeues_job_before_starting_worker(upload_ui):
    upload_ui.jobs.append({
        "job_id": "failed-job", "video_id": "failed-video", "original_name": "failed-video.mp4",
        "status": "failed", "n_windows": 0, "error": "Synthetic processing failure",
    })
    app = upload_ui.app()
    _healthy(app)
    upload_ui.start.assert_not_called()
    _element(app.sidebar.button, "Retry processing").click().run()
    _healthy(app)
    upload_ui.retry.assert_called_once_with("failed-job")
    upload_ui.start.assert_called_once()
    assert upload_ui.events == [("retry", "failed-job"), ("start", None)]
    assert upload_ui.jobs[0]["status"] == "queued"
    app.run()
    _healthy(app)
    assert any(item.value == "Queued for processing" for item in app.caption)


def test_older_unfinished_uploads_remain_visible_and_retryable(upload_ui):
    # list_jobs returns newest first; unfinished work must survive a long history.
    upload_ui.jobs = [
        {"job_id": f"completed-{i}", "video_id": f"completed-{i}", "original_name": f"completed-{i}.mp4",
         "status": "completed", "n_windows": 1, "error": None}
        for i in range(25)
    ]
    upload_ui.jobs.extend([
        {"job_id": "old-failure", "video_id": "old-failure", "original_name": "old-failure.mp4",
         "status": "failed", "n_windows": 0, "error": "Older processing failure"},
        {"job_id": "old-running", "video_id": "old-running", "original_name": "old-running.mp4",
         "status": "running", "n_windows": 0, "error": None},
        {"job_id": "old-queued", "video_id": "old-queued", "original_name": "old-queued.mp4",
         "status": "queued", "n_windows": 0, "error": None},
    ])
    app = upload_ui.app()
    _healthy(app)
    displayed = [item.value for item in app.text]
    assert all(name in displayed for name in ("old-failure.mp4", "old-running.mp4", "old-queued.mp4"))
    assert any("Processing locally" in item.value for item in app.caption)
    assert any(item.value == "Queued for processing" for item in app.caption)
    _element(app.sidebar.button, "Retry processing").click().run()
    _healthy(app)
    upload_ui.retry.assert_called_once_with("old-failure")
    assert next(job for job in upload_ui.jobs if job["job_id"] == "old-failure")["status"] == "queued"


def test_queue_read_failure_preserves_existing_review_and_hides_uploads(upload_ui, monkeypatch):
    index_path = upload_ui.write_index(["existing", "uploaded"])
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["videos"][1]["upload_job_id"] = "unreadable-job"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    monkeypatch.setattr(uploads, "list_jobs", Mock(side_effect=OSError("Synthetic queue unavailable")))
    app = upload_ui.app()
    _element(app.text_input, "Reviewer name").set_value("Synthetic Reviewer").run()
    _healthy(app)
    assert any("Synthetic queue unavailable" in item.value for item in app.error)
    choices = _element(app.sidebar.selectbox, "Video")
    assert choices.value == "existing"
    assert len(choices.options) == 1
    assert "existing  |  0 s to 5 s" in app.subheader[0].value
    upload_ui.start.assert_not_called()
