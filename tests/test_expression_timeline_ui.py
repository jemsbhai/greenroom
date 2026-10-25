"""Exercise timestamped expression browsing using synthetic records and media."""
import csv
import io
import json
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


JSON_DOWNLOAD = "Download expression timeline (JSON)"
CSV_DOWNLOAD = "Download expression timeline (CSV)"


def _element(elements, label):
    return next(element for element in elements if element.label == label)


def _healthy(app):
    assert not app.exception, [item.message for item in app.exception]
    assert not app.error, [item.value for item in app.error]


def _table(app):
    return next(
        item.value for item in app.dataframe
        if "AI expression (unverified)" in item.value.columns
    )


def _open_timeline(app):
    _element(app.sidebar.radio, "View").set_value("Expression timeline").run()
    _healthy(app)
    return app


@pytest.fixture
def timeline_ui(tmp_path, monkeypatch):
    monkeypatch.setenv("GREENROOM_DATA_DIR", str(tmp_path))
    (tmp_path / "videos").mkdir()
    (tmp_path / "derived").mkdir()
    placeholder = tmp_path / "videos" / "synthetic.mp4"
    placeholder.write_bytes(b"Synthetic fixture: decoding and inference are intercepted")
    cut_subclip = Mock(return_value=placeholder)
    monkeypatch.setattr(media, "cut_subclip", cut_subclip)
    monkeypatch.setattr(uploads, "list_jobs", lambda: [])
    start_worker = Mock()
    monkeypatch.setattr(uploads, "start_worker", start_worker)

    index = {"videos": [], "windows": []}
    snapshots = {}
    first_window = None
    for video_id, bounds, labels in (
        ("synthetic-a", [(0.0, 5.0), (5.0, 10.125), (10.125, 12.345)], ["Positive", "Neutral", None]),
        ("synthetic-b", [(0.0, 4.75)], ["Negative"]),
    ):
        index["videos"].append({
            "video_id": video_id, "source_file": placeholder.name,
            "duration_s": bounds[-1][1], "n_windows": len(bounds),
        })
        records = []
        for i, ((start_s, end_s), label) in enumerate(zip(bounds, labels)):
            window = {
                "window_id": f"{video_id}:{i:05d}", "video_id": video_id,
                "index": i, "start_s": start_s, "end_s": end_s,
            }
            index["windows"].append(window)
            if video_id == "synthetic-a" and i == 0:
                first_window = window
            suggestion = None if label is None else {
                "label": label, "share": 0.8, "face_readings": 3,
                "mixture": {bucket: 0.8 if bucket == label.lower() else 0.1
                            for bucket in ("positive", "neutral", "negative")},
                "upset_share": 0.0, "frames_with_faces": 2, "frames_checked": 5,
                "status": "ai_suggested_unverified", "source": "face_expression_model",
                "model_id": "synthetic/expression-model",
            }
            records.append(build_record(
                window,
                {lid: vacuous(0.1) for lid in taxonomy.ALL_LABEL_IDS},
                {lid: 0.0 for lid in taxonomy.ALL_LABEL_IDS},
                Decision("dance", [], "ai_suggested", False, 0.1, 0.0, []),
                {}, {"model_ids": ["synthetic/expression-model"]},
                expression_suggestion=suggestion,
            ))
        record_path = tmp_path / "records" / video_id / "records.jsonld"
        record_path.parent.mkdir(parents=True)
        record_path.write_text(json.dumps({"@graph": records}), encoding="utf-8")
        snapshots[record_path] = record_path.read_bytes()

    index_path = tmp_path / "derived" / "windows.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    snapshots[index_path] = index_path.read_bytes()
    ledger.append_decision(
        "synthetic-a", first_window, "Fixture Reviewer", "edited",
        {"primary_activity": "dance"}, "dance", [], None,
        "Staff observation kept separate from the AI label", "Confident", "Fixture note", 1,
    )
    ledger_path = ledger.reviews_path("synthetic-a")
    snapshots[ledger_path] = ledger_path.read_bytes()
    append_decision = Mock(return_value={})
    monkeypatch.setattr(ledger, "append_decision", append_decision)

    downloads = {}
    original_download = st.download_button

    def capture_download(label, data, *args, **kwargs):
        downloads[label] = data
        return original_download(label, data, *args, **kwargs)

    monkeypatch.setattr(st, "download_button", capture_download)
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "app" / "review_app.py"), default_timeout=20,
    ).run()
    _healthy(app)
    yield SimpleNamespace(
        app=app, downloads=downloads, cut_subclip=cut_subclip, append_decision=append_decision,
    )
    append_decision.assert_not_called()
    start_worker.assert_not_called()
    assert list(tmp_path.rglob("reviews.jsonl")) == [ledger_path]
    assert all(path.read_bytes() == data for path, data in snapshots.items())


def test_timeline_available_without_reviewer_and_keeps_labels_separate(timeline_ui):
    app = timeline_ui.app
    assert _element(app.sidebar.radio, "View").value == "Review clips"
    assert not any(item.label == "Video" for item in app.sidebar.selectbox)
    _open_timeline(app)
    assert _element(app.sidebar.text_input, "Reviewer name").value == ""
    assert _element(app.sidebar.selectbox, "Video").value == "synthetic-a"
    assert any(item.value == "Timestamped expression tags" for item in app.subheader)
    assert any("unverified" in item.value.lower() for item in [*app.caption, *app.warning])
    assert not any("Enter your name" in item.value for item in app.info)
    assert not any(item.label in {"Accept", "Reject", "Defer", "Save edit"} for item in app.button)
    table = _table(app)
    assert list(table.columns) == [
        "Start", "End", "AI expression (unverified)", "Share", "Face readings",
        "Staff observation", "Review decision",
    ]
    assert table["Start"].tolist() == ["00:00:00.000", "00:00:05.000", "00:00:10.125"]
    assert table["End"].tolist() == ["00:00:05.000", "00:00:10.125", "00:00:12.345"]
    assert table["AI expression (unverified)"].tolist() == ["Positive", "Neutral", "No AI suggestion"]
    assert table["Staff observation"].tolist() == ["Confident", "—", "—"]
    assert table.iloc[0]["Review decision"] == "edited"
    assert table.iloc[0]["Share"] == pytest.approx(0.8)
    assert table.iloc[0]["Face readings"] == 3
    assert any("Model: synthetic/expression-model" in item.value for item in app.caption)


def test_timeline_plays_exact_selected_interval_without_context(timeline_ui):
    app = _open_timeline(timeline_ui.app)
    timeline_ui.cut_subclip.reset_mock()
    _element(app.selectbox, "Play timestamped window").select("synthetic-a:00002").run()
    _healthy(app)
    call = timeline_ui.cut_subclip.call_args
    assert call.args[2] == pytest.approx(10.125)
    assert call.args[3] == pytest.approx(12.345 - 10.125)
    assert Path(call.args[1]).name == "00002.mp4"
    assert any(
        "Source interval:" in item.value and "00:00:10.125" in item.value and "00:00:12.345" in item.value
        for item in app.caption
    )


@pytest.mark.parametrize("selected, expected", [
    (["Positive"], ["Positive"]),
    (["Confident"], ["Positive"]),
    (["No AI suggestion"], ["No AI suggestion"]),
    (["Positive", "No AI suggestion"], ["Positive", "No AI suggestion"]),
])
def test_timeline_filters_ai_staff_or_missing_suggestion(timeline_ui, selected, expected):
    app = _open_timeline(timeline_ui.app)
    _element(app.multiselect, "Filter expression tags").set_value(selected).run()
    _healthy(app)
    assert _table(app)["AI expression (unverified)"].tolist() == expected
    assert len(_element(app.selectbox, "Play timestamped window").options) == len(expected)


def test_empty_filter_result_keeps_full_video_exports(timeline_ui):
    app = _open_timeline(timeline_ui.app)
    original_json = timeline_ui.downloads[JSON_DOWNLOAD]
    original_csv = timeline_ui.downloads[CSV_DOWNLOAD]
    _element(app.multiselect, "Filter expression tags").set_value(["Upset"]).run()
    _healthy(app)
    assert app.info
    assert not any(item.label == "Play timestamped window" for item in app.selectbox)
    assert timeline_ui.downloads[JSON_DOWNLOAD] == original_json
    assert timeline_ui.downloads[CSV_DOWNLOAD] == original_csv
    assert len(list(csv.DictReader(io.StringIO(original_csv)))) == 3
    # Schema validation lives in the export tests; all window IDs must survive UI filtering.
    payload = json.dumps(json.loads(original_json))
    assert all(f"synthetic-a:{i:05d}" in payload for i in range(3))


def test_review_queue_filters_do_not_hide_timeline_windows(timeline_ui):
    app = timeline_ui.app
    _element(app.sidebar.text_input, "Reviewer name").set_value("Synthetic Reviewer").run()
    _healthy(app)
    assert _element(app.sidebar.toggle, "Pending only").value is True
    _element(app.sidebar.radio, "Queue").set_value("Review required").run()
    _healthy(app)
    assert any("Nothing in this queue" in item.value for item in app.success)
    _open_timeline(app)
    assert len(_table(app)) == 3
    assert _table(app).iloc[0]["Review decision"] == "edited"
    assert not any(item.label == "Queue" for item in app.sidebar.radio)


def test_timeline_switches_videos_without_reviewer(timeline_ui):
    app = _open_timeline(timeline_ui.app)
    _element(app.sidebar.selectbox, "Video").select("synthetic-b").run()
    _healthy(app)
    assert _table(app)["AI expression (unverified)"].tolist() == ["Negative"]
    assert _table(app)["End"].tolist() == ["00:00:04.750"]
    assert _element(app.selectbox, "Play timestamped window").value == "synthetic-b:00000"
    assert len(list(csv.DictReader(io.StringIO(timeline_ui.downloads[CSV_DOWNLOAD])))) == 1
