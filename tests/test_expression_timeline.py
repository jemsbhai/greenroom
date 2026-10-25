"""Timestamped expressions stay distinct from staff review and source artifacts."""
import csv
import io
import json
from copy import deepcopy

import pytest

from greenroom import cli
from greenroom.expression_timeline import (
    AI_STATUS, CSV_COLUMNS, SCHEMA_VERSION, build_expression_timeline,
    expression_timeline_csv, format_timestamp,
)


def suggestion(label="Positive"):
    return {
        "label": label, "share": 0.7,
        "mixture": {"positive": 0.7, "neutral": 0.2, "negative": 0.1},
        "upset_share": 0.05, "face_readings": 8, "frames_with_faces": 4,
        "frames_checked": 5, "model_id": "fixture/model", "status": AI_STATUS,
    }


def record(number, expression=None):
    return {
        "video_id": "vid", "window_id": f"vid:{number:05d}",
        "start_s": number * 5, "end_s": (number + 1) * 5,
        "expression_suggestion": expression,
        "speech": {"transcript": "Do not export this"}, "reviewer_note": "Private note",
        "reviewer_decision": "accepted", "expression_label": "Not a ledger entry",
    }


def test_chronological_all_windows_without_mutation_or_non_expression_content():
    records = [record(2, suggestion("Neutral")), record(0), record(1, suggestion())]
    before = deepcopy(records)
    timeline = build_expression_timeline("vid", records, source_file="original name.mp4")
    assert timeline["schema_version"] == SCHEMA_VERSION
    assert timeline["time_unit"] == "seconds"
    assert timeline["interval_semantics"] == "start_inclusive_end_exclusive"
    assert timeline["source_file"] == "original name.mp4"
    windows = timeline["windows"]
    assert [w["window_id"] for w in windows] == ["vid:00000", "vid:00001", "vid:00002"]
    assert windows[0]["ai_expression"] is None  # Missing is distinct from Neutral.
    assert windows[2]["ai_expression"]["label"] == "Neutral"
    assert all(w["staff_expression"] is None and w["review_decision"] == "pending" for w in windows)
    assert "Do not export this" not in json.dumps(timeline)
    assert "Private note" not in json.dumps(timeline)
    assert records == before
    windows[1]["ai_expression"]["mixture"]["positive"] = 0
    assert records == before  # Nested result objects are independent, too.


def test_explicit_latest_staff_expression_has_separate_provenance_and_never_verifies_ai():
    records = [record(0, suggestion()), record(1, suggestion()), record(2)]
    records[0]["expression_suggestion"]["status"] = "incorrectly_verified"
    decisions = {
        "vid:00000": {"decision": "accepted", "expression_label": None},
        "vid:00001": {"decision": "edited", "expression_label": "Neutral", "reviewer": "Staff A",
                      "timestamp": "2026-10-25T00:00:00+00:00", "hash": "ledger-hash", "note": "Private review note"},
        "vid:00002": {"decision": "deferred", "expression_label": "Positive", "reviewer": "Staff B"},
    }
    before = deepcopy(decisions)
    windows = build_expression_timeline("vid", records, decisions)["windows"]
    assert windows[0]["review_decision"] == "accepted"
    assert windows[0]["staff_expression"] is None
    assert windows[0]["ai_expression"]["status"] == AI_STATUS
    assert windows[1]["ai_expression"]["label"] == "Positive"
    assert windows[1]["ai_expression"]["status"] == AI_STATUS
    assert windows[1]["staff_expression"] == {
        "label": "Neutral", "reviewer": "Staff A", "reviewed_at": "2026-10-25T00:00:00+00:00",
        "ledger_hash": "ledger-hash",
    }
    assert windows[2]["ai_expression"] is None
    assert windows[2]["staff_expression"]["label"] == "Positive"
    assert windows[2]["review_decision"] == "deferred"
    assert "Private review note" not in json.dumps(windows)
    assert decisions == before


@pytest.mark.parametrize(("seconds", "expected"), [
    (0, "00:00:00.000"), (5.1234, "00:00:05.123"), (59.9995, "00:01:00.000"),
    (3599.9996, "01:00:00.000"), (90000.001, "25:00:00.001"),
])
def test_display_timestamp_precision_and_carry(seconds, expected):
    assert format_timestamp(seconds) == expected


def test_numeric_seconds_keep_submillisecond_precision():
    window = record(0)
    window.update(start_s=0.0004, end_s=5.123456)
    exported = build_expression_timeline("vid", [window])["windows"][0]
    assert exported["start_s"] == 0.0004 and exported["end_s"] == 5.123456
    assert exported["start_time"] == "00:00:00.000" and exported["end_time"] == "00:00:05.123"


@pytest.mark.parametrize(("start", "end"), [(-1, 5), (5, 5), (6, 5), (0, float("inf")), (float("nan"), 5)])
def test_invalid_intervals_are_rejected(start, end):
    window = record(0)
    window.update(start_s=start, end_s=end)
    with pytest.raises(ValueError, match="finite timestamps"):
        build_expression_timeline("vid", [window])


def test_duplicate_and_other_video_windows_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        build_expression_timeline("vid", [record(0), record(0)])
    with pytest.raises(ValueError, match="different video"):
        build_expression_timeline("other", [record(0)])


def test_csv_flattens_mixture_and_separates_missing_values_from_zero():
    ai = suggestion()
    ai["mixture"]["negative"] = 0.0
    document = build_expression_timeline("vid", [record(0), record(1, ai)], {
        "vid:00001": {"decision": "edited", "expression_label": "Upset", "reviewer": 'Staff, "A"',
                      "timestamp": "reviewed-at", "hash": "hash"},
    })
    rows = list(csv.DictReader(io.StringIO(expression_timeline_csv(document))))
    assert list(rows[0]) == CSV_COLUMNS
    assert len(rows) == 2
    assert rows[0]["ai_label"] == rows[0]["ai_status"] == rows[0]["ai_negative_share"] == ""
    assert rows[1]["ai_label"] == "Positive" and rows[1]["staff_label"] == "Upset"
    assert rows[1]["ai_negative_share"] == "0.0" and rows[1]["ai_share"] == "0.7"
    assert rows[1]["ai_face_readings"] == "8" and rows[1]["ai_model_id"] == "fixture/model"
    assert rows[1]["ai_status"] == AI_STATUS and rows[1]["staff_reviewer"] == 'Staff, "A"'
    assert rows[1]["staff_ledger_hash"] == "hash" and rows[1]["review_decision"] == "edited"
    assert rows[1]["start_time"] == "00:00:05.000"


def test_empty_timeline_and_csv_are_valid():
    document = build_expression_timeline("vid", [])
    assert document["windows"] == [] and document["source_file"] is None
    assert expression_timeline_csv(document) == ",".join(CSV_COLUMNS) + "\n"


@pytest.fixture
def source_exports(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "videos").mkdir(parents=True)
    (data / "derived").mkdir()
    (data / "videos" / "original.mp4").write_bytes(b"synthetic source media")
    index = {
        "videos": [{"video_id": "vid", "source_file": "original.mp4", "n_windows": 2}],
        "windows": [{key: window[key] for key in ("video_id", "window_id", "start_s", "end_s")}
                    for window in (record(0), record(1))],
    }
    (data / "derived" / "windows.json").write_text(json.dumps(index), encoding="utf-8")
    records_dir = data / "records" / "vid"
    records_dir.mkdir(parents=True)
    (records_dir / "records.jsonld").write_text(json.dumps({"@graph": [record(1, suggestion()), record(0)]}), encoding="utf-8")
    decisions = [
        {"window_id": "vid:00000", "decision": "edited", "expression_label": "Positive", "reviewer": "Old"},
        {"window_id": "vid:00000", "decision": "accepted", "expression_label": None},
        {"window_id": "vid:00001", "decision": "edited", "expression_label": "Neutral", "reviewer": "New", "hash": "latest"},
    ]
    (records_dir / "reviews.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in decisions), encoding="utf-8")
    monkeypatch.setenv("GREENROOM_DATA_DIR", str(data))
    monkeypatch.setattr(cli.config, "load_profile", lambda *args: pytest.fail("Export must not load models or profile"))
    return data


@pytest.mark.parametrize("file_format", ["json", "csv"])
def test_cli_exports_latest_reviews_without_changing_any_source_files(source_exports, tmp_path, file_format):
    originals = {path: path.read_bytes() for path in source_exports.rglob("*") if path.is_file()}
    destination = tmp_path / "exports" / f"timeline.{file_format}"
    assert cli.main(["export-expressions", "--video", "vid", "--output", str(destination), "--format", file_format]) == 0
    if file_format == "json":
        output = json.loads(destination.read_text(encoding="utf-8"))
        assert output["source_file"] == "original.mp4"
        assert output["windows"][0]["staff_expression"] is None  # Latest entry cleared the old label.
        assert output["windows"][1]["staff_expression"]["label"] == "Neutral"
        assert output["windows"][1]["ai_expression"]["status"] == AI_STATUS
    else:
        with destination.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 2 and rows[0]["staff_label"] == "" and rows[1]["staff_label"] == "Neutral"
    assert originals == {path: path.read_bytes() for path in source_exports.rglob("*") if path.is_file()}


def test_cli_all_preflights_every_video_before_writing_exports(source_exports, tmp_path):
    destination = tmp_path / "exports"
    assert cli.main(["export-expressions", "--all", "--output", str(destination)]) == 0
    assert [path.name for path in destination.iterdir()] == ["vid.expressions.json"]
    index_path = source_exports / "derived" / "windows.json"
    index = json.loads(index_path.read_text())
    index["videos"].append({"video_id": "missing", "source_file": "missing.mp4"})
    index_path.write_text(json.dumps(index))
    new_destination = tmp_path / "fail-before-writing"
    assert cli.main(["export-expressions", "--all", "--output", str(new_destination)]) == 2
    assert not new_destination.exists()


@pytest.mark.parametrize("target", ["records/vid/records.jsonld", "records/vid/records.csv", "derived/windows.json", "videos/export.json"])
def test_cli_refuses_destinations_within_protected_data_folders(source_exports, target):
    originals = {path: path.read_bytes() for path in source_exports.rglob("*") if path.is_file()}
    destination = source_exports / target
    assert cli.main(["export-expressions", "--video", "vid", "--output", str(destination)]) == 2
    assert originals == {path: path.read_bytes() for path in source_exports.rglob("*") if path.is_file()}


def test_cli_unknown_video_or_format_mismatch_does_not_create_output(source_exports, tmp_path):
    destination = tmp_path / "no-output.csv"
    assert cli.main(["export-expressions", "--video", "unknown", "--output", str(destination)]) == 2
    assert cli.main(["export-expressions", "--video", "vid", "--output", str(destination)]) == 2
    assert not destination.exists()


@pytest.mark.parametrize("state", ["queued", "running", "failed", "missing", "mismatched_video", "mismatched_job", "invalid_id", "completed"])
def test_cli_requires_confirmed_completed_upload_without_modifying_queue(source_exports, tmp_path, state):
    job_id = "0123456789abcdef0123456789abcdef"
    index_path = source_exports / "derived" / "windows.json"
    index = json.loads(index_path.read_text())
    index["videos"][0]["upload_job_id"] = "../../escape" if state == "invalid_id" else job_id
    index_path.write_text(json.dumps(index))
    jobs = source_exports / "derived" / "uploads" / "jobs"
    jobs.mkdir(parents=True)
    if state != "missing":
        job = {"job_id": "wrong" if state == "mismatched_job" else job_id,
               "video_id": "different" if state == "mismatched_video" else "vid",
               "status": state if state in ("queued", "running", "failed") else "completed"}
        (jobs / f"{job_id}.json").write_text(json.dumps(job))
    originals = {path: path.read_bytes() for path in source_exports.rglob("*") if path.is_file()}
    destination = tmp_path / "upload-timeline.json"
    result = cli.main(["export-expressions", "--video", "vid", "--output", str(destination)])
    assert result == (0 if state == "completed" else 2)
    assert destination.exists() == (state == "completed")
    assert originals == {path: path.read_bytes() for path in source_exports.rglob("*") if path.is_file()}


@pytest.mark.parametrize("problem", [
    "partial", "empty", "duplicate", "unknown_window", "start", "end",
    "n_windows", "empty_index", "duplicate_index", "missing_index_window",
])
def test_cli_rejects_records_that_do_not_match_complete_index(source_exports, tmp_path, problem):
    records_path = source_exports / "records" / "vid" / "records.jsonld"
    index_path = source_exports / "derived" / "windows.json"
    document = json.loads(records_path.read_text())
    index = json.loads(index_path.read_text())
    if problem == "partial":
        document["@graph"].pop()
    elif problem == "empty":
        document["@graph"] = []
    elif problem == "duplicate":
        document["@graph"].append(deepcopy(document["@graph"][0]))
    elif problem == "unknown_window":
        document["@graph"][0]["window_id"] = "vid:unknown"
    elif problem in ("start", "end"):
        document["@graph"][0][f"{problem}_s"] += 0.001
    elif problem == "n_windows":
        index["videos"][0]["n_windows"] = 3
    elif problem == "empty_index":
        index["windows"] = []
        index["videos"][0]["n_windows"] = 0
        document["@graph"] = []
    elif problem == "duplicate_index":
        index["windows"].append(deepcopy(index["windows"][0]))
        index["videos"][0]["n_windows"] = 3
    elif problem == "missing_index_window":
        index["windows"].pop()
        index["videos"][0]["n_windows"] = 1
    records_path.write_text(json.dumps(document))
    index_path.write_text(json.dumps(index))
    originals = {path: path.read_bytes() for path in source_exports.rglob("*") if path.is_file()}
    destination = tmp_path / "incomplete.json"
    assert cli.main(["export-expressions", "--video", "vid", "--output", str(destination)]) == 2
    assert not destination.exists()
    assert originals == {path: path.read_bytes() for path in source_exports.rglob("*") if path.is_file()}
