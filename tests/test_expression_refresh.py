"""Expression refresh preserves activity predictions and human review artifacts."""
import csv
import json
from copy import deepcopy

import pytest

from greenroom import cli, pipeline
from greenroom.expressions import WindowExpression


@pytest.fixture
def existing_exports(tmp_path, monkeypatch):
    (tmp_path / "videos").mkdir()
    monkeypatch.setenv("GREENROOM_DATA_DIR", str(tmp_path))
    windows = [
        {"window_id": f"vid:{i:05d}", "video_id": "vid", "index": i, "start_s": 5 * i, "end_s": 5 * (i + 1)}
        for i in range(2)
    ]
    records = [
        {
            **window,
            "primary_activity": "singing_vocal_performance",
            "speech": {"word_count": 7, "rationale": "Synthetic speech fixture"},
            "review_status": "ai_suggested",
            "reviewer_decision": "accepted",
            "staff_contextual_observation": "Synthetic staff observation",
            "provenance": {"model_ids": ["activity/model"], "profile": "original", "generated_at": "original-time"},
            "opinions_by_modality": {"audio": {"synthetic": "unchanged"}},
            "expression_suggestion": {"label": "Old suggestion"},
        }
        for window in reversed(windows)
    ]
    out = tmp_path / "records" / "vid"
    out.mkdir(parents=True)
    document = {"@context": {"fixture": "keep"}, "@graph": records, "extension": "unchanged"}
    (out / "records.jsonld").write_text(json.dumps(document), encoding="utf-8")
    fields = ["window_id", "video_id", "start_s", "end_s", "primary_activity", "custom_column", "expression_suggestion"]
    with (out / "records.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for window in windows:
            writer.writerow({**{key: window[key] for key in fields[:4]}, "primary_activity": "singing_vocal_performance",
                             "custom_column": "preserve me", "expression_suggestion": "Old suggestion"})
    meta = {
        "video_id": "vid", "provenance": deepcopy(records[0]["provenance"]), "processing_wall_seconds": 100,
        "stages": ["audio"], "channels": ["audio"], "profile": "original",
        "stage_meta": {"audio": {"wall_seconds": 100}}, "channel_outputs": {"audio": [{"original": True}]},
    }
    (out / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (out / "segments.json").write_text('[{"label": "original segment"}]', encoding="utf-8")
    (out / "reviews.jsonl").write_text('{"synthetic": "never rewrite"}\n', encoding="utf-8")
    (tmp_path / "videos" / "vid.mp4").touch()
    index = {"videos": [{"video_id": "vid", "source_file": "vid.mp4", "duration_s": 10}], "windows": windows}
    profile = {"name": "test", "energy": {"nvml": False},
               "expression": {"enabled": True, "model": "configured/model", "min_face_readings": 3}}
    return tmp_path, out, index, profile


def fake_model(monkeypatch, scores):
    class Model:
        model_id = "actual/fallback"

        def score_windows(self, frames_dir, fps, windows, *, max_frames_per_window):
            assert frames_dir.is_dir()
            return scores, {"windows_with_suggestion": 99, "face_readings_total": 3, "model_id": self.model_id}

    def load(profile, stages, want_gpu, log):
        assert stages == ("expression",)
        return {"expression": Model()}

    monkeypatch.setattr(pipeline, "_load_models", load)
    monkeypatch.setattr(pipeline, "sample_frames", lambda *args, **kwargs: [])
    monkeypatch.setattr(pipeline, "_release_gpu", lambda: None)


def test_refresh_preserves_all_non_expression_data_and_clears_no_face_suggestion(existing_exports, monkeypatch):
    root, out, index, profile = existing_exports
    original = json.loads((out / "records.jsonld").read_text())
    original_meta = json.loads((out / "run_meta.json").read_text())
    with (out / "records.csv").open(newline="") as handle:
        original_csv = list(csv.DictReader(handle))
    protected = {name: (out / name).read_bytes() for name in ("segments.json", "reviews.jsonl")}
    scores = [WindowExpression("vid:00001", 0, 0, 5),
              WindowExpression("vid:00000", 3, 2, 5, {"positive": 0.8, "neutral": 0.1, "negative": 0.1})]
    fake_model(monkeypatch, scores)

    summaries = pipeline.tag_expressions(["vid"], profile, index=index, log=lambda _: None)

    refreshed = json.loads((out / "records.jsonld").read_text())
    assert refreshed["@context"] == original["@context"] and refreshed["extension"] == original["extension"]
    for before, after in zip(original["@graph"], refreshed["@graph"]):
        assert {k: v for k, v in before.items() if k != "expression_suggestion"} == {
            k: v for k, v in after.items() if k != "expression_suggestion"
        }
    assert refreshed["@graph"][0]["expression_suggestion"] is None
    assert refreshed["@graph"][1]["expression_suggestion"] == scores[1].suggestion("actual/fallback", 3)
    with (out / "records.csv").open(newline="") as handle:
        refreshed_csv = list(csv.DictReader(handle))
    assert [row.pop("expression_suggestion") for row in refreshed_csv] == ["Positive", ""]
    for row in original_csv:
        row.pop("expression_suggestion")
    assert refreshed_csv == original_csv
    meta = json.loads((out / "run_meta.json").read_text())
    for key in original_meta:
        if key not in ("stage_meta", "channel_outputs"):
            assert meta[key] == original_meta[key]
    assert meta["stage_meta"]["audio"] == original_meta["stage_meta"]["audio"]
    assert meta["channel_outputs"]["audio"] == original_meta["channel_outputs"]["audio"]
    assert meta["channel_outputs"]["expression"] == [score.to_dict() for score in scores]
    assert meta["stage_meta"]["expression"]["windows_with_suggestion"] == 1
    assert meta["expression_refresh"] == summaries[0]
    assert summaries[0]["provenance"]["model_ids"] == ["actual/fallback"]
    assert summaries[0]["provenance"]["profile"] == "test"
    assert summaries[0]["provenance"]["generated_at"]
    assert summaries[0]["expression_config"] == profile["expression"]
    assert not (root / "records" / "runs.jsonl").exists()
    assert not list((root / "derived" / "frames").iterdir())
    assert protected == {name: (out / name).read_bytes() for name in protected}


@pytest.mark.parametrize("problem", ["missing_file", "missing_window", "duplicate_window", "timestamp", "csv_timestamp"])
def test_refresh_refuses_missing_or_mismatched_exports_before_loading_models(existing_exports, monkeypatch, problem):
    _, out, index, profile = existing_exports
    path = out / "records.jsonld"
    doc = json.loads(path.read_text())
    if problem == "missing_file":
        path.unlink()
    elif problem == "csv_timestamp":
        csv_path = out / "records.csv"
        csv_path.write_text(csv_path.read_text().replace("vid:00000,vid,0,5", "vid:00000,vid,1,5"))
    else:
        if problem == "missing_window":
            doc["@graph"].pop()
        elif problem == "duplicate_window":
            doc["@graph"][1] = deepcopy(doc["@graph"][0])
        else:
            doc["@graph"][0]["end_s"] += 1
        path.write_text(json.dumps(doc))
    originals = {p.name: p.read_bytes() for p in out.iterdir()}
    monkeypatch.setattr(pipeline, "_load_models", lambda *args: pytest.fail("preflight must finish before model loading"))
    with pytest.raises(ValueError, match="Restore matching exports or run:"):
        pipeline.tag_expressions(["vid"], profile, index=index)
    assert originals == {p.name: p.read_bytes() for p in out.iterdir()}


def test_refresh_preflights_all_videos_before_updating_any(existing_exports, monkeypatch):
    _, out, index, profile = existing_exports
    index["videos"].append({"video_id": "missing", "source_file": "missing.mp4", "duration_s": 10})
    originals = {p.name: p.read_bytes() for p in out.iterdir()}
    monkeypatch.setattr(pipeline, "_load_models", lambda *args: pytest.fail("all selected exports must pass preflight"))
    with pytest.raises(ValueError, match="Cannot refresh expressions for missing"):
        pipeline.tag_expressions(["vid", "missing"], profile, index=index)
    assert originals == {p.name: p.read_bytes() for p in out.iterdir()}


@pytest.mark.parametrize("score_ids", [["vid:00000"], ["vid:00000", "vid:00000"], ["vid:00000", "other:00001"]])
def test_refresh_rejects_incomplete_or_misaligned_model_results(existing_exports, monkeypatch, score_ids):
    _, out, index, profile = existing_exports
    originals = {p.name: p.read_bytes() for p in out.iterdir()}
    fake_model(monkeypatch, [WindowExpression(wid, 0, 0, 5) for wid in score_ids])
    with pytest.raises(ValueError, match="expression scorer returned"):
        pipeline.tag_expressions(["vid"], profile, index=index, log=lambda _: None)
    assert originals == {p.name: p.read_bytes() for p in out.iterdir()}


@pytest.mark.parametrize(("selection", "expected"), [(["--all"], ["a", "b"]), (["--video", "b", "--video", "a"], ["b", "a"])])
def test_cli_selects_expression_refresh_only(monkeypatch, selection, expected):
    monkeypatch.setattr(cli.config, "load_profile", lambda name: {"name": name})
    monkeypatch.setattr(cli.config, "windows_index_path", lambda: "unused")
    monkeypatch.setattr(cli, "load_index", lambda path: {"videos": [{"video_id": "a"}, {"video_id": "b"}]})
    calls = []
    monkeypatch.setattr(pipeline, "tag_expressions", lambda videos, profile, index: calls.append((videos, profile)))
    assert cli.main(["tag-expressions", *selection]) == 0
    assert calls == [(expected, {"name": "rtx4090"})]
