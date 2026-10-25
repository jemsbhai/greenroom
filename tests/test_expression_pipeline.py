"""Expression suggestions stay separate from activity evidence. No models or media."""
import csv
import json

import pytest

from greenroom import expressions, pipeline
from greenroom.expressions import WindowExpression


def test_expression_stage_saves_window_suggestions_without_activity_evidence(tmp_path, monkeypatch):
    (tmp_path / "videos").mkdir()
    monkeypatch.setenv("GREENROOM_DATA_DIR", str(tmp_path))
    windows = [
        {"window_id": f"vid:{i:05d}", "video_id": "vid", "index": i, "start_s": 5.0 * i, "end_s": 5.0 * (i + 1)}
        for i in range(2)
    ]
    scores = [
        WindowExpression("vid:00001", 3, 2, 4, {"positive": 0.8, "neutral": 0.1, "negative": 0.1}),
        WindowExpression("vid:00000", 2, 1, 4, {"positive": 0.1, "neutral": 0.8, "negative": 0.1}),
    ]

    class FakeTagger:
        def __init__(self, model_id, fallback_id, *, device, min_face_px):
            self.model_id = fallback_id

        def score_windows(self, frames_dir, fps, received_windows, *, max_frames_per_window):
            assert frames_dir == tmp_path / "derived" / "frames" / "vid_expr"
            assert fps == 2.0 and max_frames_per_window == 4
            assert received_windows == windows
            return scores, {"model_id": self.model_id, "face_readings_total": 5, "windows_with_suggestion": 2}

    def fake_sample_frames(source, out_dir, *, fps, height):
        assert source == tmp_path / "videos" / "vid.mp4"
        assert out_dir == tmp_path / "derived" / "frames" / "vid_expr"
        assert fps == 2.0 and height == 540
        return [out_dir / f"{i:06d}.jpg" for i in range(1, 21)]

    monkeypatch.setattr(expressions, "ExpressionTagger", FakeTagger)
    monkeypatch.setattr(pipeline, "sample_frames", fake_sample_frames)
    monkeypatch.setattr(pipeline, "_release_gpu", lambda: None)
    profile = {
        "name": "test",
        "device": "cpu",
        "energy": {"nvml": False},
        "expression": {
            "enabled": True,
            "model": "test/main",
            "fallback_model": "test/fallback",
            "sample_fps": 2.0,
            "frame_height": 540,
            "max_frames_per_window": 4,
            "min_face_readings": 3,
        },
    }
    index = {"videos": [{"video_id": "vid", "source_file": "vid.mp4", "duration_s": 10.0}], "windows": windows}

    pipeline.run_videos(["vid"], profile, stages=("expression",), index=index, log=lambda _: None)

    out_dir = tmp_path / "records" / "vid"
    records = json.loads((out_dir / "records.jsonld").read_text(encoding="utf-8"))["@graph"]
    assert records[0]["expression_suggestion"] is None
    assert records[1]["expression_suggestion"] == scores[0].suggestion("test/fallback", 3)
    assert all(record["opinions_by_modality"] == {} for record in records)
    assert all(record["staff_contextual_observation"] is None for record in records)
    assert all(record["provenance"]["model_ids"] == ["test/fallback"] for record in records)
    meta = json.loads((out_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["channels"] == []
    assert meta["stages"] == ["expression"]
    assert meta["stage_meta"]["expression"]["windows_with_suggestion"] == 1
    assert meta["stage_meta"]["expression"]["face_readings_total"] == 5
    assert meta["stage_meta"]["sample_frames_expression"]["n_frames"] == 20
    assert meta["channel_outputs"]["expression"] == [score.to_dict() for score in scores]
    with (out_dir / "records.csv").open(encoding="utf-8", newline="") as handle:
        assert [row["expression_suggestion"] for row in csv.DictReader(handle)] == ["", "Positive"]


@pytest.mark.parametrize(
    ("profile", "stages"),
    [
        ({}, ("expression",)),
        ({"expression": {"enabled": False, "model": "test/main"}}, ("expression",)),
        ({"expression": {"enabled": True}}, ("expression",)),
        ({"expression": {"enabled": True, "model": "test/main"}}, ()),
    ],
)
def test_expression_model_is_not_loaded_when_not_requested_or_configured(monkeypatch, profile, stages):
    def unexpected_load(*args, **kwargs):
        pytest.fail("ExpressionTagger must not load without a requested, enabled, configured stage")

    monkeypatch.setattr(expressions, "ExpressionTagger", unexpected_load)
    assert pipeline._load_models(profile, stages, False, lambda _: None) == {}
