"""Exercise review interactions with synthetic records and intercepted ledger writes."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from greenroom import ledger, media, taxonomy
from greenroom.expressions import WindowExpression
from greenroom.fuse import Decision
from greenroom.opinions import vacuous
from greenroom.records import build_record


EXPRESSION_TOGGLE = "Show unverified AI expression suggestion"
STAFF_EXPRESSION = "Staff-only contextual expression (optional; never an automated claim)"


def _element(elements, label):
    return next(element for element in elements if element.label == label)


def _healthy(app):
    assert not app.exception, [item.message for item in app.exception]
    assert not app.error, [item.value for item in app.error]


@pytest.fixture
def review_app(tmp_path, monkeypatch):
    (tmp_path / "videos").mkdir()
    (tmp_path / "derived").mkdir()
    placeholder = tmp_path / "videos" / "synthetic.mp4"
    placeholder.write_bytes(b"Synthetic media fixture; no decoding or inference")
    index = {"videos": [], "windows": []}
    original_records = {}
    for video_id, count in (("synthetic-a", 2), ("synthetic-b", 1)):
        index["videos"].append(
            {"video_id": video_id, "source_file": placeholder.name, "duration_s": 5.0 * count, "n_windows": count}
        )
        records = []
        for i in range(count):
            window = {
                "window_id": f"{video_id}:{i:05d}", "video_id": video_id,
                "index": i, "start_s": i * 5.0, "end_s": (i + 1) * 5.0,
            }
            index["windows"].append(window)
            suggestion = None
            if video_id == "synthetic-a" and i == 0:
                suggestion = WindowExpression(
                    window["window_id"], 3, 2, 5, {"positive": 0.8, "neutral": 0.1, "negative": 0.1}
                ).suggestion("synthetic/model")
            records.append(build_record(
                window,
                {label: vacuous(0.1) for label in taxonomy.ALL_LABEL_IDS},
                {label: 0.0 for label in taxonomy.ALL_LABEL_IDS},
                Decision("dance", [], "ai_suggested", False, 0.1, 0.0, []),
                {}, {"model_ids": ["synthetic/model"]}, expression_suggestion=suggestion,
            ))
        path = tmp_path / "records" / video_id / "records.jsonld"
        path.parent.mkdir(parents=True)
        original_records[path] = json.dumps({"@graph": records})
        path.write_text(original_records[path], encoding="utf-8")
    (tmp_path / "derived" / "windows.json").write_text(json.dumps(index), encoding="utf-8")
    monkeypatch.setenv("GREENROOM_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(media, "cut_subclip", lambda *args, **kwargs: placeholder)
    append_decision = Mock(return_value={})
    monkeypatch.setattr(ledger, "append_decision", append_decision)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "review_app.py"), default_timeout=20).run()
    _healthy(app)
    _element(app.text_input, "Reviewer name").set_value("Synthetic Reviewer").run()
    _healthy(app)
    yield SimpleNamespace(app=app, append_decision=append_decision)
    assert not list(tmp_path.rglob("reviews.jsonl"))
    assert all(path.read_text(encoding="utf-8") == original for path, original in original_records.items())


def test_expression_visibility_persists_across_windows_and_videos(review_app):
    app = review_app.app
    assert _element(app.radio, "Queue").value == "Everything"
    assert _element(app.sidebar.toggle, EXPRESSION_TOGGLE).value is False
    assert not app.warning
    assert _element(app.selectbox, STAFF_EXPRESSION).value == "none"
    assert any("Expression suggestions available: 1/2 windows." in item.value for item in app.sidebar.caption)

    _element(app.sidebar.toggle, EXPRESSION_TOGGLE).set_value(True).run()
    _healthy(app)
    warning = app.warning[0].value
    assert "Positive (share 0.80, 3 face readings)" in warning
    assert "Unverified model output" in warning
    assert "not a statement about how anyone feels" in warning
    assert "Use the Edit form" in warning
    assert _element(app.selectbox, STAFF_EXPRESSION).value == "none"

    _element(app.button, "Next").click().run()
    _healthy(app)
    assert "5 s to 10 s" in app.subheader[0].value
    assert _element(app.sidebar.toggle, EXPRESSION_TOGGLE).value is True
    assert not app.warning
    assert any("No expression suggestion is available for this window." in item.value for item in app.info)

    _element(app.button, "Previous").click().run()
    _healthy(app)
    assert "0 s to 5 s" in app.subheader[0].value
    assert _element(app.sidebar.toggle, EXPRESSION_TOGGLE).value is True
    assert "Positive" in app.warning[0].value

    _element(app.sidebar.selectbox, "Video").select("synthetic-b").run()
    _healthy(app)
    assert _element(app.sidebar.toggle, EXPRESSION_TOGGLE).value is True
    assert any("Expression suggestions available: 0/1 windows." in item.value for item in app.sidebar.caption)
    assert any("No expression suggestion is available for this window." in item.value for item in app.info)
    assert _element(app.selectbox, STAFF_EXPRESSION).value == "none"
    review_app.append_decision.assert_not_called()


def test_expression_control_remains_available_when_queue_is_empty(review_app):
    app = review_app.app
    _element(app.sidebar.radio, "Queue").set_value("Review required").run()
    _healthy(app)
    assert any("Nothing in this queue" in item.value for item in app.success)
    assert _element(app.sidebar.toggle, EXPRESSION_TOGGLE).value is False
    _element(app.sidebar.toggle, EXPRESSION_TOGGLE).set_value(True).run()
    _healthy(app)
    _element(app.sidebar.radio, "Queue").set_value("Everything").run()
    _healthy(app)
    assert _element(app.sidebar.toggle, EXPRESSION_TOGGLE).value is True
    assert "Positive" in app.warning[0].value
    review_app.append_decision.assert_not_called()


def test_accept_does_not_adopt_suggestion_or_unsaved_staff_expression(review_app):
    app = review_app.app
    _element(app.sidebar.toggle, EXPRESSION_TOGGLE).set_value(True).run()
    _healthy(app)
    assert _element(app.selectbox, STAFF_EXPRESSION).value == "none"
    _element(app.selectbox, STAFF_EXPRESSION).select("Positive").run()
    _healthy(app)
    review_app.append_decision.assert_not_called()

    _element(app.button, "Accept").click().run()
    _healthy(app)
    review_app.append_decision.assert_called_once()
    args = review_app.append_decision.call_args.args
    assert args[3] == "accepted"
    assert args[8] is None  # No staff contextual observation was submitted.
    assert args[9] is None  # The AI suggestion and unsaved Edit form are not accepted.
