"""Hash-chained review ledger: append, latest-wins, verify, tamper detection. Uses a temp data dir."""
import json

import pytest

from greenroom import ledger


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    (tmp_path / "videos").mkdir()
    monkeypatch.setenv("GREENROOM_DATA_DIR", str(tmp_path))
    return tmp_path


def _window(i: int) -> dict:
    return {"window_id": f"vid:{i:05d}", "video_id": "vid", "index": i, "start_s": 5.0 * i, "end_s": 5.0 * (i + 1)}


def _ai(primary: str) -> dict:
    return {"primary_activity": primary, "secondary_activities": [], "review_status": "ai_suggested", "confidence": 0.7, "provenance": {"git_sha": "abc"}}


def test_append_verify_and_latest_wins(data_dir):
    e1 = ledger.append_decision("vid", _window(0), "Sam", "accepted", _ai("dance"), "dance", [], None, None, None, "", 3.2)
    e2 = ledger.append_decision("vid", _window(1), "Sam", "rejected", _ai("dance"), None, [], None, None, None, "not dance", 2.0)
    e3 = ledger.append_decision("vid", _window(0), "Sam", "edited", _ai("dance"), "singing_vocal_performance", ["dance"], True, "", "Positive", "", 9.0)
    assert e1["prev_hash"] == ledger.GENESIS and e2["prev_hash"] == e1["hash"] and e3["prev_hash"] == e2["hash"]
    report = ledger.verify_chain("vid")
    assert report["ok"] and report["entries"] == 3
    latest = ledger.latest_decisions("vid")
    assert latest["vid:00000"]["decision"] == "edited"
    assert latest["vid:00000"]["final_primary"] == "singing_vocal_performance"
    assert latest["vid:00000"]["final_secondary"] == ["dance"]
    assert latest["vid:00001"]["decision"] == "rejected"


def test_tampering_is_detected(data_dir):
    ledger.append_decision("vid", _window(0), "Sam", "accepted", _ai("dance"), "dance", [], None, None, None, "", 1.0)
    ledger.append_decision("vid", _window(1), "Sam", "accepted", _ai("dance"), "dance", [], None, None, None, "", 1.0)
    path = ledger.reviews_path("vid")
    lines = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["decision"] = "rejected"  # edit history without recomputing the hash
    lines[0] = json.dumps(first, ensure_ascii=False)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = ledger.verify_chain("vid")
    assert not report["ok"]
    assert any("line 0" in p for p in report["problems"])


def test_bad_decision_rejected(data_dir):
    with pytest.raises(ValueError):
        ledger.append_decision("vid", _window(0), "Sam", "maybe", _ai("dance"), "dance", [], None, None, None, "", None)
