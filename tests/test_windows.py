"""Pure unit tests. No media files, no models, no data directory required."""
from greenroom import taxonomy as T
from greenroom.config import is_local_ollama_model
from greenroom.windows import frame_indices_for_window, make_windows, merge_runs, slugify


def test_make_windows_exact_multiple():
    ws = make_windows(20.0, "v", "v.mp4", window_s=5.0)
    assert [(w.start_s, w.end_s) for w in ws] == [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 20.0)]
    assert ws[0].window_id == "v:00000"
    assert ws[-1].window_id == "v:00003"


def test_make_windows_long_tail_becomes_window():
    ws = make_windows(23.0, "v", "v.mp4", window_s=5.0, min_tail_s=2.0)
    assert len(ws) == 5
    assert (ws[-1].start_s, ws[-1].end_s) == (20.0, 23.0)


def test_make_windows_short_tail_merges():
    ws = make_windows(21.0, "v", "v.mp4", window_s=5.0, min_tail_s=2.0)
    assert len(ws) == 4
    assert ws[-1].end_s == 21.0


def test_make_windows_short_video_single_window():
    ws = make_windows(1.5, "v", "v.mp4", window_s=5.0)
    assert len(ws) == 1 and ws[0].end_s == 1.5


def test_slugify():
    assert slugify("THIS IS ME-intro.mp4") == "this-is-me-intro"
    assert slugify("this is me-songwriting-camp2-deshaun.mp4") == "this-is-me-songwriting-camp2-deshaun"


def test_frame_indices_for_window():
    # 1 fps sampling: frame k covers [k-1, k); window [5, 10) holds frames 6..10
    assert frame_indices_for_window(5.0, 10.0, fps=1.0, n_frames=10) == [6, 7, 8, 9, 10]
    picked = frame_indices_for_window(5.0, 10.0, fps=1.0, n_frames=2)
    assert len(picked) == 2 and all(6 <= k <= 10 for k in picked)


def test_merge_runs():
    windows = [{"video_id": "v", "window_id": f"v:{i}", "start_s": 5.0 * i, "end_s": 5.0 * (i + 1)} for i in range(4)]
    segs = merge_runs(["dance", "dance", "singing_vocal_performance", "singing_vocal_performance"], windows)
    assert [(s["label"], s["start_s"], s["end_s"]) for s in segs] == [("dance", 0.0, 10.0), ("singing_vocal_performance", 10.0, 20.0)]


def test_validate_label_set():
    assert T.validate_label_set("dance", ["singing_vocal_performance"]) == []
    assert T.validate_label_set("dance", ["dance"])
    assert T.validate_label_set("setup_idle_empty", ["dance"])
    assert T.validate_label_set("not_a_label", [])


def test_ollama_cloud_guard():
    assert is_local_ollama_model("qwen2.5:7b-instruct")
    assert not is_local_ollama_model("gpt-oss:120b-cloud")
    assert not is_local_ollama_model("deepseek-v4-flash:cloud")
