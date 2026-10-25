"""CPU routing and batching without model downloads or accelerator access."""
import json
import sys
from types import SimpleNamespace

import pytest

from greenroom import audio_events, config, expressions, pipeline, transcribe, transcript_cues, vision_tags


def test_cpu_loaders_pin_all_models_and_ollama_to_cpu(monkeypatch):
    profile = config.load_profile("cpu")
    calls = {}
    thread_counts = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=thread_counts.append))
    monkeypatch.setattr(pipeline.StageMeter, "_init_nvml", lambda _: pytest.fail("CPU must never initialize NVML"))

    def constructor(channel):
        def build(*args, **kwargs):
            calls[channel] = (args, kwargs)
            return SimpleNamespace(model_id=args[0], device=kwargs.get("device"), num_frames=8,
                                   resolved_mapping=lambda: {})
        return build

    monkeypatch.setattr(audio_events, "AudioEventScorer", constructor("audio"))
    monkeypatch.setattr(audio_events, "AudioSetScorer", lambda *a, **k: pytest.fail("CPU profile disables AST"))
    monkeypatch.setattr(vision_tags, "XClipScorer", constructor("xclip"))
    monkeypatch.setattr(vision_tags, "VideoMaeScorer", constructor("videomae"))
    monkeypatch.setattr(transcribe, "Transcriber", constructor("asr"))
    monkeypatch.setattr(transcript_cues, "OllamaCueModel", constructor("cues"))
    monkeypatch.setattr(expressions, "ExpressionTagger", constructor("expression"))
    # A caller or custom profile cannot accidentally enable GPU work for device=cpu.
    profile["energy"]["nvml"] = True
    profile["transcript_llm"]["options"]["num_gpu"] = 99
    models = pipeline._load_models(profile, pipeline.STAGES, True, lambda _: None)

    assert set(models) == {"audio", "xclip", "videomae", "asr", "cues", "expression"}
    assert all(calls[name][1]["device"] == "cpu" for name in models if name != "cues")
    assert calls["asr"] == (("small.en", "int8"), {"device": "cpu", "cpu_threads": 4})
    assert calls["cues"][0] == ("qwen2.5:1.5b-instruct",)
    assert calls["cues"][1]["options"] == {"num_gpu": 0, "num_thread": 4}
    assert calls["cues"][1]["timeout_s"] == 600
    assert thread_counts == [4]


def test_existing_gpu_profile_keeps_asr_and_ollama_defaults(monkeypatch):
    calls = {}

    def build_asr(*args, **kwargs):
        calls["asr"] = args, kwargs
        return SimpleNamespace(model_id="asr")

    def build_cues(*args, **kwargs):
        calls["cues"] = args, kwargs
        return SimpleNamespace(model_id="cues")

    monkeypatch.setattr(transcribe, "Transcriber", build_asr)
    monkeypatch.setattr(transcript_cues, "OllamaCueModel", build_cues)
    pipeline._load_models(config.load_profile("rtx4090"), ("transcript",), False, lambda _: None)
    assert calls["asr"] == (("large-v3", "float16"), {"device": "cuda"})
    assert calls["cues"] == (("qwen2.5:7b-instruct",), {"host": "http://127.0.0.1:11434", "temperature": 0.0})


def test_faster_whisper_receives_cpu_int8_and_thread_limit(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(
        WhisperModel=lambda *args, **kwargs: calls.append((args, kwargs))))
    model = transcribe.Transcriber("small.en", "int8", device="cpu", cpu_threads=4)
    assert calls == [(("small.en",), {"device": "cpu", "compute_type": "int8", "cpu_threads": 4})]
    assert model.model_id == "faster-whisper:small.en:int8"


@pytest.mark.parametrize("options", [None, {"num_gpu": 0, "num_thread": 4}])
def test_ollama_request_uses_cpu_options_only_when_requested(monkeypatch, options):
    import requests

    posts = []
    monkeypatch.setattr(requests, "get", lambda *a, **k: SimpleNamespace(
        json=lambda: {"models": [{"name": "local-test"}]}))

    def post(url, **kwargs):
        posts.append((url, kwargs))
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"message": {"content": "{}"}})

    monkeypatch.setattr(requests, "post", post)
    model = transcript_cues.OllamaCueModel("local-test", options=options, timeout_s=600)
    assert model.ask("synthetic words only", "", "") == "{}"
    assert posts[0][0] == "http://127.0.0.1:11434/api/chat"
    assert posts[0][1]["json"]["options"] == {"temperature": 0.0, "num_predict": 300, **(options or {})}
    assert posts[0][1]["timeout"] == 600


def test_cpu_run_and_refresh_keep_timestamps_and_never_touch_gpu(tmp_path, monkeypatch):
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / "synthetic.mp4").touch()
    monkeypatch.setenv("GREENROOM_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(pipeline.StageMeter, "_init_nvml", lambda _: pytest.fail("CPU must never initialize NVML"))
    monkeypatch.setattr(pipeline, "_release_gpu", lambda: pytest.fail("CPU must not touch CUDA"))
    monkeypatch.setattr(pipeline, "extract_audio", lambda source, dest: dest)
    frame_calls = []
    monkeypatch.setattr(pipeline, "sample_frames", lambda source, dest, **kwargs: frame_calls.append((dest, kwargs)) or [])
    windows = [{"window_id": "synthetic:00000", "video_id": "synthetic", "index": 0, "start_s": 0.0, "end_s": 5.0}]
    index = {"videos": [{"video_id": "synthetic", "source_file": "synthetic.mp4", "duration_s": 5.0}], "windows": windows}
    calls = []

    class Scorer:
        def __init__(self, channel):
            self.model_id = channel
            self.device = "cpu"
            self.channel = channel

        def score_windows(self, *args, **kwargs):
            calls.append((self.channel, kwargs))
            if self.channel == "expression":
                return [expressions.WindowExpression("synthetic:00000", 2, 2, 5,
                        {"positive": 0.8, "neutral": 0.1, "negative": 0.1})], {"device": "cpu"}
            score = SimpleNamespace(window_id="synthetic:00000", label_scores={"dance": 0.8},
                                    quality=1.0, class_probs={"dance": 0.8}, to_dict=lambda: {"synthetic": True})
            return [score], {"device": "cpu"}

        def resolved_mapping(self):
            return {}

    def load(profile, stages, want_gpu, log):
        assert want_gpu is False
        names = ["expression"] if stages == ("expression",) else ["audio", "ast", "xclip", "videomae", "expression"]
        return {name: Scorer(name) for name in names}

    monkeypatch.setattr(pipeline, "_load_models", load)
    profile = config.load_profile("cpu")
    profile["energy"]["nvml"] = True
    pipeline.run_videos(["synthetic"], profile, index=index, log=lambda _: None)

    assert dict(calls)["audio"]["batch_size"] == dict(calls)["ast"]["batch_size"] == 1
    assert dict(calls)["xclip"]["batch_size"] == dict(calls)["videomae"]["batch_size"] == 1
    assert dict(calls)["expression"]["batch_faces"] == 4
    assert frame_calls[0][0].name == "synthetic_cpu_4fps_224px"
    record_path = tmp_path / "records" / "synthetic" / "records.jsonld"
    before = json.loads(record_path.read_text())["@graph"][0]
    assert before["start_s"] == 0 and before["end_s"] == 5
    assert before["expression_suggestion"]["label"] == "Positive"
    summaries = pipeline.tag_expressions(["synthetic"], profile, index=index, log=lambda _: None)
    assert summaries[0]["windows_with_suggestion"] == 1
    after = json.loads(record_path.read_text())["@graph"][0]
    assert {k: v for k, v in before.items() if k != "expression_suggestion"} == {
        k: v for k, v in after.items() if k != "expression_suggestion"}
    for stage in summaries[0]["stage_meta"].values():
        assert stage["gpu_energy_measured"] is False
