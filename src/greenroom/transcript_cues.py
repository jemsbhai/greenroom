"""Transcript evidence channel: a local instruct model reads the words of a window.

The model receives the transcribed words of a 5-second window plus context on
either side and returns strict JSON: which activities the words support, which
it rules out, whether someone is facilitating (directing, teaching, giving
feedback), and what kind of speech it is (performance, conversation,
instruction, interview, narration). It is told to judge only what the words
show and never to describe anyone's feelings, mood or character.

Served by Ollama on localhost. Any cloud-routed model tag is refused before a
single word is sent (config.assert_local_ollama_model). Transcript text never
leaves the machine and is never written into records or run metadata.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from . import config
from . import taxonomy as T

SPEECH_KINDS = ("performance", "conversation", "instruction", "interview", "narration", "unclear")

LABEL_GUIDE = "\n".join(
    [
        "singing_vocal_performance: sung lyrics, a melody being sung",
        "songwriting: writing, revising or discussing lyrics or a melody in progress; trying lines",
        "lyricism_spoken_word_rap: rapped or spoken-word lyrics performed with rhythm",
        "dance: counting steps, choreography talk, music cues for dancing",
        "acting_theater: scripted dialogue, lines delivered in character, scene direction",
        "film_media_production: filming, camera, shots, interview questions and answers to camera",
        "recording_arts_audio_production: recording takes, booth, mix, beat, track, headphones talk",
        "podcast_production: hosted conversation for an audience, episode and segment talk",
        "visual_arts: painting, drawing, colors, canvas, sketch talk",
        "fashion_design: fabric, sewing, garments, fitting, design sketches",
        "modeling_runway: runway, walk, poses, photo shoot talk",
        "group_performance_or_showcase: an ensemble piece, many performers at once",
        "instruction_facilitation: an instructor or peer directing, teaching or giving feedback",
        "audience_applause_transition: host or MC between acts, introductions, thank-yous, applause",
        "setup_idle_empty: logistics, waiting, unrelated chat, nothing happening",
    ]
)

SYSTEM_PROMPT = f"""You classify what is happening in a 5-second window of a video from a youth performing-arts program, using only the transcribed speech you are given. Judge only what the words show.

Labels and what the words would look like:
{LABEL_GUIDE}

Return strict JSON with exactly these keys:
{{"activities": [{{"label": "<label_id>", "confidence": <0.0 to 1.0>}}],
 "ruled_out": ["<label_id>", ...],
 "facilitation": <true or false>,
 "facilitation_confidence": <0.0 to 1.0>,
 "speech_kind": "<one of: performance, conversation, instruction, interview, narration, unclear>",
 "rationale": "<one short sentence about what kind of speech it is; do not quote the words>"}}

Rules:
- A sung, rapped or recited lyric is a performance. Text cannot tell sung delivery from rapped delivery, so for lyrics give singing_vocal_performance and lyricism_spoken_word_rap the same confidence. Directive language such as "from the top", "again", "try it with", "everyone" is facilitation.
- An interview or testimonial answer spoken to camera is speech_kind "interview" and supports film_media_production, not acting_theater.
- Voice-over narration over footage is speech_kind "narration" and supports no activity by itself.
- Put a label in ruled_out only when the words make it clearly inapplicable.
- Never describe or guess anyone's feelings, mood, confidence, character or state. Never use names and never quote the transcript in the rationale. If the words are too few to tell, return an empty activities list and speech_kind "unclear"."""


@dataclass
class WindowCue:
    window_id: str
    word_count: int
    quality: float
    label_scores: dict[str, float] = field(default_factory=dict)
    facilitation: bool | None = None
    facilitation_confidence: float = 0.0
    speech_kind: str = "unclear"
    rationale: str = ""
    raw_ok: bool = True

    def to_dict(self) -> dict:
        return {
            "window_id": self.window_id,
            "word_count": self.word_count,
            "quality": round(self.quality, 4),
            "label_scores": {k: round(v, 4) for k, v in self.label_scores.items()},
            "facilitation": self.facilitation,
            "facilitation_confidence": round(self.facilitation_confidence, 4),
            "speech_kind": self.speech_kind,
            "rationale": self.rationale,
            "parsed": self.raw_ok,
        }


def _clamp(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if v < 0 else 1.0 if v > 1 else v


def parse_cue_json(text: str) -> dict | None:
    """Extract the first JSON object from a model reply; None if there is none."""
    if not text:
        return None
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def cue_from_reply(window_id: str, word_count: int, quality: float, data: dict | None) -> WindowCue:
    cue = WindowCue(window_id=window_id, word_count=word_count, quality=quality)
    if data is None:
        cue.raw_ok = False
        cue.quality = 0.0
        return cue
    scores: dict[str, float] = {}
    for item in data.get("activities") or []:
        if not isinstance(item, dict):
            continue
        lid = str(item.get("label", "")).strip()
        if lid in T.ALL_LABEL_IDS:
            scores[lid] = max(scores.get(lid, 0.0), _clamp(item.get("confidence", 0.0)))
    for lid in data.get("ruled_out") or []:
        lid = str(lid).strip()
        if lid in T.ALL_LABEL_IDS and lid not in scores:
            scores[lid] = 0.0
    facilitation = data.get("facilitation")
    cue.facilitation = bool(facilitation) if isinstance(facilitation, bool) else None
    cue.facilitation_confidence = _clamp(data.get("facilitation_confidence", 0.0))
    if cue.facilitation is True:
        scores["instruction_facilitation"] = max(scores.get("instruction_facilitation", 0.0), cue.facilitation_confidence)
    kind = str(data.get("speech_kind", "unclear")).strip().lower()
    cue.speech_kind = kind if kind in SPEECH_KINDS else "unclear"
    if cue.speech_kind == "interview":
        scores["film_media_production"] = max(scores.get("film_media_production", 0.0), 0.5)
    # Words cannot tell sung from rapped delivery: a lyric supports both labels equally and CLAP decides.
    vocal = [lid for lid in ("singing_vocal_performance", "lyricism_spoken_word_rap") if lid in scores]
    if vocal:
        best = max(scores[lid] for lid in vocal)
        scores["singing_vocal_performance"] = best
        scores["lyricism_spoken_word_rap"] = best
    cue.rationale = str(data.get("rationale", ""))[:240]
    cue.label_scores = scores
    return cue


class OllamaCueModel:
    def __init__(self, model: str, host: str = "http://127.0.0.1:11434", temperature: float = 0.0, timeout_s: float = 120.0):
        import requests

        self.model = config.assert_local_ollama_model(model)
        self.host = host.rstrip("/")
        self.temperature = float(temperature)
        self.timeout_s = float(timeout_s)
        self._requests = requests
        tags = requests.get(f"{self.host}/api/tags", timeout=10).json()
        names = {m.get("name", "") for m in tags.get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            raise RuntimeError(f"Ollama model {self.model!r} is not pulled locally. Run: ollama pull {self.model}")

    @property
    def model_id(self) -> str:
        return f"ollama:{self.model}"

    def ask(self, window_text: str, before: str, after: str) -> str:
        user = (
            f"Window text: {window_text or '(no words)'}\n"
            f"Context before: {before or '(none)'}\n"
            f"Context after: {after or '(none)'}\n"
            "Return the JSON only."
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature, "num_predict": 300},
            "keep_alive": "10m",
        }
        reply = self._requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout_s)
        reply.raise_for_status()
        return (reply.json().get("message") or {}).get("content", "")


def score_windows(
    model: OllamaCueModel,
    transcript: dict,
    windows: list[dict],
    context_pad_s: float = 20.0,
    min_words: int = 3,
    words_for_full_quality: int = 10,
    no_speech_threshold: float = 0.6,
    log=None,
) -> tuple[list[WindowCue], dict]:
    from .transcribe import text_of, words_in

    t0 = time.perf_counter()
    cues: list[WindowCue] = []
    asked = 0
    failed = 0
    for i, w in enumerate(windows):
        own = words_in(transcript, w["start_s"], w["end_s"], no_speech_threshold)
        if len(own) < min_words:
            cues.append(WindowCue(w["window_id"], len(own), 0.0))
            continue
        before = words_in(transcript, max(0.0, w["start_s"] - context_pad_s), w["start_s"], no_speech_threshold)
        after = words_in(transcript, w["end_s"], w["end_s"] + context_pad_s, no_speech_threshold)
        quality = min(1.0, len(own) / float(words_for_full_quality))
        try:
            reply = model.ask(text_of(own), text_of(before), text_of(after))
            data = parse_cue_json(reply)
        except Exception:  # noqa: BLE001
            data = None
        asked += 1
        cue = cue_from_reply(w["window_id"], len(own), quality, data)
        if not cue.raw_ok:
            failed += 1
        cues.append(cue)
        if log and asked % 25 == 0:
            log(f"  transcript cues: {asked} windows asked ({i + 1}/{len(windows)})")
    meta = {
        "model_id": model.model_id,
        "windows_with_speech": asked,
        "windows_without_speech": len(windows) - asked,
        "parse_failures": failed,
        "context_pad_seconds": context_pad_s,
        "wall_seconds": round(time.perf_counter() - t0, 3),
    }
    return cues, meta
