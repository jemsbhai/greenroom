"""Transcript-cue parsing and window word selection. No models, no Ollama."""
from greenroom.transcribe import text_of, words_in
from greenroom.transcript_cues import cue_from_reply, parse_cue_json


def test_parse_cue_json_tolerates_prose_around_the_object():
    reply = 'Sure. {"activities": [{"label": "songwriting", "confidence": 0.8}], "ruled_out": ["dance"], "facilitation": true, "facilitation_confidence": 0.7, "speech_kind": "conversation", "rationale": "talk about a verse"} done'
    data = parse_cue_json(reply)
    assert data["speech_kind"] == "conversation"
    assert parse_cue_json("no json here") is None
    assert parse_cue_json("") is None


def test_cue_from_reply_maps_fields():
    data = {
        "activities": [{"label": "songwriting", "confidence": 0.8}, {"label": "not_a_label", "confidence": 0.9}],
        "ruled_out": ["dance", "songwriting"],
        "facilitation": True,
        "facilitation_confidence": 0.7,
        "speech_kind": "interview",
        "rationale": "x" * 500,
    }
    cue = cue_from_reply("v:00001", 12, 1.0, data)
    assert cue.label_scores["songwriting"] == 0.8  # ruled_out does not override a positive score
    assert cue.label_scores["dance"] == 0.0
    assert "not_a_label" not in cue.label_scores
    assert cue.facilitation is True and cue.label_scores["instruction_facilitation"] == 0.7
    assert cue.speech_kind == "interview" and cue.label_scores["film_media_production"] >= 0.5
    assert len(cue.rationale) <= 240


def test_cue_from_reply_gives_sung_and_rapped_lyrics_equal_support():
    data = {"activities": [{"label": "lyricism_spoken_word_rap", "confidence": 1.0}], "ruled_out": ["singing_vocal_performance"], "facilitation": False, "facilitation_confidence": 0.0, "speech_kind": "performance", "rationale": ""}
    cue = cue_from_reply("v:00003", 12, 1.0, data)
    assert cue.label_scores["singing_vocal_performance"] == cue.label_scores["lyricism_spoken_word_rap"] == 1.0


def test_cue_from_reply_handles_parse_failure():
    cue = cue_from_reply("v:00002", 15, 1.0, None)
    assert cue.raw_ok is False and cue.quality == 0.0 and cue.label_scores == {}


def test_words_in_window_and_no_speech_filter():
    transcript = {
        "segments": [
            {"start": 0.0, "end": 4.0, "no_speech_prob": 0.1, "words": [{"w": "from", "s": 0.5, "e": 0.8, "p": 0.9}, {"w": "the", "s": 0.9, "e": 1.0, "p": 0.9}, {"w": "top", "s": 1.1, "e": 1.4, "p": 0.9}]},
            {"start": 4.5, "end": 9.0, "no_speech_prob": 0.9, "words": [{"w": "hallucinated", "s": 5.0, "e": 6.0, "p": 0.3}]},
            {"start": 9.0, "end": 12.0, "no_speech_prob": 0.05, "words": [{"w": "again", "s": 9.5, "e": 9.9, "p": 0.9}]},
        ]
    }
    first = words_in(transcript, 0.0, 5.0)
    assert text_of(first) == "from the top"
    assert words_in(transcript, 5.0, 10.0) == [{"w": "again", "s": 9.5, "e": 9.9, "p": 0.9}]
