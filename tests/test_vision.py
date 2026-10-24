"""Pure tests for the vision channel plumbing. No models, no media."""
from greenroom import taxonomy as T
from greenroom.audio_events import map_audioset_classes
from greenroom.vision_tags import XCLIP_PROMPTS, _pick_exactly, apply_visual_equivalence, map_kinetics_classes


def test_every_label_has_xclip_prompts():
    for lid in T.ALL_LABEL_IDS:
        assert lid in XCLIP_PROMPTS and len(XCLIP_PROMPTS[lid]) >= 2


def test_kinetics_mapping_rules():
    id2label = {
        0: "dancing ballet",
        1: "breakdancing",
        2: "singing",
        3: "applauding",
        4: "drawing",
        5: "writing",
        6: "playing drums",
        7: "eating spaghetti",
        8: "adjusting glasses",  # must not match the dj rule
        9: "sewing",
        10: "recording music",
    }
    mapping = map_kinetics_classes(id2label)
    assert mapping[0] == ("dance",) and mapping[1] == ("dance",)
    assert mapping[2] == ("singing_vocal_performance",)
    assert mapping[3] == ("audience_applause_transition",)
    assert mapping[4] == ("visual_arts",)
    assert mapping[5] == ("songwriting", "visual_arts")
    assert mapping[6] == ("recording_arts_audio_production",)
    assert mapping[9] == ("fashion_design",)
    assert mapping[10] == ("recording_arts_audio_production",)
    assert 7 not in mapping and 8 not in mapping


def test_visual_equivalence_shares_support_between_singing_and_rap():
    scores = {"singing_vocal_performance": 0.5, "lyricism_spoken_word_rap": 0.2, "dance": 0.1}
    out = apply_visual_equivalence(scores)
    assert out["singing_vocal_performance"] == out["lyricism_spoken_word_rap"]
    assert abs(out["singing_vocal_performance"] - 0.7) < 1e-9
    assert out["dance"] == 0.1
    # a channel that only scores singing (VideoMAE) passes that support to rap as well
    out2 = apply_visual_equivalence({"singing_vocal_performance": 0.3})
    assert out2["lyricism_spoken_word_rap"] == 0.3


def test_audioset_mapping_rules():
    id2label = {
        0: "Singing",
        1: "Choir",
        2: "Rapping",
        3: "Applause",
        4: "Cheering",
        5: "Silence",
        6: "Speech",
        7: "Music",
        8: "Electric guitar",
        9: "Female singing",
    }
    mapping = map_audioset_classes(id2label)
    # the parent class "Singing" covers rapping in the AudioSet ontology, so it is context only
    assert 0 not in mapping
    assert mapping[1] == ("singing_vocal_performance",)
    assert mapping[9] == ("singing_vocal_performance",)
    assert mapping[2] == ("lyricism_spoken_word_rap",)
    assert mapping[3] == ("audience_applause_transition",) and mapping[4] == ("audience_applause_transition",)
    assert mapping[5] == ("setup_idle_empty",)
    # speech, music and instruments are context only
    assert 6 not in mapping and 7 not in mapping and 8 not in mapping


def test_pick_exactly_spreads_and_repeats():
    # linspace(0, 9, 5) = 0, 2.25, 4.5, 6.75, 9; Python rounds 4.5 to 4
    assert _pick_exactly([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5) == [1, 3, 5, 8, 10]
    assert _pick_exactly([4, 5], 4) == [4, 4, 5, 5]
    assert _pick_exactly([], 4) == []
