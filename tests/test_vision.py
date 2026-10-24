"""Pure tests for the vision channel plumbing. No models, no media."""
from greenroom import taxonomy as T
from greenroom.vision_tags import XCLIP_PROMPTS, _pick_exactly, map_kinetics_classes


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
    }
    mapping = map_kinetics_classes(id2label)
    assert mapping[0] == ("dance",) and mapping[1] == ("dance",)
    assert mapping[2] == ("singing_vocal_performance",)
    assert mapping[3] == ("audience_applause_transition",)
    assert mapping[4] == ("visual_arts",)
    assert mapping[5] == ("songwriting", "visual_arts")
    assert mapping[6] == ("recording_arts_audio_production",)
    assert mapping[9] == ("fashion_design",)
    assert 7 not in mapping and 8 not in mapping


def test_pick_exactly_spreads_and_repeats():
    # linspace(0, 9, 5) = 0, 2.25, 4.5, 6.75, 9; Python rounds 4.5 to 4
    assert _pick_exactly([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5) == [1, 3, 5, 8, 10]
    assert _pick_exactly([4, 5], 4) == [4, 4, 5, 5]
    assert _pick_exactly([], 4) == []
