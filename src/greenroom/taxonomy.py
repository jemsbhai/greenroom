"""Label taxonomy and guideline version.

The guideline text lives in docs/ANNOTATION_GUIDELINE.md. The version string
below is written into every gold label and every annotation record; bump it
whenever the guideline changes meaning.
"""
from __future__ import annotations

GUIDELINE_VERSION = "1.0"

# (label_id, display name). Activities are multi-label at the window level.
ACTIVITY_LABELS: list[tuple[str, str]] = [
    ("singing_vocal_performance", "Singing / vocal performance"),
    ("songwriting", "Songwriting (writing or working out a song)"),
    ("lyricism_spoken_word_rap", "Lyricism / spoken word / rap"),
    ("dance", "Dance"),
    ("acting_theater", "Acting / theater"),
    ("film_media_production", "Film / media production (camera, interview, editing)"),
    ("recording_arts_audio_production", "Recording arts / audio production"),
    ("podcast_production", "Podcast production"),
    ("visual_arts", "Visual arts (painting, drawing, design work)"),
    ("fashion_design", "Fashion design"),
    ("modeling_runway", "Modeling / runway"),
    ("group_performance_or_showcase", "Group performance / showcase piece"),
]

# Exactly one of these may be primary when no activity is present in the window.
NON_ACTIVITY_LABELS: list[tuple[str, str]] = [
    ("instruction_facilitation", "Instruction / facilitation (no participant activity in view)"),
    ("audience_applause_transition", "Audience / applause / transition"),
    ("setup_idle_empty", "Setup / idle / empty room"),
]

PRIMARY_CHOICES: list[tuple[str, str]] = ACTIVITY_LABELS + NON_ACTIVITY_LABELS

ACTIVITY_IDS = [label for label, _ in ACTIVITY_LABELS]
NON_ACTIVITY_IDS = [label for label, _ in NON_ACTIVITY_LABELS]
ALL_LABEL_IDS = ACTIVITY_IDS + NON_ACTIVITY_IDS
DISPLAY_NAME = dict(PRIMARY_CHOICES)

# Staff-only contextual observation vocabulary from the brief. Never a gold
# label, never an automated claim; shown in the review app behind a toggle.
EXPRESSION_LABELS = ["Confident", "Upset", "Positive", "Neutral", "Negative"]

# Words the observable-indicator generator must not emit (post-filter, logged).
AFFECT_BLOCKLIST = {
    "happy", "sad", "angry", "anxious", "nervous", "confident", "upset", "depressed", "excited", "scared",
    "afraid", "joyful", "joy", "frustrated", "bored", "shy", "proud", "ashamed", "lonely", "distressed",
    "trauma", "traumatized", "emotional", "emotion", "mood", "feeling", "feels", "felt", "engaged",
    "disengaged", "enthusiastic", "passionate", "insecure", "hesitant", "comfortable", "uncomfortable",
}


def validate_label_set(primary: str, secondary: list[str]) -> list[str]:
    """Return a list of problems (empty when the combination is valid)."""
    problems: list[str] = []
    if primary not in ALL_LABEL_IDS:
        problems.append(f"unknown primary label {primary!r}")
    for s in secondary:
        if s not in ALL_LABEL_IDS:
            problems.append(f"unknown secondary label {s!r}")
        if s == primary:
            problems.append("a label cannot be both primary and secondary")
    if primary in NON_ACTIVITY_IDS and any(s in ACTIVITY_IDS for s in secondary):
        problems.append("when an activity is present it must be the primary label")
    return problems
