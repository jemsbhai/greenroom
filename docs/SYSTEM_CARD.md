# greenroom system card (draft, to be completed in Phase 5)

## Purpose and intended use

greenroom indexes recordings of The Performance Academy programs into 5-second windows labeled with performing-arts and creative-practice activities, with explicit uncertainty, so that authorized TPA staff can find, review and document program moments faster. Every model output is a suggestion; the record of what happened in a session is the set of staff decisions.

## Out of scope, by design

- Identifying any person. No face detection for identity, no face embeddings, no voiceprints, no identity clusters, no per-person statistics.
- Any claim about a participant's emotional state, well-being, risk, development or character. The five expression words (Confident, Upset, Positive, Neutral, Negative) are offered only as unverified text-derived suggestions that a staff member may accept or discard; they are never aggregated unless staff-accepted.
- Ranking participants or sessions.

## Models (fill from the Hugging Face model cards; no guesses)

| Stage | Model id | License | Parameters | Training data as stated on the card | Limitations or bias statements on the card | Link |
|---|---|---|---|---|---|---|
| ASR (rtx4090) | Systran/faster-whisper-large-v3 (converted openai/whisper-large-v3) | | | | | |
| ASR (t4) | Systran/faster-distil-whisper-large-v3 (converted distil-whisper/distil-large-v3) | | | | | |
| Audio events | laion/larger_clap_music_and_speech | | | | | |
| Audio events (fallback) | laion/clap-htsat-unfused | | | | | |
| Audio events (second channel) | MIT/ast-finetuned-audioset-10-10-0.4593 | | | | | |
| Vision tags (rtx4090) | google/siglip-so400m-patch14-384 | | | | | |
| Vision tags (t4) | google/siglip-base-patch16-224 | | | | | |
| Person count | facebook/detr-resnet-50 | | | | | |
| Transcript cues (rtx4090) | Qwen/Qwen2.5-7B-Instruct (served by Ollama as qwen2.5:7b-instruct) | | | | | |
| Transcript cues (t4) | Qwen/Qwen2.5-3B-Instruct (served by Ollama as qwen2.5:3b-instruct) | | | | | |
| Observable indicators (rtx4090) | Qwen/Qwen2.5-VL-7B-Instruct | | | | | |
| Observable indicators (t4) | Qwen/Qwen2.5-VL-3B-Instruct | | | | | |

## Uncertainty handling

To be written from PLAN.md Section 6 once the calibration fit exists: how scores become Subjective Logic opinions, how modalities are fused, what triggers "staff review required", and the fitted thresholds with the size of the split they were fitted on.

## Evaluation summary

To be filled from the Phase 4 results: gold set size and inter-annotator agreement, per-label precision/recall/F1 with cluster-bootstrap confidence intervals, confusion matrix, calibration (ECE, Brier), risk-coverage results, ablations, throughput and energy per hour of video per profile. Numbers only from the evaluation scripts; nothing typed by hand.

## Known limitations (to be completed)

- All models are zero-shot and were trained on web data; youth arts footage shot in rooms with mixed lighting is out of distribution for them.
- Edited promotional clips contain fast cuts and music beds; a 5-second window can span several shots.
- Facilitation is detected from words only; silent instruction (pointing, demonstrating) is missed.
- Expression suggestions come from transcript text alone and are weak by design.

## Privacy and bias risks (to be completed)

- Zero-shot vision and audio models may perform unevenly across skin tones, body types, accents and dialects; this is reported where the gold set allows and otherwise named as untested.
- Transcripts of participants' own words are sensitive narrative content; they stay on the processing machine and are never exported in samples.
