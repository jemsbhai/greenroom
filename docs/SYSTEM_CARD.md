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

- Every channel's score for a label becomes a binomial Subjective Logic opinion: positive evidence r = E_pos * q * p and negative evidence s = E_neg * q * (1 - p), with belief r / (r + s + 2), disbelief s / (r + s + 2), uncertainty 2 / (r + s + 2). q is an input-quality factor (audio signal above the noise floor, frame brightness, word count); a channel that saw nothing contributes a vacuous opinion.
- Positive and negative weights differ per channel (profiles/*.yaml): 10 and 4 for the zero-shot channels, 10 and 1.5 for the supervised classifiers whose label sets barely overlap ours, 5 and 2 for the transcript model. These values are placeholders chosen on the clip-level sanity checks below, not fitted; they are reported here for that reason.
- Softmax channels are normalized against their own leader and against chance (score mode relative_top) so a runner-up at a third of the leader lands below the presence floor.
- Opinions are fused per label with cumulative fusion; a pairwise belief conflict mass is computed alongside. A window is routed to "staff review required" when the primary's uncertainty is 0.5 or more, when conflict is 0.4 or more, or when no label reaches a projected probability of 0.5; it is marked mixed when a second activity also clears 0.5 within a 0.1 margin.
- The vision channels give singing and rap equal support (a camera cannot separate them) and the transcript channel does the same (words cannot either); only the audio channels decide between the two.

## Evaluation summary (run at commit c01741b, rtx4090 profile, all five channels)

No hand-labeled ground truth exists for this footage, so the numbers are of four kinds, each stated for what it is. All come from `greenroom report`.

- Corpus: 9 provided files, 26.8 minutes, 323 windows of 5 s.
- File-name weak labels (the organizers' own names for the single-activity clips): the dominant primary label agrees with the file name on 5 of 6 clips (both songwriting clips, the dance, rap and singing promos; 31 of 31, 45 of 46, 5 of 7, 3 of 4 and 2 of 2 windows carry an expected label). The painting promo disagrees: a song with lyrics plays over painting footage, two audio-side channels hear a performance, and visual arts is primary on 2 of 9 windows. This is the system's clearest known failure mode (Section Known limitations).
- Triage: 56% of windows confident, 28% mixed, 16% review required overall; on the 15-minute final video 48%, 31%, 21%.
- Staff review: 4 decisions by the pipeline author (all accepted), median 28 s per decision. Too few to report an agreement rate; the mechanism and the ledger are what is demonstrated.
- Inter-channel diagnostics: each channel's own top label matches the fused primary on 57% (audio), 49% (X-CLIP), 33% (AST), 27% (transcript, over 211 windows with speech) and 25% (VideoMAE) of windows; the fused primary is supported by all five channels on 118 windows, by one channel only on 29. Mean uncertainty of the primary 0.11, mean conflict 0.28, conflict at or above 0.4 on 16% of windows.
- Throughput and energy on the RTX 4090 laptop, sustained run with all channels: 2.6 video-minutes per wall-minute, 118 kJ of GPU energy per hour of video (NVML, Ollama included). Single short clips run at 4 to 5 video-minutes per wall-minute; the sustained number is lower because the laptop GPU throttles thermally during a long run. The t4 profile has not yet been measured on a T4.

## Known limitations

- Zero-shot and web-trained models; youth arts footage in rooms with mixed lighting is out of distribution for all of them.
- A song with lyrics over montage footage is heard as a performance by CLAP, AST and the transcript model; only the vision channels can say the picture shows painting or sewing. Staff review catches it; the system cannot, and the painting promo shows it.
- Acting/theater is supported mostly by X-CLIP alone (no audio class corresponds to it), so talking heads can read as monologue; the transcript channel reduces this (acting fell from 4.9 to 3.6 minutes when it was added) but does not remove it.
- Singing and rap flip between adjacent windows inside vocal segments because only the audio channels can tell them apart and they are not always sure; segment-level smoothing is not implemented.
- Facilitation comes from words only; silent instruction (pointing, demonstrating) is missed.
- Edited promotional clips contain fast cuts and music beds; a 5-second window can span several shots.
- Evidence weights and thresholds are unfitted placeholders; the review ledger is the data they should be fitted on.
- Expression suggestions: see Section 3 of docs/DATA_GOVERNANCE.md for the face-expression stage and its limits; the transcript-based suggestions are weak by design.

## Privacy and bias risks

- Zero-shot vision and audio models perform unevenly across skin tones, body types, accents and dialects; nothing here measures that, so it is named as untested rather than absent.
- Face expression classifiers are trained on posed, mostly adult faces and are known to perform unevenly across skin tones, ages and lighting; an expression is not an emotion and performers act. Their output is a suggestion behind a toggle, never a claim.
- Transcripts of participants' own words are sensitive narrative content; they stay on the processing machine and are never exported in samples or shown to any external model.
- No identity is computed anywhere; records are about windows of time, not people.
