# Data governance note

greenroom, AI4Good Hackathon 2026, Problem Case 1 (The Performance Academy), Track A.

This note describes how the system and the team handle the video footage provided for the hackathon. The footage shows youth participants of The Performance Academy's 2025 "This Is Me" program. Everything below is a description of what the code does and what the team did, not a promise about the future; the sections marked "after the event" describe the commitments the team makes for post-event custody.

## 1. Where the data is and where it is not

- The provided videos were downloaded from the organizers' link to one team laptop (the pipeline owner's machine). Every model in the pipeline runs on that laptop. No provided file, and nothing derived from one (audio tracks, frames, clips, transcripts, label records, review decisions), has been sent to any website, cloud service, shared drive, chat tool, or external AI model.
- The data directory sits outside the code repository. The code refuses to run if the data directory is found inside the repository (config.data_dir), and the repository's .gitignore excludes data folders, media files and databases as a second fence. The repository holds code and documentation only; it is private during the event, with teammates and organizers as collaborators.
- Teammates who needed the data for setup received it on a USB stick or downloaded it from the organizers' link to their own laptops, with the same rules.
- The Hugging Face credit is used only on openly licensed public footage of adult performers (for the T4 timing run); never on the provided videos.
- Public artifacts (sample output files, screenshots, the demo recording, the DevPost page, the slides) use openly licensed public footage only, with attribution in docs/PUBLIC_FOOTAGE.md. The numbers computed from the provided videos appear on the slides; frames from them do not.

## 2. What the system computes, and what it never computes

- The unit of analysis is a 5-second window of a video. For each window the system records activity labels (singing, dance, songwriting, and so on), a confidence with explicit uncertainty, a review status, and provenance (model ids, pipeline version, git commit).
- Identity: the system never identifies anyone. It builds no face embeddings, no voiceprints, no identity registries and no tracks of a person across frames or windows. The person detector that supports participation counts keeps only the number of people in a frame. The hashed review ledger holds reviewer names (staff), never participant names.
- Transcripts: speech is transcribed locally so the system can recognize instruction, songwriting conversation and spoken word. Transcripts are the participants' own words and are treated as the most sensitive derived data: they stay on the processing laptop, are never exported in sample files, and are never shown to any external model. The pipeline refuses cloud-routed language models by name.
- Emotional and clinical claims: the system makes none. The five generalized expression words the organizers listed (Confident, Upset, Positive, Neutral, Negative) exist only as unverified suggestions that a staff member may accept, edit, reject or defer. They are hidden behind a toggle in the review app, carry the status ai_suggested_unverified, and never enter any count or report unless a staff member accepted or edited them. "Confident" is never produced by a model.
- No risk scores, no mental-health assessments, no rankings of participants or sessions, anywhere.

## 3. Face expression analysis: organizer clarification and design limits

The written brief prohibits biometric facial recognition and penalizes profiling of a youth participant's internal state. The team asked the organizers whether face-based expression analysis was permitted for the hackathon. The clarification, received by text message on October 24, 2026, from Dr. Karthikeyan Umapathy (Florida Data Science for Social Good), reads:

> Hi Muntaser,
>
> Just confirming that as long as no identifying data leaves your system, recognition and expression classification is allowed for the hackathon.
>
> - Karthikeyan

On that basis the team uses expression classification only. Identity recognition is not built and is not needed for the problem. The expression stage is designed so that the condition in the clarification holds by construction:

- Faces are detected with a classical frontal-face detector that has no learned identity features. Each detected region is classified in memory by an image model fine-tuned for facial expressions and then discarded.
- Nothing per face is written anywhere: no crop, no bounding box, no embedding, no track. The only stored output per 5-second window is an aggregate (the share of face readings the model placed in positive, neutral and negative classes, an "upset" share, and the number of readings) plus the model id.
- The output is a suggestion with status ai_suggested_unverified, shown in the review app behind a toggle, subject to staff decision like every other suggestion.
- Known limits, stated here and in the system card: expression classifiers are trained on posed, mostly adult faces; their performance varies with skin tone, age, lighting and camera angle; an expression is not an emotion; performers act. These limits are the reason the output is a suggestion to a staff member who knows the participant and the context, not a claim.
- The stage has a switch in the run profile and is off in the deployable (T4) profile by default.

## 4. Consent and the role of staff

- The organizers obtained the footage from The Performance Academy for the hackathon. The team did not contact participants and did not attempt to identify them.
- Nothing the system outputs becomes a record of a participant's development or state. Authorized TPA staff decide which suggestions become annotations, using their knowledge of the participant, the program and the session. The system's role is to shorten the time staff spend finding and reviewing moments.
- Any future use beyond the hackathon demonstration (training, reporting, publication) requires TPA's and the organizers' written permission. The team makes no such use.

## 5. Access control

- During the event: one laptop, one operating-system login, the pipeline owner present. Teammates who acted as reviewers did so at that laptop. The review app binds to 127.0.0.1 only and has no network exposure.
- The review ledger is append-only and hash-chained. Every decision carries the reviewer's name, the model's original suggestion, the final labels and a timestamp; `greenroom verify` recomputes every hash and reports any edit, deletion or reordering. Nothing is ever deleted from the ledger; a corrected decision is a new entry.
- After the event, any deployment at TPA would need accounts, encryption at rest and a retention policy set by TPA. This prototype does not provide them and is not to be exposed on a network.

## 6. Retention and deletion

- Derived artifacts (audio tracks, sampled frames, review clips, records) live under the data directory and are removed with `python -m greenroom purge` (scripts/purge_derived.py), which leaves the provided videos untouched.
- Records and ledgers are keyed by video id, so removing one recording removes everything derived from it, including review decisions.
- After the event: the team keeps its copies only as long as the organizers ask for post-event handover support, then deletes the provided videos, all derived artifacts and all records from every team machine and USB stick, and confirms the deletion to the organizers in writing. Hugging Face model weights and the public footage are not covered by this, since they contain no participant data.

## 7. Post-event custody and use restrictions

- The code (without data) may be handed to Florida Data Science for Social Good and The Performance Academy as the organizers request.
- The team will not reuse the footage, frames, audio, transcripts, records, review decisions or any numbers computed from them for portfolios, papers, marketing or other projects without explicit written permission from the organizers and TPA.
- Erasure under a withdrawn consent is supported today at the level of a whole recording (Section 6). A finer design, in which individual annotations can be erased from the hash-chained ledger while the chain stays verifiable, is described as future work in the system card and is not built.

## 8. Public footage

- Openly licensed footage of adult performing-arts events (public domain, CC0, CC BY, CC BY-SA), sourced from the Internet Archive and Wikimedia Commons with the license of each file recorded in docs/PUBLIC_FOOTAGE.md, is used for the long-form test, the public demo, the sample output files and the T4 timing run. No footage of children is included even where a license would allow it.

## 9. Contacts

- Pipeline owner and data custodian during the event: Muntaser Syed (jemsbhai@gmail.com).
- Organizer contact for clarifications: Dr. Karthikeyan Umapathy, Florida Data Science for Social Good.
