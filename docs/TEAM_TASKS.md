# Team tasks (Saturday afternoon onward)

Muntaser owns the pipeline. Nobody on the team hand-labels video: labeling is part of the product, done by staff in the review app. The tasks below are all desk work. Teammate 1 is kushulreddy39, Teammate 2 is AP10-24.

## Ground rules, read before anything else

- The provided videos contain youth participants. They stay on Muntaser's laptop. Never upload them, or anything made from them (clips, frames, audio, transcripts, label files), to any website, chat tool, shared drive, or AI service, and never commit them to git.
- Recording video in the venue is not permitted, so there is no team-recorded footage. Anything that has to be shown or shared publicly (demo recording, screenshots, sample output files, DevPost page) uses openly licensed public footage of adult performers (Task 1).
- The only cloud compute the rules allow is private Hugging Face compute, and we use it only on the public footage, never on the provided videos.
- Never write a name, nickname, or identifying description of a participant anywhere, and never write a judgment about how a participant feels.

## Task 1 (both, start now): source openly licensed public footage

This footage is our only material for the long-form test, the public demo, the sample output files, the T4 timing run and the DevPost page, because the provided videos can never appear in any of those.

What we need: 45 to 90 minutes in total, across 2 to 4 files, of adult performing-arts events that resemble a TPA session: recitals, open mics, poetry slams, rap or spoken-word showcases, dance rehearsals or classes, theater rehearsals, music or songwriting workshops, community talent shows. Prefer long uncut recordings that include dead time, transitions, applause and someone giving instructions, since those are what we need to test. 480p or better, English speech preferred.

Where to look, and the license rule:

- Internet Archive (archive.org): search terms such as "open mic", "poetry slam", "dance recital", "talent show", "theater rehearsal", "community theater", "music workshop"; set Media Type to Movies. On each item page, check the rights or license field. Accept only "Public Domain", "CC0", or "Creative Commons Attribution" (CC BY, CC BY-SA). Skip items that show no license or "All rights reserved". Download the MP4 derivative from the item page.
- Wikimedia Commons (commons.wikimedia.org): every file page states its license; the same accepted licenses apply. Look in categories such as videos of dance performances, poetry readings, concerts, theater.
- Do not download from YouTube or other streaming sites, even when a video is marked Creative Commons: their terms forbid downloading and we do not want that argument in the room.
- No footage of children, even when the license allows it. Adults only.
- Nothing from the provided TPA videos is ever mixed into this set.

For every file you keep, add a row to `docs/PUBLIC_FOOTAGE.md` (create it): title, source URL, license (exact wording from the page), author or uploader to credit, duration, what activities it contains. Put the files on Muntaser's laptop (USB, or download them directly there) under `data\public\videos`.

## Task 2 (Teammate 1, kushulreddy39): public image set for prompt tuning

The vision model is zero-shot: it scores each window against written phrases such as "a person painting on a canvas". Which phrasing works best is an empirical question, and we answer it on public images, not on the provided data.

- For each of the 15 labels in `docs/ANNOTATION_GUIDELINE.md` (12 activities plus instruction, applause/transition, setup/idle), collect 15 to 20 openly licensed photos (Wikimedia Commons, Pexels, Unsplash) into `prompt_images\<label_id>\` on your laptop, with varied settings and adult performers. Keep a text file listing each image's source URL and license.
- Muntaser will add `scripts/scout_prompts.py`, which scores candidate phrasings against your image set and ranks them per label. It runs on a Hugging Face Job with a T4 GPU using your $10 credit; the exact command comes with the script.

## Task 3 (Teammate 2, AP10-24): model card facts

- Open the Hugging Face model card of every model listed in `profiles/rtx4090.yaml` and `profiles/t4.yaml` (CLAP, X-CLIP, VideoMAE, faster-whisper, the Qwen models, DETR) and fill the table in `docs/SYSTEM_CARD.md`: license, parameter count, what the card says it was trained on, and any limitation or bias statement, each with a link to the card. No guessing: if the card does not say, write "not stated on the card".

## Task 4 (Teammate 1, after Task 2): data governance note

- Draft `docs/DATA_GOVERNANCE.md` from Section 9 of `docs/PLAN.md`: consent, privacy preservation, access control, retention and deletion, post-event custody, and the statement that no provided data left Muntaser's laptop. Plain sentences, no marketing language.

## Task 5 (Teammate 2): demo script, DevPost text, slides

- `docs/DEMO_SCRIPT.md`: the end-to-end walkthrough the brief asks for (a video enters the system, candidate moments appear with labels, a reviewer accepts, edits, rejects and defers, the record is exported), written against the public footage.
- DevPost page text and the slide deck in the organizers' template, with the Results slide left for Muntaser's numbers.

## Task 6 (either teammate, 30 minutes, when Muntaser says the review app is ready)

- Sit at Muntaser's laptop as "staff" and review suggestions on the provided videos: accept, edit, reject or defer, with the timer running. This is the staff-time measurement and the first review-based agreement numbers for the presentation. It is using the product, not labeling a dataset.

## Later tasks that use the Hugging Face credits

- Run the `t4` profile end to end on the public footage as a private Hugging Face Job on a T4 and save the timing log (decision D11 in PLAN.md).
- Run the heavier model stages on the public footage in the cloud so the laptop stays free for the provided videos.
