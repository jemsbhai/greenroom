# Team tasks (Saturday afternoon onward)

Muntaser owns the pipeline. Tasks 1 and 2 decide the Model Performance score (40%) and the demo, so they come first. Everything else fits around them. Teammate 1 is kushulreddy39, Teammate 2 is AP10-24.

## Ground rules, read before anything else

- The provided videos contain youth participants. They never leave your laptop. Never upload them, or anything made from them (clips, frames, audio, transcripts, label files), to any website, chat tool, shared drive, or AI service, and never commit them to git.
- Gold label files (`data\gold\<yourname>.jsonl`) are participant-level annotations: hand them to Muntaser by USB stick or a private direct message, never through the repo.
- Recording video in the venue is not permitted, so there is no team-recorded footage. Anything that has to be shown or shared publicly (demo recording, screenshots, sample output files, DevPost page) uses openly licensed public footage of adult performers (Task 2).
- The only cloud compute the rules allow is private Hugging Face compute, and we use it only on the public footage, never on the provided videos.
- Never write a name, nickname, or identifying description of a participant anywhere, and never write a judgment about how a participant feels.

## Setup on your own laptop (about 30 minutes, Windows)

1. Install Python 3.12 from python.org (tick "Add python.exe to PATH") and Git for Windows.
2. Accept the GitHub invitation to the greenroom repository, then clone it and install:

   ```powershell
   git clone https://github.com/jemsbhai/greenroom.git
   cd greenroom
   python -m pip install -e ".[dev]"
   ```

3. Get the data folder from Muntaser on a USB stick (it holds `videos\` and the already cut `derived\` clips) and copy it somewhere outside the repository, for example `C:\ai4good\data`. Alternatively download the same provided videos from the organizers' link into `C:\ai4good\data\videos` (the 9 files Muntaser has: the final video, the intro, the two songwriting clips and the five promos) and let step 5 cut the clips.
4. Point the tools at the data folder, then open a new PowerShell window so the setting is picked up:

   ```powershell
   [Environment]::SetEnvironmentVariable("GREENROOM_DATA_DIR", "C:\ai4good\data", "User")
   ```

5. Check the setup and make sure the window index and review clips exist (instant if they came on the USB stick, about ten minutes otherwise):

   ```powershell
   cd greenroom
   python -m greenroom env
   python -m greenroom windows --subclips
   ```

6. Start the labeling app and enter your name in the sidebar:

   ```powershell
   python -m streamlit run app/gold_app.py --server.address 127.0.0.1
   ```

If step 5 complains about ffmpeg, run `python -m pip install imageio-ffmpeg` and try again. If anything else fails, send Muntaser the full error text.

## Task 1 (both): label every window

- 323 windows of 5 seconds across the 9 files (26.8 minutes of footage): final video 185, intro 35, songwriting 46 and 31, promos 26 in total. Expect 80 to 100 minutes of labeling. Do the final video first, then the intro and the songwriting clips, then the promos.
- Read `docs/ANNOTATION_GUIDELINE.md` first. Label alone; do not discuss individual windows with each other until you have both finished. The point is two independent opinions.
- Use the context toggle whenever a 5-second window is hard to read on its own, but label only what is inside the window.
- Keep a short list (window number and video) of cases where the guideline was unclear; that list becomes guideline version 1.1.
- When finished, send `data\gold\<yourname>.jsonl` to Muntaser (USB or private DM). He will produce the list of windows where you two disagree; you then review those together and agree on a final label. Muntaser takes no part in labeling or adjudication.

## Task 2 (both): source openly licensed public footage

This footage is our only material for the long-form test, the public demo, the sample output files, the T4 timing run and the DevPost page, because the provided videos can never appear in any of those.

What we need: 45 to 90 minutes in total, across 2 to 4 files, of adult performing-arts events that resemble a TPA session: recitals, open mics, poetry slams, rap or spoken-word showcases, dance rehearsals or classes, theater rehearsals, music or songwriting workshops, community talent shows. Prefer long uncut recordings that include dead time, transitions, applause and someone giving instructions, since those are what we need to test. 480p or better, English speech preferred.

Where to look, and the license rule:

- Internet Archive (archive.org): search terms such as "open mic", "poetry slam", "dance recital", "talent show", "theater rehearsal", "community theater", "music workshop"; set Media Type to Movies. On each item page, check the rights or license field. Accept only "Public Domain", "CC0", or "Creative Commons Attribution" (CC BY, CC BY-SA). Skip items that show no license or "All rights reserved". Download the MP4 derivative from the item page.
- Wikimedia Commons (commons.wikimedia.org): every file page states its license; the same accepted licenses apply. Look in categories such as videos of dance performances, poetry readings, concerts, theater.
- Do not download from YouTube or other streaming sites, even when a video is marked Creative Commons: their terms forbid downloading and we do not want that argument in the room.
- No footage of children, even when the license allows it. Adults only.
- Nothing from the provided TPA videos is ever mixed into this set.

For every file you keep, add a row to `docs/PUBLIC_FOOTAGE.md` (create it): title, source URL, license (exact wording from the page), author or uploader to credit, duration, what activities it contains. Put the files on Muntaser's laptop (USB, or download them directly there) under `data\public\videos`.

## Task 3 (Teammate 1, kushulreddy39): public image set for prompt tuning

The vision model is zero-shot: it scores each frame against written phrases such as "a person painting at an easel". Which phrasing works best is an empirical question, and we answer it on public images, not on the provided data.

- For each of the 12 activity labels in `docs/ANNOTATION_GUIDELINE.md`, collect 15 to 20 openly licensed photos (Wikimedia Commons, Pexels, Unsplash) into `prompt_images\<label_id>\` on your laptop, with varied settings and adult performers. Keep a text file listing each image's source URL and license.
- Muntaser will add `scripts/scout_prompts.py`, which scores candidate phrasings against your image set and ranks them per label. It runs on a Hugging Face Job with a T4 GPU using your $10 credit; the exact command comes with the script.

## Task 4 (Teammate 2, AP10-24): model card facts

- Open the Hugging Face model card of every model listed in `profiles/rtx4090.yaml` and `profiles/t4.yaml` and fill the table in `docs/SYSTEM_CARD.md`: license, parameter count, what the card says it was trained on, and any limitation or bias statement, each with a link to the card. No guessing: if the card does not say, write "not stated on the card".

## Task 5 (Teammate 1, after Task 3): data governance note

- Draft `docs/DATA_GOVERNANCE.md` from Section 9 of `docs/PLAN.md`: consent, privacy preservation, access control, retention and deletion, post-event custody, and the statement that no provided data left the team's laptops. Plain sentences, no marketing language.

## Later tasks that use the Hugging Face credits

These wait until the pipeline exists (Muntaser will say when):

- Run the `t4` profile end to end on the public footage as a private Hugging Face Job on a T4 and save the timing log. This is the measured "runs on a T4" number for the presentation (decision D11 in PLAN.md).
- Run the VLM indicator stage on the public footage in the cloud so the laptop stays free for the provided videos.
- Label a 200-window sample of the public footage in the gold app (Muntaser will build the index and the sample list), so the long-form results have a gold set of their own.
