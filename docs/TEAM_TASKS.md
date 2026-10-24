# Team tasks (Saturday afternoon onward)

Muntaser owns the pipeline. The two tasks below marked "both" are the ones that decide the Model Performance score (40%) and the long-form demo, so they come first. Everything else fits around them.

## Ground rules, read before anything else

- The provided videos contain youth participants. They never leave your laptop. Never upload them, or anything made from them (clips, frames, audio, transcripts, label files), to any website, chat tool, shared drive, or AI service, and never commit them to git.
- Gold label files (`data\gold\<yourname>.jsonl`) are participant-level annotations: hand them to Muntaser by USB stick or a private direct message, never through the repo.
- The only cloud compute the rules allow is private Hugging Face compute, and we use it only on footage we record ourselves (Task 2). Never on the provided videos.
- Never write a name, nickname, or identifying description of a participant anywhere, and never write a judgment about how a participant feels.

## Setup on your own laptop (about 30 minutes, Windows)

1. Install Python 3.12 from python.org (tick "Add python.exe to PATH") and Git for Windows.
2. Clone the repository (URL from Muntaser), then in PowerShell:

   ```powershell
   cd greenroom
   python -m pip install -e ".[dev]"
   ```

3. Download the 8 provided videos from the organizers' link into a folder outside the repository, for example `C:\ai4good\data\videos` (create the `videos` folder; the file names stay as they are).
4. Point the tools at the data folder, then open a new PowerShell window so the setting is picked up:

   ```powershell
   [Environment]::SetEnvironmentVariable("GREENROOM_DATA_DIR", "C:\ai4good\data", "User")
   ```

5. Check the setup, build the window index, and cut the 138 review clips (a few minutes):

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

- 138 windows of 5 seconds across the 8 files, about 35 to 45 minutes.
- Read `docs/ANNOTATION_GUIDELINE.md` first. Label alone; do not discuss individual windows with each other until you have both finished. The point is two independent opinions.
- Use the context toggle whenever a 5-second window is hard to read on its own, but label only what is inside the window.
- Keep a short list (window number and video) of cases where the guideline was unclear; that list becomes guideline version 1.1.
- When finished, send `data\gold\<yourname>.jsonl` to Muntaser (USB or private DM). He will produce the list of windows where you two disagree; you then review those together and agree on a final label. Muntaser takes no part in labeling or adjudication.

## Task 2 (both): record the long-form footage

This footage is our only material for the long-form demo, the public sample files, the T4 timing run and the DevPost page, because the provided videos can never appear in any of those.

- 45 to 60 minutes, one continuous take, phone on a tripod or propped on a table, landscape, no cuts, no editing. Record somewhere in the venue where nobody else walks into frame; the two of you are the only people who may appear.
- Fill the hour with what a TPA session contains, in blocks of a few minutes each, in any order: someone sings a song (quality does not matter), someone raps or does spoken word over a beat played from a phone, a short dance, one of you "teaches" the other with corrections and "again from the top", a seated conversation like a podcast, someone sketches or paints, applause, and two or three stretches of nothing at all (empty frame, moving chairs, chatting about lunch). Change distance to the camera and lighting between blocks.
- Copy the file to Muntaser's laptop by USB. It goes into `data\team\videos`, separate from the provided data.

## Task 3 (Teammate A): public image set for prompt tuning

The vision model is zero-shot: it scores each frame against written phrases such as "a person painting at an easel". Which phrasing works best is an empirical question, and we answer it on public images, not on the provided data.

- For each of the 12 activity labels in `docs/ANNOTATION_GUIDELINE.md`, collect 15 to 20 openly licensed photos (Wikimedia Commons, Pexels, Unsplash) into `prompt_images\<label_id>\` on your laptop, with varied settings and performers of different ages. Keep a text file listing each image's source URL.
- Muntaser will add `scripts/scout_prompts.py`, which scores candidate phrasings against your image set and ranks them per label. It runs on a Hugging Face Job with a T4 GPU using your $10 credit; the exact command comes with the script. Public images only; nothing from the provided videos goes near it.

## Task 4 (Teammate B): model card facts

- Open the Hugging Face model card of every model listed in `profiles/rtx4090.yaml` and `profiles/t4.yaml` and fill the table in `docs/SYSTEM_CARD.md`: model id, license, parameter count, what the card says it was trained on, and any limitation or bias statement, each with a link to the card. No guessing: if the card does not say, write "not stated on the card".

## Task 5 (Teammate A, after Task 3): data governance note

- Draft `docs/DATA_GOVERNANCE.md` from Section 9 of `docs/PLAN.md`: consent, privacy preservation, access control, retention and deletion, post-event custody, and the statement that no provided data left the laptop. Plain sentences, no marketing language.

## Later tasks that use the Hugging Face credits

These wait until the pipeline exists (Muntaser will say when):

- Run the `t4` profile end to end on the team footage as a private Hugging Face Job on a T4, and save the timing log. This is the measured "runs on a T4" number for the presentation (decision D11 in PLAN.md).
- Run the VLM indicator stage on the team footage in the cloud so the laptop stays free for the provided videos.
