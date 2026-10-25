# greenroom

Staff-reviewed indexing of long-form youth arts video for The Performance Academy.

AI4Good Hackathon 2026 (Florida Data Science for Social Good and UNF, Jacksonville, October 23 to 25, 2026).
Problem Case 1: The Performance Academy Video Intelligence Challenge. Track A: Automated AI Video Indexing and Review Support.

Team: TODO names and contact emails.

## What it does

A local, open-source pipeline cuts a long recording into 5-second windows, scores each window for performing-arts and creative-practice activities from three independent evidence channels (audio events, transcript, visual tags), fuses the channels into Subjective Logic opinions so every label carries belief, disbelief and uncertainty, routes uncertain or conflicting windows to a review queue, and writes one annotation record per window in the schema the organizers specified. Staff accept, edit, reject or defer each suggestion in a review app; every decision is sealed into a tamper-evident ledger. The system never identifies individuals, never stores face crops or embeddings, and never asserts an emotional state.

Design and decision log: [docs/PLAN.md](docs/PLAN.md). Label definitions: [docs/ANNOTATION_GUIDELINE.md](docs/ANNOTATION_GUIDELINE.md).

For teammates integrating this pipeline into another system, start with [INTEGRATION_HANDOFF.md](INTEGRATION_HANDOFF.md): portable setup, local job APIs, the timestamped expression contract, export commands, and staff-review safeguards.

## Data rules (read before running anything)

- The provided videos contain youth participants. They never enter this repository, any public service, or any cloud model. All models run locally.
- The data directory is outside the repository and is named by the environment variable `GREENROOM_DATA_DIR`. It must contain a `videos/` folder. Everything derived (audio, subclips, indexes, gold labels, records, ledgers) is written under that directory.
- No face recognition, face embeddings, voiceprints or identity structures exist anywhere in this code base, and none may be added.
- Screenshots, demo recordings, sample output files and slides use footage recorded by the team of consenting adults, never the provided data.

## Quickstart (Windows, PowerShell)

```powershell
# one-time
[Environment]::SetEnvironmentVariable("GREENROOM_DATA_DIR", "E:\path\to\data", "User")   # folder that contains videos\
cd greenroom
python -m venv --system-site-packages .venv  # reuse the installed CUDA torch build
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

# check the environment, build the window index, cut review subclips
python -m greenroom env
python -m greenroom windows --subclips
python -m pytest -q

# gold labeling app (human labels, no model output on screen)
python -m streamlit run app/gold_app.py --server.address 127.0.0.1

# pipeline (audio, vision, transcript, expression), records land under <data>/records/<video_id>/
python -m pip install -e ".[ml]"
python -m greenroom run --video this-is-me-songwriting-camp2-reese
python -m greenroom run --all --profile rtx4090

# refresh only expression suggestions in existing records; preserve activities and reviews
python -m greenroom tag-expressions --all --profile rtx4090

# staff review app (accept, edit, reject, defer; hash-chained ledger) and ledger check
python -m streamlit run app/review_app.py --server.address 127.0.0.1
python -m greenroom verify
```

The `ml` extra (`python -m pip install -e ".[ml]"`) installs the model dependencies for the pipeline stages; `torch` is expected to be installed already (CUDA build). The project virtual environment reuses that installation. OpenCV is constrained to 4.x because the expression stage uses its bundled Haar `CascadeClassifier`; the base OpenCV 5 wheel no longer provides that API. If activation is unavailable, use `.\.venv\Scripts\python.exe` in place of `python`.

The RTX 4090 profile enables local expression suggestions. They are stored separately from activity evidence, remain unverified, and in **Review clips** are hidden until staff turn on **Show unverified AI expression suggestion** in the sidebar. This control stays available even when a window has no suggestion, and stays on while browsing windows and videos. The sidebar reports how many windows have suggestions; a window without one shows an explanation. The default **Everything** queue includes all activity statuses. Staff observations still require the Edit form.

Use `python -m greenroom tag-expressions --all` to refresh expressions for videos that already have pipeline records, or replace `--all` with `--video <id>`. This preserves activity predictions, speech, staff reviews and activity provenance. The refresh records its own model, configuration and time in each video's run metadata. Videos without records must first be processed with `python -m greenroom run --video <id>`.

To add new recordings in the review app, open **Add videos** in the sidebar, choose one or more files, and select **Add to processing queue**. MP4 (including `.mpeg4`), MOV, MKV, AVI, WEBM and M4V are supported, up to 2 GB per file. **Upload progress** updates while you review other videos; completed uploads appear automatically in the Video selector. Failed uploads show an error and **Retry processing**. New installations can upload before a window index or review records exist, provided the external data directory and its `videos/` folder are present.

Uploaded recordings stay in `<data>/videos/` and are analyzed locally with the RTX 4090 profile, including expressions. Silent videos use the vision and expression stages. The app validates each file and assigns a unique video ID without rebuilding existing windows or replacing recordings and staff reviews. Repeated uploads of the same file with the same profile reuse their existing job. Background jobs and logs live under `<data>/derived/uploads/`; processing continues when a browser tab is closed. Restart Streamlit after changing its upload limit in `.streamlit/config.toml`. The app requires the same model dependencies and local Ollama setup as the command-line pipeline.

For timestamped expression tags across a video, choose **Expression timeline** under **View**. It shows source-relative start/end times, keeps unverified AI suggestions separate from staff observations, supports label filtering and window playback, and offers JSON/CSV downloads without requiring a reviewer name. For integration, use `python -m greenroom export-expressions --video <id> --output <external-path>/timeline.json` or `--all --output <external-directory>`; add `--format csv` for CSV. Windows without an expression remain in the export, and accepting an activity never promotes an AI expression to a staff observation.

## Repository layout

```
app/            Streamlit apps: staff review (review_app.py), blind labeling (gold_app.py)
docs/           PLAN.md, ANNOTATION_GUIDELINE.md, later SYSTEM_CARD.md, DATA_GOVERNANCE.md, ARCHITECTURE.md
profiles/       rtx4090.yaml (development), t4.yaml (deployable target)
samples/        sample output files generated from openly licensed public footage only
src/greenroom/  the package
tests/          unit tests that need no media and no models
```

## License

MIT.
