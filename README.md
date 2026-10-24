# greenroom

Staff-reviewed indexing of long-form youth arts video for The Performance Academy.

AI4Good Hackathon 2026 (Florida Data Science for Social Good and UNF, Jacksonville, October 23 to 25, 2026).
Problem Case 1: The Performance Academy Video Intelligence Challenge. Track A: Automated AI Video Indexing and Review Support.

Team: TODO names and contact emails.

## What it does

A local, open-source pipeline cuts a long recording into 5-second windows, scores each window for performing-arts and creative-practice activities from three independent evidence channels (audio events, transcript, visual tags), fuses the channels into Subjective Logic opinions so every label carries belief, disbelief and uncertainty, routes uncertain or conflicting windows to a review queue, and writes one annotation record per window in the schema the organizers specified. Staff accept, edit, reject or defer each suggestion in a review app; every decision is sealed into a tamper-evident ledger. The system never identifies individuals, never stores face crops or embeddings, and never asserts an emotional state.

Design and decision log: [docs/PLAN.md](docs/PLAN.md). Label definitions: [docs/ANNOTATION_GUIDELINE.md](docs/ANNOTATION_GUIDELINE.md).

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
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"          # installs into the system Python; torch (CUDA build) is already there

# check the environment, build the window index, cut review subclips
python -m greenroom env
python -m greenroom windows --subclips
python -m pytest -q

# gold labeling app (human labels, no model output on screen)
python -m streamlit run app/gold_app.py --server.address 127.0.0.1

# pipeline (Phase 1: audio channel), records land under <data>/records/<video_id>/
python -m pip install -e ".[ml]"
python -m greenroom run --video this-is-me-songwriting-camp2-reese
python -m greenroom run --all --profile rtx4090
```

The `ml` extra (`python -m pip install -e ".[ml]"`) installs the model dependencies for the pipeline stages; `torch` is expected to be installed already (CUDA build). No virtual environment is used on the development machine; pins are captured with `pip freeze > requirements.lock` once the pipeline runs.

## Repository layout

```
app/            Streamlit apps: gold labeling (now), staff review (Phase 3)
docs/           PLAN.md, ANNOTATION_GUIDELINE.md, later SYSTEM_CARD.md, DATA_GOVERNANCE.md, ARCHITECTURE.md
profiles/       rtx4090.yaml (development), t4.yaml (deployable target)
samples/        sample output files generated from team-recorded footage only
src/greenroom/  the package
tests/          unit tests that need no media and no models
```

## License

MIT.
