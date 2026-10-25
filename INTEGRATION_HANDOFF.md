# Greenroom integration handoff

This file is for the teammate and AI agent integrating Greenroom into another local system. Greenroom imports videos, analyzes timestamped windows, lets staff review suggestions, and exports expression timelines. The integration boundary is the Python package or CLI plus local JSON/CSV files. There is no Greenroom REST API, webhook, shared database, or cloud service in this repository.

## Start here

1. Run the existing application and a synthetic clip before changing its behavior.
2. Use `greenroom.expression_timeline` or `greenroom export-expressions` to obtain the timestamped expression contract below.
3. Keep AI suggestions and staff observations in separate fields in the receiving system.
4. For new videos, use `greenroom.uploads.submit_upload` and its worker. Wait for the job to become `completed` before importing its outputs.
5. Preserve video IDs, window IDs, timestamps, original files, and the append-only review ledger. Add your adapter around these interfaces.

All provided participant media and derived data stay on the authorized local machine, outside this repository. Do not send videos, frames, face crops, transcripts, records, or review ledgers to a hosted model, public repository, external logging service, or a teammate's AI chat. Share code and synthetic examples through Git. Face recognition, identity embeddings, person tracking, and speaker identity inference are outside this implementation.

## Windows setup

Use Python 3.11 or newer. The current development profile targets an NVIDIA CUDA machine with approximately 16 GB of GPU memory; actual requirements depend on the selected stages and other GPU processes. Install a compatible CUDA PyTorch build for the target machine first. `torch` is deliberately not installed by the project extras. Do not replace an existing working CUDA build with a CPU wheel.

Run these commands from PowerShell, substituting your own repository and data paths. The data directory must be outside the repository. The `$env:` assignment affects this shell and processes started from it; persisting a user environment variable alone does not update an already open shell.

```powershell
$repo = "D:\projects\greenroom"
$env:GREENROOM_DATA_DIR = "D:\greenroom-data"
New-Item -ItemType Directory -Force -Path (Join-Path $env:GREENROOM_DATA_DIR "videos") | Out-Null
Set-Location $repo

# Use the Python installation that already has the appropriate CUDA torch build.
python -m venv --system-site-packages .venv
$python = Join-Path $repo ".venv\Scripts\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install -e ".[ml,dev]"
& $python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
& $python -c "import cv2; print(cv2.__version__, hasattr(cv2, 'CascadeClassifier'))"

# Optional persistence for newly opened terminals:
[Environment]::SetEnvironmentVariable("GREENROOM_DATA_DIR", $env:GREENROOM_DATA_DIR, "User")
```

`requirements.lock` is a broad snapshot of the original machine, including unrelated packages and an editable reference to another project. It is not a portable application lockfile. Install from `pyproject.toml` as above. Its dependency lower bounds are not a guarantee that every future dependency combination works; record a tested environment for your target. In particular, retain `opencv-python-headless>=4.8,<5`: the expression detector uses the OpenCV 4 bundled Haar cascade API. A clean environment avoids collisions among multiple OpenCV distributions. If you use a venv without system packages, install the correct CUDA PyTorch build inside it before the ML extra.

Install FFmpeg and ffprobe on PATH and install Ollama separately. `imageio-ffmpeg` provides a bundled FFmpeg fallback; ffprobe is optional, with metadata parsing as the fallback. For the full default pipeline, Ollama must serve the local model at `http://127.0.0.1:11434`:

```powershell
ollama pull qwen2.5:7b-instruct
ollama list
# If the Ollama desktop service is not already running, run `ollama serve`
# in a separate terminal. Do not start a second server on the same port.
& $python -m greenroom env
& $python -m pytest -q
& $python -m streamlit run app/review_app.py --server.address 127.0.0.1 --server.port 8501
```

First inference downloads model weights when absent from the local caches. Model download access is different from sending participant data to an inference service: inference remains local. Pre-cache models if the final system must run without network access. The environment check is diagnostic; a successful `env` command does not prove every model or Ollama is ready. Verify with a synthetic full run on the target machine.

`.streamlit/config.toml` binds the app to localhost, sets the upload limit to 2,048 MiB per file, and disables usage telemetry. Launch from the repository root to pick up this configuration. Restart Streamlit after changing server configuration. The app has no user authentication or deployment access-control layer; a teammate adding a network service must design that boundary explicitly before exposing participant data.

## Running and reviewing

For a new installation, open the sidebar's **Add videos**, select one or more recordings, and click **Add to processing queue**. Review mode and upload mode share the same external data directory. Jobs continue in a separate local process if the browser closes. Failed jobs expose **Retry processing**. The UI accepts MP4, MPEG4, MOV, MKV, AVI, WEBM, and M4V; the backend also checks that the file has a decodable video stream.

Select a processed video and set **View** to **Expression timeline** to see tags across the whole video. Each row carries source-relative start/end timestamps, the unverified AI expression, and any separate staff observation. Filter labels, select a timestamped window for playback, and download JSON or CSV. Timeline viewing does not require a reviewer name. **Review clips** requires a reviewer name before recording decisions. Use the Edit form to record a staff expression observation; accepting an activity suggestion alone does not verify its AI expression.

For initial bulk ingestion, copy authorized files into `<data>/videos/`, then build the index once:

```powershell
& $python -m greenroom windows --subclips
& $python -m greenroom run --all --profile rtx4090

# IDs come from derived/windows.json, not from a guessed filename conversion.
$index = Get-Content (Join-Path $env:GREENROOM_DATA_DIR "derived\windows.json") -Raw | ConvertFrom-Json
$index.videos | Select-Object video_id, source_file, n_windows
$videoId = $index.videos[0].video_id

# Subsequent expression-only refresh of existing records preserves activity data and staff reviews.
& $python -m greenroom tag-expressions --video $videoId --profile rtx4090
# Or: & $python -m greenroom tag-expressions --all --profile rtx4090

# Run a new silent video without loading audio/transcript models:
# & $python -m greenroom run --video $videoId --profile rtx4090 --stages vision,expression
& $python -m greenroom verify
```

`windows` rebuilds the index; it is not incremental intake. Once uploads or reviews exist, use the upload API for additions instead of rebuilding the index. `run` rewrites model records for its selected videos; do not use `run --stages expression` as a refresh of reviewed activity records. Use `tag-expressions` for that purpose. Coordinate all inference writes: upload workers serialize themselves with an OS lock, but unrelated direct CLI pipeline runs and expression refreshes do not share a universal scheduler. Do not run overlapping writers against the same data root/video. `purge` is destructive and is not an integration or refresh step.

## Source map and models

| Code | Responsibility |
|---|---|
| `src/greenroom/config.py`, `profiles/*.yaml` | Data paths, supported formats, profile settings |
| `src/greenroom/cli.py` | CLI commands and arguments |
| `src/greenroom/uploads.py` | Validated intake, unique IDs, durable job state, local worker |
| `src/greenroom/windows.py`, `media.py` | Timestamp windows, FFmpeg probing/extraction/subclips |
| `src/greenroom/pipeline.py` | Orchestration; stages `audio`, `vision`, `transcript`, `expression` |
| `src/greenroom/audio_events.py`, `vision_tags.py` | Audio/video activity scores |
| `src/greenroom/transcribe.py`, `transcript_cues.py` | Local ASR and transcript activity cues |
| `src/greenroom/expressions.py` | Haar face detection, expression inference, window aggregation |
| `src/greenroom/opinions.py`, `fuse.py`, `taxonomy.py` | Activity evidence fusion, routing, label definitions |
| `src/greenroom/records.py`, `ledger.py` | AI records and separate hash-chained staff decisions |
| `src/greenroom/expression_timeline.py` | Versioned expression timeline export, shared by CLI/UI |
| `app/review_app.py` | Upload, review, timestamped expression display and downloads |
| `tests/` | Unit, data-contract, and Streamlit UI tests using synthetic data |

Default `rtx4090` configuration:

| Stage | Model/component | Role |
|---|---|---|
| Audio | `laion/larger_clap_music_and_speech` | Audio/text activity scoring |
| Audio | `MIT/ast-finetuned-audioset-10-10-0.4593` | AudioSet event classifier |
| Vision | `microsoft/xclip-base-patch16-zero-shot` | Video/text activity scoring |
| Vision | `MCG-NJU/videomae-base-finetuned-kinetics` | Kinetics activity classifier |
| Transcript | faster-whisper `large-v3` (`Systran/faster-whisper-large-v3`) | Local speech transcription, FP16 |
| Transcript | Bundled Silero VAD through ONNX Runtime | Speech-region filtering; ASR wrapper has a fallback if ONNX Runtime is unavailable |
| Transcript | Ollama `qwen2.5:7b-instruct` | Local transcript activity/facilitation cues |
| Expression | OpenCV `haarcascade_frontalface_default.xml` | Frontal-face detection, no recognition |
| Expression | `trpakov/vit-face-expression` | Face-expression probabilities |

CLAP's fallback is `laion/clap-htsat-unfused`; the expression fallback is `dima806/facial_emotions_image_detection`. Read the actual `model_id` in the output because a fallback can be selected at load time. The T4 profile substitutes X-CLIP patch32, faster-whisper `distil-large-v3`/`int8_float16`, and Ollama `qwen2.5:3b-instruct`; it disables AST and currently has no enabled expression block. Do not select it expecting expression output without adding a compatible expression configuration and testing it.

DETR (`facebook/detr-resnet-50`) and Qwen2.5-VL entries in the profiles are future configuration, not running stages. SigLIP is unset. PySceneDetect is a dependency but is not used to create the fixed windows. There is no face recognition, cross-frame person tracking, speaker diarization, or emotion-specific language model. Runtime support includes PyTorch/Transformers, faster-whisper/CTranslate2/PyAV/ONNX Runtime, OpenCV/Pillow/NumPy, librosa/SoundFile, FFmpeg, local Ollama/Requests, and Streamlit/pandas/PyYAML; optional NVML measures GPU use.

## Local files and completion boundary

```text
<GREENROOM_DATA_DIR>/
  videos/<source_file>
  derived/windows.json
  derived/audio/<video_id>.wav
  derived/frames/...
  derived/transcripts/<video_id>.json
  derived/subclips/<video_id>/...
  derived/uploads/jobs/<job_id>.json
  derived/uploads/logs/<job_id>.log
  derived/uploads/worker.log
  records/<video_id>/records.jsonld
  records/<video_id>/records.csv
  records/<video_id>/segments.json
  records/<video_id>/run_meta.json
  records/<video_id>/reviews.jsonl
```

`records.jsonld` is an object containing `@context` and an `@graph` list of per-window AI records; it is not a top-level array. Its `expression_suggestion` field is an object or `null`. The ordinary `records.csv` only flattens the suggestion label; use the dedicated expression export for its richer fields. `segments.json` merges consecutive **activity** labels and is not an expression timeline. `run_meta.json` contains actual model IDs, stage metrics, channel outputs, and expression refresh metadata. Transcripts and sampled full frames remain sensitive derived files, even though face crops, boxes, and embeddings are not persisted.

An upload job transitions `queued` → `running` → `completed` or `failed`. Only `completed` is the successful ingestion boundary. An index entry or an existing output file alone is insufficient: indexing precedes inference and failed jobs may have partial files. A direct synchronous CLI run completes when it returns successfully. The export CLI checks record windows against the index and requires a matching completed job for uploaded videos. Serialize exports against ongoing writes to the same video; these checks are not a distributed transaction.

## Expression timeline contract

Export one video or the whole library to an external directory:

```powershell
$exports = Join-Path $env:GREENROOM_DATA_DIR "exports"
New-Item -ItemType Directory -Force -Path $exports | Out-Null
& $python -m greenroom export-expressions --video $videoId --output (Join-Path $exports "timeline.json")
& $python -m greenroom export-expressions --video $videoId --format csv --output (Join-Path $exports "timeline.csv")
& $python -m greenroom export-expressions --all --output (Join-Path $exports "timelines")
```

For one video, the output filename suffix must match the selected format. `--all` writes `<video_id>.expressions.json` (or `.csv`) inside the destination directory and preflights every indexed video before writing: a missing record set or an unfinished upload fails the entire export. Use `--video` for a ready subset. Destinations inside the repository or the data root's `videos`, `derived`, and `records` trees are rejected to prevent accidental source-artifact overwrite; `<data>/exports/expressions/` is a suitable destination.

JSON is the preferred integration format. Synthetic example, unrelated to participant media:

```json
{
  "schema_version": "greenroom.expression-timeline.v1",
  "video_id": "synthetic-demo",
  "source_file": "synthetic-demo.mp4",
  "time_unit": "seconds",
  "interval_semantics": "start_inclusive_end_exclusive",
  "windows": [
    {
      "window_id": "synthetic-demo:00000",
      "start_s": 0.0,
      "end_s": 5.0,
      "start_time": "00:00:00.000",
      "end_time": "00:00:05.000",
      "ai_expression": {
        "label": "Positive",
        "share": 0.7,
        "mixture": {"positive": 0.7, "neutral": 0.2, "negative": 0.1},
        "upset_share": 0.05,
        "face_readings": 4,
        "frames_with_faces": 3,
        "frames_checked": 5,
        "status": "ai_suggested_unverified",
        "source": "face_expression_model",
        "model_id": "trpakov/vit-face-expression",
        "note": "Generalized expression read from faces by a model; not a statement about how anyone feels."
      },
      "staff_expression": null,
      "review_decision": "accepted"
    },
    {
      "window_id": "synthetic-demo:00001",
      "start_s": 5.0,
      "end_s": 7.0,
      "start_time": "00:00:05.000",
      "end_time": "00:00:07.000",
      "ai_expression": null,
      "staff_expression": null,
      "review_decision": "pending"
    }
  ]
}
```

Rules the receiving system must retain:

- Check `schema_version` before importing. `source_file` is the source filename relative to `<data>/videos/`, not a public media URL; the pure Python helper permits it to be null when omitted. Resolve media only within your authorized data root and provide playback through your own local integration.
- Times are seconds from the **start of the original video**, not wall-clock timestamps or offsets within a rendered subclip. Match playback using `start_s <= currentTime < end_s`. At exactly 5.0 seconds in this example, select the second window. Display strings are `HH:MM:SS.mmm` and can exceed 24 hours; use numeric values for seeking and comparisons.
- Default windows are five seconds. A remainder shorter than two seconds merges into the preceding window, so some final windows exceed five seconds; short videos and other remainders also vary. Never calculate timestamps from a row number or assume every duration is five seconds. The index's `window_seconds` controls construction; changing a profile value alone does not rebuild existing windows.
- `windows` is sorted by time and includes windows without AI output. `ai_expression: null` means no suggestion is available, not Neutral, no people, or an error-free inference of absence. It can result from missing/insufficient detectable faces or from expression processing not having been run.
- AI labels are `Positive`, `Neutral`, `Negative`, or `Upset`. These are generalized model buckets; surprise is included in Positive, and sad/angry probabilities can lead to Upset. They are not direct measurements of a person's feelings.
- `share` and `mixture` summarize average mapped model probabilities across detected face readings. They are not calibrated correctness probabilities, percentages of people, or verified observations. A face seen in three frames can count three times. Multiple people are mixed together; no per-person or cross-window identity exists.
- `ai_expression.status` remains `ai_suggested_unverified` regardless of an activity review decision. The first synthetic row above deliberately demonstrates an accepted activity with an unverified AI expression and no staff expression.
- `staff_expression` is `null` unless the **latest ledger decision for that window** explicitly contains an `expression_label`. When present its keys are `label`, `reviewer`, `reviewed_at`, and `ledger_hash`. Do not resurrect an older expression after a later decision clears it. Staff labels can also include `Confident`, which the AI classifier does not produce.
- `review_decision` is `pending`, `accepted`, `edited`, `rejected`, or `deferred`. It describes the latest review decision, not AI expression verification. The export intentionally omits free-text reviewer notes, transcripts, face crops, and identity fields.
- Upsert by `(dataset_id, video_id, window_id)` in a multi-library receiving system, with your own stable `dataset_id`. Within one library `(video_id, window_id)` is enough. Preserve raw numeric times, the AI object, and the separate staff object. A model refresh updates AI fields on the same key; a later ledger decision updates staff fields. Null values must clear prior imported values rather than silently retaining stale labels.

CSV repeats schema/video/time-unit metadata on each row and flattens model fields under `ai_` and staff fields under `staff_`. Null values become empty cells. The canonical column list is `CSV_COLUMNS` in `src/greenroom/expression_timeline.py`; numeric timestamps keep the same meaning as JSON. UI downloads include the whole video's timeline even when the displayed table is filtered.

## Python adapter examples

The package can be imported without loading inference models to build a timeline from completed records. This example is intended to run in the target's local process with `GREENROOM_DATA_DIR` set:

```python
from greenroom import config, ledger
from greenroom.expression_timeline import build_expression_timeline, expression_timeline_csv
from greenroom.records import load_records
from greenroom.windows import load_index

def export_completed_video(video_id: str) -> dict:
    # Caller must first establish job completion / successful synchronous run.
    index = load_index(config.windows_index_path())
    video = next(v for v in index["videos"] if v["video_id"] == video_id)
    records = load_records(config.records_dir() / video_id)
    expected = {w["window_id"]: (w["start_s"], w["end_s"])
                for w in index["windows"] if w["video_id"] == video_id}
    actual = {r["window_id"]: (r["start_s"], r["end_s"]) for r in records}
    if len(records) != video["n_windows"] or len(actual) != len(records) or actual != expected:
        raise RuntimeError("Incomplete or mismatched window records; do not import yet")
    verification = ledger.verify_chain(video_id)
    if not verification["ok"]:
        raise RuntimeError(f"Invalid review ledger: {verification['problems']}")
    return build_expression_timeline(
        video_id, records,
        decisions=ledger.latest_decisions(video_id),
        source_file=video["source_file"],
    )

# Your database adapter should store both objects under the same stable key:
# for row in export_completed_video(video_id)["windows"]:
#     db.upsert((dataset_id, video_id, row["window_id"]), row)
# csv_text = expression_timeline_csv(export_completed_video(video_id))
```

To enqueue a local file from another Python system:

```python
from pathlib import Path
from greenroom import uploads

with Path("D:/incoming/consenting-adult-demo.mp4").open("rb") as video:
    job = uploads.submit_upload(video, profile_name="rtx4090")
uploads.start_worker()  # Uses this interpreter, environment, and data directory.
print(job["job_id"], job["video_id"])

# Poll from your job scheduler; do not block an HTTP request until inference ends.
job_by_id = {item["job_id"]: item for item in uploads.list_jobs()}
status = job_by_id[job["job_id"]]["status"]
# Import only when status == "completed".
# For a failed job: uploads.retry_job(job["job_id"]); uploads.start_worker()
```

`submit_upload` requires a seekable binary file-like object with a `name`, and accepts an optional `max_bytes`. The backend defaults to 2,048 MiB, assigns a collision-resistant source filename/video ID, and deduplicates identical file bytes with the same profile. A repeated submission may return an already completed or failed job, so inspect its status. Retry explicitly; do not create a new job by renaming an unchanged file. `python -m greenroom.uploads` drains pending jobs synchronously, and `uploads.process_queue()` is the equivalent Python entry point. The worker automatically uses only `vision,expression` for silent clips. It refuses to overwrite an upload that already has staff reviews.

The current app plays a selected window as a subclip: its player clock starts at zero, and the caption shows the interval in the original video. It does not automatically synchronize an expression overlay while a full video plays. If the receiving UI plays the original source, use its player seek API with the row's `start_s`. If it plays a generated subclip starting at that timestamp, map player time back with `source_time = clip_start_s + player_current_time`; show the original interval in the label. Keep separate visual columns/badges for **AI expression (unverified)** and **Staff observation**. Do not infer that a reviewed activity makes an expression verified.

## Staff decisions and integration checks

`reviews.jsonl` is append-only, hash-chained, and authoritative for staff decisions. `ledger.latest_decisions(video_id)` returns the last entry per window; raw `records.jsonld` remains AI output. Never edit or truncate old ledger lines, flatten the ledger into model records, or automatically append acceptance decisions during import. If adding a staff review UI, use the existing `ledger.append_decision` interface, supply a real reviewer and explicit staff expression choice, and preserve its decision fields. Serialize review writes: the ledger currently has no general multi-writer database transaction layer.

Before handing an adapter to another teammate:

1. Run `python -m pytest -q` and `python -m greenroom verify`.
2. Upload a synthetic seven-second video. Wait for `completed`; expect two intervals `[0,5)` and `[5,7)`, a selectable review clip, and all windows in JSON/CSV. A pattern with no faces should not fabricate expression tags. Use a consenting-adult test recording separately to inspect face detection/classification.
3. Verify that Expression timeline works without a reviewer name and that selecting a window plays its matching source interval. At an exact boundary the following window must own that time.
4. Accept an activity without setting a staff expression. Its exported AI expression must remain unverified and `staff_expression` must remain null. Enter an explicit staff observation through Edit, export again, and confirm the reviewer/time/hash. A later cleared label must clear the receiving system's value.
5. Confirm that JSON and CSV preserve the same windows and numeric timestamps; filters in the UI must not be mistaken for missing source records. Keep null-expression windows in the receiving store.
6. Re-submit the same file/profile and check job deduplication. Exercise a failed job/retry. Existing video IDs, windows, reviews, and active review selection must survive unrelated uploads.
7. Refresh expressions with `tag-expressions`; verify activities and the ledger remain unchanged. Re-import by stable keys and update model provenance. Check that no private artifacts entered Git or remote logs.

Known limits: the model samples faces rather than tracking each frame, frontal-face detection misses profiles/occlusions/small faces, predictions vary with lighting and performance context, no continuous emotion onset/offset is inferred, and a timestamp tag covers its whole analysis window. The default expression settings are 1 fps at 540-pixel height, at most five frames per window, a 40-pixel minimum face size, and at least two face readings. They are configurable in `profiles/rtx4090.yaml`. There is no live progress percentage, distributed worker queue, authentication, or stable person identity contract. Integrate those concerns as separate explicit features rather than assuming they already exist.
