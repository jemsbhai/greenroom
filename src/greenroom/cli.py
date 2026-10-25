"""Command-line entry points.

    python -m greenroom env                 check the environment and the data directory
    python -m greenroom windows [--subclips] build the window index (and review subclips)
    python -m greenroom gold-app            print the command that starts the gold labeling app
    python -m greenroom run ...             run the pipeline on one or more videos
    python -m greenroom tag-expressions ... refresh expressions while preserving existing activity and review data
    python -m greenroom export-expressions ... export timestamped expressions without running models
    python -m greenroom review-app          print the command that starts the staff review app
    python -m greenroom verify              check the hash chain of every review ledger
    python -m greenroom report              write <data>/records/report.md and report.json (results numbers)
    python -m greenroom purge --yes         delete every derived artifact (audio, frames, clips, records); videos untouched
"""
from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from . import __version__, config
from .media import cut_subclip, ffmpeg_path, ffprobe_path, probe
from .windows import WINDOW_SECONDS, build_index, load_index, save_index


def _print(label: str, value) -> None:
    print(f"{label:<28} {value}")


def cmd_env(_: argparse.Namespace) -> int:
    ok = True
    _print("greenroom", __version__)
    _print("python", f"{platform.python_version()} ({sys.executable})")
    _print("platform", platform.platform())
    try:
        import torch  # type: ignore

        _print("torch", f"{torch.__version__} cuda_available={torch.cuda.is_available()} cuda={torch.version.cuda}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            _print("gpu", f"{props.name} {props.total_memory / 1024**3:.1f} GB")
    except Exception as exc:  # noqa: BLE001
        _print("torch", f"not importable ({exc})")
    try:
        _print("ffmpeg", ffmpeg_path())
    except Exception as exc:  # noqa: BLE001
        ok = False
        _print("ffmpeg", f"MISSING ({exc}); pip install imageio-ffmpeg or winget install Gyan.FFmpeg")
    _print("ffprobe", ffprobe_path() or "not on PATH (duration will be parsed from ffmpeg output)")

    ollama = shutil.which("ollama")
    if ollama:
        try:
            out = subprocess.run([ollama, "list"], capture_output=True, text=True, timeout=20).stdout
            names = [line.split()[0] for line in out.splitlines()[1:] if line.strip()]
            local = [n for n in names if config.is_local_ollama_model(n)]
            cloud = [n for n in names if not config.is_local_ollama_model(n)]
            _print("ollama local models", ", ".join(local) or "NONE (run: ollama pull qwen2.5:7b-instruct)")
            if cloud:
                _print("ollama cloud models", ", ".join(cloud) + "  <- not permitted for hackathon data; the client refuses them")
        except Exception as exc:  # noqa: BLE001
            _print("ollama", f"present but `ollama list` failed ({exc})")
    else:
        _print("ollama", "not on PATH")

    try:
        data = config.data_dir()
        videos = config.list_videos()
        _print("data dir", data)
        _print("videos", f"{len(videos)} file(s)")
        for v in videos:
            _print("", f"{v.name} ({v.stat().st_size / 1024**2:.1f} MB)")
        idx = config.windows_index_path()
        _print("window index", idx if idx.is_file() else f"not built yet ({idx})")
    except Exception as exc:  # noqa: BLE001
        ok = False
        _print("data dir", f"PROBLEM: {exc}")
    return 0 if ok else 1


def cmd_windows(args: argparse.Namespace) -> int:
    videos_dir = config.videos_dir()
    index = build_index(videos_dir, probe, window_s=args.window_seconds)
    path = save_index(index, config.windows_index_path())
    total_windows = len(index["windows"])
    total_s = sum(v["duration_s"] for v in index["videos"])
    print(f"Indexed {len(index['videos'])} video(s), {total_s / 60:.1f} min, {total_windows} windows of {args.window_seconds:g} s -> {path}")
    for v in index["videos"]:
        print(f"  {v['video_id']:<45} {v['duration_s'] / 60:6.1f} min  {v['n_windows']:4d} windows  audio={v['has_audio']}")
    if args.subclips:
        print("Cutting review subclips (cached; re-run is cheap) ...")
        by_file = {v["video_id"]: videos_dir / v["source_file"] for v in index["videos"]}
        done = 0
        for w in index["windows"]:
            dst = config.subclips_dir() / w["video_id"] / f"{w['index']:05d}.mp4"
            cut_subclip(by_file[w["video_id"]], dst, w["start_s"], w["end_s"] - w["start_s"], height=args.height)
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{total_windows}")
        print(f"  {done}/{total_windows} subclips ready under {config.subclips_dir()}")
    return 0


def cmd_gold_app(_: argparse.Namespace) -> int:
    app = config.repo_root() / "app" / "gold_app.py"
    print(f"{sys.executable} -m streamlit run \"{app}\" --server.address 127.0.0.1")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import STAGES, run_videos

    profile = config.load_profile(args.profile)
    index = load_index(config.windows_index_path())
    if args.all:
        video_ids = [v["video_id"] for v in index["videos"]]
    else:
        video_ids = list(args.video or [])
    if not video_ids:
        print("Give --video <id> (repeatable) or --all. Known ids:")
        for v in index["videos"]:
            print(f"  {v['video_id']}")
        return 2
    stages = tuple(args.stages.split(",")) if args.stages else STAGES
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        print(f"unknown stage(s) {unknown}; available in this build: {STAGES}")
        return 2
    run_videos(video_ids, profile, stages=stages, fusion_method=args.fusion, index=index)
    return 0


def cmd_review_app(_: argparse.Namespace) -> int:
    app = config.repo_root() / "app" / "review_app.py"
    print(f"{sys.executable} -m streamlit run \"{app}\" --server.address 127.0.0.1")
    return 0


def cmd_tag_expressions(args: argparse.Namespace) -> int:
    from .pipeline import tag_expressions

    try:
        profile = config.load_profile(args.profile)
        index = load_index(config.windows_index_path())
        video_ids = [v["video_id"] for v in index["videos"]] if args.all else list(args.video or [])
        tag_expressions(video_ids, profile, index=index)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


def cmd_export_expressions(args: argparse.Namespace) -> int:
    import json
    import re

    from .expression_timeline import build_expression_timeline, expression_timeline_csv
    from .ledger import latest_decisions

    try:
        index = load_index(config.windows_index_path())
        videos = {video["video_id"]: video for video in index["videos"]}
        selected = list(videos) if args.all else [args.video]
        if not selected:
            raise ValueError("No videos are indexed")
        unknown = [video_id for video_id in selected if video_id not in videos]
        if unknown:
            raise ValueError(f"Unknown video id(s): {', '.join(unknown)}")
        output = Path(args.output).expanduser().resolve()
        # Export paths must not replace inference, review, or source artifacts.
        data = config.data_dir()
        protected = [data / "videos", data / "derived", data / "records", config.repo_root()]
        if any(output.is_relative_to(root.resolve()) for root in protected):
            raise ValueError("Choose an export destination outside the repository and videos/, derived/, records/; use <data>/exports/expressions/")
        if args.all:
            if output.exists() and not output.is_dir():
                raise ValueError("--all requires --output to name a directory")
        elif output.suffix.lower() != f".{args.format}":
            raise ValueError(f"--output must end in .{args.format} (or select the matching --format)")

        exports = []
        for video_id in selected:
            # Index ids are generated by the app; also validate before making paths.
            if not video_id or "/" in video_id or "\\" in video_id or video_id in (".", ".."):
                raise ValueError(f"Unsafe video id in window index: {video_id!r}")
            if "upload_job_id" in videos[video_id]:
                job_id = videos[video_id]["upload_job_id"]
                if not isinstance(job_id, str) or not re.fullmatch(r"[0-9a-f]{32}", job_id):
                    raise ValueError(f"Invalid upload job id for {video_id}; cannot confirm completed processing")
                job_path = data / "derived" / "uploads" / "jobs" / f"{job_id}.json"
                if not job_path.is_file():
                    raise ValueError(f"Upload status is missing for {video_id}; cannot confirm completed processing")
                job = json.loads(job_path.read_text(encoding="utf-8"))
                if (not isinstance(job, dict) or job.get("job_id") != job_id
                        or job.get("video_id") != video_id or job.get("status") != "completed"):
                    raise ValueError(f"Upload processing is not confirmed complete for {video_id}; finish or retry the upload before exporting")
            source = config.records_dir() / video_id / "records.jsonld"
            if not source.is_file():
                raise ValueError(f"No records for {video_id}; finish processing that video before exporting")
            records = json.loads(source.read_text(encoding="utf-8"))["@graph"]
            expected = [window for window in index["windows"] if window["video_id"] == video_id]
            by_id = {window["window_id"]: window for window in expected}
            record_ids = [record["window_id"] for record in records]
            if (not expected or len(by_id) != len(expected)
                    or videos[video_id].get("n_windows", len(expected)) != len(expected)
                    or len(record_ids) != len(set(record_ids)) or set(record_ids) != set(by_id)):
                raise ValueError(f"Records for {video_id} do not cover the complete indexed windows; restore matching records before exporting")
            for record in records:
                window = by_id[record["window_id"]]
                if record.get("start_s") != window["start_s"] or record.get("end_s") != window["end_s"]:
                    raise ValueError(f"Record timestamps for {record['window_id']} differ from the window index; restore matching records before exporting")
            document = build_expression_timeline(
                video_id, records, latest_decisions(video_id), source_file=videos[video_id].get("source_file"),
            )
            destination = output / f"{video_id}.expressions.{args.format}" if args.all else output
            content = (json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
                       if args.format == "json" else expression_timeline_csv(document))
            exports.append((destination, content, len(document["windows"])))
        # Prepare every document before creating outputs, so a bad video fails early.
        for destination, content, count in exports:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8", newline="")
            print(f"{count} timestamped expression windows -> {destination}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


def cmd_verify(_: argparse.Namespace) -> int:
    from .ledger import verify_all

    reports = verify_all()
    if not reports:
        print("no review ledgers found")
        return 0
    bad = 0
    for r in reports:
        state = "ok" if r["ok"] else "BROKEN"
        print(f"{r['video_id']:<45} {r['entries']:4d} decisions  {state}")
        for p in r["problems"]:
            print(f"    {p}")
        bad += 0 if r["ok"] else 1
    print(f"{len(reports) - bad} of {len(reports)} ledgers intact")
    return 1 if bad else 0


def cmd_report(_: argparse.Namespace) -> int:
    from .report import write_report

    out_md, out_json, md = write_report()
    print(md)
    print(f"\nwritten: {out_md}\n         {out_json}")
    return 0


def cmd_purge(args: argparse.Namespace) -> int:
    import shutil

    targets = [config.derived_dir()]
    if not args.keep_records:
        targets.append(config.records_dir())
    if not args.yes:
        print("This deletes:")
        for t in targets:
            print(f"  {t}")
        print("The provided videos are not touched. Re-run with --yes to proceed.")
        return 2
    for t in targets:
        if t.is_dir():
            shutil.rmtree(t)
            print(f"deleted {t}")
    print("derived artifacts removed; videos untouched")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="greenroom", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_env = sub.add_parser("env", help="check the environment and the data directory")
    p_env.set_defaults(func=cmd_env)

    p_win = sub.add_parser("windows", help="build the window index under the data directory")
    p_win.add_argument("--window-seconds", type=float, default=WINDOW_SECONDS)
    p_win.add_argument("--subclips", action="store_true", help="also cut low-resolution review subclips for every window")
    p_win.add_argument("--height", type=int, default=480, help="subclip height in pixels")
    p_win.set_defaults(func=cmd_windows)

    p_gold = sub.add_parser("gold-app", help="print the command that starts the gold labeling app")
    p_gold.set_defaults(func=cmd_gold_app)

    p_run = sub.add_parser("run", help="run the pipeline and write records under <data>/records/<video_id>/")
    p_run.add_argument("--profile", default="rtx4090", help="profiles/<name>.yaml")
    p_run.add_argument("--video", action="append", help="video id from the window index (repeatable)")
    p_run.add_argument("--all", action="store_true", help="every video in the index")
    p_run.add_argument("--stages", default=None, help="comma-separated stages; default: all available")
    p_run.add_argument("--fusion", default="cumulative", choices=["cumulative", "mean", "max"])
    p_run.set_defaults(func=cmd_run)

    p_expr = sub.add_parser("tag-expressions", help="refresh expressions on existing records, preserving activity and review data")
    p_expr.add_argument("--profile", default="rtx4090", help="profiles/<name>.yaml")
    selection = p_expr.add_mutually_exclusive_group(required=True)
    selection.add_argument("--video", action="append", help="video id from the window index (repeatable)")
    selection.add_argument("--all", action="store_true", help="every video in the index")
    p_expr.set_defaults(func=cmd_tag_expressions)

    p_export = sub.add_parser("export-expressions", help="export timestamped AI and staff expressions as JSON or CSV without running models")
    selection = p_export.add_mutually_exclusive_group(required=True)
    selection.add_argument("--video", help="one video id from the window index")
    selection.add_argument("--all", action="store_true", help="every video in the index; all must have processed records")
    p_export.add_argument("--output", required=True, help="output file for --video, or output directory for --all; keep outside source/derived/record folders")
    p_export.add_argument("--format", choices=["json", "csv"], default="json")
    p_export.set_defaults(func=cmd_export_expressions)

    p_review = sub.add_parser("review-app", help="print the command that starts the staff review app")
    p_review.set_defaults(func=cmd_review_app)

    p_verify = sub.add_parser("verify", help="check the hash chain of every review ledger")
    p_verify.set_defaults(func=cmd_verify)

    p_report = sub.add_parser("report", help="results report from records, ledgers and run metadata")
    p_report.set_defaults(func=cmd_report)

    p_purge = sub.add_parser("purge", help="delete derived artifacts under the data directory; the videos are not touched")
    p_purge.add_argument("--yes", action="store_true", help="actually delete (without it, only lists what would go)")
    p_purge.add_argument("--keep-records", action="store_true", help="keep records and review ledgers")
    p_purge.set_defaults(func=cmd_purge)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
