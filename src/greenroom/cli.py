"""Command-line entry points.

    python -m greenroom env                 check the environment and the data directory
    python -m greenroom windows [--subclips] build the window index (and review subclips)
    python -m greenroom gold-app            print the command that starts the gold labeling app
    python -m greenroom run ...             run the pipeline on one or more videos
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
