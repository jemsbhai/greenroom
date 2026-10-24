"""Command-line entry points.

    python -m greenroom env                 check the environment and the data directory
    python -m greenroom windows [--subclips] build the window index (and review subclips)
    python -m greenroom gold-app            print the command that starts the gold labeling app
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
