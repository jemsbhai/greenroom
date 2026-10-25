"""Paths and run configuration.

All data lives outside the repository. The data directory is taken from the
GREENROOM_DATA_DIR environment variable; it must contain a videos/ folder.
Everything the pipeline derives (audio tracks, low-resolution subclips, window
indexes, gold labels, records, ledgers) is written under that directory, never
under the repository.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ENV_DATA_DIR = "GREENROOM_DATA_DIR"
ENV_ALLOW_DATA_IN_REPO = "GREENROOM_ALLOW_DATA_IN_REPO"

VIDEO_EXTENSIONS = {".mp4", ".mpeg4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def repo_root() -> Path:
    """The repository root (the folder that holds pyproject.toml)."""
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """Resolve and validate the data directory.

    Raises FileNotFoundError when the directory or its videos/ folder is
    missing, and RuntimeError when the data directory sits inside the
    repository (a second fence behind .gitignore; override only knowingly).
    """
    value = os.environ.get(ENV_DATA_DIR)
    path = Path(value).expanduser().resolve() if value else (repo_root().parent / "data").resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"Data directory not found: {path}. Set {ENV_DATA_DIR} to the folder that contains videos/."
        )
    if not (path / "videos").is_dir():
        raise FileNotFoundError(f"Expected a videos/ folder inside the data directory: {path}")
    if os.environ.get(ENV_ALLOW_DATA_IN_REPO) != "1":
        try:
            path.relative_to(repo_root())
        except ValueError:
            pass
        else:
            raise RuntimeError(
                f"The data directory {path} is inside the repository {repo_root()}. "
                f"Move it outside the repo (or set {ENV_ALLOW_DATA_IN_REPO}=1 if you accept the risk)."
            )
    return path


def videos_dir() -> Path:
    return data_dir() / "videos"


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def derived_dir() -> Path:
    return _ensure(data_dir() / "derived")


def subclips_dir() -> Path:
    return _ensure(derived_dir() / "subclips")


def audio_dir() -> Path:
    return _ensure(derived_dir() / "audio")


def frames_dir() -> Path:
    return _ensure(derived_dir() / "frames")


def gold_dir() -> Path:
    return _ensure(data_dir() / "gold")


def records_dir() -> Path:
    return _ensure(data_dir() / "records")


def windows_index_path() -> Path:
    return derived_dir() / "windows.json"


def list_videos() -> list[Path]:
    return sorted(p for p in videos_dir().iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


def profiles_dir() -> Path:
    return repo_root() / "profiles"


def load_profile(name: str) -> dict:
    """Load profiles/<name>.yaml (rtx4090 or t4)."""
    path = profiles_dir() / f"{name}.yaml"
    if not path.is_file():
        available = ", ".join(sorted(p.stem for p in profiles_dir().glob("*.yaml")))
        raise FileNotFoundError(f"Unknown profile {name!r}. Available: {available}")
    with path.open("r", encoding="utf-8") as handle:
        profile = yaml.safe_load(handle) or {}
    profile.setdefault("name", name)
    return profile


def is_local_ollama_model(name: str) -> bool:
    """True when an Ollama model tag refers to a locally served model.

    The hackathon forbids sending the provided data to any cloud-hosted
    generative model. Ollama lists cloud-routed models with a ':cloud' tag;
    those are rejected here and in the LLM client.
    """
    lowered = name.strip().lower()
    return bool(lowered) and ":cloud" not in lowered and not lowered.endswith("-cloud")


def assert_local_ollama_model(name: str) -> str:
    if not is_local_ollama_model(name):
        raise RuntimeError(
            f"Refusing to use Ollama model {name!r}: cloud-routed models are not permitted for hackathon data. "
            "Pull a local model, for example: ollama pull qwen2.5:7b-instruct"
        )
    return name
