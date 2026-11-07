"""Run metadata that training and evaluation both record."""

from __future__ import annotations

import subprocess
from importlib import metadata


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_dirty() -> bool | None:
    """True if the working tree has uncommitted changes to tracked or new files."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(out.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions(packages: list[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in packages:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out
