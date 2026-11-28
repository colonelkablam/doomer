"""play.sh and play.bat: committed runnable, with the line endings their
shells need (bash chokes on Windows ones, cmd misreads Unix ones)."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout


def test_play_sh_is_executable_in_git_and_has_unix_line_endings() -> None:
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    mode = _git("ls-files", "--stage", "play.sh").split()[0]
    assert mode == "100755"  # so ./play.sh runs after a clone
    attrs = _git("check-attr", "eol", "play.sh", "play.bat")
    assert "play.sh: eol: lf" in attrs and "play.bat: eol: crlf" in attrs


def test_play_sh_parses() -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("no bash here")
    subprocess.run([bash, "-n", str(ROOT / "play.sh")], check=True)
