"""A freeze is logged: no frame for HANG_SECONDS, and every thread's stack
goes to the hang log; frames coming keep it quiet."""

import time
from pathlib import Path

import pytest

from ok_doomer.client import watchdog
from ok_doomer.client.watchdog import Watchdog


@pytest.fixture
def _log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "hang.log"
    monkeypatch.setattr(watchdog, "hang_log_path", lambda: path)
    monkeypatch.setattr(watchdog, "HANG_SECONDS", 0.3)
    return path


def test_frames_coming_keep_it_quiet(_log: Path) -> None:
    dog = Watchdog()
    try:
        for _ in range(10):  # a frame every 0.1 s: never 0.3 s without one
            dog.pet()
            time.sleep(0.1)
    finally:
        dog.close()
    assert _log.read_text() == ""


def test_a_freeze_says_where(_log: Path) -> None:
    dog = Watchdog()
    try:
        dog.pet()
        time.sleep(0.8)  # frozen: no frame
    finally:
        dog.close()
    text = _log.read_text()
    assert "Thread" in text and "test_a_freeze_says_where" in text  # where it was stuck
