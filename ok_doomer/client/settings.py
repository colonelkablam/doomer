"""What the game remembers between runs (your name), in a small JSON file
in your home folder: ~/.ok-doomer/settings.json, on Linux and Windows
alike. Settings you edit by hand are client.toml's; this is the game's.
Anything wrong with the file (missing, unreadable, odd) means defaults:
it's never worth failing to start over."""

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_NAME = "player"
MAX_NAME = 16  # as the server keeps


def settings_path() -> Path:
    return Path.home() / ".ok-doomer" / "settings.json"


def _load() -> dict[str, Any]:
    try:
        data = json.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def load_name() -> str:
    name = _load().get("name")
    return name[:MAX_NAME] if isinstance(name, str) and name.strip() else DEFAULT_NAME


def save_name(name: str) -> None:
    data = _load()
    data["name"] = name[:MAX_NAME]
    path = settings_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    except OSError as e:
        log.warning("couldn't save settings to %s: %s", path, e)
