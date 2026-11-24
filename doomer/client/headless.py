"""Helpers for rendering without a display (cloud VMs, CI, tests).

SDL picks its video driver when pygame.display is initialised, so the
environment variable must be set before that happens.
"""
import os
from pathlib import Path


def use_dummy_video() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def save_png(surface, path: str | Path, scale: int = 1) -> Path:
    """Save a surface as PNG, optionally scaled up (nearest neighbour)."""
    import pygame

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if scale != 1:
        w, h = surface.get_size()
        surface = pygame.transform.scale(surface, (w * scale, h * scale))
    pygame.image.save(surface, str(path))
    return path
