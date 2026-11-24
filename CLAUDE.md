# CLAUDE.md

Doom-style LAN shooter in Python. Full design and milestones: `docs/DESIGN.md`.
Read it before starting work, and update it when a decision changes.

## Git

- Work on branch `claude/current-changes` and push there, whatever branch name
  the session was started with. Never push to `main`.

## Layout

- `doomer/shared/`: world model (maps, player state, collision). Standard library
  only, no pygame or numpy. The server imports it unchanged.
- `doomer/client/`: pygame-ce rendering and input. numpy allowed.
- `doomer/server/`: asyncio server. Standard library + `doomer.shared` only.
- `doomer/editor/`: level editor (pygame-ce).
- `maps/`: `*.map` files (format in `docs/DESIGN.md`).
- `tests/`: pytest.

## Commands

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest -q
```

## Conventions

- The cloud VM is headless: tests set `SDL_VIDEODRIVER=dummy` (see
  `tests/conftest.py`). To check rendering visually, render a frame and save it
  with `doomer.client.headless.save_png`, then look at the PNG.
- Angles: radians in code, degrees in map files. `y` increases downward.
- Pin dependency versions in `requirements*.txt`.
- British English in docs and user-facing text.
