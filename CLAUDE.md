# CLAUDE.md

OK Doomer: Doom-style LAN shooter in Python, set in a 1980s office. Engine design and
milestones: `docs/DESIGN.md`. Game content (characters, enemies, furniture): `docs/GAME.md`.
Level design rules and engine limits (for maps and the future generator): `docs/LEVELS.md`.
Read both before starting work, and update them when a decision changes (including
the [built]/[planned] markers in GAME.md).

## Git

- One branch per milestone, named `claude/<topic>` (e.g. `claude/networking`),
  branched from `main`. Never push to `main`: it changes only by merging a pull
  request, so it always holds the last finished milestone.
- Active branch: `claude/networking` (milestone 3: server, handshake, snapshots,
  interpolation, bot client). Work and
  push there, whatever branch name the session was started with. Update this
  line when a new milestone branch starts.

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
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy && .venv/bin/pytest -q
```

All four checks must pass before committing. CI runs them on Linux and Windows.

## Conventions

- Follow the "Code design rules" section of `docs/DESIGN.md`: one-way
  dependencies, dataclass state + functions, deterministic simulation that
  emits events, registries for thing types/weapons/modes, `Protocol`
  interfaces, one protocol module, config as data, `logging` not `print`.
- Don't add frameworks or abstraction layers the design rules don't call for.

- The cloud VM is headless: tests set `SDL_VIDEODRIVER=dummy` (see
  `tests/conftest.py`). To check rendering visually, render a frame and save it
  with `doomer.client.headless.save_png`, then look at the PNG.
- Angles: radians in code, degrees in map files. `y` increases downward.
- Pin dependency versions in `requirements*.txt`.
- British English in docs and user-facing text.
