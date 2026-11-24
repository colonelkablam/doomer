# OK Doomer

A Doom-style LAN shooter set in a 1980s office block: fight haywire
photocopiers and angry bosses among beige PCs and water coolers, as the IT
Guy, the Caretaker or the Temp. Python, raycast rendering with pygame-ce, an
authoritative server on a Raspberry Pi, co-op and deathmatch, and a level
editor.

- Setting, characters, enemies, furniture: [docs/GAME.md](docs/GAME.md)
- Engine, code design and milestones: [docs/DESIGN.md](docs/DESIGN.md)

Current state: single-player walking around the ground floor, with solid desk
pods and counters (milestone 2b).

## Play (Linux)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/python -m doomer.client
```

Controls: mouse to look, WASD to move, arrow keys also work. Esc releases the
mouse; Esc again quits. Click to recapture.

Options: a map path as the first argument (default `maps/office.map`),
`--mode coop|deathmatch`, `--config client.toml` (see `client.example.toml`).

## Develop

```bash
.venv/bin/pip install -r requirements-dev.txt
```

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy && .venv/bin/pytest -q
```

Render a frame without a display:

```bash
.venv/bin/python -m doomer.client.screenshot maps/office.map --at 20.5,11.2,90 --out out/frame.png
```
