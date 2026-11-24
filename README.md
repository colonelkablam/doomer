# Doomer

A Doom-style LAN shooter in Python: raycast rendering with pygame-ce, an
authoritative server on a Raspberry Pi, co-op and deathmatch, and a level
editor. See [docs/DESIGN.md](docs/DESIGN.md) for the design and milestones.

Current state: single-player walking around a textured map (milestone 1b).

## Play (Linux)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/python -m doomer.client
```

Controls: mouse to look, WASD to move, arrow keys also work. Esc releases the
mouse; Esc again quits. Click to recapture.

Options: `--mode coop|deathmatch`, `--config client.toml` (see
`client.example.toml`), or a different map path as the first argument.

## Develop

```bash
.venv/bin/pip install -r requirements-dev.txt
```

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy && .venv/bin/pytest -q
```

Render a frame without a display:

```bash
.venv/bin/python -m doomer.client.screenshot maps/test.map --at 9.5,2.5,90 --out out/frame.png
```
