# Doomer

A Doom-style LAN shooter in Python: raycast rendering with pygame-ce, an
authoritative server on a Raspberry Pi, co-op and deathmatch, and a level
editor. Early days: see [docs/DESIGN.md](docs/DESIGN.md) for the plan.

## Setup (Linux)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
```

```bash
.venv/bin/pytest -q
```
