# OK Doomer

A Doom-style LAN shooter set in a 1980s office block: fight haywire
photocopiers and angry bosses among beige PCs and water coolers, as the IT
Guy, the Caretaker or the Temp. Python, raycast rendering with pygame-ce, an
authoritative server on a Raspberry Pi, co-op and deathmatch, and a level
editor.

- Setting, characters, enemies, furniture: [docs/GAME.md](docs/GAME.md)
- Engine, code design and milestones: [docs/DESIGN.md](docs/DESIGN.md)

Current state: single-player on the ground floor: a two-storey glazed lobby
with the main doors, a reception desk and stairs up to a railed balcony; desk
pods you can jump onto; look up/down, jump and crouch. Multiplayer: a server,
the game joining it, and bots.

## Play (Linux)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/python -m ok_doomer.client
```

That opens the start screen: single player, join a game on your network
(or type an address), and your name (remembered). Arrow keys and Enter or
the mouse; Esc goes back.

Controls: mouse to look around (including up and down), WASD to move, arrow
keys also work, Space to jump (you can get onto desks), Ctrl or C to crouch.
Esc releases the mouse; Esc again quits. Click to recapture. F11 (or Alt+Enter)
toggles full screen; the window can be resized too. Tab shows the map; F3 the
network stats; the backquote key (`) a console of everything that happens.

Options: a map path as the first argument plays it straight away (no start
screen), as does `--connect HOST:PORT`; `--mode coop|deathmatch`, `--name`,
`--config client.toml` (see `client.example.toml`). In a game, Esc releases
the mouse and Esc again goes back to the start screen.

## Server and bots

Run a server, then join it from the game (and add bots to have company):

```bash
.venv/bin/python -m ok_doomer.server --port 5029
```

```bash
.venv/bin/python -m ok_doomer.client --connect 127.0.0.1:5029 --name ann
```

```bash
.venv/bin/python -m ok_doomer.bot --connect 127.0.0.1:5029 --count 3
```

From another machine, use the server's IP address instead of 127.0.0.1. To see
the games running on your network (servers answer on UDP port 5028):

```bash
.venv/bin/python -m ok_doomer.client --find
```

To feel a slower network on one machine, put the lag proxy in between and
join its port instead (here 30 ms each way, up to 10 ms of jitter, 1% of
data stalled as a lost packet would be):

```bash
.venv/bin/python -m ok_doomer.server.lagproxy --to 127.0.0.1:5029 --listen 5030 --lag 30 --jitter 10 --loss 1
```

```bash
.venv/bin/python -m ok_doomer.client --connect 127.0.0.1:5030 --name ann
```

In the game, F3 shows the connection's round trip, jitter, updates a
second and prediction corrections; `--no-predict` turns prediction off to
compare.

The server starts on the ground floor in co-op; `--map maps/x.map` plays a
map instead, `--config server.toml` reads settings (`ServerConfig` in
`ok_doomer/server/config.py`). Ctrl+C (or SIGTERM) stops it, telling
everyone. It needs only Python: no install step on the Pi.

## Develop

```bash
.venv/bin/pip install -r requirements-dev.txt
```

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy && .venv/bin/pytest -q
```

Render a frame without a display:

```bash
.venv/bin/python -m ok_doomer.client.screenshot maps/office.map --at 20.5,11.2,90 --out out/frame.png
```
