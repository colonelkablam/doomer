# Doomer: design

A Doom-style shooter in Python for home LAN play (Fedora and Windows clients,
Raspberry Pi 5 server), with co-op, deathmatch and a level editor.

## Architecture

- **Server** (Pi 5): authoritative. Owns player positions, health, scores and
  hits. Never renders. asyncio TCP, JSON lines, 60 Hz fixed tick.
  `TCP_NODELAY` on every socket.
- **Clients**: render locally with pygame-ce (numpy for per-column maths).
  They send inputs and receive state snapshots. Never pixels.
- **Model/view/input separation**: `doomer/shared` holds the map, player state
  and collision. It is standard library only (no pygame, no numpy), so the
  server runs it unchanged. `tests/test_architecture.py` enforces this.
- **One code path for input**: `PlayerInput` -> `step(world, inputs)`
  (`shared/sim.py`) -> new world state. Single-player calls it in a
  `LocalSession`; networked play calls it on the server (and later on the
  client for prediction). The client game loop talks to a `GameSession`
  Protocol, so swapping `LocalSession` for a `NetworkSession` doesn't change it.
- **Fixed timestep**: the model advances in fixed steps (`shared/timestep.py`).
  Rendering runs as fast as it can and interpolates the camera between the last
  two simulation states, so a 60 Hz simulation looks smooth on a 144 Hz screen.
- **Collision**: the player is an axis-aligned square (half-width 0.25 cells),
  moved one axis at a time, which gives wall sliding for free. Moves are
  sub-stepped so nothing can tunnel through a wall.

## Code design rules

The aim is that adding a feature (a weapon, a pickup, a door, a game mode)
means adding a module and registering it, not editing a dozen files.

1. **Dependencies point one way**: `client`, `server` and `editor` import
   `shared`; `shared` imports nothing of theirs. Enforced by
   `tests/test_architecture.py` (extend it as packages are added).
2. **Data and logic are separate.** State is plain typed `@dataclass(slots=True)`
   objects. Behaviour lives in functions that take state and return or mutate
   it. No god-objects with fifty methods.
3. **The simulation is deterministic and side-effect free.** `step(world,
   inputs)` (fixed `dt` from the world's config) touches no clock, socket, file or random module. Randomness
   comes from a seeded `random.Random` stored in the world. This makes it
   testable, replayable, and usable for client prediction.
4. **The simulation emits events, it doesn't call out.** A shot, a death or a
   pickup is appended to an event list (`ShotFired`, `PlayerDied`, ...). Sound,
   HUD, networking and scoring consume events. New consumers never need
   simulation changes.
5. **Registries for things that grow.** Thing types, weapons and game modes
   are looked up by name in a registry (a dict populated by a decorator or an
   explicit table). Adding a pickup = one new definition + one registration.
6. **Interfaces as `typing.Protocol`, not base classes.** e.g. an
   `InputSource` (keyboard, network, bot), a `Transport` (local, TCP). Swap
   implementations without inheritance trees.
7. **Wire format is defined in one place.** Every network message is a typed,
   frozen dataclass in `shared/protocol.py` with explicit `to_dict` /
   `from_dict`, plus the protocol version. Nothing else builds JSON by hand.
8. **Config is data.** Tunables (speeds, tick rate, resolution, key bindings)
   live in typed config dataclasses, loadable from TOML (`tomllib`), not as
   magic numbers scattered through the code.
9. **Logging via `logging`**, never `print`, outside CLI entry points.

### Deliberately not doing

- **Full ECS (entity-component-system).** It pays off with thousands of
  entities and many combinable behaviours. We'll have tens of entities, and in
  Python an ECS mostly adds indirection. The registry + event approach gives
  the extensibility without that cost. Revisit if entity behaviours start
  combining in awkward ways.
- **Plugin loading, dependency-injection frameworks, abstract factories.**
  Plain functions and Protocols are enough at this size.

### Tooling

- `ruff` for lint and formatting, `mypy --strict` for types, `pytest` for
  tests. All three must pass before a commit; CI runs them on Linux and
  Windows (`.github/workflows/ci.yml`).
- Every public function is type-annotated. Tests sit alongside every module in
  `shared` and `server`; rendering is checked by rendering to PNG.

## Networking

- Join handshake: client sends protocol version and name; server rejects on
  version mismatch, otherwise sends the map file text and the player id.
  (No map hash check: the server is the source of the map.)
- Client owns its view angle: turning is instant locally and the angle is sent
  with each input. Movement is server-authoritative.
- Remote players are interpolated between the last two snapshots.
- Client-side prediction of own movement only if it still feels laggy.
- Still to define: mid-game join, disconnect and timeout handling.
- Later: UDP broadcast so clients find the Pi without typing an IP.

## Coordinates

- Map units are cells. `x` increases to the right (east), `y` increases down
  the file (south), so the file reads like a top-down view.
- Angles: degrees in map files, radians in code. 0 = east; with `y` pointing
  down, positive angles turn clockwise on the top-down view (90 = south).
- Players have `z` and `z_velocity` from the start (zero until heights exist),
  so the protocol and collision code don't change shape later.

## Map format

One plain-text file per map in `maps/`, `*.map`. `#` starts a comment only at
the start of a line outside the grid sections.

```
name: Test Room

[walls]
#######
#.....#
#..2..#
#.....#
#######

[things]
# type        x     y     angle
coop_start    1.5   1.5   0
exit          5.5   3.5   0
dm_spawn      1.5   3.5   0
```

- `[walls]`: `.` = empty floor, `1`-`9` = wall type (texture), `#` = alias for
  `1`. Rows may be ragged; missing cells are walls. The map edge is treated as
  solid.
- `[floor]` / `[ceiling]` (optional): one digit `0`-`9` per cell, defaults
  0 and 4; same dimensions as `[walls]`. Height units TBD when
  milestone 4 starts.
- `[things]`: one per line. Positions are floats in cell units (`1.5` = centre
  of cell 1). Unknown types are an error at load time.
- Supported modes are derived, not declared: **co-op** if there is a
  `coop_start` and an `exit`; **deathmatch** if there are at least 4
  `dm_spawn`s. The server ignores things not relevant to the current mode.

## Rendering

- Grid raycaster (DDA) at 320x200 internal resolution, scaled up to the window.
- The raycaster returns a **list of hits per ray**, not just the first one, so
  floor/ceiling heights (milestone 4) extend it rather than replace it: draw
  front to back with per-column clipping, like Doom's clip arrays.
- Look up/down by y-shearing (moving the horizon), limited to about +/-30 deg.
- Other players and items are billboard sprites, depth-clipped against the
  per-column wall depth buffer.

## Gameplay defaults

- Weapons: hitscan first (a server-side ray tested against players as circles;
  reuses the DDA). No lag compensation needed on a LAN. Projectiles later.
- Respawn: deathmatch = spawn point farthest from other players; co-op = start.
- Scoring: frags, first to N.
- Sound: `pygame.mixer` on each client, triggered by events in snapshots
  (e.g. `{"ev": "shot", "by": 2}`).

## Level editor

Separate pygame-ce tool, top-down grid. Paint walls and wall types (number
keys), later floor/ceiling heights; place things. Shows mode readiness
("co-op ready", "needs 2 more DM spawns"). Saves the map format above.

## Milestones

| # | Milestone | Done when |
|---|---|---|
| 0 ✅ | Scaffold: layout, requirements, CLAUDE.md, pytest, headless PNG helper | `pytest` passes; a test frame saves as PNG |
| 1 ✅ | Map loader, fixed-timestep model, movement with wall sliding, flat-colour DDA raycaster at 320x200, mouse look | Walkable test map at 60 fps on Fedora; ray-distance and collision tests pass |
| 1b | Textured walls by wall type | Wall types visibly differ |
| 2 | Billboard sprites, depth-clipped | A static thing renders correctly in front of and behind walls |
| 3 | Server: handshake, 60 Hz tick, snapshots, interpolation, bot client | Two clients and a bot move together; protocol tests pass |
| 3b | Hitscan, health, death/respawn, DM scoring | A playable deathmatch |
| 4 | Multi-hit rendering, per-cell heights, step-up/gravity, y-shearing | Stairs and ledges render and are walkable |
| 5 | Level editor (pull earlier if hand-editing height grids gets tedious) | Round-trips a map; shows mode readiness |

## Setup

- Linux: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`,
  then `.venv/bin/python -m doomer.client` (single-player, `maps/test.map`).
- Client settings: copy `client.example.toml` to `client.toml` and pass
  `--config client.toml`.
- Windows: native Python (not WSL), venv, launched from `play.bat` which cds to
  its own folder, runs the client with `--host`/`--name` and ends with `pause`.
- Use the Pi's IP rather than `.local` names (Windows mDNS is unreliable).
- Server on the Pi: standard library only, no install step.
