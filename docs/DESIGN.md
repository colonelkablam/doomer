# OK Doomer: design

A Doom-style shooter in Python for home LAN play (Fedora and Windows clients,
Raspberry Pi 5 server), with co-op, deathmatch and a level editor. Set in a
1980s office block. This file is the engine and code design; the setting,
characters, enemies and furniture are in [GAME.md](GAME.md); level design
rules and limits in [LEVELS.md](LEVELS.md). The Python
package is `ok_doomer` (the project name, `ok-doomer`, with the hyphen a
module name can't hold).

## Architecture

- **Server** (Pi 5): authoritative. Owns player positions, health, scores and
  hits. Never renders. asyncio TCP, JSON lines, 60 Hz fixed tick.
  `TCP_NODELAY` on every socket.
- **Clients**: render locally with pygame-ce (numpy for per-pixel maths,
  numba to compile the ray casting).
  They send inputs and receive state snapshots. Never pixels.
- **Model/view/input separation**: `ok_doomer/shared` holds the map, player state
  and collision. It is standard library only (no pygame, no numpy), so the
  server runs it unchanged. `tests/test_architecture.py` enforces this.
- **One code path for input**: `PlayerInput` -> `step(world, inputs)`
  (`shared/sim.py`) -> new world state. Networked play calls it on the
  server (and later on the client for prediction). The client game loop
  talks to a `GameSession` Protocol, so swapping `LocalSession` for a
  `NetworkSession` doesn't change it.
- **Single-player runs the server in-process** (decided for milestone 3, as
  Quake III and Minecraft since 1.3 do): `LocalSession` becomes a thin
  wrapper round the same `GameServer` core the Pi runs, joined by a direct
  in-memory link instead of a socket (no JSON). One code path for spawning,
  floor changes and pickups, so single-player exercises the multiplayer
  code. Hosting from the game is the same plus a TCP listener; the
  dedicated Pi server is the same core with no client in it. [built:
  `client/local.py`: `LocalSession` is a `NetworkSession` on a
  `LocalLink`, the server ticking in lockstep, one tick per input sent;
  it plays to the same numbers as calling `step` directly (tested), and
  costs ~0.2 ms a tick more (a snapshot each way, the predictor).]
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
   `shared`; `shared` imports nothing of theirs. The client also imports
   the server's core (`server/game.py`, for single-player in-process);
   the server never imports the client. Enforced by
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

- **Who does what.** The server owns the one real `World`: each tick it
  takes every client's queued input, runs `step`, and sends each client a
  snapshot and the tick's events; it decides anything two players could
  disagree about (who got the cup, hits, when the lift goes, the next
  floor), and handles joins, ids and timeouts. The client reads input and
  sends one input per tick, keeps a copy of the world (built from the map
  the server sent, its changing state overwritten from each snapshot),
  renders from it, and turns events into the HUD and sound; it decides
  nothing. Keeping a whole `World` on the client means every renderer
  helper works unchanged, and prediction is `step` run on that copy.
- **Inputs are per tick, numbered, and queued** on the server, one used per
  tick. "Use" is a press (this tick held, last tick not: `use_held`), so a
  server that kept only each client's latest input could lose a tap
  shorter than a tick. A player whose input hasn't arrived gets
  `sim.idle_input` (stands still, keeps their view and crouch); the server
  may repeat their last input instead.
- **Snapshots, measured**: the whole changing state of `office.map` with 4
  players is ~1.2 KB of JSON (12 doors, 10 cubicle doors, 12 pickups); at
  60 Hz to 4 clients ~2.3 Mbit/s, fine on a LAN. So full snapshots first,
  deltas only if a measurement says so. A tick with 4 players took 0.58 ms
  on the cloud VM (budget 16.7 ms); a Pi 5 is perhaps 2-3x slower.
- **Snapshots** (`shared/snapshot.py`): only what changes in play. The
  client builds its own `World` from the map the server sent, so a
  snapshot's doors, pickups, switches and dispensers go positionally (each
  a row of its changing fields, in map order); players go whole, with ids,
  as they join and leave. `capture(world)` and `apply(world, snap)`
  (players updated in place, so a session holding one sees it move).
  Each entity type's fields are listed as `STATE` (sent) or `FIXED` (by
  the map: a lift's `here`, a cubicle's `locked`); the tests check every
  field is listed, that every field play changes is sent (a run that uses
  every switch, dispenser and door, the lift arriving), and that a fresh
  world given a snapshot then plays on identically. Not sent: `rng` (the
  server's alone), events (alongside, in the same message). Entities
  that appear in play (enemies, drops) will need ids, as players do.
  815 bytes of JSON for `office.map` with 4 players, before rounding.
- **The protocol** (`shared/protocol.py`, version 2; bumped whenever the
  wire format changes, so different code refuses at once with both
  versions rather than half-working: a version 1 server, from before
  join events, let newer clients in and they never heard who came and
  went. `schema_fingerprint()` hashes every message, event and field,
  and a test pins it to the version): `Hello(version,
  name)` -> `Welcome(player_id, mode, config, map_text)` or `Reject`;
  then `Input(seq, input)` from the client each tick and `Update(ack,
  snapshot, events)` from the server each tick; `NewFloor(map_text)` when
  the lift arrives; `Bye(reason)` either way. One JSON object a line, a
  `type` key; `encode` and `decode` are the only way in or out. The
  config goes in the welcome so `step` runs the same on both sides.
  Events convert by walking the `Event` subclasses: a new event needs no
  protocol change. Everything decoded is checked (a client's data is
  untrusted): malformed is a `ProtocolError`, never a crash; numbers in
  inputs must be finite (NaN, `Infinity` and `1e999` refused: a NaN angle
  would spread through the world); lines over 1 MB are refused. An
  `Update` of the ground floor with 4 players, moving, is ~1.2 KB.
- **The server core** (`server/game.py`, `GameServer`): no sockets, no
  clock. The transport calls `join(hello)` (a `Welcome`, or a `Reject`:
  wrong version, full), `receive(player_id, message)` and `leave`, and
  `tick()` at the tick rate, which returns what to send whom (each
  client's `Update`, same snapshot and events, its own `ack`). Inputs:
  one queued input used per tick; more than `input_slack` (2) left
  waiting are merged into it (the later one's movement, a press of use
  or jump from either), since one arrives and one's used each tick, so a
  burst would otherwise stand as lag for good (measured: a 1.5 s first
  frame left 6 queued, 152 ms from key to screen; merged, 52 ms); the
  client also sends at most 2 inputs a frame; repeats and stale ones (seq not newer)
  ignored; the queue capped (`input_queue`, the oldest dropped, so a
  flooding client can't build up lag); a client with none queued has
  their last input repeated for `hold_ticks` (6: 100 ms, smoothing
  jitter), then stands still (one who dropped off doesn't walk on).
  Timeouts count the server's own ticks (`GameServer.ticks`: the world's
  start again on each floor), `timeout_seconds` of silence. The lift
  arriving: `NewFloor` to everyone, then the new floor's first `Update`;
  inputs queued for the old floor are dropped (and acked). Player ids are
  never reused. Names tidied (spaces collapsed, 16 characters, never
  empty). Settings in `server/config.py` (`ServerConfig`).
- **The transport** (`server/net.py`, `TcpServer`; run with `python -m
  ok_doomer.server`): asyncio, one thread, so the `GameServer` is never
  touched by two things at once. `TCP_NODELAY` on every socket (Nagle's
  algorithm would hold small packets back to batch them); a `Hello`
  within 5 s; anything that won't decode ends the connection with a
  `Bye` saying why; a client whose unsent data passes 1 MB is dropped
  (it can't keep up). The tick loop ticks by the loop's monotonic clock,
  each tick due `dt` after the last (no drift), skipping ahead after a
  stall of more than 8 ticks. SIGTERM (systemd stopping it) and Ctrl+C
  stop it cleanly, a `Bye` to everyone.
- **Bots** (`ok_doomer/bot`, `python -m ok_doomer.bot --connect HOST:PORT
  --count N`): standard library and `shared` only, asyncio, many to a
  process. A `WorldMirror` (`shared/mirror.py`: the client's copy of the
  world, built from the `Welcome`, kept in step by `Update` and
  `NewFloor`, numbering inputs and noting the `ack`) and a brain
  (`bot/brain.py`, `Wanderer`: walk ahead; stuck for 10 ticks, press use
  and turn a random way). The game client's `NetworkSession` uses the
  same mirror.
- **Load, measured** (the cloud VM, bots in the same process as the
  server): 60 ticks/s held at 4, 8 and 16 bots; tick median 0.96 / 1.32 /
  2.37 ms, p99 1.28 / 2.38 / 7.33 ms; sent 2.1 / 5.5 / 16.4 Mbit/s.
  Bandwidth grows with the square of the players (each update carries
  every player, to every client): fine on a LAN at 8.
- **Server CPU, cut ~3x** (profiled, then measured; a tick's work, 8 / 16
  players: 2.02 / 4.56 ms before, 0.69 / 1.46 ms after). Two-thirds of
  the simulation was collision checking every player against every desk
  and door on the floor: `sim.move_player` now keeps only obstacles,
  doors and cubicle doors within reach of the move (conservative; tested
  bit-identical to unfiltered over 3,000 moves aimed at all three, and
  each too-tight filter fails it). Encoding was a third of the rest, the
  same snapshot once per client: `protocol.Encoder` encodes a tick's
  shared snapshot and events once and splices on each client's `ack`
  (byte-identical to `encode`). Not done: rounding floats (would make the
  client's world differ from the server's by rounding, so prediction would
  correct every update) or deltas (not needed at these sizes).
- **The game client on a server** (`client/network.py`; `python -m
  ok_doomer.client --connect HOST:PORT --name NAME`): a `Connection` (a
  plain socket, `TCP_NODELAY`; a reader thread blocks on it and queues
  each message, since the game loop is pygame's, not asyncio's) and a
  `NetworkSession`, the same `GameSession` interface as `LocalSession`.
  It sends one input per tick (its own `FixedTimestep`), drains the queue
  once a frame onto a `WorldMirror`, and keeps players' poses in a short
  history on the server's timeline (tick x dt: receipt times bunch up
  when two updates land in one frame). It predicts you (below; without prediction, you're
  drawn 1 tick behind the newest snapshot, the server's word), and draws
  others 3 ticks behind (50 ms, room for jitter), interpolating between the snapshots
  either side, angles the short way round, never extrapolating. Other
  players are drawn as `office_worker` (`client/sprites.py`): a voxel
  model baked into 16 views (white shirt, red tie, navy trousers, a dark
  quiff; ~1.8 m), turned by the player's angle, so you see their back as
  they walk away. One look for everyone until the characters (GAME.md)
  are drawn. Posed (`client/people.py`): standing, a 4-pose walk (left
  foot forward, passing, right forward, passing; arms against the legs),
  crouching and jumping, each a model baked to its own 16 views and
  sprite name. Legs and arms are straight segments swung from hip and
  shoulder; the crouch leans the upper body 30 degrees about the hip (a
  resampled rotation: a shear would keep its height); smoothed normals
  (`VoxelModel.smooth`) stop staircase slopes shading in stripes. The pose
  is picked by `client/animation.py` (`Stride`): the walk advances by
  distance (`STRIDE_CELLS`, a cycle per cell, set by eye: the game moves
  at ~7x a real walk), so feet keep pace, a blocked player stops
  mid-stride, backwards runs it backwards; standing after 4 still frames.
  From the interpolated positions: nothing new on the wire. The crouch is
  33 voxels against the game's crouch height of 27 (a big head; a lean
  steep enough looks like falling): decide hit boxes with 3b. It shows:
  the reception counter's ledge (0.425) hides a crouched player's body
  (0.42) but not the sprite's head (0.52), which gets a name over it. The server
  going away or saying `Bye` ends the game (`GameSession.ended`).
- **A bad network on one machine** (`server/lagproxy.py`, `python -m
  ok_doomer.server.lagproxy --to HOST:PORT --listen 5030 --lag 30
  --jitter 10 --loss 1`): a TCP proxy delaying each direction by `lag`
  plus up to `jitter` ms. By TCP's rules: nothing lost or reordered; loss
  is a stall of 200 ms (about TCP's shortest resend delay) holding up
  everything behind it, as head-of-line blocking does. A separate
  process, so the server keeps no test-only paths.
- **Network stats** (F3 in the game; `NetworkSession.net_stats`): round
  trip (an input sent to the update confirming it, median of 60: wire,
  plus the waits for a tick at each end), jitter (spread of the gaps
  between updates), updates a second, inputs unconfirmed, prediction
  corrections. All measured by the client: nothing extra on the wire.
- **Who's here.** The server tells everyone when someone joins or leaves
  (`PlayerJoined`, `PlayerLeft` with why: quit, timed out, lost
  connection), as events: the HUD says "Bob joined", "Bob timed out"
  (your own joining isn't news); on joining, "Here: Bob, Cath" (from the
  first update; anyone joining in that same tick is in it, not news
  twice). HUD messages stack, up to 6 (`max_messages`), each for its own 3 s, so bots
  joining together all show (it was one line, the newest replacing the
  last). Messages are drawn into the frame like the names: the 5x7 font
  (`pixelart.FONT_5X7`, capitals, digits, the punctuation messages use;
  completed from the lobby sign's letters, which are unchanged), green on
  black panels, wrapped to the frame's width (`message_scale`). A test
  checks every character any message can produce is in the font. All
  the game's words are drawn this way now (no system font): a floor's
  name on arrival (5x7 at twice; the map's name under it unless it says
  the same), the "complete" banner across a black band, the fps and F3
  stats top left, "Click to play" (`Hud.draw_frame`, `hud.draw_text`).
- **Whose news.** Every event goes to every client, but your HUD tells you
  only yours and the team's (`hud.news_for`): someone else at a locked
  door, the cooler or a lift button isn't news to you (a stuck bot
  pressing use was filling everyone's screen); a pass anyone finds is
  ("BOB PICKED UP THE BLUE PASS": the team holds it); joins, leaves and
  the power are everyone's. Everything, everyone's, with who and the
  tick, goes to the console (backquote; `client/console.py`, the system
  font: a debugging tool), and to the log at debug level. Names over heads: the renderer records
  what it drew of each sprite, after walls and desks hid their share
  (`Renderer.seen`: pixels, top row, middle); a name goes just over the
  top of what's seen of someone, if at least a few pixels are and
  they're within 12 cells. Exact, with no depth guessing: a head over a
  desk gets a name, someone behind a wall doesn't. Drawn into the frame
  before it's scaled (chunky, like the rest): the 3x5 pixel font in
  capitals, phosphor green on a black panel, a terminal readout
  (`name_scale`, `name_panel` in the client's `[render]` config).
- **Finding games on the LAN** (protocol 3: `Seek`, `GameInfo`). A client
  asks, by UDP broadcast on 5028 (`client/discovery.py`, `GameFinder`);
  every server that hears answers the asker directly with its name
  (`ServerConfig.name`, else the machine's), floor, map, mode, players
  and TCP port (`server/discovery.py`, started by `TcpServer.start` given
  a discovery port; `discovery = false` turns it off). Asking rather than
  announcing: the list is there as soon as someone looks, and servers
  say nothing while nobody does. Seeks go to 255.255.255.255, each local
  network's own broadcast address (/24 assumed: the standard library
  can't ask for a netmask; on Windows the first can leave by a virtual
  adapter) and 127.0.0.1 (a game on this machine). A game heard two ways
  is listed once (`server_id`), by its LAN address; one silent for 3 s is
  forgotten; one from other code is listed as needing updating (its
  version and name are read before the rest, which may not decode). The
  responder can't be used to flood: datagrams over 1 KB ignored, small
  replies, one address answered at most every 0.2 s. `python -m
  ok_doomer.client --find` lists what it hears. Wi-Fi with AP isolation
  passes no broadcasts: join by address there.
- **The start screen** (`client/menu.py`; `python -m ok_doomer.client`
  with no map and no `--connect`, which still go straight in): SINGLE
  PLAYER, HOST GAME (to come), JOIN GAME (the games found, live, and an
  address to type), NAME (remembered: `client/settings.py`,
  `~/.ok-doomer/settings.json`), QUIT; keys or the mouse, Esc back a
  screen. Green 5x7 text, bare (no panels), on flat navy: the title in
  the middle, the items down the left, a `>` before the selected (the
  others indented to line up). On the right (main and name screens; the
  game list wants the width) one of the game's own objects, picked at
  random, turning once every 12 s (`client/showcase.py`): the Mac SE/30,
  the boss's PC, a cup of coffee. Box models, so drawn by the renderer
  itself, in an empty room from 48 points round: each view rendered with
  the object and without, the pixels that differ are the object (the room
  cut away). The camera's distance is guessed from the model's size, then
  measured and put right so its widest view fills 85% of a 120-pixel
  square. Baked on a thread (~0.5-1 s), the first view at once. (An 80s
  office block at night, turning, was tried first: not quite right.) A failed join says why on the
  menu; leaving a game (Esc twice) comes back to it, closing the window
  quits. `Menu` is the state and what keys and clicks do, apart from
  pygame's loop (`run_menu`) and the program (`start`), so it's tested
  without a window. The window is a `Display` (`client/app.py`), shared
  by the menu and the game (`play`). The font gained `>` and `_`; a test
  checks everything the menu writes is in it.
- **TCP and JSON lines** are fine on a LAN. Their weakness, head-of-line
  blocking (a lost packet holds up everything behind it until resent),
  matters on the internet, not here. Framing and encoding stay behind
  `shared/protocol.py`, so UDP stays possible.
- Join handshake: client sends protocol version and name; server rejects on
  version mismatch, otherwise sends the map file text and the player id.
  (No map hash check: the server is the source of the map.)
- Client owns its view angle: turning is instant locally and the angle is sent
  with each input. Movement is server-authoritative.
- Remote players are interpolated between the last two snapshots.
- **Client-side prediction of your own movement** [built:
  `shared/prediction.py`, on by default; `--no-predict` to compare].
  Built after testing showed "a little pause before I move": measured, a
  key press took 52-68 ms to show on one machine without it (a tick
  waiting to send, a tick waiting at the server, drawn a tick behind, a
  frame), 17 ms (one frame, as single-player) with it. Each tick the
  input moves a private copy of your player at once (`sim.predict_move`:
  look, walk, jump, fall, never use or pickups: those change the world
  and are the server's); each `Update`, the inputs the server has applied
  (`ack`) are dropped and the rest replayed from where the server has
  you. The same `step` code both sides, so walking, sliding along walls,
  jumping and crouching need no corrections at all (tested up to 200 ms
  each way); a disagreement (a door that opened sooner there, a merged
  input) goes into a drawn offset that fades over 0.1 s (a glide), more
  than half a cell snaps.
- Still to define: mid-game join, disconnect and timeout handling.
- Later: UDP broadcast so clients find the Pi without typing an IP.
- **Pickups, dispensers and meters are server state** (GAME.md, Pickups).
  The server runs the one deterministic `step`; it decides who touched a
  pickup first, applies damage and coffee's absorption, and hands out from
  dispensers. Clients predict their own movement only, never pickups or
  meters: they wait for the server, so a cup can't vanish for two players
  at once. In the world state:
  - `PlayerState.health` and `PlayerState.coffee` [both there; coffee
    topped up by cups, `pickups.PICKUP_EFFECTS` ({type: {meter: amount}})
    and `METER_MAX`; a full meter leaves a pickup lying; `Refreshed`
    events].
  - `World.pickups`: passes already, with a `taken` flag; coffee cups,
    snacks and the like join them, each pickup type's amounts from the
    thing registry (`health=`, `coffee=`; config as data).
  - `World.dispensers` [built: `shared/dispensers.py`]: per dispenser, its
    type (`DISPENSERS`: the meter, how much a use, the allowance per player)
    and what it has given each player, `{player_id: given}`. Used with E
    (switches first, then dispensers, then cubicle doors, then doors);
    `Dispensed` events,
    with a `reason` when nothing's given ("full", "empty", "no power"). A player who joins later starts with
    the full budget. On the main circuit (the vending machine), it works
    only while there's power.
  - Snapshots carry each client its own meters and budgets (a HUD hint
    when the cooler's nearly dry) and which pickups are gone; events
    (`PickedUp`, `Dispensed`, `Hurt`) drive the HUD and, later, sounds.
  - Deathmatch: pickups respawn (a tick to come back on), budgets refill
    on respawn.
  Build order: the data model now or with networking (so snapshots carry it
  from the start), damage and the HUD meters with weapons (3b).

## Coordinates

- Map units are cells. `x` increases to the right (east), `y` increases down
  the file (south), so the file reads like a top-down view.
- Angles: degrees in map files, radians in code. 0 = east; with `y` pointing
  down, positive angles turn clockwise on the top-down view (90 = south).
- Heights are in walls (1 = one storey, 2.7 m; a cell is as wide as a storey
  is tall). Players have feet height `z`, `z_velocity`, `on_ground`,
  `crouching` and `pitch`.
- **Vertical movement** (`shared/sim.py`, `MovementConfig`): gravity ~3.4 g;
  jump apex 0.32 (clears a desk at 0.25, not a counter at 0.375); body 0.67
  standing / 0.42 crouched, eye 0.6 / 0.35; steps up to 0.125 climbed
  automatically; crouching halves speed and holds while there's no headroom.
  Collision is height-aware: a cell blocks you if its floor is above your feet
  plus the step height, or its ceiling leaves no room for your body. You
  stand on the highest floor under your collision box. A box may always move
  out of a cell it's already in. Furniture obstacles still block at any height.

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

- `[walls]`: `.` = empty floor, `1`-`9` then `a`-`z` = wall types 1-35
  (texture), `#` = alias for `1`. Rows may be ragged; missing cells are walls. The map edge is treated as
  solid.
- `[floor]` / `[ceiling]` (optional): one character per cell, `0`-`9` then
  `a`-`z` for 10-35; defaults 0 and 8; same dimensions as `[walls]`. Heights
  are in **eighths of a wall** (~34 cm with a 2.7 m office ceiling), so a
  double-height atrium has `[ceiling]` `g` (16).
- **Low blocks (raised floors)**: a wall cell (digit in `[walls]`) whose
  `[floor]` value is 1-7 is a block of that height instead of a full wall:
  2 = desk (0.68 m), 3 = counter (1 m). Its sides use the wall type's texture;
  its top colour comes from `TOP_COLOURS` in `client/textures.py`. `0` (or 8+)
  keeps the full-height wall, so maps without `[floor]` are unchanged. A low
  block is also a raised floor: you can stand on it if you can get up (steps
  of 1 unit are climbed automatically; a desk needs a jump). Wall type `e` is a
  plain raised floor (galleries, steps). Things may stand on a low block and
  are drawn on its top. A raised floor must have a wall type (its sides need a
  texture), so `[floor]` on an empty `.` cell is an error.
- **Stairs** (wall type `h`): a stair cell's floor slopes up across the cell
  towards its `[facing]`, by half a storey (`STAIR_RISE` = 4 units); its
  `[floor]` value is the height at the top edge. Two stair cells climb a full
  storey. `GameMap.floor_at(x, y)` gives the floor at any point. Movement
  uses the smooth slope; the renderer draws real steps (see Rendering).
- **Balconies** (wall type `i`): a raised floor with a 1.1 m railing along
  every edge that drops to a lower floor. Where stairs arrive level with it
  there's no railing, so the opening appears by itself. Railings stop you
  walking off; plain raised floors (`e`) don't.
- **Ceilings**: `[ceiling]` sets each cell's ceiling. Walls are drawn up to the
  ceiling of the space in front of them; where the ceiling drops between cells
  a bulkhead face is drawn. Wall types may register an `UPPER` texture for
  the part above one storey (plain wallpaper above an office wall's dado).
  **No room over room**: one floor and one ceiling per cell, as in Doom. Other
  storeys of the building are separate maps.
- `[facing]` (optional): one of `n`, `e`, `s`, `w` per cell (`.` or missing =
  `s`): which way a cell's front faces, for wall types with different
  textures per side (desks: the front is where you sit).
- `[areas]` (optional): one of `o`, `e`, `l`, `s`, `c`, `w` per cell (`.` or
  missing = `o`): open-plan office, executive, lobby, service, lift car,
  washroom. Cosmetic (the server
  ignores it): it picks the cell's floor and ceiling textures, and how plain
  walls (`#`) look from that side. A plain wall face takes the look of the
  area it faces, so one wall can be stone on its lobby side and painted
  block on its plant-room side. A bulkhead (where the ceiling steps down)
  follows the same rule: it looks like the area it's seen from. Looks are
  in `client/areas.py`.
- `storey: <floor>` (optional header, after `name:`): which floor of the
  building the map is (`shared/building.py`; G if none).
- `[things]`: one per line, `type x y angle [key=value ...]`. Positions are
  floats in cell units (`1.5` = centre of cell 1). Unknown types are an
  error at load time. Properties after the angle are declared per thing
  type (`ThingType.properties`, with their allowed values): a door's
  `lock=blue|red|gold`. An unknown property or value is a load error.
- **`at=`** (any thing but a door; `shared/placement.py`): places the thing
  against a wall of its cell from its own size, so neither a hand-made map
  nor the generator works out coordinates. `at=n|e|s|w`: moved across to
  0.01 off that wall's face (its footprint, turned to its angle; flat
  things like posters have none), turned to face out of it; its position
  along the wall stays as written, so a row of servers or posters shares a
  wall. `at=nw|ne|sw|se`: into that corner, angle kept, plants (ficus,
  fern) turned up to 25 degrees by a random number seeded by their cell.
  No full wall there, or too big for the cell, is a load error. The parser
  resolves it (the sim and renderer see the final x, y, angle); the
  `Thing` keeps what was written (`written`) so `format_map` writes the
  `at=` line back. Why the thing places itself rather than the generator
  nudging x, y: the generator thinks in cells and walls, a model's size
  lives in one place (the thing type), and resizing one doesn't mean
  moving every copy in every map. A wall over a doorway isn't a wall cell,
  so signs above doors are still written by hand.
- Supported modes are derived, not declared: **co-op** if there is a
  `coop_start` and an `exit`; **deathmatch** if there are at least 4
  `dm_spawn`s. The server ignores things not relevant to the current mode.

## Rendering

- Grid raycaster (DDA) at 320x200 internal resolution, scaled up to the window
  (nearest-neighbour): a 4x window by default (1280x800), resizable, or full
  screen at the monitor's resolution (F11 / Alt+Enter, or `fullscreen`). The
  picture keeps its shape, centred; `integer_scaling` (default) scales by a
  whole number so every game pixel is the same size, with black borders
  (1600x1000 on a 1080p screen), else as large as fits (1728x1080), pixels
  unevenly sized. The HUD draws inside the picture (`app.view_rect`).
- **Two-phase frame**: the DDA (one ray per column, 320 a frame) is compiled;
  everything per-pixel is vectorised numpy over the whole frame.
- **Compiled ray casting** (`client/fastcast.py`): the renderer's DDA is the
  same algorithm as `shared/raycast.py`, compiled to machine code with numba
  (`@njit(cache=True)`: ~3 s to compile on first launch, cached on disk
  after). All 320 rays in one call, written as struct-of-arrays (field x span
  x column) that the numpy passes read directly; it also adds the camera's
  own-space spans, face texture slots and back-face flags. Why: in Python the
  per-column loop was ~40% of a frame, nearly all interpreter overhead (~17k
  Python calls a frame, an object per hit); compiled it's ~0.3 ms, and frames
  went from 12.5 to 8.4 ms in the lobby (9.0 -> 5.9 ms open plan) on the
  cloud VM. `shared/raycast.py` stays pure Python (the server will use it
  for hitscan, and it's the readable reference); `tests/test_fastcast.py`
  checks the two agree exactly, ray for ray. numba over C/C++: no compiler
  toolchain needed on Windows (wheels for Linux x86/ARM and Windows), and it
  stays Python. `NUMBA_DISABLE_JIT=1` runs it uncompiled, for debugging.
- **Per-pixel z-buffer** (w x h perpendicular distances). Walls write it;
  low blocks and sprites draw only where nearer, so a desk hides the bottom
  of the wall and of any sprite behind it regardless of draw order. (Doom used
  per-column clip ranges instead; a z-buffer was unaffordable in 1993, but is
  a handful of numpy comparisons now.)
- **Low blocks**: the ray continues through them (`RayHit` carries `top`,
  `base` and `exit_distance`) and stops at the first full wall. Each block is
  drawn as a front face (base..top at the entry distance, texture anchored to
  the floor) plus a top surface if below eye level. A row y on a horizontal
  surface at height h is at perpendicular distance
  `(eye - h) * projection / (y - horizon)`, the same similar-triangles rule as
  the floor. Blocks are drawn in layers (each column's k-th block), each layer
  cropped to the band of rows it can occupy.
- **Per-side textures**: a wall type may register a different texture for
  each side, from its own point of view: FRONT, RIGHT, BACK, LEFT
  (`client/textures.py`). Every ray hit records the compass face it entered
  and left by; the side is `(face - facing) mod 4`, with compass faces
  numbered clockwise from north. The atlas lookup is `slot_table[type, side]`.
  Types registered once use the same texture on every side. A back face seen
  through an open desk is viewed from inside, so its `u` is mirrored to match
  the outside view.
- **Masked textures (open desks)**: a texture may carry an opacity mask
  (`build_opacity()`, parallel to the atlas). Low-block faces skip transparent
  texels without touching the z-buffer, so the floor, walls and sprites behind
  show through the space under a desk. For see-through blocks the ray's exit
  face is drawn too (the desk's back legs), unless the ray goes straight into
  another block of the pod. Full walls ignore masks: the ray stops at them, so
  there's nothing behind to show. Costs ~2-3 ms with desk pods filling the
  view (~8.5 ms/frame total): the first place to optimise if needed.
- **Contact shadow**: floor seen through the gap under an open desk, at a
  distance between where the ray enters and leaves the desk's cell, is
  darkened by a fixed number of colormap levels, so desks sit on the floor
  instead of floating.
- **Texture tiling rule**: frames and trims are split evenly between opposite
  edges of a tile, and desk legs are flush with face edges, so blocks line up
  and corners meet (see `client/textures.py`).
- **8-bit indexed colour.** Every pixel is a byte indexing a 256-colour palette
  (`client/palette.py`), laid out as `index = (ramp << 4) | shade`: 16 hue
  ramps x 16 brightness steps. Brightness is linear in the low nibble, so
  darkening never changes hue and can be computed exactly.
- **Light by lookup table**: `COLORMAP[level, index]` (32 light levels, like
  Doom's COLORMAP) gives the darkened index. Shading a frame is one numpy
  fancy-index.
- **Textures are 64x64**, a power of two (coordinates wrap with `& 63`), stored
  as palette indices, transposed to `[u][v]` so a wall column is contiguous.
  64 matches the 320x200 resolution: larger textures would be minified and
  alias (shimmer) without mipmapping. Currently generated procedurally with
  numpy (`client/textures.py`), registered per wall type; slot 0 is a magenta
  checker for missing textures. Imported image files can come later
  (quantised to the palette).
- Ceiling and floor darken with distance, with 4x4 ordered (Bayer) dithering to
  hide banding. Precomputed, so free per frame.
- **Camera**: position, angle, eye height `z` and pitch. A point at height h and
  perpendicular depth d is drawn at row `horizon + (eye - h) * projection / d`;
  everything uses that formula or its inverse.
- **Look up/down by y-shearing**: `horizon = middle + tan(pitch) * projection`,
  clamped to +/-25 deg. Verticals stay vertical (not true pitch, as in Heretic
  and Duke Nukem 3D).
- **Span renderer**: the raycaster reports changes of space (low blocks,
  ceiling changes); each change can draw a riser face, a bulkhead face, and
  floor/ceiling surfaces at its heights until the next change, z-tested. The
  default floor (0) and ceiling (1) are a background cached per (horizon, eye
  height). Runs of identical blocks are merged into one hit. Each layer is
  drawn in a lower pass (risers, back faces, shadows, floors) and an upper
  pass (bulkheads, ceilings), each cropped to its rows. Explicit surfaces are
  dithered like the background. A ceiling stretch carries on over desks
  inside it (desks don't interrupt the ceiling); the camera's own space is
  split into its ceiling (to the next ceiling change) and the floor it
  stands on (to its cell's edge). ~4 ms in corridors, ~10 ms among desk pods
  or in the double-height lobby.
- **Textured floors and ceilings** (`client/areas.py`, the kernel in
  `client/fastcast.py`): floor casting. Each floor or ceiling pixel's
  distance (the row formula, inverted) puts it at a world point, camera +
  ray x distance; the texture is the one for that cell's area, and the
  texel `(frac(x), frac(y)) * 64`. The default floor and ceiling are one
  compiled pass that also shades, lights and dithers (the light map still
  sampled once per 2x2 pixels); ceilings at other heights (the atrium) and
  the tops of raised floors (`areas.TOP_KIND`: raised floors' carpet; a
  balcony takes its area's floor;
  desk tops stay flat colours, with their seams) use the same lookup per
  pixel. Textures come from a table by (area, kind): floor, ceiling, soffit
  (a ceiling lower than a storey, over a doorway: a lining, not ceiling
  tiles), then the raised tops. Desk pairs' tops are per cell instead
  (`areas.cell_tops`: a kind past the table's columns reads a per-cell
  grid): mahogany with paperwork, four variants by position, turned to the
  pod's facing. Paperwork as texture, not boxes: flat things cost nothing
  that way, as the pixels are shaded anyway; each box adds a slab test per
  screen column it covers. Stair treads are box tops, stretched once
  over each step (a cell wide, an eighth deep), so their carpet is drawn
  pre-squashed 8:1 to come out square.
- **Texture shading rule**: a texture's gentle variation (noise, ± a
  fraction of a shade) must sit mid-shade, e.g. 10.0 ± 0.25, never 8.5 ±
  0.5: textures are quantised to whole shades, and variation that crosses a
  step turns into big blotches up close. **Mipmaps**: seen at a slant, many texels fall
  in one pixel and a fine pattern shimmers as you move. Each texture has 7
  levels, each averaging 2x2 texels of the last (in RGB, then the nearest
  palette colour) down to one colour; a pixel takes the level where a texel
  is about a pixel: log2 of its footprint in texels, depth / projection
  across the ray, x depth / eye height along it at a slant. The level is
  Bayer-dithered between neighbours so there's no seam. Cost: ~1 ms per
  frame over flat colours (the level splits into a per-row and a per-column
  part, so there's no log per pixel). `RenderConfig.floor_colour` /
  `ceiling_colour` draw flat colours instead (tests use them).
- **Default ceiling depth**: background ceiling pixels get their real depth in
  the z-buffer wherever the default ceiling really is the ceiling, so a low
  ceiling hides a higher one beyond it (looking from a normal room into the
  atrium). Inside stretches with a different ceiling it doesn't exist, so
  those pixels stay infinitely far. The default floor stays infinitely far
  (open floor is never raised).
- **Sloped floors**: hits carry the floor height where the ray enters and
  leaves a cell (`top`, `exit_top`); along a ray a slope's height is linear,
  so each pixel's distance is closed-form: `d = (eye - top + m near) /
  ((y - horizon) / projection + m)` (m = 0 for flat floors). Surfaces may be
  above the horizon. Stair cells are not drawn this way any more (their span
  draws nothing; it still carries heights for the spans around it): their
  steps are box models, below.
- **Desk-pair seams**: the top of a back-to-back desk pair (`d`,
  `textures.TOP_SEAM_WALL_TYPES`) gets a dark line across its middle,
  perpendicular to its facing, where the two desks meet, and a fainter join
  at each cell edge shared with another pair facing the same way (between
  pairs in a row, not at its ends). Each top pixel is put back into the
  world (camera + ray x distance), and its own cell looked up in a per-map
  grid of pair facings (a ray merges a run of identical cells into one span,
  so the span's first cell isn't enough); the line is at least half a pixel's footprint wide (depth^2 /
  (projection x eye height above the top) along the ray, at a slant), or it
  falls between pixel centres and vanishes.
- **Railings** are drawn above balcony edges where the ray climbs onto the
  balcony (from below) or leaves it over a drop (from on top), as a masked
  texture.
- **Sparse pixel work**: in the lower pass, masks are built with cheap
  comparisons over the band of rows, then textures, shading and dithering
  are evaluated only at the pixels drawn (`np.nonzero`). Texture slots per
  (cell, face) are precomputed per map. With these, the double-height lobby
  with its stairs and balcony is ~13-14 ms/frame, the tightest part of the
  map.
- **Ceiling lights** are panels set into the ceiling, not sprites: ceiling
  casting. Each `ceiling_light` thing marks a cell; the panel's corners are
  projected to a screen bounding box, and for those pixels the world point on
  the ceiling plane decides whether it's inside the panel: the middle 2 x 2 ceiling tiles (1.35 m square), the grid's bars carried across it. Drawn
  only where nearer than what's there (the default ceiling's own depth where
  the z-buffer is empty). Full-bright diffuser, shaded grey frame.
- **Sprites** (`client/sprites.py`): 64x64 palette art with an opacity mask,
  spanning one cell wide by one wall tall and standing on the floor, so texel
  size matches the walls; ceiling-hung things are drawn at the top of the
  image (`on_ceiling`, so they ignore desks below). Keyed by thing type name;
  things without art (spawns) are invisible. A sprite in a low-block cell
  stands on the block's top (`SpriteInstance.z`). Z-tested per pixel.
  `fullbright` sprites (lights, signs) skip distance shading.
- **Rotating sprites** (`client/models.py`): a sprite can have 16 frames,
  one per 22.5 degrees round the thing (Doom had 8); the renderer picks the
  frame from the direction the camera is in relative to the thing's map
  angle. Frames aren't hand-drawn: the thing is a small voxel model (boxes in
  a 64^3 grid, 1 voxel = 1 sprite texel), rendered orthographically to each
  view at startup (~25 ms) with per-voxel normals lit from a light fixed to
  the model, so all views agree. Views look down 10 degrees, since you mostly
  see furniture from above; the nearest parts then hang below the base line
  (`sink` rows), and the sprite is z-tested from its nearest point (`reach`)
  so a desk top in front of its centre doesn't hide them. Meant for enemies
  (a model per animation pose) and irregular things; the beige PC
  (`terminal`) uses it.
- **Box models** (`client/boxes.py`): small box-shaped things drawn as real
  geometry: a few axis-aligned boxes in the thing's frame, turned to its map
  angle, a 64x64 texture per face (with an opacity mask). Each box gets the
  screen columns its bounding circle can cover (`_in_view`; conservative,
  so most boxes, behind or beside you, are dropped and far ones cover a few
  columns), and each (box, column) pair is slab-tested in one numpy pass;
  the face a ray enters by is drawn like a wall slice at the entry depth,
  the top (or underside) like a floor between entry and exit depth, then
  see-through texels are dropped and all box pixels are z-tested together.
  Correct from any angle and height with no frame popping, which sprites
  can't do. ~2.5 ms for ~220 boxes in view (testing every box against all
  320 columns took ~10). Box-model things are drawn within 16 cells. Box
  models: the Mac SE/30 (`mac`, with keyboard and mouse), the filing
  cabinet, the water cooler and the ficus. A box can be turned about the
  model's origin (`Box.turn`); round things are octagonal prisms made of
  four strips, two of them turned 45 degrees (two squares would make an
  eight-pointed star: a union keeps the corners). `Box.glow`: lights itself
  (lit signs), full brightness regardless of light map and distance. Text
  is one thin box per letter, each with its own see-through letter texture
  from a 5x7 font, since one texture stretched across a whole sign would
  smear it. Foliage is crossed cards:
  thin boxes with see-through leaf textures, four upright at 45 degrees to
  each other and two flat, as games have long done for plants. Big
  furniture stays as map cells.
- **Cubicle doors** (`shared/cubicles.py`, drawn by
  `boxes.cubicle_door_boxes`): each run of toilet cubicles has a door per
  cubicle in `World.cubicle_doors`, how far open (0-1, swinging over 0.3 s)
  and whether it's locked (ENGAGED: the second of each run, for good). Use
  (E) toggles a vacant one, from outside or in; a locked one emits
  `CubicleDoorUsed` with `engaged` and the HUD says someone's in there.
  Hinged on the cubicle's +y side (so a run's first door folds back against
  the end wall), they swing out, a quarter turn. Collision follows the
  state: shut (under half open), a door is a thin wall across the
  cubicle's front; open, its leaf is a thin obstacle standing out into the
  room. A shutting door waits while someone's in its sweep (it never shuts
  through anyone); an opening one doesn't (whoever it reaches walks out of
  the leaf, as obstacles let a box that's already overlapping them leave).
  The run's partitions, pans and cisterns are footprints like any
  furniture's. A cubicle is sized for a player (0.6 wide, its door 0.52 between
  fixed returns, 0.9 deep: 0.54 clear in front of the pan), so you can shut yourself in, and jump onto
  the toilet's lid so your feet don't show under the door. The run's fixed
  parts (partitions, headrail, toilets, the feet under the engaged door)
  are an ordinary box model; the doors are drawn from their state, a box
  turned about the model origin by the swing, offset so it turns on its
  hinge.
- **Standing on furniture**: an `Obstacle` carries its footprint's `top`.
  It blocks only a body whose feet are more than a step below that
  (`collision.obstacles_in_the_way`), and you stand on the highest top under
  you that you're up on (`collision.obstacle_support`, with the floor in
  `sim.standing_height`). So furniture low enough to jump onto (apex 0.32)
  can be stood on, as desks can: a toilet's lid (0.18), a coffee table,
  a sofa, a vanity counter; taller things (a counter, a cabinet, the
  boss's desk at 0.33) still block. Things with only a radius block at
  any height. The map checker uses the same rules.
- **Lifts and the building** (`shared/doors.py`, `shared/building.py`):
  the building's floors are data (`FLOOR_LABELS` bottom to top, B3 to the
  penthouse; `FLOOR_MAPS` the ones built), and a map's `storey:` header
  says which it is. A lift `Door` carries where its car is (`car_floor`, a
  floor index, fractional while moving), this floor's index (`here`),
  `called`, and `arrive_in`. The call button (E outside) brings the car,
  a floor every `FLOOR_SECONDS`, and opens it; E inside the car (its
  panel, whichever way you face) shuts or opens the doors; shutting doors
  reopen for anyone in them (`in_lift_doorway`). The ride (`sim._ride`):
  once every player is in a car that's here, shut, unlocked for the team
  and not just arrived, `World.ride` runs for `RIDE_SECONDS`, the occupied
  cars' `car_floor` climbing a floor (the panel counts), then
  `LevelComplete(next_floor)`. The session then moves the team on
  (`spawn.arrive_on_floor`: a new world, every player at its
  `coop_start` in a car, health and coffee carried, passes not); the
  server will do the same, and tell the clients. A floor whose `coop_start` is in a
  car is an arrival: that car starts here and opens after
  `ARRIVE_SECONDS`. The map checker treats it as open, and counts a lift
  as a way out only once the team can unlock it.
- **Doors** (prototype; `shared/doors.py`, drawn by `boxes.door_boxes`): a
  door is a thing in a doorway cell (open floor between two wall cells),
  standing at one end of it, `DOOR_SET` (0.45) from the cell's middle:
  0.05 cells (~13 cm, a frame's depth) in from that edge, flush with the
  room it serves, the rest of the doorway a recess on the other side (a
  corridor's). Hinged and double doors stand at the end away from their
  swing (facing) and swing out into that recess: a leaf (0.6) is shorter
  than the recess (0.95), so the swing never reaches the corridor itself or
  the room's furniture. A leaf swings through anyone in its way (collision
  is only the doorway's closed part); it just won't shut on them. (Pushing
  people out of the swing was tried: walking in from the corridor, through
  the swing, you were pushed back every tick, so doors only worked one
  way.) Sliding and lift doors stand at the end they
  face, flush with that room; their leaves slide inside the walls either
  side, not over the room's wall face (on the edge itself they'd poke out
  and flicker against it).
  `Door.plane` is that plane (collision, light); fittings (card readers, a
  lift's surround) keep the doorway's centre. The doorway's side walls
  (plain wall faces facing into a door's cell) take a door lining
  (`areas.REVEAL_WALL_TYPE`: plaster, timber architraves at each end), as
  a doorway belongs to neither room. State (how far open,
  opening or closing) lives in the simulation's `World.doors`, so the server
  will own it; the client draws boxes from it each frame (passed to the
  renderer as `boxes`). Hinged: two fixed jambs and a leaf whose model
  origin is the hinge, turned by how far open it is, so it really swings;
  use (E) opens and closes it, and it won't shut on someone in the doorway.
  Double: two leaves hinged at either side of the doorway, meeting in the
  middle and swinging the same way, between narrow jambs; used the same way.
  Sliding: two glass leaves (a dithered, see-through texture) that part
  towards the walls either side, stopping with ~32 cm of each still in the
  doorway (`SLIDE`, as real ones park behind fixed side panels; lift doors
  the same), opening by themselves for anyone within 1.6
  cells. Collision: the closed part of the doorway as thin walls. A lowered
  ceiling over the doorway (6/8) gives a lintel through the existing
  bulkhead drawing; with lighting on, a ceiling lower than a storey (the
  lintel's soffit) takes the plain light times `soffit_shade` (0.85), not
  the square root other ceilings get, so it sits in shadow like the walls.
- **Circuits and powered things**: a thing is on the main circuit unless
  its map line says `circuit=emergency` (ceiling lights, and things with
  screens: `mac`, `pc`, `terminal`). Its boxes carry that
  (`PlacedBox.emergency`, from `SpriteInstance.emergency`), and the
  renderer decides per box whether it's live: the floor has power, or
  it's on the emergency circuit (`render._powered`). A dead box on the
  mains with `off_faces` (a screen) shows those, unlit; a dead indicator
  lamp isn't drawn; a dead mains glow (a lit sign) is lit like anything
  else. Screens are boxes of their own over the case (the Mac's too), so
  only they change. The lifts' indicators, call buttons and car panels
  work the same way, so `door_boxes` takes no power: the renderer decides.
- **Indicator lamps** (`Box.indicator`, `blink`, `blink_phase`;
  `boxes.lit_now`): a small lit box on a panel, drawn only while lit, the
  dark socket painted on the panel behind showing otherwise. Unlit when it's
  on the mains (`mains`) and there's no power, or in the off half of its
  blink (seconds per cycle; things of a model offset by where they stand,
  so a row doesn't blink in step). The breaker's lamps (battery: red
  blinking, then green) and the servers' (mains) use it; the renderer
  filters lamps by `now` and `power` before drawing boxes.
- **Things that move by the clock** (`boxes.SPINNING`): the renderer takes
  `now` (seconds) and turns those box models by it, with a small bob, as
  an ID pass card does. Cosmetic, so not part of the simulation: each
  client turns its own.
- **The automap** (`client/automap.py`): Tab toggles it; it draws an 88 x 88 map into
  the 8-bit frame's top-right corner before it's scaled up (chunky pixels
  like the rest). North up: each zoom (2, 4, 8 px a cell) has an image of
  the whole floor drawn once (`floor_image`: wall faces, desk outlines,
  door lines, lift cars); each frame a square is cut from it round you,
  masked to the cells you've seen, given a one-pixel shadow, and written
  over the scene (see-through). Seen cells: each frame, every 4th
  column's ray is sampled out to the z-buffer's depth at the horizon row
  (`Renderer.eye_level_rays`) and the cells it crosses are marked.
  Client-side state.
- **The map checker** (`shared/validate.py`, `check_floor`; on the command
  line `python -m ok_doomer.shared.validate maps/*.map`): can a co-op floor
  be finished? It floods the space a player can reach from the co-op start
  on a quarter-cell grid of positions, each step tested with the
  simulation's own `move_box` and height rules (steps, a jump onto a desk,
  railings, furniture, cubicle screens, door leaves), so it can't disagree
  with how players move. Doors open only if the team can open them (its
  pass in hand; a lift's with power); passes and the breaker within the
  game's reach are taken, which can open more: flood again until nothing
  changes. Problems come back as sentences ("the red pass is only
  reachable through the red-locked door at (11.5, 12.5)"). A test runs it
  on every map in `maps/`; the level generator will run it on what it
  makes. ~1 s for the office.
- **Posters** (`poster` things, `kind=`): one thin box each against a
  wall, drawn as `type:kind` from BOX_MODELS. The fire plan (`kind=plan`)
  is drawn from the map with a dot where it hangs, so its texture is made
  when a map's boxes are (`Renderer._map_boxes_for`, `boxes.poster_plans`):
  box textures can grow after import (`boxes.box_textures()` restacks them
  when they do).
- **The view through glass** (`textures.panorama`, `Renderer._outside`):
  glass in wall textures (windows `4`, curtain wall `f`, entrance doors
  `g`) is a reserved palette index, `GLASS`; when walls are drawn, those
  pixels show a panorama instead, looked up by the column's ray angle
  (512 texels round) and the row's height over the horizon (a tangent,
  square texels), as Doom draws its sky: infinitely far, so it turns with
  you but never moves as you walk. Unshaded (the city lights itself). Only
  computed when glass is on screen: ~0.3 ms a frame. Ground floor for now;
  higher floors would lower the horizon line of the ground (more city).
- **Light through doorways lights floors, not walls above**: the light map
  is a floor plan, and faces take their light from the floor just in front
  of them; so a pool of lobby light through the glass doors lit the face
  over the doors on the office side, which that light passes under and
  can't reach. A second map (`LightMap.upper`, `high()`) is baked with
  lowered ceilings (lintels) and every door plane stopping light; faces
  above head height (bulkheads, the face over a doorway) take their light
  from it. Floors use the first. A third (`LightMap.walls`,
  `wall_faces()`), baked only when the map has glass doors, has every door
  plane stopping light but no lintels: full walls and full-height ceilings
  take their light from it, so light through a glass door lights the floor
  beyond, not the walls either side (it only grazes them) or the ceiling
  (out of its line through a door-high opening). In the dark this showed:
  the exit sign over the office side of the glass doors lit the lobby's
  wall up its two storeys, and a pool on its ceiling. Soffits (low, over a
  doorway) keep the floor map. (Three bakes: ~0.2 s for the office on this VM;
  per frame, no extra lookups: each pixel reads one map or the other.)
- **Furniture footprints** (`things.Footprint`, `shared/furniture.py`): a
  blocking thing type can give its floor plan as rectangles in its own frame,
  each with a top height, instead of the square `radius`. Placed at the
  thing's angle (snapped to quarter turns), they are its collision
  (`collision.Obstacle` is a rectangle) and the surfaces other things rest
  on: a thing inside another's footprint stands on its top
  (`resting_height`, used for sprites, box models and pickups). The
  reception desk is the first: a ledge at the front (~1.15 m), a work
  surface behind it (~0.73 m).
- **Lifts** (`door_lift`, `shared/doors.py`): brushed steel doors parting
  like the glass ones. A lift is one cell off its lobby: the car is the
  doorway cell itself (`Door.car`), walled in on three sides, its doors its
  outer wall at the lobby end (they face out), under a 6/8 ceiling (~2 m). Use calls the lift, only with
  power (else `LiftNoPower`); a called lift's doors open for anyone near.
  When every player stands in a lift car, the doors close, and once those
  with people behind them are shut the floor is done (`LevelComplete`,
  `World.complete`; the HUD shows a banner). Its fittings are box models
  drawn from its state (`boxes._lift_fittings`), kept simple: a steel
  architrave on the lobby wall and a sill; a plain call plate beside it
  with one round button (amber once called); a black indicator strip over
  it (an amber G with power); inside the car, a panel of round buttons by
  the door (G lit with power). Co-op needs a lift or an `exit`. Riding to the next floor comes
  with the tower of floors.
- **Dark rooms and the palette**: each ramp's 16 shades are evenly spaced
  (shade s is (s + 1)/16 of full), so dim surfaces sit where steps are big:
  shade 2 -> 3 is +33% (12 -> 13 is +8%). Texture detail there either
  vanishes or jumps a whole step, and dithering between two far-apart
  shades shows as a checkerboard: dark rooms looked blotchy. Measured
  options: spacing the shades on a curve only helps the very darkest
  (step at 10% brightness 62% -> ~45%; at 20% no better) and coarsens the
  bright end (6% -> 10-13%), so the palette stays linear. Instead, the
  baked light is bent by `LightingConfig.lift` (0.65): light^0.65 raises
  dim light most (0.1 -> 0.22, 0.3 -> 0.46; 0.9 -> 0.93), putting dark
  surfaces where shade steps are smaller. A power cut stays dark; what
  disappears is the murky middle. Truecolour lighting (multiplying RGB,
  not palette shades) would remove the steps entirely, at the cost of the
  banded 8-bit look; not done.
- **Power** (`shared/switches.py`): level flags in `World.flags`, set by
  switches. The breaker sets `power`; a map with a breaker starts without it
  (a map without one is powered). Use (E) near the breaker, facing it,
  throws it (`SwitchThrown`); it's one-way. The client draws the breaker from
  its state (lever down/up, lamp red/green). Without power, the light map
  is recomposed from the panels' patches: a lower ambient
  (`unpowered_ambient`, 0.1) plus only the emergency lights: exit signs
  (`exit_sign`, a dim light at each, `exit_sign_intensity` 0.7 of a panel,
  so pools of light mark the doors on the way out) and any
  `circuit=emergency` panels; main panels draw dark; ceilings take the plain light instead of its square root (with
  little light there's little to bounce); and glowing boxes marked `mains`
  (the lobby sign) are lit like anything else until the power's back.
- **ID passes and locks** (`shared/pickups.py`, `door lock=`): passes are
  pickups (walk within 0.75 cells, about an arm's reach beyond your box, at
  most 0.5 storeys above or below your feet); the team shares them
  (`World.passes`), so any player can open a door locked with that colour.
  A locked door stays shut for a team without its pass (using it emits
  `DoorLocked`, which the HUD shows); a card reader on the wall either side
  shows the colour wanted and a lamp, red until the team holds that pass.
  Deathmatch (`GameMode.all_passes`) starts with every pass. The HUD
  (`client/hud.py`) shows passes held and messages from events.
- **Cubicle screens** (`shared/screens.py`, drawn by `boxes.desk_screens`):
  on `j` desk pairs, thin world-aligned panels (5 cm thick, to ~1.25 m) down
  the middle of each pair and across every cell edge (a shared edge once).
  Worked out once per map in `shared/` so the simulation and the renderer
  agree; drawn as boxes, built once per map with the stairs. Collision: thin
  walls (`collision.ThinWall`) resolved per axis like obstacles, active only
  while your feet are between the desk top and the screen top (the railing
  rule), so they shape movement on the desks and nowhere else.
- **Stairs as steps** (`boxes.stair_boxes`): each stair cell is drawn as 8
  step boxes (1/16-storey risers, 1/8-cell treads), built once per map and
  never distance-culled. Each step's top is level with the slope at its back
  edge, so feet following the slope are at most a riser below the tread.
  Step sides are textured in world units (`tiled_sides`) so the 8 read as one
  wall; a step's front and back faces are drawn only one riser tall at the
  top (`riser`), since the step in front hides the rest. ~2 ms in the lobby.
- **Blocking things** (`ThingType.blocking`, `radius`) are square footprints
  that the shared collision code treats like walls.

### Lighting [planned]

Today brightness depends only on distance from the eye (Doom's "fog",
`shade_distance`), plus a fixed darkening of north/south faces. Ceiling
light panels are drawn full-bright but light nothing. The plan: light that
comes from the fittings, keeping the 8-bit palette and the one-lookup shading
(`COLORMAP[level, texel]`), and the frame budget.

**Model: a baked light map** (as Doom's per-sector light and Quake's light
maps, simplified). Per map, at load, client-side:

- A grid of light values over the floor plan, 4 samples per cell (0.25 cell
  resolution; 96 x 64 floats for the office). Each sample:
  `ambient + sum over lights of intensity * falloff * visible`.
- **Falloff**: from the panel at its ceiling height `c` above the floor,
  `1 / (1 + (d / r)^2)` with `d = sqrt(horizontal^2 + c^2)`, cut off at ~6
  cells. A two-storey ceiling then gives dimmer, wider pools without a
  special case.
- **Visibility**: a sample is lit by a panel only if the segment between
  them crosses no full wall (grid DDA, cell resolution) and no shut opaque
  door (hinged, double, lift: the map is baked once, with doors shut; glass
  doors let light through). A door blocks at its plane only, so the
  doorway either side of it is lit (the lift doors' lobby face). Without
  this, lift cars' panels shone out through their doors and lit the lift
  wall in patches. Low blocks don't block light (the contact shadow under
  desks already exists).
- **Ambient**: a map header value (default ~0.35), with an optional
  `[light]` grid later for per-cell overrides (a dark server room, a lobby
  lit by its glazing).
- Each panel's contribution is kept as a separate patch (its radius only),
  so dynamic changes add or subtract one patch, not a rebake. Bake: well
  under 0.5 s; cached per map.

**Applying it**: brightness = light(x, y) x distance fade (the fade stays,
gentler, for atmosphere). Both multiply into one colormap level, so every
surface is still a single lookup.

- Walls and block faces: one light value per column, sampled just in front
  of the face (0.1 cell into the space it faces). Cheap.
- Floors and ceilings: no longer one cached background. Floor-cast: each
  pixel's world position (camera + ray x distance, as the desk seams do)
  samples the map bilinearly. Ceilings use a flatter curve (lit mostly by
  bounce). The biggest cost: ~2-3 ms at 320 x 200; if needed, evaluate at
  half resolution and let the dither hide it.
- Desk tops, box models: per pixel from their world positions (already
  computed). Sprites: one value at the thing.
- Light panels and the exit sign stay full-bright (emissive). Light stays
  white: the palette's ramps can darken a colour but not tint it.

**Dynamic light** (cosmetic first, gameplay later):

- Flickering or dead tubes (a `ceiling_light` variant): toggle that panel's
  patch; pattern seeded, client-side.
- Muzzle flashes, explosions: a short-lived patch at the source, added to a
  frame-local copy of the affected region.
- Power events (a switch; the CEO cutting the lights, emergency lighting
  only): the simulation owns which lights are on, so all clients agree; it
  emits events and the snapshot carries the light states. Lighting maths
  stays client-only.

**Phases**:

1. ✅ Bake (ambient, panels, falloff, wall occlusion) + tests (brighter
   under a panel than between; a wall blocks light; a two-storey room is
   dimmer under its panels); a top-down PNG of the map for tuning:
   `client/lighting.py`, `LightingConfig`, `python -m
   ok_doomer.client.lightmap_png maps/office.map --out out/light.png`. ~0.2 s
   for the office. Panels also fade smoothly to nothing at the cutoff, so
   there's no ring. First bake: the lobby and offices have pools; the open
   plan (a panel every 2 cells) saturates past 1 everywhere, so phase 3
   needs fewer panels there or a tighter falloff.
2. ✅ Apply to walls, faces, boxes and sprites: `Renderer(config,
   lighting)`; faces take the light `FACE_LIGHT_BACKOFF` (0.13 cell) back
   along the ray from where it meets them, in the space they face; box
   pixels at their own position, sprites at the thing's. Light is clamped
   to 1 and multiplies the distance fade (one colormap lookup still).
   `LightingConfig.enabled` switches it off. ~1 ms in the lobby. Floors and
   ceilings are still unlit until phase 3, so dark walls stand on a bright
   floor for now.
3. ✅ Floors and ceilings lit per pixel: the default floor and ceiling are
   built each frame (`_lit_background`) instead of cached, each pixel's
   world position sampling the map; light is looked up once per 2x2 pixels
   (it changes slowly) and shading stays per pixel. Desk tops, raised floors
   and non-default ceilings are lit per pixel too. Ceilings take the square
   root of the light (bounce light varies less). Tuned: ambient 0.3; with
   lighting on, the distance fade reaches its darkest at 20 cells instead
   of 14 (`LightingConfig.shade_distance`), since the two multiply. The
   open plan's 35 panels became 8 (one over each pod, four over the aisles)
   so it has pools. Cost on the cloud VM: +1.5-3 ms per frame (lobby ~17 ->
   ~20 ms). `LightingConfig.enabled = false` switches it off.
4. Dynamic: power events ✅ (emergency lighting until the breaker is
   thrown: see Power); flicker and muzzle flashes later.

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
| 1b ✅ | Textured walls by wall type | Wall types visibly differ |
| 2 ✅ | Billboard sprites, depth-clipped; office furniture, office textures, office map | A static thing renders correctly in front of and behind walls |
| 3 | Server: handshake, 60 Hz tick, snapshots, interpolation, bot client | Two clients and a bot move together; protocol tests pass |
| 3b | Character classes, weapons (hitscan first), health, death/respawn, DM scoring | A playable deathmatch with the three characters |
| 4 | Enemies: entities, behaviours, sight, flow-field pathing, projectiles (see GAME.md "NPC design") | Co-op floor with staplers, chairs and a photocopier |
| 5 | Level editor | Round-trips a map; shows mode readiness |
| 2b ✅ | Raised floors: low blocks (desk pods, counters) rendered with faces and tops, per-pixel z-buffer, things on blocks | Desks hide what's behind them; terminals sit on desks |
| 2d ✅ | Lighting: baked light map from the ceiling panels (falloff, wall occlusion, ambient), applied to walls, floors, ceilings, boxes and sprites; dynamic lights later (see Rendering, "Lighting") | Pools of light under the panels, dark corners; no frame over budget |
| 2e ✅ | The ground-floor loop: passes and locks, power and emergency lighting, lifts; areas (textured floors, ceilings, walls dressed by area); doors flush with rooms; the boss's desk; compiled ray casting (numba) | Lobby -> blue pass -> office -> red pass -> plant room -> breaker -> lift: "GROUND FLOOR COMPLETE" |
| 2c ✅ | Heights and movement: per-cell ceilings, span renderer, look up/down, jump, crouch, stand on desks, stairs; double-height lobby with a gallery | Walk up the steps to the gallery, jump onto a desk |
| 6 | Ramp cells (sloped floors) for proper-looking stairs and full-storey mezzanines | A storey-high mezzanine reached by a ramp/stair |

Heights were pulled forward (2b, 2c) because they changed how the renderer
and collision work, and the network protocol carries the results (`z`, pitch,
jump, crouch). Steps are a whole cell deep (2.7 m), so stairs look terraced;
ramp cells (a sloped floor, drawn with a closed-form ray-plane intersection)
are the fix.

## Working method

How decisions get made here, so a new session carries on the same way:

1. **Look before judging.** Render the view in question headless
   (`ok_doomer.client.headless`, or a script calling `Renderer.render` and
   `save_png`), crop and enlarge the problem area, and look at it. Measure
   when it's numeric (frame times with a fixed set of views; shade counts;
   light-map values at points) before saying what's wrong.
2. **Find the cause, not the symptom.** Most "looks wrong" reports so far
   had a mechanism behind them: light-map spill through doorways, texture
   noise quantising into blotches, a ceiling cutting a texture, a use point
   left behind when doors moved. Say what it is, with the evidence.
3. **Prefer the simplest thing that's right**: textures over boxes for flat
   things; one rule over per-thing options (doors at the end of their
   doorway, faced by how they open); config knobs for taste (lift,
   integer_scaling) rather than hard-coded choices.
4. **Be honest about costs and mistakes.** Benchmark changes to the frame
   (lobby / open plan / corridor views); when an estimate was wrong (the
   shade-curve palette), say so and change course; when a fix breaks
   something (the door push made doors one-way), revert it and write down
   why.
5. **Every change: tests, the four checks, docs, commit, push.** Tests pin
   behaviour (and fail without the change: check that). DESIGN.md (engine),
   GAME.md (content, [built]/[planned]), LEVELS.md (rules for maps and the
   generator) are updated with the change, including dead ends worth
   remembering.

Rules that have come out of it (details in the sections above):

- Textures: gentle variation sits mid-shade (10.0 +- 0.25, never 8.5 +- 0.5)
  or it blotches up close; avoid gradients and noise on things seen close
  (doors, surrounds); thin lines read best near the axes; small things are
  a few pixels, keep them rare and square-on; draw for the stretch of the
  face (box faces and screens stretch textures once).
- Flat things are textures (desk paperwork, Post-its on screens); things
  with height are boxes (the Mac, the boss's desk); a one-off centrepiece
  of a dozen boxes is cheap, dozens of repeated boxes aren't.
- Light: the light map is a floor plan; faces take their light from the
  floor in front of them, faces above head height from a second map in
  which lintels and doors stop light. Shut opaque doors block light at
  their plane. Emergency light is the exit signs.
- Areas dress a floor: each cell's floor, ceiling and soffit, and the look
  of plain walls facing into it; doorways' side walls are linings.
- Doors: flush with the room they serve, the doorway a recess on the
  corridor side; hinged ones swing out into the recess (never into the
  corridor); glass and lift doors face their room and park leaving a strip
  of each leaf. Lifts are one cell off the lobby, their doors ~3 cm behind
  the lobby wall's face (`LIFT_SET`), flush in a steel architrave. Corridors 2 cells wide;
  recesses are cover. See LEVELS.md, "Doors and corridors".

## Open work

Noted at the end of a session, to pick up next:

- **The single-player loop is closed**: the ground floor, the lift ride,
  floor 1 (GAME.md, Game loop). Next: more floors; sound (lift music, the
  ding); a start screen when networking needs one (host or join). Enemies, weapons and pickups (milestones 3b,
  4) come after networking.
- **Health and coffee** (GAME.md, Pickups; Networking above): two meters
  per player, coffee as armour (soaks a third of each hit). Built: the
  meters, coffee cups, the water cooler, the vending machine (a dispenser
  that needs power: `DispenserType.needs_power`). To do: snack pickups,
  damage and absorption with 3b, the meters on the HUD.
- **Office objects to make** (GAME.md, Office furniture, the `planned`
  rows): a kitchen area (counter, sink, kettle, microwave, fridge, mugs);
  a photocopier, a fax
  on a stand, a coat stand, an ashtray bin, a flip chart, an overhead
  projector on a trolley, a fire extinguisher.
- **Cosmetic to-dos**, smaller polish noted along the way:
  - The lift ride's lurch: the car doesn't move (each storey is its own
    map; the ride is the panel counting, then the next map), so sell it
    with the camera: a dip of a few cm as it sets off (`LiftDeparting`),
    eased back, and a smaller bounce as it stops, as the eye height eases
    for a crouch. Client-side only. Sound (the hum, a ding, the doors)
    would do more, once there's audio.
  - A flickering fluorescent panel (one of the washrooms'): the light map
    is baked, so its pool would have to be switched on and off as a
    separate patch.
  - A flashing amber beacon over the plant room door (an indicator lamp,
    `blink`), drawing players to the power.
  - Wall clocks, stopped at the moment the power failed; ticking again
    once it's back (a powered box, its hands by the clock).
  - Lamps that cast light: the breaker's red lamp pulsing on its wall in
    the dark (needs light patches added and removed at run time, and
    colour, which the light map lacks).
  - More face parts: a quiff (the one asymmetric hairstyle), a ponytail,
    a hard hat (facilities); then desk photo frames and a staff photo
    wall, with the face generator.
  - Big glasses at poster distance can merge into a dark band: drop the
    top rim if it bothers.
  - The view from higher floors: lower the panorama's ground line per
    floor (more city), the moon in view from the offices too.
  - Lift doors on the emergency circuit (a firefighting lift): doors
    don't take `circuit=` yet.
- **A face generator, when something needs many faces** (a wall of staff
  photos, a photo frame on each desk, NPC portraits): a seeded
  `random_face(rng)` in `client/faces.py` picking each part of a `Face`
  (seed it from where it's used, so faces are the same every load). Add a
  few combination rules or weights then, once its use is known (no beard
  with a perm, a tie with a collar already holds, variety across a row);
  the cast (`CAST`) stays hand-picked. Until then the templates are the
  bank: every new template multiplies the faces possible.
- **Rendering speed** (a laptop ran 45 fps where the desktop ran 90;
  networking costs the client ~0.1 ms a frame, measured: it's the
  renderer). A lobby frame on the cloud VM: 17 ms, spread over a dozen
  stages of 1-2 ms each (box models 2.2, light lookups 1.6, what's in
  view 1.3, wall texels, floors and ceilings, low blocks...): numpy's
  overhead per call and its temporary arrays, not one hot spot, and all
  of it on the CPU. Target: 60 fps on the laptop. In order:
  1. **Box models first** (measured, a lobby view: building the list of
     ~590 boxes 0.75 ms, under 4% of a 20 ms frame, so caching the ones
     that don't move would save ~0.5 ms at most: an early guess said
     more; drawing them 5.6 ms, 28%: the biggest stage). Compile
     `_draw_boxes` with numba; cache static placements along the way.
  2. **Then the other hot stages with numba**, as the ray casting was
     (`client/fastcast.py`, now 0.25 ms): light lookups, what's in view,
     sprites, floors and ceilings, a stage at a time, the dearest first,
     measured each time. Plain loops compiled to machine code: no Python
     per call, no temporary arrays. Guard: fixed views rendered by the
     old code and the new must match pixel for pixel. Guess, to be
     measured: 2-5x a stage, the frame 17 -> 8-10 ms.
  3. **Later, the graphics card.** First the cheap part: let it do the
     final nearest-neighbour scale to the window (a CPU cost every frame,
     worst full screen on a big monitor). Then, maybe, a GPU renderer
     (OpenGL, e.g. `moderngl`): the world as real 3D triangles drawn at
     320x200 and scaled up, the palette look kept by a shader, as GZDoom
     draws Doom's maps; it would lift the ray caster's limits (sloped
     floors, milestone 6, come free) but is a rewrite of the renderer.
     The server never draws: the Pi isn't affected either way.
- **Networking (milestone 3)**: built (Networking above, in build
  order: spawning in `shared`, snapshots, the protocol, the server core,
  TCP, bots, the networked client, lag measured and prediction, then
  single-player on the in-process server, the CPU cut, the lag proxy and
  F3 stats, who's here and names). To do, in this order:
  1. **Host from the game, find games on the LAN, a start screen.**
     [finding games and the start screen: built (Networking above);
     hosting to do] Host
     = `LocalSession`'s in-process server plus a TCP listener (no Pi or
     server terminal needed). Servers announce themselves by UDP
     broadcast about once a second; clients list what they hear (Fedora's
     firewall needs that port opened too). A start screen in the green
     terminal style: SINGLE PLAYER / HOST GAME / JOIN GAME (games found,
     and an address to type as a fallback).
  2. **Rejoin after a dropout**: back within ~30 s with the same name, you
     get your player back (where you were, health, coffee), not a respawn.
  3. **Text chat**: T, type, Enter; a message type of its own, shown in
     the HUD's stacked messages.
  4. **Close the milestone**: the Pi over Wi-Fi tested (the lag proxy
     stands in until then), then a pull request of `claude/networking`
     into `main`; milestone 3b next (the crouch sprite's head above the
     body's height waits there: hit boxes).
  5. **Playing over the internet, safely.** Recommended: no open port
     at all, but a private encrypted network between the players
     (Tailscale, or WireGuard / ZeroTier): install it on the Pi and each
     player's machine, share the Pi, join its address (`--connect
     100.x.y.z:5029`) as on the LAN. Nothing to scan or flood, only those
     invited, encrypted, no router setup, and it works behind an ISP's
     CGNAT (where forwarding a port is impossible). Where the server
     stands today (reviewed): it never runs or loads anything a client
     sends (messages only steer a player; maps are the server's own
     files) and refuses malformed data, but anyone who finds the port
     can join, traffic isn't encrypted, and floods aren't limited. If a
     port is ever opened instead, build first: (a) a server password in
     the hello; (b) TLS (`ssl`, asyncio has it), so the password isn't
     sent in the clear; (c) flood limits: clients' lines capped at a few
     KB (only the server sends big ones: maps), connections per address,
     handshakes pending at once; (d) the server as a systemd service, an
     unprivileged user, sandboxed (`NoNewPrivileges`, `ProtectSystem`);
     (e) the router forwards TCP 5029 to the Pi and nothing else (never
     SSH), the Pi kept updated. (a) and (c) are worth having on a LAN
     too. Over the internet, prediction keeps your own movement prompt;
     a lossy link stalls everything behind a lost packet (TCP): the lag
     proxy's `--loss` shows what that's like.
  Also: `play.bat` for Windows (Setup below: double-click to play; once
  the start screen exists it needs no options). LAN discovery on
  Windows: a broadcast to 255.255.255.255 can leave by a virtual adapter
  (WSL, Hyper-V, a VPN), so send to each local network's own broadcast
  address too (e.g. 192.168.1.255); keep typing an address as the
  fallback (Wi-Fi with AP isolation blocks broadcasts altogether).
  Later: server admin (kick, change map); joining mid-ride (you'd arrive
  on the floor the others are leaving). Debug text (fps, F3 stats) may go
  back to the system font if the pixel font gets in the way: fine either
  way.

## Setup

- Linux: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`,
  then `.venv/bin/python -m ok_doomer.client` (single-player: the ground floor,
  `maps/office.map`; or name a map, e.g. `maps/floor1.map`).
- Client settings: copy `client.example.toml` to `client.toml` and pass
  `--config client.toml`.
- Windows: native Python 3.11+ (not WSL) and a venv: `py -m venv .venv;
  .venv\Scripts\pip install -r requirements.txt`, then `.venv\Scripts\python
  -m ok_doomer.client` (`--connect HOST:PORT` to join). Hosting: allow
  Python through Windows Defender Firewall on Private networks when it
  asks, and the Wi-Fi must be a Private network, not Public. [planned]
  `play.bat`: cds to its own folder, runs the client, ends with `pause`
  (so an error stays on screen).
- Use the Pi's IP rather than `.local` names (Windows mDNS is unreliable).
- Server on the Pi: standard library only, no install step:
  `python3 -m ok_doomer.server` from the checkout (port 5029). Bots:
  `python3 -m ok_doomer.bot --connect HOST:5029 --count 3`.
