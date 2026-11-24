# OK Doomer: design

A Doom-style shooter in Python for home LAN play (Fedora and Windows clients,
Raspberry Pi 5 server), with co-op, deathmatch and a level editor. Set in a
1980s office block. This file is the engine and code design; the setting,
characters, enemies and furniture are in [GAME.md](GAME.md); level design
rules and limits in [LEVELS.md](LEVELS.md). The Python
package keeps the short name `doomer`.

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
- `[things]`: one per line. Positions are floats in cell units (`1.5` = centre
  of cell 1). Unknown types are an error at load time.
- Supported modes are derived, not declared: **co-op** if there is a
  `coop_start` and an `exit`; **deathmatch** if there are at least 4
  `dm_spawn`s. The server ignores things not relevant to the current mode.

## Rendering

- Grid raycaster (DDA) at 320x200 internal resolution, scaled up to the window.
- **Two-phase frame**: the DDA is a Python loop (one ray per column, 320 a
  frame); everything per-pixel is vectorised numpy over the whole frame.
  About 3 ms per frame for the office without blocks in view, ~6 ms with desk
  pods filling the view.
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
  the ceiling plane decides whether it's inside the 1.2 x 0.6 m panel. Drawn
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
  angle, a 64x64 texture per face. Every box is slab-tested against every
  column's ray at once (numpy, boxes x columns); the face a ray enters by is
  drawn like a wall slice at the entry depth, the top (or underside) like a
  floor between entry and exit depth, then all box pixels are z-tested
  together. Correct from any angle and height with no frame popping, which
  sprites can't do; the cost is per box per column, so keep models to a few
  boxes, only within 16 cells. The Mac SE/30 (`mac`, with keyboard and
  mouse) is the first. Big furniture stays as map cells.
- **Stairs as steps** (`boxes.stair_boxes`): each stair cell is drawn as 8
  step boxes (1/16-storey risers, 1/8-cell treads), built once per map and
  never distance-culled. Each step's top is level with the slope at its back
  edge, so feet following the slope are at most a riser below the tread.
  Step sides are textured in world units (`tiled_sides`) so the 8 read as one
  wall; a step's front and back faces are drawn only one riser tall at the
  top (`riser`), since the step in front hides the rest. ~2 ms in the lobby.
- **Blocking things** (`ThingType.blocking`, `radius`) are square footprints
  that the shared collision code treats like walls.

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
| 2c ✅ | Heights and movement: per-cell ceilings, span renderer, look up/down, jump, crouch, stand on desks, stairs; double-height lobby with a gallery | Walk up the steps to the gallery, jump onto a desk |
| 6 | Ramp cells (sloped floors) for proper-looking stairs and full-storey mezzanines | A storey-high mezzanine reached by a ramp/stair |

Heights were pulled forward (2b, 2c) because they changed how the renderer
and collision work, and the network protocol carries the results (`z`, pitch,
jump, crouch). Steps are a whole cell deep (2.7 m), so stairs look terraced;
ramp cells (a sloped floor, drawn with a closed-form ray-plane intersection)
are the fix.

## Setup

- Linux: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`,
  then `.venv/bin/python -m doomer.client` (single-player, `maps/test.map`).
- Client settings: copy `client.example.toml` to `client.toml` and pass
  `--config client.toml`.
- Windows: native Python (not WSL), venv, launched from `play.bat` which cds to
  its own folder, runs the client with `--host`/`--name` and ends with `pause`.
- Use the Pi's IP rather than `.local` names (Windows mDNS is unreliable).
- Server on the Pi: standard library only, no install step.
