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
- `[things]`: one per line, `type x y angle [key=value ...]`. Positions are
  floats in cell units (`1.5` = centre of cell 1). Unknown types are an
  error at load time. Properties after the angle are declared per thing
  type (`ThingType.properties`, with their allowed values): a door's
  `lock=blue|red|gold`. An unknown property or value is a load error.
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
- **Doors** (prototype; `shared/doors.py`, drawn by `boxes.door_boxes`): a
  door is a thing in a doorway cell (open floor between two wall cells),
  standing in the plane through the cell's middle. State (how far open,
  opening or closing) lives in the simulation's `World.doors`, so the server
  will own it; the client draws boxes from it each frame (passed to the
  renderer as `boxes`). Hinged: two fixed jambs and a leaf whose model
  origin is the hinge, turned by how far open it is, so it really swings;
  use (E) opens and closes it, and it won't shut on someone in the doorway.
  Double: two leaves hinged at either side of the doorway, meeting in the
  middle and swinging the same way, between narrow jambs; used the same way.
  Sliding: two glass leaves (a dithered, see-through texture) that part
  into the walls either side, opening by themselves for anyone within 1.6
  cells. Collision: the closed part of the doorway as thin walls. A lowered
  ceiling over the doorway (6/8) gives a lintel through the existing
  bulkhead drawing; with lighting on, a ceiling lower than a storey (the
  lintel's soffit) takes the plain light times `soffit_shade` (0.85), not
  the square root other ceilings get, so it sits in shadow like the walls.
- **Furniture footprints** (`things.Footprint`, `shared/furniture.py`): a
  blocking thing type can give its floor plan as rectangles in its own frame,
  each with a top height, instead of the square `radius`. Placed at the
  thing's angle (snapped to quarter turns), they are its collision
  (`collision.Obstacle` is a rectangle) and the surfaces other things rest
  on: a thing inside another's footprint stands on its top
  (`resting_height`, used for sprites, box models and pickups). The
  reception desk is the first: a ledge at the front (~1.15 m), a work
  surface behind it (~0.73 m).
- **Power** (`shared/switches.py`): level flags in `World.flags`, set by
  switches. The breaker sets `power`; a map with a breaker starts without it
  (a map without one is powered). Use (E) near the breaker, facing it,
  throws it (`SwitchThrown`); it's one-way. The client draws the breaker from
  its state (lever down/up, lamp red/green). Without power, the light map
  is recomposed from the panels' patches: a lower ambient
  (`unpowered_ambient`) plus only `circuit=emergency` panels; main panels
  draw dark; ceilings take the plain light instead of its square root (with
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
  them crosses no full wall (grid DDA, cell resolution). Low blocks don't
  block light (the contact shadow under desks already exists).
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
   doomer.client.lightmap_png maps/office.map --out out/light.png`. ~0.2 s
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
| 2d | Lighting: baked light map from the ceiling panels (falloff, wall occlusion, ambient), applied to walls, floors, ceilings, boxes and sprites; dynamic lights later (see Rendering, "Lighting") | Pools of light under the panels, dark corners; no frame over budget |
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
