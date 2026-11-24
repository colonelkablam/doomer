# Level design rules and limits

What a map can contain, where things can go, and what the engine can't do.
Written for hand-built maps now, and as the rule set for a procedural map
generator later: every rule here should become an automatic check.

All numbers come from the code (`doomer/shared/config.py`, `mapfile.py`,
`things.py`, `modes.py`); if this file and the code disagree, the code wins
and this file needs fixing. File format details: `DESIGN.md`, "Map format".

## Scale

| Thing | Size |
|---|---|
| A cell | 1 x 1, square: 2.7 m (a cell is as wide as a storey is tall) |
| A storey (default ceiling) | 1 wall = 8 height units = 2.7 m |
| A height unit | 1/8 wall, ~34 cm |
| Heights in a file | `0`-`9`, then `a`-`z` = 10-35 units (max ~4.4 storeys) |
| Wall types | `1`-`9`, then `a`-`z` = 1-35 |
| Furthest you can see | 64 cells (rays stop there) |

**Walls are whole cells.** There are no thin walls: a partition, a window or
a door is a full 2.7 m cell drawn with a texture. Doors and windows are only
textures (none open yet). Plan rooms on a grid where every wall costs a cell.

## Cells: what each one can be

| Kind | How | Walkable? | Notes |
|---|---|---|---|
| Open floor | `.` in `[walls]` | yes | Floor is always 0. A `[floor]` value here is an error |
| Full wall | wall type, `[floor]` 0 (or >= its ceiling) | no | Drawn from the floor to the ceiling of the space in front |
| Low block / raised floor | wall type, `[floor]` 1 .. ceiling-1 | on top, if you can get up | Desks, counters, platforms. Sides use the wall type's texture |
| Stairs (`h`) | `h`, `[floor]` = height at the top edge, `[facing]` = uphill | yes, from the low end | Rises 4 units across the cell. Two cells = a storey |
| Balcony (`i`) | `i` + `[floor]` | yes | Railing on every edge that drops more than a step |

**One floor and one ceiling per cell.** No room over room, no overhangs you
can walk under, no bridges over a walkway. A balcony is solid underneath.
Different storeys of the building are separate maps.

## Moving around (what the player can do)

| Ability | Limit | Map consequence |
|---|---|---|
| Fit through a gap | Collision box 0.5 x 0.5 | Any 1-cell gap fits |
| Walk up a step | Rises up to 1 unit (0.125) | Steps of 1 unit, or stairs (`h`) |
| Jump onto something | Apex 0.32 walls | 2 units (desk) yes; 3 units (counter) no |
| Walk upright | Body 0.67 | Needs ceiling - floor >= 6 units |
| Crouch | Body 0.42 | 4-5 units of headroom: crouch-only (crawlways, low bulkheads) |
| Stand on a desk | Desk top 2 + body 0.67 | Needs ceiling >= 8 over a desk (the default is fine) |
| Fall | No fall damage (yet) | Any drop is survivable; only railings stop you |
| Climb stairs | Only from the low end | The cell before a stair's low end must be at the low height; stair sides block if more than a step |
| Look up/down | +/- 25 degrees | Very tall spaces and deep drops are partly out of view |

**Getting onto raised floors:** by 1-unit steps, by stairs, or by jumping (2
units at most). Anything higher with no stairs is unreachable.

## Things

| Thing | Purpose | Rules |
|---|---|---|
| `coop_start` | Where co-op players arrive | Exactly 1 for co-op. On open floor or a raised floor, never in a full wall |
| `exit` | Co-op level exit (drawn as a hanging EXIT sign) | Exactly 1 for co-op. Must be reachable on foot from `coop_start` |
| `dm_spawn` | Deathmatch spawn | At least 4 for deathmatch. Reachable; spread out (see checks) |
| `desk`, `water_cooler`, `filing_cabinet`, `ficus` | Blocking furniture | Square footprints, half-widths 0.32 / 0.12 / 0.16 / 0.16 |
| `terminal` | PC on a desk | Put it in a desk/counter cell; it stands on the top |
| `ceiling_light` | Light panel in the ceiling | Marks the cell; drawn at that cell's ceiling height |

- Positions are cell coordinates (`3.5` = centre of cell 3). Use cell centres
  unless there's a reason not to.
- **Blocking furniture in a 1-cell-wide passage blocks it completely**: the
  player is 0.5 wide and any footprint leaves less than that. Keep blocking
  things out of 1-wide corridors, balcony walkways and the cells at the top
  and bottom of stairs.
- Things in a low-block cell stand on its top (on stairs, on the slope).
- Unknown thing types are a load error.

Supported modes are derived from the things: **co-op** needs a `coop_start`
and an `exit`; **deathmatch** needs at least 4 `dm_spawn`s.

## Furniture and layout conventions

- **Desk pods**: rows of `d` cells (two desks back to back), `[facing]` `n`
  (sitters north and south) or `e`. Leave an aisle along both long sides for
  the sitters. A row of identical cells is drawn as one (cheaper; see
  Performance).
- **Single desks** `c` face where you sit; **executive desks** `b` likewise.
- **Counters**: wall type `2`, height 3: too high to jump onto (cover).
- **Tall spaces**: `[ceiling]` `g` (16) for two storeys. Walls get their
  upper-storey texture above one storey; a bulkhead face appears where the
  ceiling drops.
- **Two-storey glazing**: `f` (curtain wall); main doors `g`.
- **Balconies**: `i` at 8 (one storey) along a wall, reached by two stair
  cells arriving level with it.

## Performance budget

60 fps means 16.7 ms per frame. Current costs on the office map: corridors
~5 ms, desk pods ~10 ms, the two-storey lobby with stairs and balcony ~14 ms.
What costs:

- **Changes along a ray.** Every time a ray crosses into a different kind of
  space (a desk, a raised floor, a different ceiling) is another layer of
  drawing. Keep views to roughly 5 or fewer such changes. Runs of identical
  cells (same type, height, ceiling, facing) count as one.
- **Large open spaces full of low blocks** (big open-plan floors with many
  pods, tall rooms with balconies) are the most expensive. Break long
  sightlines with full walls.
- **Map size**: no hard limit, but rays stop at 64 cells; keep maps within
  about 48 x 48.
- Lights and sprites are cheap; dozens are fine.

## Checks a generator (or a hand-made map) must pass

1. **Loads**: `parse_map` succeeds (valid characters, things not in walls,
   raised floors have a wall type, ceilings above floors).
2. **Modes**: co-op ready (1 `coop_start`, 1 `exit`) and/or deathmatch ready
   (>= 4 `dm_spawn`).
3. **Reachable on foot**: from `coop_start`, the `exit` and every `dm_spawn`
   can be reached by walking (steps of 1 unit, stairs from the low end; no
   jumping assumed), with headroom >= 6 units all the way.
4. **No blocked passages**: no blocking furniture in a 1-cell-wide passage or
   at the ends of stairs.
5. **Stairs are well formed**: the cell before the low end is at the low
   height; the cell beyond the top end is level with the top (a landing,
   balcony or raised floor).
6. **Desks and pods**: every `d` has an open aisle cell on both long sides;
   every `c`/`b` has an open cell on its front.
7. **Spawns**: `dm_spawn`s at least ~4 cells apart; none on a 1-cell ledge or
   in a crouch-only space.
8. **Exit away from start**: path length from `coop_start` to `exit` above a
   minimum (a level shouldn't end in the first room).
9. **Performance**: estimate changes-of-space per ray over a sample of
   viewpoints; reject layouts that exceed the budget.

These are worth implementing as `doomer/shared/validate.py` (standard library
only, so the server can check a generated map too) before the generator.

## Level structure (later)

Two options, from the brainstorm:

**A. The tower** (recommended first). One map per floor of the office block,
ground floor to the penthouse boss fight. Each level: arrive by the lift
(`coop_start` in the lift lobby), find the way to the lift or stairwell up
(`exit`). Fits the setting, every level is a bounded map the existing engine
handles, and the networking plan already sends the map to clients on join:
the server generates a floor from a seed and sends it (or just the seed and
generator version, if generation is deterministic).

A generator sketch for a floor: split the floor plate into rooms (binary
space partition or a template grid), assign room types (open plan, offices,
meeting rooms, kitchen, washrooms, server room) each with a furniture
template, connect them with corridors, place the lift lobby and the exit
far apart, then run the checks above and retry on failure. Mix in a few
hand-made floors (the lobby, the boss floor).

**B. Endless crawl**: an open map generated in chunks as players move, to a
final boss. Much more work: the map stops being a fixed object (the renderer,
collision and pathfinding all assume one), chunks must be generated
identically on server and clients or streamed, and co-op players far apart
need different chunks loaded. Possible later, but A first.
