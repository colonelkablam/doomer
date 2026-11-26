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
| `desk`, `water_cooler`, `filing_cabinet`, `ficus` | Blocking furniture | Square footprints, half-widths 0.32 / 0.12 / 0.16 / 0.16. Cabinets, coolers and ficuses are box models: set the angle so the front faces into the room (away from the wall they stand against); the ficus's crown spreads wider than its footprint |
| `terminal` | PC on a desk | Put it in a desk/counter cell; it stands on the top. Its angle is where the screen faces (towards the sitter): in a pod, 270 on the north half, 90 on the south |
| `mac` | Macintosh SE/30, keyboard and mouse | As `terminal` (same angle rule). Solid boxes: costs more than a sprite per thing in view, fine for dozens. Its screen is blank until the power's on; `circuit=emergency` (also on `pc`, `terminal`) keeps it lit in a power cut: use it for one that matters (security, reception) |
| `welcome_sign` | "Boomer Industries" in lit letters | Stand it just in front of a wall's face (e.g. x = 7.99 for the face at 8), angle out of the wall. About 4 cells wide, 1.09 to 1.46 storeys up: needs a two-storey wall |
| `reception_desk` | Two-level reception desk | Angle: the way its front (the visitors' side) faces. 1.84 cells long, 0.61 deep (0.31 in front of its position, 0.30 behind). Things placed over it rest on it: the ledge (front 0.17 cells) or the work surface (behind). Leave the receptionist a way round to the back |
| `pass_blue`, `pass_red`, `pass_gold` | ID passes: staff, facilities, executive | Walk over (or reach) to take; the team shares it. On a counter or desk, put it within 0.5 of the edge you approach from (you're stopped at the edge; reach is 0.75 from your centre) |
| `door_lift` | Lift doors and the floor's exit | The lift is one cell off the lobby: put the door in that cell, angle out of it (towards the lobby); the cell is the car, its doors on the lobby line. Ceiling `6`, area `c`, a `ceiling_light` in it (main circuit: dark until the power's on). Wall it in with `k`: the cell behind, [facing] towards the car (its mirror); the cells either side, [facing] away from the lobby (their lobby faces take the lobby's look). Co-op needs a lift or an `exit` |
| `door_hinged`, `door_double`, `door_sliding` | Office door / double office doors (use to open) / automatic glass doors | In a doorway cell: open floor between two full walls, the doorway one cell wide. Angle: across the doorway, the way it opens: hinged and double towards the corridor (they stand at the room's end and swing out into the doorway), sliding towards the room (they stand at that end); see Doors and corridors. Give the doorway cell a ceiling of `6` (a lintel at ~2 m: the face over the door). `lock=blue` (or red, gold): opens only for a team with that pass; card readers appear beside it. Not in a two-storey cell, and nothing blocking within a cell either side |
| `breaker` | Main breaker: restores the power | Stand it just in front of a solid wall's face (not a window), angle out of the wall (a wall-mounted isolator: it blocks 0.24 out from the wall, its lever's reach, 0.44 wide; its keep-clear mat in front doesn't block). A map with one starts on emergency lighting: put `exit_sign`s over the main doors on the way there (and out), including the plant room's, so the route is lit in pools. Put it behind a lock (the plant room), before the lifts |
| `executive_desk` | The boss's absurd desk, with its chair (box model) | Mid-room, two cells long across its facing: angle = towards visitors, the boss's chair behind it (1 cell back). Leave a cell clear round it. Things on it rest on its top: put a pass within 0.5 of the visitors' edge |
| `phone` | 1980s desk phone (box model, 3 boxes) | On a desk, angle towards its user (as a Mac: +x is the user's side), clear of the Mac and keyboard. Keep them few (the boss's desk, reception) |
| `pc` | Late-80s PC with a green screen (box model, 5 boxes) | The boss's computer, in place of a Mac: on a big desk (it's ~0.3 wide, ~0.45 deep with its keyboard), angle towards its user as a Mac |
| `chair` | Office swivel chair (box model, 4 boxes; blocks, radius 0.1) | One per sitter: ~0.25-0.36 out from the desk edge in front of their Mac, angle towards the desk; nudge each a few cm and turn it up to ~30 degrees at random so a row looks used. Between a pod and a wall one cell away, keep chairs at least 0.6 from the wall so a player (0.5 wide) gets past |
| `poster` | A poster or picture, `kind=` synergy, teamwork, excellence, employee_mar/apr/may, founder, accident, high_voltage, plan | At a wall's face (0.01 off it), angle out of the wall. Hang a `plan` by each way in (it draws the floor, marking where it hangs) and a `high_voltage` by the plant room door and its breaker, so the power is signposted |
| `server` | Minicomputer cabinet, `kind=tape` or `disk`; its lamps blink once the power's on | In the plant room (or a computer room): at a solid wall's face, angle out of it, in a row 0.38 apart (0.38 wide, blocks 0.36 out). Leave the breaker and the way to it clear |
| `sofa_two`, `sofa_three`, `sofa_corner`; `coffee_table`; `fern` | Lobby seating: modular sofa runs (0.28 a seat, 0.32 deep) and a corner unit; a low table; a small fern | Angle = the way the sitters face, backs against a wall. Make an L: a corner unit in the room's corner (backs west and north at angle 0), runs either side of it, the table in front, 0.2 or so clear of the seats (a nook: you needn't fit between). Keep it out of the way in: the map checker tells you if it cuts the route. The fern on the table: it rests on its top |
| `coffee_cup` | A cup of coffee: a pickup, +5 coffee | On a desk or counter, within reach of where a player stands (0.75 from their centre: near the edge they walk up to), clear of the Mac and keyboard; angle turns its handle. A few on a floor, not every desk |
| `vending_machine` | Snack machine: a health dispenser (10 a use, 40 a player), working once the power's on | At a solid wall's face, angle out of it (0.36 wide, blocks 0.3 out): in a corridor where it widens, a kitchen, a break area. Not on the critical path: it's a reward |
| `exit_sign` | Green EXIT sign over a door, always lit; in a power cut, a dim emergency light (a pool ~1-2 cells round it) | At the face of the wall over the doorway, on the side people leave from (e.g. x = 9.02 for a doorway whose office-side wall face is x = 9), angle out of the wall. One per main door on the route: office -> lobby -> street, and the plant room. Over glazed doors that run up to the storey line (the entrance's), `kind=transom`: on the transom bar above them, not on the glass |
| `ceiling_light` | Light panel in the ceiling; lights the room (baked light map). `circuit=emergency`: stays lit without power | Marks the cell; drawn at that cell's ceiling height. Space them 3-6 cells apart for pools of light with darker gaps (every 2 cells saturates a room flat). Light stops at full walls, goes through doorways, and reaches ~6 cells. A room with none sits at ambient (dark): use that on purpose |

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
  Performance). `j` is the same with cubicle screens (a booth per sitter):
  give neighbouring cells in a row the same facing so their shared edge has
  one screen. Screens block you on the desk tops; under a storey-high
  ceiling you can't jump over them from a desk (no headroom), so a `j` pod
  is no desk-top route, only booths. Under a tall ceiling they can be jumped.
- **Single desks** `c` face where you sit; **executive desks** `b` likewise.
- **Areas** (`[areas]`): mark every open cell of a room with its kind of
  space: `e` the boss's suite, `l` the entrance and lift lobby, `s` plant
  rooms and back corridors, the rest office (default). Build room walls from
  plain `#` and let the areas dress them: a wall between two areas shows
  each side its own look. Use an explicit wall type only for something
  that's the same from both sides (a window, brick, the curtain wall).
  Doorway cells take the area of the room they open from (the floor changes
  under the door). Keep areas to whole rooms: a change of carpet mid-room
  looks like a mistake.
- **Corners** [planned]: a `corner=nw|ne|sw|se` property on things (pot
  plants first), placing the thing in that corner of its cell, inset by its
  own footprint (a ficus: 0.2 from each wall, its pot's radius 0.16 plus a
  gap), so neither a hand-made map nor the generator has to work out the
  numbers. Until then, write them: cell (x, y)'s NW corner is (x + 0.2,
  y + 0.2), SE (x + 0.8, y + 0.8). Leaves reaching past a wall are hidden
  by it.
- **Macs on desk pairs** go at the sitter's end of each half, at (0.68,
  0.25) and (0.32, 0.75) of a north/south pod's cell (a quarter turn for
  east/west), facing their sitter: the tops' paperwork (`areas.desk_top`)
  keeps to the other end of each half, so a Mac elsewhere sits on it.
- **Desk-top things** (`mac`): turn each a few degrees off square (3-10 deg,
  either way, varied between neighbours), so a row of them looks used rather
  than lined up, and nudge it 0.02-0.06 cells along the desk (up to 0.015
  towards or away from the sitter). Keep the turn under ~12 deg: the
  keyboard turns with the Mac and would reach past the desk's edge. Every
  corner of the Mac, keyboard and mouse should stay ~0.05 inside the cell.
- **Counters**: wall type `2`, height 3: too high to jump onto (cover).
- **Tall spaces**: `[ceiling]` `g` (16) for two storeys. Walls get their
  upper-storey texture above one storey; a bulkhead face appears where the
  ceiling drops.
- **Two-storey glazing**: `f` (curtain wall); main doors `g`.
- **Balconies**: `i` at 8 (one storey) along a wall, reached by two stair
  cells arriving level with it.

## Doors and corridors

Walls are whole cells (2.7 m thick), so every door is in a doorway cell: a
one-cell gap in a wall line. The door stands at the end of it on the side
it opens into, flush with that room's wall; the rest is a recess on the
other side. The rules that follow from that, for hand-made and generated
floors:

1. **Doors are flush with rooms, recessed off circulation.** A door between
   a corridor (or the lobby) and a room stands at the room's end of the
   doorway: the room gets a clean, flush wall with the door in it; the
   corridor gets a one-cell recess. Hinged and double doors face the
   corridor (they swing out, into the recess, never into the corridor
   itself); glass sliding
   doors face the room. Between two rooms, the door belongs to the more
   private one (the boss's office off his secretary's).
2. **Lift doors are the exception**: flush with the lobby (they face out of
   the car), as real landing doors are, the car behind.
3. **The doorway cell belongs to the corridor**: its area, a `6` ceiling
   (the lintel), and its side walls are door linings (automatic).
4. **A room at the end of a corridor**: the corridor's last cell is the
   doorway, its door flush with the room's wall.
5. **Corridors at least 2 cells wide**, so co-op players pass and fights
   have room. Door recesses are cover: step into one to break line of
   sight, or wait in one; a natural place for enemies to lurk.
6. **No door in a two-storey cell, nothing blocking a cell either side.**

A future option, if 2.7 m walls feel too thick for offices: thin walls on
cell edges (Doom's linedefs), which the ray caster would test as it
crosses each grid line. Not needed for the rules above.

## Performance budget

60 fps means 16.7 ms per frame. Current costs on the office map: corridors
~5 ms, desk pods with Macs ~10 ms, the two-storey lobby with stairs and
balcony ~16 ms (on the cloud VM; faster desktops have more headroom).
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
- Lights and sprites are cheap to draw; dozens are fine. Lighting (the baked
  light map) costs ~1.5-3 ms per frame regardless of how many panels. Box models (Macs) and
  stairs (8 boxes per cell) cost per box per column: dozens of boxes in view
  are fine, hundreds aren't.

## Checks a generator (or a hand-made map) must pass

1. **Loads**: `parse_map` succeeds (valid characters, things not in walls,
   raised floors have a wall type, ceilings above floors).
2. **Modes**: co-op ready (1 `coop_start`, 1 `exit`) and/or deathmatch ready
   (>= 4 `dm_spawn`).
3. **Can be finished** [built: `shared/validate.py`; run
   `python -m doomer.shared.validate maps/<name>.map`]: from `coop_start`,
   walking by the game's own movement rules (steps up to 1 unit, stairs
   from the low end, a jump onto a desk but not a counter, railings,
   furniture and screens in the way), every locked door's pass is reachable
   before the door it opens (no pass behind its own door), the `breaker`
   (if any) is reachable, and a lift car (with the power on) or an `exit`
   after them all. It says what's wrong in a sentence. A test runs it on
   every map in `maps/`.
4. **No blocked passages**: covered by 3 where it matters (furniture that
   cuts off the route fails it); still worth keeping blocking furniture out
   of 1-cell-wide passages and the ends of stairs. Deathmatch spawns
   reachable from each other: not checked yet.
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
