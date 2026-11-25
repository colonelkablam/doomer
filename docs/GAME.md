# OK Doomer: game design

The setting, cast and content. Engine and code design live in `DESIGN.md`.
Status markers: **[built]** exists in the game, **[planned]** is design only.

## Setting

**When**: 1983–1991. Beige PCs with green-screen and amber CRTs, dot-matrix
printers, fax machines going mainstream, cubicle farms, shoulder pads, ashtrays
on desks being quietly removed. Ends before Windows 95, email and the
internet. The baby boomers (born 1946–64) are 19–45 and running the place.

**Where**: one corporate office block, climbed floor by floor. Each map is a
floor; the exit is the lift or the stairwell. You start in the ground-floor
lobby and finish in the penthouse boardroom (a basement may come later: see
Game loop).

**Tone**: deadpan office comedy played straight as a shooter. Nothing is gory:
enemies break, spark, jam, or storm off.

Suggested floor order:

| Floor | Theme | New things introduced |
|---|---|---|
| B1 | Mailroom and post | Staplers, photocopiers |
| G | Two-storey glazed entrance lobby: main doors, reception desk, stairs up to a railed balcony | Swivel chairs, water coolers |
| 1 | Open-plan sales floor | Middle managers |
| 2 | Accounts (all filing cabinets) | Fax swarms |
| 3 | IT and the server room | Haywire mainframe terminals |
| 4 | HR and meeting rooms | HR reps, flip charts |
| 5 | Executive floor, wood panelling | The Boss (mini-boss) |
| P | Penthouse boardroom | The CEO (final boss) |

## Game loop

Each floor is a short puzzle-and-fight route: find passes, open the way to
the plant room, restore power, take the lift up. The team shares passes and
power, so in co-op anyone can do each step.

The ground floor, as built [built]:

1. Start in the lobby on emergency lighting: only `circuit=emergency` panels
   are lit, the "Boomer Industries" sign is dark, the lifts are dead.
2. The **staff (blue) pass** lies on the reception desk. It opens the sliding
   glass doors into the open-plan office.
3. The **facilities (red) pass** is on the boss's desk. It opens the plant
   room's door.
4. Throw the **main breaker** in the plant room (use, E): the power comes
   back. The lights come on, the sign lights up, the lifts get power.
5. **Call a lift** (E at its doors), step into the car. When every player is
   in a lift car the doors close: "GROUND FLOOR COMPLETE".

Planned, roughly in order [planned]:

- **A full-size ground floor**: about 32 x 24 cells, laid out for the loop
  above (the current map is the test bed).
- **Automap** (Tab): a HUD map of the floor plan as you've seen it.
- **Map checker** (`shared/validate.py`): a floor is completable: every
  locked door's pass is reachable before it, the breaker is reachable,
  there's a lift or exit.
- **The lift ride**: the doors close, a short ride (indicator counting up,
  maybe a lift-music sting), then the next floor loads with the team in its
  lift lobby. Until then the floor just ends on the banner.
- **Later floors** reuse the loop with variations: the **executive (gold)
  pass** for the upper floors, a stairwell as an alternative exit, power
  that's already on but a different blocker (a jammed lift, a fire door).
- **Enemies** on the route (see Enemies), so the passes and breaker are
  fought for, not just collected.
- **Where you start** (open, revisit): the game starts in the ground-floor
  lobby for now. The basement mailroom (B1 in the floor order below) is
  undecided: maybe added later, e.g. reached by taking the lift down.

## Player characters

Three classes, each loosely a D&D archetype: a damage dealer, a tank and a
healer, which suits two-player co-op. Same movement code; they differ in
health, speed, weapons and one special ability. All are gender-neutral roles;
the sprite can come in variants.

| Character | Archetype | Health | Speed | Weapons | Special |
|---|---|---|---|---|---|
| **The IT Guy** | Wizard: fragile, strong at range | Low | Medium | Thrown floppy disks (fast, ranged); Modem Screech (short cone, dial-up shriek, stuns) | **Ctrl-Alt-Del**: clears everything on screen; long recharge |
| **The Caretaker** | Fighter / barbarian: tough, close range | High | Slow | Mop (melee, wide arc); fire extinguisher (cone, knockback) | **Overtime**: brief damage resistance |
| **The Tea Trolley** | Cleric: support and healing | Medium | Slow while pushing the trolley | Scalding teapot (short splash cone); sugar lumps (thrown, rapid, weak) | **Tea Break**: parks the trolley; allies near it heal over time, and it blocks shots like cover |

Healing in co-op: the Tea Trolley can also hand a cuppa directly to a team
mate (a short-range heal on a cooldown), so the healer is useful even when
the team is spread out.

On the bench, if a fourth slot is ever wanted: **The Temp** (rogue): fast and
sneaky, staple gun and letter opener (double damage from behind), and
"Nobody Notices the Temp" (enemies take longer to spot you).

## Enemies [planned]

Each maps to a classic shooter role, so the AI needs only a handful of
behaviours (see "NPC design" below).

| Enemy | Role | Behaviour | Attack | Notes |
|---|---|---|---|---|
| **Feral Stapler** | Swarm | Hops in groups, fast, weak | Melee bite | Come in fives; the first thing you meet |
| **Possessed Swivel Chair** | Charger | Lines up, then rolls at you fast | Ram (knockback) | Dodges badly; hits walls and spins |
| **Haywire Photocopier** | Turret | Stationary, turns to face you | Wads of jammed paper (slow projectile) | Tanky; dies in a toner cloud that briefly blinds. Inert copiers are furniture, so you never quite trust one |
| **Middle Manager** | Soldier | Walks, takes cover, shouts | Thrown memos (fast projectile) | The basic grunt. Barks lines: "Synergy!", "Let's take this offline!" |
| **Fax Swarm** | Flyer | Curls of thermal paper flapping erratically | Paper cuts (melee) | Bat-like; hard to hit |
| **Coffee Machine** | Area denial | Stationary | Sprays scalding puddles that linger | Puddles hurt; walk round them |
| **HR Rep** | Support | Keeps away from you, buffs nearby enemies | "Performance Review": slows you | Priority target |
| **The Boss** | Mini-boss | Paces, charges, enrages at half health | Memo volley; desk-slam shockwave | Executive floor |
| **The CEO** | Final boss | Phases: golf putts, then a shredder, then the golden parachute | Varies by phase | Boardroom |

## Office furniture

Passive decoration. **Blocking** things stop movement (and, later, shots).

| Thing | Blocks | Status |
|---|---|---|
| Desk pod (a row of back-to-back steel desk pairs, a seam where each pair meets and joins between pairs, a Mac over each sitter's knee space) | yes | [built] |
| Cubicle screens on desk pods (fabric, aluminium cap, seated head height; a booth per sitter) | yes (from the side) | [built] |
| Executive desk (kneehole desk: drawer pedestals, open in the middle) | yes | [built] |
| Reception counter (low block, wood, 1 m) | yes | [built] |
| Beige PC terminal (stands on a desk pod or counter; rotating sprite, 16 views, faces its map angle) | no | [built] |
| Apple Macintosh SE/30 with keyboard and mouse (on a desk; solid box model, faces its map angle) | no | [built] |
| Desk with beige PC (single desk, sprite) | yes | [built] |
| Office door: wood veneer, lever handle, kick plate; swings open on use (E) | yes, until opened | [built] (prototype) |
| Double office doors: two such leaves meeting in the middle, for full-width doorways | yes, until opened | [built] (prototype) |
| Sliding glass doors: aluminium frame, frosted band, push bar; open by themselves | yes, until open | [built] (prototype) |
| Reception desk (box model): a cream laminate transaction ledge with a chrome edge over a slatted wood front with "WELCOME" in brass letters, on a recessed plinth; behind, a grey work surface at desk height over a drawer pedestal, the switchboard phone; returns at both ends; the visitors' book on the ledge | yes | [built] |
| "Boomer Industries" sign: lit brass letters on a dark panel, high on the lobby wall behind reception above the glass doors | no | [built] |
| Main breaker: steel switchgear cabinet with louvres, a hazard band, a red-handled lever and a status lamp (red, then green) | yes | [built] |
| Emergency lighting: until the breaker is thrown, only emergency panels are lit and the lobby sign is dark | no | [built] |
| Lifts: brushed steel doors, a call button (lit amber when called) and a floor indicator (lit with power); a steel car with a handrail behind | no | [built] |
| Card readers by locked doors: stripe in the pass colour, red/green lamp | no | [built] |
| Water cooler (box model: cream cabinet, hot and cold taps, an upturned 19-litre bottle nearly as wide as the cabinet, ~1.45 m to its top; faces its map angle) | yes | [built] |
| Filing cabinet (box model: five steel drawers with pulls and label cards; faces its map angle) | yes | [built] |
| Potted ficus (box model: octagonal terracotta pot, a single trunk, a crown of crossed see-through leaf cards drawn as individual drooping leaves in a ragged outline) | yes | [built] |
| Fluorescent ceiling light (a panel set into the ceiling) | no | [built] |
| Exit sign (marks the exit) | no | [built] |
| Swivel chair (inert) | no | planned |
| Photocopier (inert) | yes | planned |
| Fax machine on stand | yes | planned |
| Vending machine | yes | planned |
| Coat stand | yes | planned |
| Ashtray bin | yes | planned |
| Flip chart | yes | planned |
| Overhead projector on trolley | yes | planned |
| Fire extinguisher | no | planned |

## Areas [built]

Each cell of a floor is one of four kinds of space (`[areas]` in the map). The
area sets the floor, the ceiling, and how plain walls (`#`) look from that
side, so the same wall can be stone in the lobby and blockwork in the plant
room behind it.

| Area | Map letter | Plain walls | Floor | Ceiling |
|---|---|---|---|---|
| Open-plan office | `o` (default) | Beige woodchip above a painted dado, skirting | Blue-grey carpet tiles, laid quarter-turned | Suspended ceiling: fissured tiles in a white grid 4 x 4 per cell, lined up with its edges; a light panel is the middle 2 x 2, the grid carried across it |
| Executive | `e` | Wood panelling: four tall raised panels across, dado rail, short panels below | Deep blue carpet tiles | Suspended ceiling |
| Lobby | `l` | Pale polished stone slabs | Terrazzo slabs | Smooth plaster |
| Service | `s` | Painted breeze block: cream gloss above a grey dado | Grey flecked lino | Bare concrete, formwork marks |
| Lift car | `c` | Brushed stainless panels with a handrail (the car walls `k`: a tinted mirror on the back wall) | Black studded rubber | Perforated brushed steel |

Windows (`4`, ribbon windows with blinds, in offices; `f`, the curtain wall,
in the lobby), lift doors and furniture keep their own textures wherever
they are. Where a ceiling steps down (over a doorway), the face above
looks like the area you see it from.

Soffits, the underside of a doorway's lintel, are lined rather than tiled:
painted plaster framed in timber (office), timber boards in a moulding
(executive), veined stone in a steel edge (lobby), concrete (service),
perforated steel (lift car). A balcony carries on its room: its front is
the area's wall (stone in the lobby) and its floor the area's floor
(terrazzo); the stairs up to it have the area's walls for sides and a blue
carpet runner on the treads, each with a metal nosing. Plain raised floors
(`e`) are fitted blue carpet.

Lifts [built]: brushed stainless doors in a doorway lined with polished
steel (architrave, grooved sill), a floor indicator above (amber G between
up and down arrows, dark without power), a call plate beside (the up
button glows once called), and inside the car a button panel by the door
(B, G, 1-5, P; G lit).

## Walls [built]

| Wall type | Texture | Where |
|---|---|---|
| 1 | Plain wall: looks like the area it faces (see Areas) | Everywhere |
| 2 | Wood panelling, four tall panels across (the executive look, wherever it's placed) | Executive offices |
| 3 | Grey cubicle partition | Open-plan floors |
| 4 | Window with venetian blinds | Outer walls |
| 5 | Lift doors | Exits |
| 6 | Stairwell brick | Stairs, basement |
| 7 | Toilet / kitchen tiles | Washrooms |
| 8 | Steel panels | Server room |
| 9 | Mainframe cabinet (blinking lights) | Server room |
| a (10) | Plain table: laminate slab on steel legs, open underneath (low block, height 2) | Spare: canteen, meeting rooms |
| b (11) | Executive desk: kneehole between drawer pedestals at the front; solid wood modesty panel and sides (low block, height 2, uses [facing]) | Executive offices |
| c (12) | 1980s steel desk: mahogany laminate top, grey steel frame and legs, one drawer pedestal at its left end; open underneath (low block, height 2, uses [facing]) | Single desks |
| d (13) | Two steel desks back to back in one cell, sitters facing each other: both long sides show a desk front (pedestal on the right), both ends are half pedestal panel, half open (low block, height 2; [facing] sets the axis). Its top carries paperwork beside each sitter's Mac (textured, flat things only): manila files shaped like a desktop's folder icons, splayed about, stacked with sheets on, or open flat; loose A4 with lines of type; on the bare desk beside them, pink "While You Were Out" message slips (and, in one variant, a pencil). Four variants | Open-plan pods |
| e (14) | Raised floor: plain sides with a nosing line every 1/8, blue carpet top (low block; walk up 1/8 steps) | Lobby gallery and its steps |
| f (15) | Glazed curtain wall: aluminium mullions and glass, two storeys meeting at a transom | Lobby outer walls |
| g (16) | Main entrance: closed glass double doors, curtain wall above | Lobby entrance |
| h (17) | Stairs: half a storey per cell ([facing] = up); you walk a smooth slope, drawn as 8 carpeted steps | Lobby stairs |
| i (18) | Balcony floor: raised floor with a railing along edges that drop away; front and floor dressed by its area | Lobby balcony |
| k (20) | Lift car walls: brushed stainless panels and a handrail inside the car, a tinted mirror on the back wall (its front: [facing] points into the car); outside, a plain wall dressed by its area | Lift cars |
| l (21) | Pale polished stone slabs (the lobby look, wherever it's placed) | Round lift doors |
| j (19) | Desk pair with cubicle screens: as `d`, plus fabric screens to seated head height (~1.25 m) down the middle and across each cell edge (box models; they block you on the desk tops). Each side of a screen is bare or has things pinned to it: a couple of Post-its, a cluster of them, or a wall calendar | Open-plan cubicles |

## Pickups

- **ID passes** [built]: lanyard passes in three colours, staff (blue),
  facilities (red), executive (gold), for locked doors. Shared by the team:
  one player picks it up, everyone can use it. A floor's loop: the staff
  pass gets you off the lobby, the facilities pass into the plant room,
  where the power comes on and the lifts wake up (see Game loop).
- **Health** [planned]: cup of coffee (small), box of doughnuts (large), sick
  note (full).
- **Armour** [planned]: shoulder pads.
- **Ammo** [planned]: box of floppies (IT Guy), tea bags (Tea Trolley), fire
  extinguisher refill (Caretaker).

## NPC design (how enemies will work in code)

Built to the code design rules in `DESIGN.md`.

- **Server-authoritative.** Enemies live in the shared `World` as entities and
  are moved by `step()`, deterministically (randomness from `world.rng`).
  Clients only draw them from snapshots, interpolated like other players.
- **Data-driven types.** An `EnemyType` registry entry holds the numbers
  (health, speed, radius, sight range, attack, pain chance) and names a
  behaviour. Adding an enemy = one registry entry + sprite art, unless it needs
  a genuinely new behaviour.
- **Behaviours are small state machines**, registered by name and shared
  between enemy types:
  `IDLE -> ALERT -> CHASE -> ATTACK -> (PAIN) -> CHASE ... -> DEAD`.
  The roles above (swarm, charger, turret, soldier, flyer, area denial,
  support) need about five behaviours between them.
- **Sight** is a raycast from the enemy to each player using the shared DDA
  (with a field-of-view check and a reaction delay), so walls block it
  exactly as they block rendering.
- **Movement**: steer straight at the target when it's visible. When it isn't,
  follow a flow field: a breadth-first search over the grid outward from each
  player, recomputed a few times a second. On a 30x30 grid that's ~900 cells,
  cheap even in Python, and every enemy reads the same field, so 50 enemies
  cost about the same as 1.
- **Attacks**: melee (range check), hitscan (the DDA again), or projectiles
  (projectiles are entities with their own movement and collision).
- **Events** (`EnemyAlerted`, `EnemyAttacked`, `EnemyHurt`, `EnemyDied`) drive
  sounds and barks on the client; the AI never touches audio.
- **Sprites**: frame chosen by state (walk, attack, pain, death) and, for
  enemies that face somewhere, by viewing angle, as in Doom (16 rotations,
  baked from voxel models, `client/models.py`, as the PC terminal's are
  [built]; animation poses would be one model each). Boxy enemies (the
  photocopier) could instead be box models (`client/boxes.py`) [planned].
- **Modes**: enemies appear in co-op. Deathmatch ignores enemy things in the
  map, like other mode-specific things.
