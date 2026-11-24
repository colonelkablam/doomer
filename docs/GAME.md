# OK Doomer: game design

The setting, cast and content. Engine and code design live in `DESIGN.md`.
Status markers: **[built]** exists in the game, **[planned]** is design only.

## Setting

**When**: 1983–1991. Beige PCs with green-screen and amber CRTs, dot-matrix
printers, fax machines going mainstream, cubicle farms, shoulder pads, ashtrays
on desks being quietly removed. Ends before Windows 95, email and the
internet. The baby boomers (born 1946–64) are 19–45 and running the place.

**Where**: one corporate office block, climbed floor by floor. Each map is a
floor; the exit is the lift or the stairwell. You start in the basement
mailroom and finish in the penthouse boardroom.

**Tone**: deadpan office comedy played straight as a shooter. Nothing is gory:
enemies break, spark, jam, or storm off.

Suggested floor order:

| Floor | Theme | New things introduced |
|---|---|---|
| B1 | Mailroom and post | Staplers, photocopiers |
| G | Reception and lobby | Swivel chairs, water coolers |
| 1 | Open-plan sales floor | Middle managers |
| 2 | Accounts (all filing cabinets) | Fax swarms |
| 3 | IT and the server room | Haywire mainframe terminals |
| 4 | HR and meeting rooms | HR reps, flip charts |
| 5 | Executive floor, wood panelling | The Boss (mini-boss) |
| P | Penthouse boardroom | The CEO (final boss) |

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
| Desk pod (2x2 steel desks back to back, drawers on the sitters' side) | yes | [built] |
| Executive desk (kneehole desk: drawer pedestals, open in the middle) | yes | [built] |
| Reception counter (low block, wood, 1 m) | yes | [built] |
| Beige PC terminal (stands on a desk pod or counter) | no | [built] |
| Desk with beige PC (single desk, sprite) | yes | [built] |
| Water cooler | yes | [built] |
| Filing cabinet | yes | [built] |
| Potted ficus | yes | [built] |
| Fluorescent ceiling light | no | [built] |
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

## Walls [built]

| Wall type | Texture | Where |
|---|---|---|
| 1 | Beige office wall with dado rail | Everywhere |
| 2 | Wood panelling | Executive offices |
| 3 | Grey cubicle partition | Open-plan floors |
| 4 | Window with venetian blinds | Outer walls |
| 5 | Lift doors | Exits |
| 6 | Stairwell brick | Stairs, basement |
| 7 | Toilet / kitchen tiles | Washrooms |
| 8 | Steel panels | Server room |
| 9 | Mainframe cabinet (blinking lights) | Server room |
| a (10) | Plain table: laminate slab on steel legs, open underneath (low block, height 2) | Spare: canteen, meeting rooms |
| b (11) | Executive desk: kneehole between drawer pedestals at the front; solid wood modesty panel and sides (low block, height 2, uses [facing]) | Executive offices |
| c (12) | 1980s steel desk: mahogany laminate top, grey steel frame and legs, one drawer pedestal at its left end; open underneath (low block, height 2, uses [facing]) | Open-plan desk pods, back to back |

## Pickups [planned]

- **Health**: cup of coffee (small), box of doughnuts (large), sick note (full).
- **Armour**: shoulder pads.
- **Ammo**: box of floppies (IT Guy), staples (Temp), fire extinguisher refill
  (Caretaker).
- **Keys**: colour-coded security passes (red, blue, yellow) for locked doors.

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
  enemies that face somewhere, by viewing angle (8 rotations), as in Doom.
- **Modes**: enemies appear in co-op. Deathmatch ignores enemy things in the
  map, like other mode-specific things.
