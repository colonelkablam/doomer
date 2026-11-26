"""Snapshots: the part of a World that changes during play, to send from the
server to clients (and, in single-player, from the in-process server to the
game).

A client builds its own World from the same map the server sent, so
everything the map fixes (walls, where each door is, what each pickup is)
is already there. A snapshot carries only what changes: the players, how
far open each door is, which pickups are gone, the passes and flags. It's
applied onto the client's world, overwriting those fields.

Doors, pickups and the rest are lists built from the map in the same order
on both sides, so they go positionally: each a tuple of its changing
fields' values, in the order listed in STATE below. Players come and go, so
they go whole, with their ids.

Every field of every entity type is listed below as changing (sent) or
fixed by the map (not sent); `tests/test_snapshot.py` checks that none is
left out, so a new field forces the choice.

Entities that appear during play (enemies, dropped items) aren't
positional: they'll need ids and to go whole, as players do.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any

from ok_doomer.shared.cubicles import CubicleDoor
from ok_doomer.shared.dispensers import Dispenser
from ok_doomer.shared.doors import Door
from ok_doomer.shared.pickups import Pickup
from ok_doomer.shared.player import PlayerState
from ok_doomer.shared.switches import Switch
from ok_doomer.shared.world import World

# A field's value as sent: JSON-able scalars (a dispenser's `used` dict goes
# as sorted (player id, given) pairs).
Value = int | float | bool | str | None | tuple[tuple[int, int], ...]
Row = tuple[Value, ...]

# Per entity type: the fields that change during play (sent), and those the
# map fixes (not sent).
STATE: dict[type, tuple[str, ...]] = {
    Door: ("open", "opening", "car_floor", "called", "arrive_in"),
    CubicleDoor: ("open", "opening"),
    Pickup: ("taken",),
    Switch: ("on",),
    Dispenser: ("used",),
}
FIXED: dict[type, tuple[str, ...]] = {
    Door: ("id", "kind", "x", "y", "facing", "lock", "here"),  # here: the floor it's on
    CubicleDoor: ("id", "run_x", "run_y", "run_angle", "index", "count", "locked"),
    Pickup: ("id", "type", "x", "y", "z", "angle"),
    Switch: ("id", "type", "x", "y", "facing", "flag"),
    Dispenser: ("id", "type", "x", "y"),
}
# Players go whole: every field, `id` first.
PLAYER_FIELDS = tuple(f.name for f in dataclasses.fields(PlayerState))

# The World's own fields: in the snapshot, or not (and why).
WORLD_SENT = (
    "players", "tick", "doors", "pickups", "passes", "switches", "dispensers",
    "cubicle_doors", "flags", "ride", "complete",
)  # fmt: skip
WORLD_NOT_SENT = {
    "map": "the client loads it from the map the server sent",
    "mode": "sent once, when joining",
    "config": "sent once, when joining",
    "rng": "the server's alone: clients never roll dice",
    "events": "sent alongside, in the same message",
    "obstacles": "fixed by the map",
    "screens": "fixed by the map",
}


@dataclass(frozen=True, slots=True)
class Snapshot:
    tick: int
    players: tuple[Row, ...]  # PLAYER_FIELDS values, one row per player, by id
    doors: tuple[Row, ...]  # STATE[Door] values, in World.doors order
    cubicle_doors: tuple[Row, ...]
    pickups: tuple[Row, ...]
    switches: tuple[Row, ...]
    dispensers: tuple[Row, ...]
    passes: tuple[str, ...]  # sorted
    flags: tuple[str, ...]  # sorted
    ride: float
    complete: bool


class SnapshotMismatch(ValueError):
    """The snapshot doesn't fit this world: it's of another map."""


def capture(world: World) -> Snapshot:
    """The changing state of a world, as it is now."""
    return Snapshot(
        tick=world.tick,
        players=tuple(
            tuple(_out(getattr(p, f)) for f in PLAYER_FIELDS)
            for p in sorted(world.players.values(), key=lambda p: p.id)
        ),
        doors=_rows(world.doors, Door),
        cubicle_doors=_rows(world.cubicle_doors, CubicleDoor),
        pickups=_rows(world.pickups, Pickup),
        switches=_rows(world.switches, Switch),
        dispensers=_rows(world.dispensers, Dispenser),
        passes=tuple(sorted(world.passes)),
        flags=tuple(sorted(world.flags)),
        ride=world.ride,
        complete=world.complete,
    )


def apply(world: World, snap: Snapshot) -> None:
    """Overwrite a world's changing state with a snapshot's. The world must
    be of the same map (built from the same map text). Players not in the
    snapshot are removed; new ones are added."""
    _apply_rows(world.doors, Door, snap.doors)
    _apply_rows(world.cubicle_doors, CubicleDoor, snap.cubicle_doors)
    _apply_rows(world.pickups, Pickup, snap.pickups)
    _apply_rows(world.switches, Switch, snap.switches)
    _apply_rows(world.dispensers, Dispenser, snap.dispensers)
    players: dict[int, PlayerState] = {}
    for row in snap.players:
        sent = player_from_row(row)
        player = world.players.get(sent.id)
        if player is None:  # joined
            player = sent
        else:  # update in place, so anything holding it (a session) sees it move
            for name in PLAYER_FIELDS:
                setattr(player, name, getattr(sent, name))
        players[sent.id] = player
    world.players = players
    world.tick = snap.tick
    world.passes = set(snap.passes)
    world.flags = set(snap.flags)
    world.ride = snap.ride
    world.complete = snap.complete


def player_from_row(row: Row) -> PlayerState:
    """A player as a snapshot sent them (PLAYER_FIELDS values)."""
    if len(row) != len(PLAYER_FIELDS):
        raise SnapshotMismatch(f"a player has {len(row)} fields, not {len(PLAYER_FIELDS)}")
    values: dict[str, Any] = dict(zip(PLAYER_FIELDS, row, strict=True))
    if not isinstance(values["id"], int):
        raise SnapshotMismatch(f"player id {values['id']!r}")
    return PlayerState(**values)


def _rows(entities: list[Any], kind: type) -> tuple[Row, ...]:
    return tuple(tuple(_out(getattr(e, f)) for f in STATE[kind]) for e in entities)


def _apply_rows(entities: list[Any], kind: type, rows: tuple[Row, ...]) -> None:
    fields = STATE[kind]
    if len(rows) != len(entities):
        raise SnapshotMismatch(f"{len(rows)} {kind.__name__}s, the world has {len(entities)}")
    for entity, row in zip(entities, rows, strict=True):
        if len(row) != len(fields):
            raise SnapshotMismatch(f"a {kind.__name__} has {len(row)} fields, not {len(fields)}")
        for name, value in zip(fields, row, strict=True):
            setattr(entity, name, _in(getattr(entity, name), value))


def _out(value: Any) -> Value:
    """A field's value as sent: dicts (a dispenser's `used`) as sorted pairs."""
    if isinstance(value, dict):
        return tuple(sorted(value.items()))
    out: Value = value
    return out


def _in(current: Any, value: Value) -> Any:
    """Back again: pairs into a dict where the field is one."""
    if isinstance(current, dict):
        if not isinstance(value, tuple):
            raise SnapshotMismatch(f"expected (key, value) pairs, got {value!r}")
        return {int(k): int(v) for k, v in value}
    return value
