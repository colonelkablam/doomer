"""The wire format: every message between client and server, in one place.

Messages are frozen dataclasses. On the wire each is one JSON object on one
line (UTF-8, ending "\\n"), with a "type" key naming it. Nothing else builds
JSON by hand: `encode` and `decode` are the only way in and out, so the
framing could change (UDP, a binary format) here alone.

A conversation:

    client                          server
    Hello(version, name)   ->
                           <-       Welcome(player_id, mode, config, map_text)
                                    (or Reject(reason), and the server hangs up)
    Input(seq, ...)        ->       one per client tick, numbered
                           <-       Update(ack, snapshot, events), each server tick
                           <-       NewFloor(map_text): the lift's arrived
    Bye(reason)            <->      either side, before hanging up

Everything decoded is checked: a client's data is untrusted. Anything
malformed raises ProtocolError (never KeyError or TypeError), and input
numbers must be finite, since a NaN angle would spread through the world.
"""

import dataclasses
import hashlib
import json
import math
import typing
from dataclasses import dataclass
from typing import Any, ClassVar

from ok_doomer.shared.config import ConfigError, MovementConfig, SimConfig, from_dict
from ok_doomer.shared.events import Event
from ok_doomer.shared.player import PlayerInput
from ok_doomer.shared.snapshot import Row, Snapshot

# Bump whenever the wire format changes (a message, an event, a field):
# then a client and server from different code refuse each other at once,
# with the versions, rather than half-work (version 1 servers sent no join
# events: newer clients joined them and never heard who came and went).
# tests/test_protocol.py pins schema_fingerprint() to each version.
PROTOCOL_VERSION = 3
MAX_LINE = 1 << 20  # bytes: the longest line either side accepts (a map is ~20 KB)


class ProtocolError(ValueError):
    """A message that can't be decoded, or isn't what was expected."""


@dataclass(frozen=True, slots=True)
class Hello:
    """Client -> server, first: who's joining, speaking which version."""

    TYPE: ClassVar[str] = "hello"
    version: int
    name: str


@dataclass(frozen=True, slots=True)
class Welcome:
    """Server -> client: you're in. Build your world from `map_text` (with
    this mode and config, so `step` runs the same on both sides)."""

    TYPE: ClassVar[str] = "welcome"
    player_id: int
    mode: str
    config: SimConfig
    map_text: str


@dataclass(frozen=True, slots=True)
class Reject:
    """Server -> client: you're not in (wrong version, server full); the
    server hangs up after sending it."""

    TYPE: ClassVar[str] = "reject"
    reason: str


@dataclass(frozen=True, slots=True)
class Input:
    """Client -> server: one tick of a player's input, numbered from 1 so
    the server can say which it's applied (`Update.ack`)."""

    TYPE: ClassVar[str] = "input"
    seq: int
    input: PlayerInput


@dataclass(frozen=True, slots=True)
class Update:
    """Server -> client, each tick: the world as it now is, what happened
    this tick, and the last of this client's inputs applied (0: none yet)."""

    TYPE: ClassVar[str] = "update"
    ack: int
    snapshot: Snapshot
    events: tuple[Event, ...] = ()


@dataclass(frozen=True, slots=True)
class NewFloor:
    """Server -> client: the team's on the next floor. Build a new world
    from `map_text`; Updates from now on are of it."""

    TYPE: ClassVar[str] = "new_floor"
    map_text: str


@dataclass(frozen=True, slots=True)
class Bye:
    """Either way, before hanging up: why (quit, kicked, timed out)."""

    TYPE: ClassVar[str] = "bye"
    reason: str = ""


@dataclass(frozen=True, slots=True)
class Seek:
    """Finding games (UDP, broadcast on DISCOVERY_PORT): "who's running a
    game?" Servers answer with a GameInfo."""

    TYPE: ClassVar[str] = "seek"
    version: int


@dataclass(frozen=True, slots=True)
class GameInfo:
    """A server's answer to a Seek: what's running, and how to join it (TCP
    `port` at the address it answered from). `server_id`: random, so one
    game heard two ways (the LAN and this machine) is listed once."""

    TYPE: ClassVar[str] = "game"
    version: int
    server_id: str
    name: str  # the server's (its machine's name, unless set)
    floor: str  # what it's called: GROUND FLOOR
    map_name: str
    mode: str
    players: int
    max_players: int
    port: int


DISCOVERY_PORT = 5028  # UDP: Seek and GameInfo; the game itself is TCP (5029)
MAX_DATAGRAM = 1024  # bytes: a Seek or a GameInfo is far smaller

Message = Hello | Welcome | Reject | Input | Update | NewFloor | Bye | Seek | GameInfo
MESSAGES: dict[str, type[Message]] = {
    m.TYPE: m for m in (Hello, Welcome, Reject, Input, Update, NewFloor, Bye, Seek, GameInfo)
}


def encode(message: Message) -> bytes:
    """One message as a line of JSON."""
    data = {"type": message.TYPE, **_message_to_dict(message)}
    text = json.dumps(data, separators=(",", ":"), allow_nan=False)
    return text.encode() + b"\n"


class Encoder:
    """`encode`, but each tick's Update encoded once, not once per client:
    a server sends every client the same snapshot and events, only the
    `ack` differs. The JSON of the shared part is kept and each client's
    line is its ack spliced on (the same bytes `encode` makes). Use one
    per tick."""

    def __init__(self) -> None:
        self._shared: tuple[Snapshot, tuple[Event, ...], bytes] | None = None

    def encode(self, message: Message) -> bytes:
        if not isinstance(message, Update):
            return encode(message)
        cached = self._shared
        if cached is None or cached[0] is not message.snapshot or cached[1] is not message.events:
            rest = {k: v for k, v in _message_to_dict(message).items() if k != "ack"}
            text = json.dumps(rest, separators=(",", ":"), allow_nan=False)
            cached = (message.snapshot, message.events, text[1:].encode())  # less its "{"
            self._shared = cached
        head = f'{{"type":"{Update.TYPE}","ack":{message.ack},'.encode()
        return head + cached[2] + b"\n"


def decode(line: bytes) -> Message:
    """A line of JSON as a message; ProtocolError if it isn't one."""
    if len(line) > MAX_LINE:
        raise ProtocolError(f"line too long ({len(line)} bytes)")
    try:
        data = json.loads(line, parse_constant=_no_constants)
    except ProtocolError:
        raise
    except (ValueError, UnicodeDecodeError) as e:
        raise ProtocolError(f"not JSON: {e}") from e
    if not isinstance(data, dict):
        raise ProtocolError("a message is a JSON object")
    kind = MESSAGES.get(data.pop("type", None))
    if kind is None:
        raise ProtocolError("unknown message type")
    try:
        return _message_from_dict(kind, data)
    except ProtocolError:
        raise
    except (KeyError, TypeError, ValueError, ConfigError) as e:
        raise ProtocolError(f"bad {kind.TYPE}: {e!r}") from e


def _no_constants(name: str) -> None:
    raise ProtocolError(f"{name} isn't allowed")


# -- messages -------------------------------------------------------------


def _message_to_dict(m: Message) -> dict[str, Any]:
    if isinstance(m, Welcome):
        return {
            "player_id": m.player_id,
            "mode": m.mode,
            "config": dataclasses.asdict(m.config),
            "map_text": m.map_text,
        }
    if isinstance(m, Input):
        return {"seq": m.seq, **dataclasses.asdict(m.input)}
    if isinstance(m, Update):
        return {
            "ack": m.ack,
            "snapshot": dataclasses.asdict(m.snapshot),
            "events": [event_to_dict(e) for e in m.events],
        }
    return dataclasses.asdict(m)


def _message_from_dict(kind: type[Message], d: dict[str, Any]) -> Message:
    if kind is Hello:
        return Hello(_int(d["version"]), _str(d["name"]))
    if kind is Welcome:
        config = d["config"]
        if not isinstance(config, dict):
            raise ProtocolError("config must be an object")
        return Welcome(
            _int(d["player_id"]), _str(d["mode"]), from_dict(SimConfig, config), _str(d["map_text"])
        )
    if kind is Reject:
        return Reject(_str(d["reason"]))
    if kind is Input:
        return Input(_int(d.pop("seq")), input_from_dict(d))
    if kind is Update:
        evs = d["events"]
        if not isinstance(evs, list):
            raise ProtocolError("events must be a list")
        return Update(
            _int(d["ack"]),
            snapshot_from_dict(d["snapshot"]),
            tuple(event_from_dict(e) for e in evs),
        )
    if kind is NewFloor:
        return NewFloor(_str(d["map_text"]))
    if kind is Bye:
        return Bye(_str(d.get("reason", "")))
    if kind is Seek:
        return Seek(_int(d["version"]))
    if kind is GameInfo:
        return GameInfo(
            _int(d["version"]),
            _str(d["server_id"]),
            _str(d["name"]),
            _str(d["floor"]),
            _str(d["map_name"]),
            _str(d["mode"]),
            _int(d["players"]),
            _int(d["max_players"]),
            _int(d["port"]),
        )
    raise ProtocolError(f"no decoder for {kind.TYPE}")  # pragma: no cover


# -- inputs ---------------------------------------------------------------

_INPUT_FIELDS = {f.name: f.type for f in dataclasses.fields(PlayerInput)}


def input_from_dict(d: dict[str, Any]) -> PlayerInput:
    """A player's input, checked: finite numbers, real booleans, nothing
    unknown. (forward and strafe are clamped by the simulation itself.)"""
    unknown = set(d) - set(_INPUT_FIELDS)
    if unknown:
        raise ProtocolError(f"unknown input fields {sorted(unknown)}")
    values: dict[str, Any] = {}
    for name, value in d.items():
        hint = _INPUT_FIELDS[name]
        values[name] = _bool(value) if hint in (bool, "bool") else _float(value)
    return PlayerInput(**values)


# -- snapshots ------------------------------------------------------------

_SNAPSHOT_FIELDS = [f.name for f in dataclasses.fields(Snapshot)]
_ROW_FIELDS = ("players", "doors", "cubicle_doors", "pickups", "switches", "dispensers")


def snapshot_from_dict(d: Any) -> Snapshot:
    if not isinstance(d, dict) or set(d) != set(_SNAPSHOT_FIELDS):
        raise ProtocolError("a snapshot has exactly the Snapshot fields")
    values: dict[str, Any] = {}
    for name in _ROW_FIELDS:
        rows = d[name]
        if not isinstance(rows, list) or not all(isinstance(r, list) for r in rows):
            raise ProtocolError(f"snapshot {name} must be a list of rows")
        values[name] = tuple(_row(r) for r in rows)
    for name in ("passes", "flags"):
        items = d[name]
        if not isinstance(items, list):
            raise ProtocolError(f"snapshot {name} must be a list")
        values[name] = tuple(_str(s) for s in items)
    values["tick"] = _int(d["tick"])
    values["ride"] = _float(d["ride"])
    values["complete"] = _bool(d["complete"])
    return Snapshot(**values)


def _row(row: list[Any]) -> Row:
    """A row's values: JSON scalars, and lists (a dispenser's pairs) back
    into tuples."""
    out: list[Any] = []
    for v in row:
        if isinstance(v, list):
            out.append(tuple((_int(k), _int(n)) for k, n in (_pair(p) for p in v)))
        elif v is None or isinstance(v, bool | int | str):
            out.append(v)
        else:
            out.append(_float(v))
    return tuple(out)


def _pair(p: Any) -> tuple[Any, Any]:
    if not isinstance(p, list) or len(p) != 2:
        raise ProtocolError(f"expected a pair, got {p!r}")
    return p[0], p[1]


# -- events ---------------------------------------------------------------


def _all_event_types() -> dict[str, type[Event]]:
    """Every Event subclass (all of shared/events.py, loaded by importing
    Event), by class name. New events need no change here."""
    found: dict[str, type[Event]] = {}
    pending = list(Event.__subclasses__())
    while pending:
        cls = pending.pop()
        found[cls.__name__] = cls
        pending += cls.__subclasses__()
    return found


EVENT_TYPES = _all_event_types()
_EVENT_HINTS = {cls: typing.get_type_hints(cls) for cls in EVENT_TYPES.values()}


def event_to_dict(event: Event) -> dict[str, Any]:
    return {"type": type(event).__name__, **dataclasses.asdict(event)}


def event_from_dict(d: Any) -> Event:
    if not isinstance(d, dict):
        raise ProtocolError("an event is a JSON object")
    d = dict(d)
    cls = EVENT_TYPES.get(d.pop("type", None))
    if cls is None:
        raise ProtocolError("unknown event type")
    fields = {f.name for f in dataclasses.fields(cls)}
    if set(d) - fields:
        raise ProtocolError(f"unknown fields for {cls.__name__}: {sorted(set(d) - fields)}")
    for name, value in d.items():
        if isinstance(value, float):
            _float(value)
        elif isinstance(value, list) and _EVENT_HINTS[cls].get(name) == tuple[str, ...]:
            d[name] = tuple(_str(v) for v in value)  # names (PlayersHere)
        elif not (value is None or isinstance(value, bool | int | str)):
            raise ProtocolError(f"{cls.__name__}.{name}: {value!r}")
    return cls(**d)


# -- checked scalars --------------------------------------------------------


def _int(v: Any) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ProtocolError(f"expected an integer, got {v!r}")
    return v


def _float(v: Any) -> float:
    if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
        raise ProtocolError(f"expected a finite number, got {v!r}")
    return float(v)


def _bool(v: Any) -> bool:
    if not isinstance(v, bool):
        raise ProtocolError(f"expected true or false, got {v!r}")
    return v


def _str(v: Any) -> str:
    if not isinstance(v, str):
        raise ProtocolError(f"expected a string, got {v!r}")
    return v


def schema_fingerprint() -> str:
    """A hash of the wire format: every message and event, its fields and
    their types, and the input, snapshot and config they carry. Changes
    when the format does (so PROTOCOL_VERSION must too)."""
    parts = []
    shapes: list[type] = [*MESSAGES.values(), *EVENT_TYPES.values(), PlayerInput, Snapshot]
    shapes += [SimConfig, MovementConfig]  # the config the welcome carries
    for cls in sorted(shapes, key=lambda c: c.__name__):
        hints = typing.get_type_hints(cls)
        fields = [f"{f.name}:{hints[f.name]}" for f in dataclasses.fields(cls)]
        parts.append(f"{cls.__name__}({','.join(fields)})")
    return hashlib.sha256(";".join(parts).encode()).hexdigest()[:16]
