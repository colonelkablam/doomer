"""The wire format: every message and event round-trips exactly, and
anything malformed is a ProtocolError, never a crash."""

import json
import random
import typing
from dataclasses import fields

import pytest

from ok_doomer.shared.building import floor_map_path
from ok_doomer.shared.config import MovementConfig, SimConfig
from ok_doomer.shared.events import Event, PassPickedUp
from ok_doomer.shared.mapfile import parse_map
from ok_doomer.shared.player import PlayerInput
from ok_doomer.shared.protocol import (
    EVENT_TYPES,
    MAX_LINE,
    MESSAGES,
    PROTOCOL_VERSION,
    Bye,
    Hello,
    Input,
    Message,
    NewFloor,
    ProtocolError,
    Reject,
    Update,
    Welcome,
    decode,
    encode,
    event_from_dict,
    event_to_dict,
)
from ok_doomer.shared.sim import step
from ok_doomer.shared.snapshot import apply, capture
from ok_doomer.shared.spawn import spawn_player
from ok_doomer.shared.world import World, new_world

OFFICE = floor_map_path("G").read_text()


def _world(players: int = 4) -> World:
    world = new_world(parse_map(OFFICE), "coop")
    for i in range(1, players + 1):
        spawn_player(world, i, f"p{i}")
    world.dispensers[0].used[2] = 10
    rng = random.Random(1)
    for _ in range(60):
        step(world, {i: PlayerInput(forward=1, angle=rng.uniform(0, 6)) for i in world.players})
    return world


def _sample(hint: object) -> object:
    """A non-default value of a field's type."""
    if hint is bool:
        return True
    if hint is int:
        return 7
    if hint is float:
        return 0.3
    if hint is str or typing.get_args(hint) == (str, type(None)):
        return "red"
    if hint == tuple[str, ...]:
        return ("Ann", "Bob")
    raise AssertionError(f"teach _sample about {hint!r}")


def _sample_event(cls: type[Event]) -> Event:
    hints = typing.get_type_hints(cls)
    values: dict[str, typing.Any] = {f.name: _sample(hints[f.name]) for f in fields(cls)}
    return cls(**values)


def _messages() -> list[Message]:
    world = _world()
    return [
        Hello(PROTOCOL_VERSION, "Ann"),
        Welcome(
            3, "coop", SimConfig(tick_rate=30, movement=MovementConfig(walk_speed=4.0)), OFFICE
        ),
        Reject("server full"),
        Input(12, PlayerInput(0.5, -1.0, 1.25, -0.2, True, False, True)),
        Update(
            12,
            capture(world),
            (PassPickedUp(9, 1, "blue"), _sample_event(EVENT_TYPES["Dispensed"])),
        ),
        NewFloor(floor_map_path("1").read_text()),
        Bye("quit"),
    ]


def test_every_message_type_is_tested() -> None:
    assert {m.TYPE for m in _messages()} == set(MESSAGES)


@pytest.mark.parametrize("message", _messages(), ids=lambda m: m.TYPE)
def test_messages_round_trip(message: Message) -> None:
    line = encode(message)
    assert line.endswith(b"\n") and line.count(b"\n") == 1  # one line, map text and all
    assert decode(line) == message


@pytest.mark.parametrize("name", sorted(EVENT_TYPES))
def test_every_event_round_trips(name: str) -> None:
    event = _sample_event(EVENT_TYPES[name])
    assert event_from_dict(json.loads(json.dumps(event_to_dict(event)))) == event


def test_a_sent_snapshot_still_plays_on_the_same() -> None:
    """Through JSON and back, floats exact: the client's world is the
    server's."""
    world = _world()
    update = decode(encode(Update(0, capture(world))))
    assert isinstance(update, Update)
    fresh = new_world(parse_map(OFFICE), "coop")
    apply(fresh, update.snapshot)
    for _ in range(60):
        step(world, {})
        step(fresh, {})
    assert capture(fresh) == capture(world)


def test_an_update_is_small() -> None:
    """The ground floor with 4 players, as sent each tick (~2.3 Mbit/s at
    60 Hz to 4 clients at 1.2 KB): keep an eye on it."""
    assert len(encode(Update(100, capture(_world())))) < 1500


GOOD_INPUT = {"type": "input", "seq": 1, "forward": 1.0, "angle": 0.5, "use": False}

BAD_LINES = {
    "not json": b"{nope\n",
    "not an object": b"[1, 2]\n",
    "unknown type": b'{"type": "teleport"}\n',
    "no type": b'{"name": "x"}\n',
    "missing field": b'{"type": "hello", "version": 1}\n',
    "name not a string": b'{"type": "hello", "version": 1, "name": 5}\n',
    "version a bool": b'{"type": "hello", "version": true, "name": "x"}\n',
    "NaN angle": json.dumps({**GOOD_INPUT, "angle": "x"}).replace('"x"', "NaN").encode(),
    "Infinity": json.dumps({**GOOD_INPUT, "forward": "x"}).replace('"x"', "Infinity").encode(),
    "1e999 (reads as infinity)": json.dumps({**GOOD_INPUT, "angle": "x"})
    .replace('"x"', "1e999")
    .encode(),
    "use not a bool": json.dumps({**GOOD_INPUT, "use": 1}).encode(),
    "angle a string": json.dumps({**GOOD_INPUT, "angle": "0.5"}).encode(),
    "unknown input field": json.dumps({**GOOD_INPUT, "noclip": True}).encode(),
    "seq a float": json.dumps({**GOOD_INPUT, "seq": 1.5}).encode(),
    "bad snapshot": b'{"type": "update", "ack": 0, "snapshot": {}, "events": []}\n',
    # A real update, its snapshot fine, with an event that isn't.
    "bad event": encode(Update(0, capture(new_world(parse_map(OFFICE), "coop")))).replace(
        b'"events":[]', b'"events":[{"type":"Nope","tick":1}]'
    ),
    "names not strings": encode(Update(0, capture(new_world(parse_map(OFFICE), "coop")))).replace(
        b'"events":[]', b'"events":[{"type":"PlayersHere","tick":1,"names":["ann",3]}]'
    ),
    "bad event field": encode(Update(0, capture(new_world(parse_map(OFFICE), "coop")))).replace(
        b'"events":[]', b'"events":[{"type":"PassPickedUp","tick":1,"player_id":1,"colour":[]}]'
    ),
    "bad config": b'{"type": "welcome", "player_id": 1, "mode": "coop", '
    b'"config": {"tick_rate": "fast"}, "map_text": ""}\n',
    "too long": b'{"type": "bye", "reason": "' + b"x" * MAX_LINE + b'"}\n',
    "not utf-8": b'{"type": "bye", "reason": "\xff"}\n',
}


@pytest.mark.parametrize("line", BAD_LINES.values(), ids=list(BAD_LINES))
def test_bad_lines_are_protocol_errors(line: bytes) -> None:
    with pytest.raises(ProtocolError):
        decode(line)


def test_a_good_input_line_decodes() -> None:
    message = decode(json.dumps(GOOD_INPUT).encode())
    assert message == Input(1, PlayerInput(forward=1.0, angle=0.5))


def test_the_encoder_makes_the_same_bytes_encoding_each_update_once() -> None:
    """A tick's updates share their snapshot and events; the Encoder's lines
    are exactly encode's, for every client's ack, every message type, and a
    new tick's snapshot."""
    from ok_doomer.shared.protocol import Encoder

    world = _world()
    snap, events = capture(world), (PassPickedUp(9, 1, "blue"),)
    encoder = Encoder()
    for ack in (0, 7, 12345):
        update = Update(ack, snap, events)
        assert encoder.encode(update) == encode(update)
    for message in _messages():
        assert encoder.encode(message) == encode(message)
    step(world, {})
    newer = Update(3, capture(world), ())
    assert encoder.encode(newer) == encode(newer)  # not the last tick's, cached


# The wire format of each protocol version (schema_fingerprint). When this
# test fails, the format changed: bump PROTOCOL_VERSION in
# shared/protocol.py and add the new fingerprint here, so clients and
# servers of different code refuse each other rather than half-work.
SCHEMAS = {2: "6b6a0110ae9d06cc"}


def test_the_wire_format_matches_its_version() -> None:
    from ok_doomer.shared.protocol import schema_fingerprint

    assert schema_fingerprint() == SCHEMAS.get(PROTOCOL_VERSION), (
        f"the wire format changed (now {schema_fingerprint()}): "
        f"bump PROTOCOL_VERSION from {PROTOCOL_VERSION} and record it in SCHEMAS"
    )


def test_a_client_of_other_code_hears_why_it_cant_join() -> None:
    from ok_doomer.server.game import GameServer

    answer = GameServer(map_text=OFFICE).join(Hello(PROTOCOL_VERSION - 1, "old"))
    assert isinstance(answer, Reject)
    assert answer.reason == f"version {PROTOCOL_VERSION - 1}; this server speaks {PROTOCOL_VERSION}"
