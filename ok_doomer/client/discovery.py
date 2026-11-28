"""Finding games on the LAN: ask, and list who answers.

A GameFinder broadcasts a Seek (UDP, DISCOVERY_PORT) and collects the
GameInfo each server sends back (server/discovery.py), with a ping (the
time from asking to the answer), forgetting a game that stops answering.
Ask again every so often (`seek`) and read the answers each frame
(`poll`): it never waits, so a menu can show the list as it fills.

Where the Seek goes:
- 255.255.255.255, everyone on the network;
- each local network's own broadcast address (192.168.1.255 for a
  machine at 192.168.1.20): on Windows the first can leave by a virtual
  adapter (WSL, Hyper-V, a VPN) and reach nobody. The netmask isn't
  something Python's standard library can ask for, so /24 is assumed: a
  home network's usual; 255.255.255.255 covers the rest;
- this machine (127.0.0.1), for a game hosted on it.
One game heard two ways is listed once (its server_id), by its LAN
address. A game from other code (another protocol version) is still
listed, as needing updating: its version is read before the rest.
Wi-Fi with "AP isolation" (guest networks) passes no broadcasts at all:
typing an address is the fallback.
"""

import contextlib
import json
import socket
import time
from collections.abc import Iterable
from dataclasses import dataclass

from ok_doomer.shared.protocol import (
    DISCOVERY_PORT,
    MAX_DATAGRAM,
    PROTOCOL_VERSION,
    GameInfo,
    ProtocolError,
    Seek,
    decode,
    encode,
)

FORGET_SECONDS = 3.0  # a game silent this long is dropped from the list
LOOPBACK = "127.0.0.1"


@dataclass(slots=True)
class FoundGame:
    address: str  # where it answered from: join at (address, info.port)
    version: int
    name: str
    info: GameInfo | None  # None: another version, its details unread
    ping_ms: float
    last_seen: float

    @property
    def joinable(self) -> bool:
        return self.version == PROTOCOL_VERSION and self.info is not None

    def describe(self) -> str:
        """One line for a list of games."""
        if self.info is None or not self.joinable:
            return (
                f"{self.name}  (needs updating: version {self.version}, this is {PROTOCOL_VERSION})"
            )
        i = self.info
        return (
            f"{i.name}  {i.floor}  {i.players}/{i.max_players} players  "
            f"{self.ping_ms:.0f} ms  {self.address}:{i.port}"
        )


def local_addresses() -> list[str]:
    """This machine's IPv4 addresses on its networks (not loopback)."""
    found: set[str] = set()
    with contextlib.suppress(OSError):
        found.update(socket.gethostbyname_ex(socket.gethostname())[2])
    with contextlib.suppress(OSError), socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("10.255.255.255", 1))  # no packet is sent: it just picks the way out
        found.add(s.getsockname()[0])
    return sorted(a for a in found if not a.startswith("127."))


def broadcast_targets() -> list[str]:
    """Where a Seek goes: everyone, each local /24's broadcast address,
    and this machine."""
    nets = [".".join(a.split(".")[:3]) + ".255" for a in local_addresses()]
    return ["255.255.255.255", *dict.fromkeys(nets), LOOPBACK]


class GameFinder:
    def __init__(self, port: int = DISCOVERY_PORT, targets: Iterable[str] | None = None) -> None:
        self.port = port
        self.targets = list(targets) if targets is not None else broadcast_targets()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.setblocking(False)
        self.sock.bind(("", 0))
        self.found: dict[str, FoundGame] = {}
        self._asked_at = 0.0

    def seek(self, now: float | None = None) -> None:
        """Ask everyone (again)."""
        self._asked_at = time.monotonic() if now is None else now
        message = encode(Seek(PROTOCOL_VERSION))
        for host in self.targets:
            with contextlib.suppress(OSError):  # a network that isn't there
                self.sock.sendto(message, (host, self.port))

    def poll(self, now: float | None = None) -> list[FoundGame]:
        """Take the answers that have come; the games known, by name."""
        now = time.monotonic() if now is None else now
        while True:
            try:
                data, (host, _) = self.sock.recvfrom(MAX_DATAGRAM + 1)
            except (BlockingIOError, InterruptedError):
                break
            except OSError:  # e.g. Windows' "connection reset" after an unanswered send
                continue
            if len(data) <= MAX_DATAGRAM:
                self._heard(data, host, now)
        self.found = {k: g for k, g in self.found.items() if now - g.last_seen < FORGET_SECONDS}
        return sorted(self.found.values(), key=lambda g: (not g.joinable, g.name.lower()))

    def _heard(self, data: bytes, host: str, now: float) -> None:
        ping = (now - self._asked_at) * 1000
        try:
            message = decode(data)
        except ProtocolError:
            message = None
        if isinstance(message, GameInfo):
            key, version, name, info = message.server_id, message.version, message.name, message
        else:  # another version's answer? its version and name, if it has them
            try:
                raw = json.loads(data)
            except ValueError:
                return
            if not isinstance(raw, dict) or not isinstance(raw.get("version"), int):
                return
            name = str(raw.get("name", host))
            key, version, info = f"{host}:{raw.get('port')}", raw["version"], None
        known = self.found.get(key)
        if known is not None and host == LOOPBACK and known.address != LOOPBACK:
            known.last_seen = now  # this machine's own game: keep its LAN address
            return
        self.found[key] = FoundGame(host, version, name, info, ping, now)

    def close(self) -> None:
        self.sock.close()


def find(seconds: float = 1.5, port: int = DISCOVERY_PORT) -> list[FoundGame]:
    """Look for games for a while; those that answered."""
    finder = GameFinder(port)
    try:
        end = time.monotonic() + seconds
        next_ask = 0.0
        games: list[FoundGame] = []
        while (now := time.monotonic()) < end:
            if now >= next_ask:
                finder.seek(now)
                next_ask = now + 0.5
            games = finder.poll(now)
            time.sleep(0.02)
        return games
    finally:
        finder.close()
