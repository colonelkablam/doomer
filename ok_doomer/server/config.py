"""Server settings: config as data, loadable from TOML (shared/config.py)."""

from dataclasses import dataclass, field

from ok_doomer.shared.config import SimConfig


@dataclass(frozen=True, slots=True)
class ServerConfig:
    host: str = "0.0.0.0"  # every interface: the LAN can reach it
    port: int = 5029
    # What the game's called in others' lists of games found ("": this
    # machine's name). Answer those looking for games on the LAN (UDP
    # `discovery_port`), or not.
    name: str = ""
    discovery: bool = True
    discovery_port: int = 5028
    storey: str = "G"  # the floor the game starts on (shared/building.py)
    mode: str = "coop"
    max_players: int = 8
    # Inputs waiting per client. One is used per tick, so this caps the lag a
    # client sending too fast can build up (8 at 60 Hz: 133 ms).
    input_queue: int = 8
    # Inputs left waiting after a tick's is taken: more are merged into it
    # (presses kept), so a burst doesn't stand as lag. A little slack
    # absorbs network jitter.
    input_slack: int = 2
    # A client whose input is late: their last input is repeated for this
    # many ticks (smoothing over jitter), then they stand still.
    hold_ticks: int = 6
    # A client not heard from for this long is dropped.
    timeout_seconds: float = 5.0
    # Chat: at most this many messages from one player in this many
    # seconds; more are dropped (a held key, a script, can't flood it).
    chat_burst: int = 4
    chat_window_seconds: float = 5.0
    # A player who leaves (or drops) and joins again with the same name
    # within this long gets their player back, where they were, health and
    # all. (The name is the key: fine on a LAN; over the internet, a token.)
    rejoin_seconds: float = 30.0
    # Joining with the name of someone still in the game who's been silent
    # this long: it's them, back on a new connection before the old one
    # timed out (a Wi-Fi blip). The old one's dropped and they rejoin.
    reconnect_silent_seconds: float = 1.0
    sim: SimConfig = field(default_factory=SimConfig)
