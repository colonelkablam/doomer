"""Server settings: config as data, loadable from TOML (shared/config.py)."""

from dataclasses import dataclass, field

from ok_doomer.shared.config import SimConfig


@dataclass(frozen=True, slots=True)
class ServerConfig:
    host: str = "0.0.0.0"  # every interface: the LAN can reach it
    port: int = 5029
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
    sim: SimConfig = field(default_factory=SimConfig)
