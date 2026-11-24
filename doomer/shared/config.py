"""Typed config dataclasses and a small loader from TOML.

Config is data: tunables live here (or in a TOML file overriding these
defaults), not as magic numbers in the code that uses them.
"""

import dataclasses
import tomllib
import typing
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


@dataclasses.dataclass(frozen=True, slots=True)
class MovementConfig:
    walk_speed: float = 3.5  # cells per second
    radius: float = 0.25  # half-width of the player's square collision box, in cells


@dataclasses.dataclass(frozen=True, slots=True)
class SimConfig:
    tick_rate: int = 60
    movement: MovementConfig = dataclasses.field(default_factory=MovementConfig)

    @property
    def dt(self) -> float:
        return 1.0 / self.tick_rate


class ConfigError(ValueError):
    pass


def from_dict(cls: type[T], data: dict[str, Any], where: str = "") -> T:
    """Build a (possibly nested) config dataclass from a dict, e.g. parsed TOML.

    Missing keys take the dataclass defaults; unknown keys are an error so that
    typos in a config file don't get silently ignored.
    """
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"unknown config key(s) in [{where or 'root'}]: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        hint = hints[name]
        path = f"{where}.{name}" if where else name
        if dataclasses.is_dataclass(hint) and isinstance(hint, type):
            if not isinstance(value, dict):
                raise ConfigError(f"{path} must be a table")
            kwargs[name] = from_dict(hint, value, path)
        else:
            kwargs[name] = _coerce(hint, value, path)
    return cls(**kwargs)


def load_toml(cls: type[T], path: str | Path) -> T:
    with open(path, "rb") as f:
        return from_dict(cls, tomllib.load(f))


def _coerce(hint: Any, value: Any, path: str) -> Any:
    origin = typing.get_origin(hint)
    if hint is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if origin in (tuple, list):
        if not isinstance(value, list):
            raise ConfigError(f"{path} must be a list")
        args = [a for a in typing.get_args(hint) if a is not Ellipsis]
        item = args[0] if args else Any
        items = [_coerce(item, v, f"{path}[{i}]") for i, v in enumerate(value)]
        return tuple(items) if origin is tuple else items
    if hint is Any or (isinstance(hint, type) and isinstance(value, hint)):
        if hint is int and isinstance(value, bool):
            raise ConfigError(f"{path} must be an integer")
        return value
    raise ConfigError(f"{path} must be {getattr(hint, '__name__', hint)}, got {value!r}")
