"""Guard the dependency rules in docs/DESIGN.md ("Code design rules", rule 1).

shared and server must run on a bare Python install (the Pi), and nothing may
depend on the client or editor.
"""

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parent.parent / "doomer"
VIEW = {"pygame", "numpy", "doomer.client", "doomer.editor"}

RULES: dict[str, set[str]] = {
    "shared": VIEW | {"doomer.server"},
    "server": VIEW,
}


def imported_modules(path: Path) -> Iterator[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


@pytest.mark.parametrize("package", sorted(RULES))
def test_forbidden_imports(package: str) -> None:
    forbidden = RULES[package]
    bad = [
        f"{path.relative_to(PKG)}: {mod}"
        for path in (PKG / package).rglob("*.py")
        for mod in imported_modules(path)
        if any(mod == f or mod.startswith(f + ".") for f in forbidden)
    ]
    assert not bad, bad
