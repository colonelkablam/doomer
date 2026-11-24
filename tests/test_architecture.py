"""Guard the model/view split: shared code must run on a bare Python install."""
import ast
from pathlib import Path

SHARED = Path(__file__).resolve().parent.parent / "doomer" / "shared"
FORBIDDEN = {"pygame", "numpy", "doomer.client", "doomer.editor"}


def imported_modules(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_shared_has_no_view_or_third_party_imports():
    bad = []
    for path in SHARED.rglob("*.py"):
        for mod in imported_modules(path):
            if any(mod == f or mod.startswith(f + ".") for f in FORBIDDEN):
                bad.append(f"{path.name}: {mod}")
    assert not bad, bad
