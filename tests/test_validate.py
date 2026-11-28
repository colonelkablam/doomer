"""The map checker (ok_doomer/shared/validate.py): can a co-op floor be finished?"""

from pathlib import Path

import pytest

from ok_doomer.shared.mapfile import load_map, parse_map
from ok_doomer.shared.validate import check_floor

# Two rooms, north (y 1) and south (y 3), joined by a doorway at (3, 2).
TWO_ROOMS = """\
[walls]
#######
#.....#
###.###
#.....#
#######

[things]
coop_start 1.5 1.5 0
door_hinged 3.5 2.5 90 lock=red
"""


def _check(things: str, walls: str = TWO_ROOMS) -> list[str]:
    return list(check_floor(parse_map(walls + things)).problems)


@pytest.mark.parametrize("path", sorted(Path("maps").glob("*.map")), ids=lambda p: p.name)
def test_every_map_can_be_finished(path: Path) -> None:
    report = check_floor(load_map(path))
    assert report.ok, report.problems


def test_the_office_needs_both_passes_and_the_power() -> None:
    report = check_floor(load_map("maps/office.map"))
    assert report.passes == {"blue", "red"} and report.power


def test_a_pass_on_the_right_side_of_its_door_is_fine() -> None:
    assert _check("pass_red 4.5 1.5 0\nexit 5.5 3.5 0\n") == []


def test_a_pass_behind_the_door_it_opens() -> None:
    problems = _check("pass_red 1.5 3.5 0\nexit 5.5 3.5 0\n")
    assert "the red pass is only reachable through the red-locked door at (3.5, 2.5)" in problems
    assert "no way out: no lift car or exit can be reached" in problems


def test_a_lock_with_no_pass_for_it() -> None:
    problems = _check("exit 5.5 3.5 0\n")
    assert problems[0] == "the red-locked door at (3.5, 2.5) can't open: no red pass"


def test_an_unreachable_breaker_means_no_power_for_the_lifts() -> None:
    # A lift in the north wall (its car the cell (2, 0)); the breaker in a
    # sealed room to the south.
    sealed = "[walls]\n##.####\n#.....#\n#######\n#.....#\n#######\n\n[things]\n"
    sealed += "coop_start 1.5 1.5 0\ndoor_lift 2.5 0.5 90\n"
    problems = _check("breaker 3.5 3.99 270\n", sealed)
    assert "the breaker at (3.5, 3.99) can't be reached" in problems
    assert "no way out: the lifts need power, and it can't be restored" in problems
    # With the breaker in the lift's room, the floor's fine.
    assert _check("breaker 4.5 1.01 90\n", sealed) == []


def test_furniture_can_close_a_gap() -> None:
    # A filing cabinet in the doorway: the south room (and its exit) is cut off.
    walls = TWO_ROOMS.replace("door_hinged 3.5 2.5 90 lock=red\n", "")
    assert _check("exit 5.5 3.5 0\n", walls) == []
    blocked = _check("exit 5.5 3.5 0\nfiling_cabinet 3.5 2.5 0\n", walls)
    assert blocked == ["no way out: no lift car or exit can be reached"]


def test_a_desk_can_be_jumped_onto_but_not_a_counter() -> None:
    # A row of low blocks across the room, the exit beyond: a desk (2/8 high)
    # you can jump onto and over, a counter (3/8) you can't.
    room = "[walls]\n#####\n#...#\n#ddd#\n#...#\n#####\n\n"
    room += "[floor]\n00000\n00000\n0{h}{h}{h}0\n00000\n00000\n\n[things]\ncoop_start 2.5 1.5 0\n"
    assert _check("exit 2.5 3.5 0\n", room.format(h=2)) == []
    assert _check("exit 2.5 3.5 0\n", room.format(h=3)) == [
        "no way out: no lift car or exit can be reached"
    ]


def test_the_command_line_says_what_is_wrong(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from ok_doomer.shared.validate import main

    broken = tmp_path / "broken.map"
    broken.write_text(TWO_ROOMS + "pass_red 1.5 3.5 0\nexit 5.5 3.5 0\n", encoding="utf-8")
    assert main(["maps/office.map"]) == 0
    assert main([str(broken)]) == 1
    out = capsys.readouterr().out
    assert "office.map: OK (passes: blue, red)" in out
    assert "only reachable through the red-locked door" in out
