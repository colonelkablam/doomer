"""Damage, death and respawning: coffee soaks a share of each hit, health
takes the rest; at none you die, the killer scores, and you're back after
the respawn delay (deathmatch: as far from the living as the map allows)."""

from ok_doomer.server.config import ServerConfig
from ok_doomer.server.game import GameServer
from ok_doomer.shared.combat import NOBODY, hurt, respawn_due
from ok_doomer.shared.config import CombatConfig, SimConfig
from ok_doomer.shared.events import Died, Hurt, PlayerSpawned
from ok_doomer.shared.mapfile import parse_map
from ok_doomer.shared.mirror import WorldMirror
from ok_doomer.shared.player import PlayerInput
from ok_doomer.shared.protocol import PROTOCOL_VERSION, Hello, Update, Welcome
from ok_doomer.shared.sim import DEAD_EYE_HEIGHT, eye_height, predict_move, step
from ok_doomer.shared.spawn import spawn_player
from ok_doomer.shared.world import World, drain_events, new_world

ROOM = """\
[walls]
##########
#........#
#........#
#........#
##########

[things]
coop_start 2.5 2.5 0
exit 8.5 1.5 0
"""

# Four corners to spawn in.
ARENA = """\
[walls]
############
#..........#
#..........#
#..........#
#..........#
############

[things]
dm_spawn 1.5 1.5 0
dm_spawn 10.5 1.5 0
dm_spawn 1.5 4.5 0
dm_spawn 10.5 4.5 0
"""


def _world(map_text: str = ROOM, mode: str = "coop", **combat: object) -> World:
    config = SimConfig(combat=CombatConfig(**combat))  # type: ignore[arg-type]
    world = new_world(parse_map(map_text), mode, config)
    for pid, name in ((1, "ann"), (2, "bob")):
        spawn_player(world, pid, name)
    drain_events(world)
    return world


def test_coffee_soaks_a_third_the_rest_comes_off_health() -> None:
    world = _world()
    ann = world.players[1]
    ann.coffee = 25
    hurt(world, ann, 30, by=2)
    assert (ann.health, ann.coffee) == (80, 15)
    assert world.events == [Hurt(0, 1, 30, 10, 2, 80)]
    ann.coffee = 3  # less than its share: it soaks what there is
    hurt(world, ann, 30, by=2)
    assert (ann.health, ann.coffee) == (53, 0)


def test_killed_the_killer_scores_and_you_come_back_later() -> None:
    world = _world(respawn_seconds=0.5)
    ann, bob = world.players[1], world.players[2]
    hurt(world, ann, 150, by=2)
    assert ann.dead and ann.health == 0 and bob.frags == 1
    assert world.events[-1] == Died(0, 1, "ann", 2, "bob")
    drain_events(world)
    hurt(world, ann, 10, by=2)  # the dead can't be hurt again
    assert not world.events and bob.frags == 1
    for _ in range(29):
        step(world, {})
        respawn_due(world)
    assert ann.dead
    step(world, {})
    respawn_due(world)
    assert not ann.dead and (ann.health, ann.coffee) == (100, 0)
    assert [type(e) for e in world.events] == [PlayerSpawned]


def test_a_suicide_loses_a_frag_the_world_scores_nobody() -> None:
    world = _world()
    ann, bob = world.players[1], world.players[2]
    hurt(world, ann, 100, by=1)
    hurt(world, bob, 100, by=NOBODY)
    assert (ann.frags, bob.frags) == (-1, 0)
    assert world.events[-1] == Died(0, 2, "bob", NOBODY, "")


def test_the_dead_dont_move_and_see_from_the_floor() -> None:
    world = _world()
    ann = world.players[1]
    hurt(world, ann, 100)
    where = (ann.x, ann.y)
    step(world, {1: PlayerInput(forward=1.0, jump=True)})
    predict_move(world, ann, PlayerInput(forward=1.0))  # a client predicting itself
    assert (ann.x, ann.y) == where
    assert eye_height(ann, world.config.movement) == DEAD_EYE_HEIGHT


def test_deathmatch_respawns_as_far_from_the_living_as_it_can() -> None:
    world = _world(ARENA, "deathmatch", respawn_seconds=0.0)
    ann, bob = world.players[1], world.players[2]
    bob.x, bob.y = 2.0, 2.0  # by the top-left spawn
    hurt(world, ann, 100, by=2)
    step(world, {})
    respawn_due(world)
    assert (ann.x, ann.y) == (10.5, 4.5)  # the far corner


def test_the_server_respawns_and_clients_see_it_all() -> None:
    server = GameServer(ServerConfig(), map_text=ROOM)
    welcomes = [server.join(Hello(PROTOCOL_VERSION, n)) for n in ("ann", "bob")]
    assert all(isinstance(w, Welcome) for w in welcomes)
    mirror = WorldMirror.from_welcome(welcomes[0])  # type: ignore[arg-type]
    hurt(server.world, server.world.players[1], 100, by=2)

    def tick() -> list[object]:
        updates = [m for m in server.tick()[1] if isinstance(m, Update)]
        for u in updates:
            mirror.handle(u)
        return [e for u in updates for e in u.events]

    seen = tick()
    assert any(isinstance(e, Died) for e in seen)
    me = mirror.world.players[1]
    assert me.dead and mirror.world.players[2].frags == 1  # snapshots carry it
    for _ in range(round(3.0 * 60)):
        seen = tick()
    assert not mirror.world.players[1].dead and mirror.world.players[1].health == 100


# -- the HUD ----------------------------------------------------------------


def test_deaths_are_news_from_where_you_stand() -> None:
    from ok_doomer.client.hud import console_line, news_for

    names = {1: "ann", 2: "bob", 3: "cat"}
    bob_kills_ann = Died(5, 1, "ann", 2, "bob")
    assert news_for(bob_kills_ann, 1, names) == "bob killed you"
    assert news_for(bob_kills_ann, 2, names) == "You killed ann"
    assert news_for(bob_kills_ann, 3, names) == "bob killed ann"
    assert news_for(Died(5, 1, "ann", 1, "ann"), 1, names) == "You killed yourself"
    assert news_for(Died(5, 1, "ann", 1, "ann"), 3, names) == "ann killed themselves"
    assert news_for(Died(5, 1, "ann", NOBODY, ""), 1, names) == "You died"
    assert news_for(Died(5, 1, "ann", NOBODY, ""), 3, names) == "ann died"
    assert news_for(Hurt(5, 1, 10, 0, 2, 90), 1, names) is None  # (a red flash, later)
    assert console_line(bob_kills_ann, names) == "[5] bob killed ann"


def test_the_meters_and_the_wait_to_come_back_are_drawn() -> None:
    import numpy as np

    from ok_doomer.client.hud import NAME_COLOUR, Hud
    from ok_doomer.shared.player import PlayerState

    hud = Hud()
    you = PlayerState(1, "ann", 2.5, 2.5, 0.0, health=80, coffee=15)
    pixels = np.zeros((320, 200), dtype=np.uint8)
    hud.draw_status(pixels, you, None)
    lit = np.argwhere(pixels == NAME_COLOUR)
    assert int(np.min(lit[:, 0])) > 200 and int(np.min(lit[:, 1])) > 170  # bottom right only
    hud.handle([Died(5, 1, "ann", 2, "bob")], 0.0, me=1, names={})
    assert hud.killed_by == "bob"
    you.respawn_at = 300
    middle = np.zeros((320, 200), dtype=np.uint8)
    hud.draw_status(middle, you, 2.4)
    assert (middle[:, 80:120] == NAME_COLOUR).any()  # KILLED BY BOB, BACK IN 3
