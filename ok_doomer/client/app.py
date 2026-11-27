"""The game loop, in a window or full screen.

The game renders at a fixed low resolution (RenderConfig.width x height) and
the picture is scaled up, nearest-neighbour, into whatever the window is:
the starting window (window_scale), a window the player resized, or the whole
screen. `view_rect` decides where it goes.
"""

import logging
import math

import pygame

from ok_doomer.client.automap import RAY_STRIDE, Automap
from ok_doomer.client.config import ClientConfig
from ok_doomer.client.console import Console
from ok_doomer.client.hud import Hud, console_line, draw_name_tags, draw_text, name_tags
from ok_doomer.client.input import KeyboardMouseInput
from ok_doomer.client.pixelart import FONT_5X7
from ok_doomer.client.render import MAX_PITCH, Renderer, make_frame_surface
from ok_doomer.client.session import GameSession

log = logging.getLogger(__name__)


def run(session: GameSession, config: ClientConfig, start_angle: float) -> None:
    rc = config.render
    pygame.init()
    try:
        windowed_size = (rc.width * rc.window_scale, rc.height * rc.window_scale)
        fullscreen = rc.fullscreen
        window = _open_window(windowed_size, rc.vsync, fullscreen)
        pygame.display.set_caption("OK Doomer")
        frame = make_frame_surface(rc)
        renderer = Renderer(rc, config.lighting if config.lighting.enabled else None)
        hud = Hud()
        hud.arrive(session.game_map, 0.0)
        automap = Automap(session.game_map)
        show_map = False
        show_net = False  # F3: the connection's stats
        console = Console()  # backquote: every event, everyone's
        console_font = pygame.font.Font(None, 20)
        controls = KeyboardMouseInput(config.controls, start_angle, MAX_PITCH)
        controls.set_captured(True)
        clock = pygame.time.Clock()
        log.info(
            "running %r; Esc releases the mouse, Esc again quits; F11 full screen; "
            "Tab shows or hides the map (mouse wheel or -/= to zoom)",
            session.game_map.name,
        )

        while True:
            real_dt = clock.tick(rc.max_fps) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and _is_fullscreen_toggle(event):
                    fullscreen = not fullscreen
                    window = _open_window(windowed_size, rc.vsync, fullscreen)
                    controls.set_captured(controls.captured)  # re-grab in the new window
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    if not controls.captured:
                        return
                    controls.set_captured(False)
                elif event.type == pygame.MOUSEBUTTONDOWN and not controls.captured:
                    controls.set_captured(True)
                elif event.type == pygame.WINDOWFOCUSLOST:
                    controls.set_captured(False)
                elif event.type == pygame.MOUSEWHEEL and event.y:
                    automap.change_zoom(1 if event.y > 0 else -1)
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
                    show_map = not show_map
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_F3:
                    show_net = not show_net
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_BACKQUOTE:
                    console.visible = not console.visible
                elif event.type == pygame.KEYDOWN and event.key in _ZOOM_KEYS:
                    automap.change_zoom(_ZOOM_KEYS[event.key])

            player_input = controls.poll(real_dt)
            now = pygame.time.get_ticks() / 1000.0
            floor = session.game_map
            events = session.update(real_dt, player_input)
            names = session.names()
            hud.handle(events, now, session.me(), names)
            for happened in events:
                console.add(console_line(happened, names))
            if (reason := session.ended()) is not None:
                log.info("game over: %s", reason)
                return
            if session.game_map is not floor:  # the lift's arrived on the next floor
                automap = Automap(session.game_map)
                controls.angle = session.spawn_angle()
                hud.arrive(session.game_map, now)
                log.info("arrived at %r", session.game_map.name)

            camera = session.camera(controls.angle, controls.pitch)
            renderer.render(
                frame,
                session.game_map,
                camera,
                session.sprites(),
                session.boxes(),
                session.power(),
                now,
            )
            # The automap: what's in view is now seen; Tab shows or hides it.
            automap.observe(camera.x, camera.y, *renderer.eye_level_rays(RAY_STRIDE))
            pixels = pygame.surfarray.pixels2d(frame)
            tags = name_tags(renderer, session.people())
            draw_name_tags(pixels, tags, rc.name_scale, rc.name_panel)
            if show_map:
                automap.draw(pixels, camera.x, camera.y, camera.angle)
            hud.draw_frame(pixels, now, rc.message_scale)  # over the map too
            corner = [f"{clock.get_fps():.0f} fps"] if rc.show_fps else []
            if show_net:
                corner += session.net_stats().lines()
            for i, text in enumerate(corner):  # top left, a line under another
                draw_text(pixels, text, FONT_5X7, 3, 9 + i * 10, centre=False)
            if not controls.captured:
                mid = rc.height // 2
                draw_text(pixels, "Click to play - Esc to quit", FONT_5X7, rc.width / 2, mid + 3)
            del pixels  # unlock the surface
            # 8-bit frame -> window format, then nearest-neighbour scale into
            # the picture's place in the window (it may have been resized).
            window = pygame.display.get_surface() or window
            rect = view_rect(window.get_size(), (rc.width, rc.height), rc.integer_scaling)
            if rect.size != window.get_size():
                window.fill((0, 0, 0))  # the borders
            view = window.subsurface(rect)
            pygame.transform.scale(frame.convert(window), rect.size, view)
            hud.draw(view, session.passes())
            console.draw(view, console_font)
            pygame.display.flip()
    finally:
        pygame.quit()


# Zoom the automap in (+1) or out (-1).
_ZOOM_KEYS = {
    pygame.K_EQUALS: 1,
    pygame.K_PLUS: 1,
    pygame.K_KP_PLUS: 1,
    pygame.K_MINUS: -1,
    pygame.K_KP_MINUS: -1,
}


def view_rect(window: tuple[int, int], frame: tuple[int, int], integer: bool = True) -> pygame.Rect:
    """Where the scaled-up picture goes in a window: as large as fits keeping
    its shape, centred. `integer`: by a whole-number factor (at least 1), so
    every game pixel is the same size; else exactly as large as fits."""
    (ww, wh), (fw, fh) = window, frame
    scale = min(ww / fw, wh / fh)
    if integer:
        scale = max(1, math.floor(scale + 1e-9))
    w, h = min(ww, round(fw * scale)), min(wh, round(fh * scale))
    return pygame.Rect((ww - w) // 2, (wh - h) // 2, w, h)


def _is_fullscreen_toggle(event: pygame.event.Event) -> bool:
    """F11, or Alt+Enter."""
    alt = bool(event.mod & pygame.KMOD_ALT)
    return bool(event.key == pygame.K_F11 or (alt and event.key == pygame.K_RETURN))


def _open_window(size: tuple[int, int], vsync: bool, fullscreen: bool = False) -> pygame.Surface:
    """A resizable window of `size`, or the whole screen at its own resolution."""
    flags, mode = (pygame.FULLSCREEN, (0, 0)) if fullscreen else (pygame.RESIZABLE, size)
    if vsync:
        try:
            return pygame.display.set_mode(mode, flags, vsync=1)
        except pygame.error as e:
            log.warning("vsync unavailable (%s); continuing without it", e)
    return pygame.display.set_mode(mode, flags)
