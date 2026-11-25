"""The game loop, in a window or full screen.

The game renders at a fixed low resolution (RenderConfig.width x height) and
the picture is scaled up, nearest-neighbour, into whatever the window is:
the starting window (window_scale), a window the player resized, or the whole
screen. `view_rect` decides where it goes.
"""

import logging
import math

import pygame

from doomer.client.config import ClientConfig
from doomer.client.hud import Hud
from doomer.client.input import KeyboardMouseInput
from doomer.client.render import MAX_PITCH, Renderer, make_frame_surface
from doomer.client.session import GameSession

log = logging.getLogger(__name__)

HUD_COLOUR = (255, 255, 120)


def run(session: GameSession, config: ClientConfig, start_angle: float) -> None:
    rc = config.render
    pygame.init()
    try:
        windowed_size = (rc.width * rc.window_scale, rc.height * rc.window_scale)
        fullscreen = rc.fullscreen
        window = _open_window(windowed_size, rc.vsync, fullscreen)
        pygame.display.set_caption("OK Doomer")
        frame = make_frame_surface(rc)
        font = pygame.font.Font(None, 22)
        renderer = Renderer(rc, config.lighting if config.lighting.enabled else None)
        hud = Hud(font)
        controls = KeyboardMouseInput(config.controls, start_angle, MAX_PITCH)
        controls.set_captured(True)
        clock = pygame.time.Clock()
        log.info(
            "running %r; Esc releases the mouse, Esc again quits; F11 full screen",
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

            player_input = controls.poll(real_dt)
            now = pygame.time.get_ticks() / 1000.0
            hud.handle(session.update(real_dt, player_input), now)

            camera = session.camera(controls.angle, controls.pitch)
            renderer.render(
                frame, session.game_map, camera, session.sprites(), session.boxes(), session.power()
            )
            # 8-bit frame -> window format, then nearest-neighbour scale into
            # the picture's place in the window (it may have been resized).
            window = pygame.display.get_surface() or window
            rect = view_rect(window.get_size(), (rc.width, rc.height), rc.integer_scaling)
            if rect.size != window.get_size():
                window.fill((0, 0, 0))  # the borders
            view = window.subsurface(rect)
            pygame.transform.scale(frame.convert(window), rect.size, view)
            hud.draw(view, session.passes(), now)
            if rc.show_fps:
                view.blit(font.render(f"{clock.get_fps():.0f} fps", True, HUD_COLOUR), (8, 8))
            if not controls.captured:
                hint = font.render("Click to play - Esc to quit", True, HUD_COLOUR)
                view.blit(hint, hint.get_rect(center=view.get_rect().center))
            pygame.display.flip()
    finally:
        pygame.quit()


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
