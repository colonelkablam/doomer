"""The windowed game loop."""

import logging

import pygame

from doomer.client.config import ClientConfig
from doomer.client.input import KeyboardMouseInput
from doomer.client.render import Renderer, make_frame_surface
from doomer.client.session import GameSession

log = logging.getLogger(__name__)

HUD_COLOUR = (255, 255, 120)


def run(session: GameSession, config: ClientConfig, start_angle: float) -> None:
    rc = config.render
    pygame.init()
    try:
        window = _open_window((rc.width * rc.window_scale, rc.height * rc.window_scale), rc.vsync)
        pygame.display.set_caption("OK Doomer")
        frame = make_frame_surface(rc)
        font = pygame.font.Font(None, 22)
        renderer = Renderer(rc)
        controls = KeyboardMouseInput(config.controls, start_angle)
        controls.set_captured(True)
        clock = pygame.time.Clock()
        log.info("running %r; Esc releases the mouse, Esc again quits", session.game_map.name)

        while True:
            real_dt = clock.tick(rc.max_fps) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    if not controls.captured:
                        return
                    controls.set_captured(False)
                elif event.type == pygame.MOUSEBUTTONDOWN and not controls.captured:
                    controls.set_captured(True)
                elif event.type == pygame.WINDOWFOCUSLOST:
                    controls.set_captured(False)

            player_input = controls.poll(real_dt)
            session.update(real_dt, player_input)  # events unused until sound/HUD exist

            camera = session.camera(controls.angle)
            renderer.render(frame, session.game_map, camera, session.sprites())
            # 8-bit frame -> window format, then nearest-neighbour scale into the window.
            pygame.transform.scale(frame.convert(window), window.get_size(), window)
            if rc.show_fps:
                window.blit(font.render(f"{clock.get_fps():.0f} fps", True, HUD_COLOUR), (8, 8))
            if not controls.captured:
                hint = font.render("Click to play - Esc to quit", True, HUD_COLOUR)
                window.blit(hint, hint.get_rect(center=window.get_rect().center))
            pygame.display.flip()
    finally:
        pygame.quit()


def _open_window(size: tuple[int, int], vsync: bool) -> pygame.Surface:
    if vsync:
        try:
            return pygame.display.set_mode(size, vsync=1)
        except pygame.error as e:
            log.warning("vsync unavailable (%s); continuing without it", e)
    return pygame.display.set_mode(size)
