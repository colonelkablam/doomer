from pathlib import Path

import pygame

from ok_doomer.client.headless import save_png


def test_render_frame_to_png(tmp_path: Path) -> None:
    pygame.init()
    try:
        pygame.display.set_mode((1, 1))  # proves the dummy driver works
        frame = pygame.Surface((320, 200))
        frame.fill((40, 40, 40))
        pygame.draw.rect(frame, (200, 30, 30), (100, 50, 120, 100))

        out = save_png(frame, tmp_path / "frame.png", scale=2)

        loaded = pygame.image.load(str(out))
        assert loaded.get_size() == (640, 400)
        assert loaded.get_at((320, 200))[:3] == (200, 30, 30)
        assert loaded.get_at((5, 5))[:3] == (40, 40, 40)
    finally:
        pygame.quit()
