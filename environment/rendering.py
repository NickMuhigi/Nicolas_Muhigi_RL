# rendering.py
# Persistent Pygame renderer to visualize all cow cases continuously

import pygame

WIDTH, HEIGHT = 640, 360

# Global variables for persistent window
_window = None
_font = None
_clock = None

def render_case(features, diagnosis=None, ground_truth=None, save_frame=None):
    """
    Draws a single-frame visualization representing the case.
    - features: observation vector
    - diagnosis: 0 or 1 (or None)
    - ground_truth: 0 or 1 (or None)
    - save_frame: path to save the frame image (optional)
    """
    global _window, _font, _clock

    # Initialize Pygame window once
    if _window is None:
        pygame.init()
        _window = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Mastitis Detection Demo')
        _font = pygame.font.SysFont(None, 24)
        _clock = pygame.time.Clock()

    temp, scc, milk_color, appetite, swelling = features
    # evidence score normalized to [0,1]
    evidence_score = ((temp - 36.0) / 5.0 + scc / 1000.0 + milk_color + (1 - appetite) + swelling) / 5.0
    evidence_score = max(0.0, min(1.0, evidence_score))

    # Draw background
    _window.fill((245, 245, 245))
    # Cow body (simple ellipse)
    pygame.draw.ellipse(_window, (200, 200, 200), (120, 80, 400, 180))
    # Head
    pygame.draw.circle(_window, (210, 210, 210), (480, 130), 40)
    # Udder swelling indicator
    if swelling > 0.5:
        pygame.draw.circle(_window, (255, 100, 100), (320, 230), 30)
    # Evidence bar background
    pygame.draw.rect(_window, (220, 220, 220), (50, 10, 540, 18))
    # Evidence bar fill: red if high evidence, green otherwise
    fill_color = (180, 20, 20) if evidence_score > 0.5 else (20, 160, 20)
    pygame.draw.rect(_window, fill_color, (50, 10, int(540 * evidence_score), 18))
    # Text
    _window.blit(_font.render(f'Evidence score: {evidence_score:.2f}', True, (20, 20, 20)), (50, 35))
    diag_text = 'None' if diagnosis is None else ('MASTITIS' if diagnosis == 1 else 'HEALTHY')
    gt_text = 'Unknown' if ground_truth is None else ('MASTITIS' if ground_truth == 1 else 'HEALTHY')
    _window.blit(_font.render(f'Diagnosis: {diag_text}    GroundTruth: {gt_text}', True, (10, 10, 10)), (50, 60))
    _window.blit(_font.render(f'Temp:{temp:.1f}C  SCC:{scc:.0f}  MilkColor:{int(milk_color)}  Appetite:{int(appetite)}  Swelling:{int(swelling)}', True, (20,20,20)), (50, 95))

    pygame.display.flip()

    # Optionally save frame
    if save_frame is not None:
        pygame.image.save(_window, save_frame)

    # Limit FPS
    _clock.tick(10)

def close_renderer():
    """Call this at the end to quit Pygame cleanly."""
    global _window
    if _window is not None:
        pygame.quit()
        _window = None
