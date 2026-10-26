import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from maze_env import (
    ACTION_NAMES,
    DOWN,
    GOAL_REWARD,
    LEFT,
    MazeEnv,
    RIGHT,
    STEP_REWARD,
    UP,
    WALL_REWARD,
)

CELL = 56
MARGIN = 16
PANEL_W = 390
MIN_HEIGHT = 640

COLOR_BG = (28, 30, 42)
COLOR_PANEL = (24, 24, 38)
COLOR_WALL = (42, 46, 66)
COLOR_FLOOR = (232, 228, 214)
COLOR_START = (72, 168, 112)
COLOR_GOAL = (240, 180, 40)
COLOR_AGENT = (45, 105, 225)
COLOR_TEXT = (244, 244, 248)
COLOR_MUTED = (168, 174, 196)
COLOR_GOOD = (120, 220, 160)
COLOR_BAD = (255, 128, 128)


def window_size(env):
    width = MARGIN + env.cols * CELL + PANEL_W
    height = max(MARGIN + env.rows * CELL + MARGIN, MIN_HEIGHT)
    return width, height


def make_font(size, bold=False):
    for name in ("malgungothic", "malgun gothic", "gulim", "dotum", "arial"):
        if pygame.font.match_font(name):
            return pygame.font.SysFont(name, size, bold=bold)
    return pygame.font.Font(None, size)


def panel_rect(screen, env):
    x = MARGIN + env.cols * CELL + 8
    return pygame.Rect(x, 12, screen.get_width() - x - 12, screen.get_height() - 24)


def draw_maze(screen, env, agent_color, cell_font):
    for row in range(env.rows):
        for col in range(env.cols):
            x = MARGIN + col * CELL
            y = MARGIN + row * CELL
            rect = pygame.Rect(x + 1, y + 1, CELL - 2, CELL - 2)

            if env.grid[row][col] == 1:
                color = COLOR_WALL
            elif (row, col) == env.goal:
                color = COLOR_GOAL
            elif (row, col) == env.start:
                color = COLOR_START
            else:
                color = COLOR_FLOOR
            pygame.draw.rect(screen, color, rect, border_radius=4)

            if (row, col) == env.start:
                label = cell_font.render("S", True, (16, 48, 32))
                screen.blit(label, (x + 6, y + 2))
            elif (row, col) == env.goal:
                label = cell_font.render("G", True, (72, 48, 8))
                screen.blit(label, (x + 6, y + 2))

    center = (
        MARGIN + env.pos[1] * CELL + CELL // 2,
        MARGIN + env.pos[0] * CELL + CELL // 2,
    )
    radius = CELL // 4
    pygame.draw.circle(screen, agent_color, center, radius)
    pygame.draw.circle(screen, (255, 255, 255), center, radius, 2)


def blit_lines(screen, font, lines, x, y):
    for item in lines:
        if isinstance(item, tuple):
            text, color = item
        else:
            text, color = item, COLOR_TEXT
        if text == "":
            y += font.get_linesize() // 2
            continue
        image = font.render(text, True, color)
        screen.blit(image, (x, y))
        y += font.get_linesize()
    return y


def describe_step(action, reward, done, bumped):
    name = ACTION_NAMES[action]
    if done:
        return f"{name}으로 이동, 보상 {reward}. 목표 도착."
    if bumped:
        return f"{name}은 벽입니다. 보상 {reward}."
    return f"{name}으로 한 칸 이동. 보상 {reward}."


def play():
    pygame.init()
    try:
        env = MazeEnv()
        env.reset()
        screen = pygame.display.set_mode(window_size(env))
        pygame.display.set_caption("미로 게임")
        clock = pygame.time.Clock()
        body_font = make_font(17)
        cell_font = make_font(16, bold=True)

        key_to_action = {
            pygame.K_UP: UP,
            pygame.K_DOWN: DOWN,
            pygame.K_LEFT: LEFT,
            pygame.K_RIGHT: RIGHT,
        }

        steps = 0
        total_reward = 0
        last_reward = None
        done = False
        message = "방향키로 한 칸씩 이동합니다."
        running = True

        while running:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        env.reset()
                        steps = 0
                        total_reward = 0
                        last_reward = None
                        done = False
                        message = "시작 칸으로 돌아왔습니다."
                    elif event.key in key_to_action and not done:
                        action = key_to_action[event.key]
                        before = env.pos
                        _state, reward, done = env.step(action)
                        steps += 1
                        total_reward += reward
                        last_reward = reward
                        bumped = env.pos == before
                        message = describe_step(action, reward, done, bumped)

            if done:
                agent_color = COLOR_GOOD
            elif last_reward == WALL_REWARD:
                agent_color = COLOR_BAD
            else:
                agent_color = COLOR_AGENT

            screen.fill(COLOR_BG)
            panel = panel_rect(screen, env)
            pygame.draw.rect(screen, COLOR_PANEL, panel, border_radius=10)
            draw_maze(screen, env, agent_color, cell_font)

            lines = [
                ("미로 게임", COLOR_TEXT),
                "",
                ("방향키  이동", COLOR_MUTED),
                ("R  다시 시작", COLOR_MUTED),
                ("ESC  종료", COLOR_MUTED),
                "",
                "한 판이 에피소드입니다.",
                "목표에 들어가면 끝납니다.",
                "",
                f"걸음  {steps}",
                f"이번 보상  {last_reward if last_reward is not None else '-'}",
                f"보상 합  {total_reward}",
                "",
                ("보상 규칙", COLOR_MUTED),
                (f"길 한 칸  {STEP_REWARD}", COLOR_TEXT),
                (f"벽  {WALL_REWARD}", COLOR_BAD),
                (f"목표  {GOAL_REWARD}", COLOR_GOOD),
                "",
                "밝은 칸  길",
                "어두운 칸  벽",
                "초록 S  시작",
                "노란 G  목표",
                "",
                message,
            ]
            blit_lines(screen, body_font, lines, panel.x + 16, panel.y + 16)
            pygame.display.flip()
    finally:
        pygame.quit()


if __name__ == "__main__":
    play()
