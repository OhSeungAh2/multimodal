import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

# STEP_REWARD: 이동을 짧게가던 길게가던 큰 의미가 없어지는데
# ::강화학습할 때 목쵸를 정하고 움직일 때 마다 리워드를 주는 방법으로 처리하면 쉽게 처리 됨
# ::걸음마다 -1이 있다고 가정하면 짧은길이 리워드가 크니까 이런 보상상수를 통해 원하는 길로 갈 수 있게 해줌
# WALL_REWARD: 이건 이동하는것보다 더 나쁜 리워드임
# GOAL_REWARD: 목표에 도착했을 때 주는 리워드임(완벽하게 많이 커야 출발점으로부터 목적지로 가야 좋다는게 적용이됨 -> 리워드 점수를 크게 줬다는 뜻)
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
    NUM_ACTIONS
)

CELL = 56
MARGIN = 16
PANEL_W = 390
MIN_HEIGHT = 640
MOVE_DELAY_MS = 450     # 재생할 때 한 걸음 사이의 시간
STAGE_HOLD_MS = 1500    # 한 판 끝난 뒤, 다음 학습 시점으로 넘어가기 전 결과 읽는 시간

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
COLOR_TRAIL = (70, 110, 200)
COLOR_ARROW_NOW = (16, 120, 72)
COLOR_ARROW = (52, 60, 96)


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

def draw_arrow(screen, row, col, action, color):
    x = MARGIN + col * CELL + CELL - 14
    y = MARGIN + row * CELL + 14
    if action == UP:
        points = [(x - 5, y + 4), (x + 5, y + 4), (x, y - 6)]
    elif action == DOWN:
        points = [(x - 5, y - 4), (x + 5, y - 4), (x, y + 6)]
    elif action == LEFT:
        points = [(x + 4, y - 5), (x + 4, y + 5), (x - 6, y)]
    else:
        points = [(x - 4, y - 5), (x - 4, y + 5), (x + 6, y)]
    pygame.draw.polygon(screen, color, points)

def draw_maze(screen, env, trail, arrows, agent_color, cell_font):
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

    if trail:
        for pos in trail:
            if pos == env.pos:
                continue
            center = (
                MARGIN + pos[1] * CELL + CELL // 2,
                MARGIN + pos[0] * CELL + CELL // 2,
            )
            pygame.draw.circle(screen, COLOR_TRAIL, center, 4)

    if arrows:
        for (row, col), action in arrows.items():
            if (row, col) == env.goal:
                continue
            color = COLOR_ARROW_NOW if (row, col) == env.pos else COLOR_ARROW
            draw_arrow(screen, row, col, action, color)

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

        # 행동 번호
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
            draw_maze(screen, env, None, None, agent_color, cell_font)

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

def draw_chart(screen, successes, rect, font):
    if len(successes) < 2 or rect.h < 70 or rect.w < 40:
        return

    window = 50
    averages = []
    total = 0.0
    for index, value in enumerate(successes):
        total += value
        if index >= window:
            total -= successes[index - window]
            averages.append(total / window)
        else:
            averages.append(total / (index + 1))

    pygame.draw.rect(screen, (40, 42, 60), rect, border_radius=6)
    title = font.render("성공률 (최근 50판 평균)", True, COLOR_MUTED)
    screen.blit(title, (rect.x + 8, rect.y + 6))

    plot = pygame.Rect(rect.x + 12, rect.y + 30, rect.w - 24, rect.h - 42)
    pygame.draw.line(screen, (90, 94, 120), (plot.left, plot.bottom), (plot.right, plot.bottom), 1)
    pygame.draw.line(screen, (90, 94, 120), (plot.left, plot.top), (plot.left, plot.bottom), 1)

    last_index = len(averages) - 1
    points = []
    for index, average in enumerate(averages):
        x = plot.left + (plot.w - 1) * index / last_index
        # 성공률 0은 아래, 1은 위에 둔다.
        y = plot.bottom - (plot.h - 1) * average
        points.append((int(x), int(y)))
    if len(points) >= 2:
        pygame.draw.lines(screen, COLOR_GOOD, False, points, 2)
        

def watch_agent(agent, successes, summary, snapshots):
    pygame.init()
    try:
        env = MazeEnv()
        stage_index = 0
        # 도착 칸 설명과 성공률 그래프가 함께 들어가도록 사람 플레이보다 조금 높게 연다.
        screen = pygame.display.set_mode(window_size(env))
        pygame.display.set_caption("Q-Learning 미로 에이전트")
        clock = pygame.time.Clock()
        body_font = make_font(17)
        small_font = make_font(15)
        cell_font = make_font(16, bold=True)

        def begin():
            agent.q = snapshots[stage_index]["q"]
            env.reset()
            shot = snapshots[stage_index]
            if shot["reached"]:
                message = "이 정책으로 목표까지 갑니다."
            else:
                message = "아직 목표까지 못 갑니다."
            return {
                "trail": [env.pos],
                "seen": {env.pos},
                "steps": 0,
                "total_reward": 0,
                "last_reward": None,
                "finished": False,
                "paused": False,
                "elapsed": 0,
                "hold": 0,
                "message": message,
            }

        play_state = begin()
        running = True

        while running:
            dt = clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        play_state = begin()
                    elif event.key == pygame.K_SPACE:
                        play_state["paused"] = not play_state["paused"]

            # 정책 지도는 지금 보고 있는 시점의 표에서 읽는다.
            arrows = {}
            for cell in env.walkable_cells():
                arrows[cell] = agent.best_action(cell, random_tie=False)

            if not play_state["finished"] and not play_state["paused"]:
                play_state["elapsed"] += dt
                if play_state["elapsed"] >= MOVE_DELAY_MS:
                    play_state["elapsed"] = 0
                    before = env.pos
                    action = agent.best_action(before, random_tie=False)
                    _state, reward, done = env.step(action)
                    play_state["trail"].append(env.pos)
                    play_state["steps"] += 1
                    play_state["total_reward"] += reward
                    play_state["last_reward"] = reward

                    if done:
                        play_state["finished"] = True
                        play_state["message"] = "목표에 도착했습니다."
                    elif env.pos == before:
                        play_state["finished"] = True
                        play_state["message"] = "벽에 막혔습니다."
                    elif env.pos in play_state["seen"]:
                        play_state["finished"] = True
                        play_state["message"] = "같은 길로 돌아왔습니다."
                    elif play_state["steps"] >= 200:
                        play_state["finished"] = True
                        play_state["message"] = "걸음 제한에 도달했습니다."
                    else:
                        play_state["seen"].add(env.pos)
            elif play_state["finished"] and not play_state["paused"]:
                # 도착하거나 막힌 장면을 잠깐 유지한 뒤 다음 학습 시점으로 넘어간다.
                play_state["hold"] += dt
                if play_state["hold"] >= STAGE_HOLD_MS:
                    stage_index = (stage_index + 1) % len(snapshots)
                    play_state = begin()

            if env.pos == env.goal:
                agent_color = COLOR_GOOD
            else:
                agent_color = COLOR_AGENT

            screen.fill(COLOR_BG)
            panel = panel_rect(screen, env)
            pygame.draw.rect(screen, COLOR_PANEL, panel, border_radius=10)
            draw_maze(
                screen,
                env,
                play_state["trail"],
                arrows,
                agent_color,
                cell_font,
            )

            at_goal = env.pos == env.goal
            planned = None if at_goal else agent.best_action(env.pos, random_tie=False)
            q_lines = []
            for action in range(NUM_ACTIONS):
                value = agent.q_get(env.pos, action)
                chosen = action == planned
                color = COLOR_GOOD if chosen else COLOR_TEXT
                mark = "  선택" if chosen else ""
                q_lines.append((f"{ACTION_NAMES[action]}  {value:8.2f}{mark}", color))

            shot = snapshots[stage_index]
            if shot["reached"]:
                stage_color = COLOR_GOOD
                stage_result = f"이 시점 예상  {shot['steps']}걸음"
            else:
                stage_color = COLOR_BAD
                stage_result = "이 시점에는 도착 못 함"

            if play_state["finished"]:
                pause_text = "잠시 후 다음 판"
            elif play_state["paused"]:
                pause_text = "일시정지 중"
            else:
                pause_text = "재생 중"
            lines = [
                "학습 과정 재생",
                "",
                f"시점  {stage_index + 1}/{len(snapshots)}",
                f"에피소드 {shot['episode']} 까지 학습",
                (stage_result, stage_color),
                f"알파 {summary['alpha']}   감마 {summary['gamma']}",
                "",
                f"상태  {env.pos}",
                f"걸음  {play_state['steps']}",
                f"보상 합  {play_state['total_reward']}",
                "",
                ("지금 칸의 Q값", COLOR_MUTED),
            ]
            lines.extend(q_lines)
            if at_goal:
                lines.append("")
                lines.append("도착 칸에서는")
                lines.append("더 이상 학습하지 않습니다.")
            lines.extend([
                "",
                play_state["message"],
                (pause_text, COLOR_MUTED),
                "",
                ("끝나면 다음 시점으로 반복", COLOR_MUTED),
                ("스페이스  일시정지", COLOR_MUTED),
                ("R  이 판 다시", COLOR_MUTED),
                ("ESC  종료", COLOR_MUTED),
                "",
                "칸의 삼각형 = 그 칸의 정책",
            ])
            blit_lines(screen, body_font, lines, panel.x + 16, panel.y + 14)

            chart_h = 110
            chart = pygame.Rect(panel.x + 12, panel.bottom - chart_h - 8, panel.w - 24, chart_h)
            draw_chart(screen, successes, chart, small_font)
            pygame.display.flip()
    finally:
        pygame.quit()


if __name__ == "__main__":
    play()
