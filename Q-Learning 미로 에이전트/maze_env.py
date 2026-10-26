UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

NUM_ACTIONS = 4
ACTION_NAMES = ("위", "아래", "왼쪽", "오른쪽")

ACTION_DELTAS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)

STEP_REWARD = -1
WALL_REWARD = -5
GOAL_REWARD = 100

MAZE_TEXT = (
    "##########",
    "#S..#....#",
    "#.#.#.##.#",
    "#.#....#.#",
    "#.####.#.#",
    "#....#...#",
    "####.###.#",
    "#......#.#",
    "#.####..G#",
    "##########",
)


def build_maze(lines):
    if not lines:
        raise ValueError("미로가 비어 있습니다.")

    width = len(lines[0])
    grid = []
    start = None
    goal = None
    start_count = 0
    goal_count = 0

    for row, line in enumerate(lines):
        if len(line) != width:
            raise ValueError("미로의 모든 줄은 같은 길이여야 합니다.")

        cells = []
        for col, symbol in enumerate(line):
            if symbol == "#":
                cells.append(1)
            elif symbol == ".":
                cells.append(0)
            elif symbol == "S":
                cells.append(0)
                start = (row, col)
                start_count += 1
            elif symbol == "G":
                cells.append(0)
                goal = (row, col)
                goal_count += 1
            else:
                raise ValueError(f"사용할 수 없는 칸 문자입니다: {symbol}")
        grid.append(cells)

    if start_count != 1 or goal_count != 1:
        raise ValueError("S와 G는 각각 하나씩 있어야 합니다.")

    return grid, start, goal


class MazeEnv:
    def __init__(self):
        self.grid, self.start, self.goal = build_maze(MAZE_TEXT)
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.pos = self.start

    def reset(self):
        self.pos = self.start
        return self.pos

    def is_walkable(self, row, col):
        if row < 0 or row >= self.rows:
            return False
        if col < 0 or col >= self.cols:
            return False
        return self.grid[row][col] == 0

    def walkable_cells(self):
        cells = []
        for row in range(self.rows):
            for col in range(self.cols):
                if self.grid[row][col] == 0:
                    cells.append((row, col))
        return cells

    def step(self, action):
        if action not in (UP, DOWN, LEFT, RIGHT):
            raise ValueError(f"행동은 0, 1, 2, 3 중 하나여야 합니다. 받은 값: {action}")

        row, col = self.pos
        d_row, d_col = ACTION_DELTAS[action]
        next_row = row + d_row
        next_col = col + d_col

        if not self.is_walkable(next_row, next_col):
            return self.pos, WALL_REWARD, False

        self.pos = (next_row, next_col)
        if self.pos == self.goal:
            return self.pos, GOAL_REWARD, True
        return self.pos, STEP_REWARD, False


def demo():
    env = MazeEnv()
    state = env.reset()
    print(f"시작 상태: {state}")
    print(f"목표 칸: {env.goal}")
    print(f"걸어갈 수 있는 칸: {len(env.walkable_cells())}개")
    print()

    for action in (RIGHT, RIGHT, DOWN):
        state, reward, done = env.step(action)
        print(f"행동 {ACTION_NAMES[action]} -> 상태 {state}, 보상 {reward}, 종료 {done}")

    before = state
    state, reward, done = env.step(LEFT)
    print(f"행동 {ACTION_NAMES[LEFT]} -> 상태 {state}, 보상 {reward}, 종료 {done}")
    print(f"벽이면 상태가 유지되는가? {state == before}")


if __name__ == "__main__":
    demo()
