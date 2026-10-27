"""
Q-Learning 알고리즘을 사용해 미로 탈출 학습하기!

maze_env.py에서 상태, 행동, 보상이 정해지는 방식을 보고
maze_game.py에서 그 미로를 직접 걸어보며 다음 학습을 진행


1. 강화학습이 문제에서 하는 일
- 정답이 적힌 최단 경로를 주지 않음
- 에이전특 칸에서 방향을 고르고, 환경이 그에 대해 돌려준 보상을 보고 어느 방향이 유리한지 숫자로 남김

2. 반환과 할인울 gamma
- 한 판에서 앞으로 받을 보상을 할인해 더한 값을 반환(return)이라 함

    G = r0 + gamma * r1 + gamma^2 * r2 + ...
    만약 gamma가 0.95라면 한 걸음 뒤의 보상은 95%만 반영됨
    - 0에 가까우면 눈 앞의 보상만 중요
    - 1에 가까우면 먼 미래의 보상도 중요하게 생각
    - 1보다 작으면 같은 목표라도 더 가까운 보상에 대한 평가가 큼

3. Q값
- Q(s, a): 상태 s에서 행동 a를 선했한 뒤, 그 다음부터 가장 좋게 행동했을 때 얻을것으로 기대되는 반환값
- 값이 클수록 그 행동이 유리함을 의미
- 숫자를 표(딕셔너리)에 저장 (Q러닝은 표에다가 저장하니까 딕셔너리에 저장하는게 좋을 것 같음!)
    - 칸이 수십개이르모 (상태, 행동)마다 숫자 하나로 표현 가능
    - 하지만 칸의 수가 매우 커지면 표를 다 채울 수 없어 그때엔 함수 근사를 고려하면 됨(지금은 굳이 상관없음)
        - 함수 근사: Q값을 표에 저장하는 대신, 신경망 같은 함수를 사용해 상태와 행동을 입력하면 Q값을 출력하도록 학습

4. 업데이트 식
target = r + gamma * max_a' Q(s', a')
target을 구하려 한다면 이번에 받은 보상 + 다음 상태에서의 최대 Q값(미래 가치)을 고려
- Q(s, a) <- Q(s, a) + alpha * (target - Q(s, a))
    - target - Q(s, a): TD오차 = 현재 Q값과 목표값의 차이, 학습에 사용되는 오차(한걸음마다 고치니까 한 판이 끝날때까지 기다리지 않아도 됨)
    - alpha: 학습률(0.5라면 기존 값과 새 target 반씩 섞음)

5. 탐험과 활용
- 이미 G에 도달한 길을 하나 찾은 뒤엔, 그 길의 Q가 아직 0인 다른 길보다 클 수 있음
- 그렇게 되면 최대 Q만 따르게되는데 그럼! 더 짧은 길을 한번도 걸어보지 않을 수 있음
- 그래서 epsilon 확률로 네방향 중 하나를 무작위로 고르게 함 > 이걸 탐험이라고 함
- 그 다음 나머지 확률로 Q값이 가장 큰 행동을 선택함 > 이걸 활용이라고 함
- 이제 판이 거듭될수록 epsilon값을 줄여 처음엔 많이 돌아다니고 나중엔 배운 길로 걷게함(이게 고전적이고 많이 사용되는 강화학습 방식)

6. 좋은 값이 전달되는 방향
- 목표에 들어가는 행동을 하면 (상태, 행동)의 target이 100이 됨
- 그래서 그 전 칸은 다음 학습에서 gamma * (다음 상태의 최대 Q값)을 보고 함께 커지게 됨
- 목표 앖에서 출발점 쪽으로 한 칸씩 평가가 퍼지게 되는데
- 목표에서 멀수록, 목표에 도착하는 경험이 적을수록 게임 판수가 더 많이 필요하게 됨
"""

import random
import time

from maze_env import (
    ACTION_NAMES,
    DOWN,
    LEFT,
    MazeEnv,
    NUM_ACTIONS,
    RIGHT,
    UP,
)

EPISODES = 500              # 학습할 게임 판 수
MAX_STEPS = 40              # 한 판에서 최대 이동 횟수 : 40번안에 도착하지 못하면 끝인것임
ALPHA = 0.5                 # 학습률
GAMMA = 0.95                # 할인율
EPSILON_START = 0.1         # 탐험 확률
EPSILON_MIN = 0.05          # 탐험 확률의 최솟값
EPSILON_DECAY = 0.992       # 탐험 확률 감소율 : 한 판 끝날때마다 EPSILON에 곱해줌
LOG_EVERY = 100             # 몇 판마다 학습 로그를 출력할지 결정 : 100판마다 로그 출력 = 5번 나올거임
EXPLAIN_EPISODE = 1    # 몇 번째 에피소드에서 업데이트 과정을 출력할지 결정
RANDOM_SEED = 2026          # 랜덤 시드 설정
POLICY_SYMBOL = {
    UP: "^",
    DOWN: "v",
    LEFT: "<",
    RIGHT: ">",
}


class QLearningAgent:
    def __init__(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma
        # 키는 (상태, 행동), 값은 Q(s, a)
        self.q = {}

    def q_get(self, state, action):
        return self.q.get((state, action), 0.0)

    def best_action(self, state, random_tie=True):
        values = [self.q_get(state, action) for action in range(NUM_ACTIONS)]
        best = max(values)
        candidates = [action for action, value in enumerate(values) if value == best]
        if random_tie:
            return random.choice(candidates)
        return candidates[0]

    def choose_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(NUM_ACTIONS), True
        return self.best_action(state, random_tie=True), False

    def update(self, state, action, reward, next_state, done, explain=False):
        old = self.q_get(state, action)
        if done:
            future = 0.0
        else:
            future = self.q_get(next_state, 0)
            for index in range(1, NUM_ACTIONS):
                value = self.q_get(next_state, index)
                if value > future:
                    future = value

        target = reward + self.gamma * future
        td_error = target - old
        new_value = old + self.alpha * td_error

        if explain:
            print_update_example(state, action, reward, next_state, done,
                old, future, target, td_error, new_value, self.alpha, self.gamma)

        self.q[(state, action)] = new_value



def print_update_example(
    state, action, reward, next_state, done,
    old, future, target, td_error, new_value, alpha, gamma,
):
    """표에 반영하기 직전의 숫자로 업데이트 식을 그대로 출력한다."""
    print(f"상태 s = {state}")
    print(f"행동 a = {ACTION_NAMES[action]}")
    print(f"보상 r = {reward}")
    print(f"다음 상태 s' = {next_state}")
    print(f"종료 = {done}")
    if next_state == state and not done:
        print("벽이라 칸이 바뀌지 않았습니다. 다음 상태도 같은 칸입니다.")
    if done:
        print("목표에 도착했으므로 미래 가치는 0입니다.")
    else:
        print(f"다음 칸의 최대 Q = {future:.4f}")
        if future == 0.0:
            print("최대값이 0입니다. 다음 칸에서 아직 유리하다고 배운 행동이 없습니다.")
    print(
        f"target = r + gamma * future = {reward} + {gamma} * {future:.4f} "
        f"= {target:.4f}"
    )
    print(f"기존 Q(s, a) = {old:.4f}")
    print(f"TD 오차 = target - 기존 Q = {td_error:.4f}")
    print(
        f"새 Q(s, a) = 기존 Q + alpha * TD 오차 "
        f"= {old:.4f} + {alpha} * {td_error:.4f} = {new_value:.4f}"
    )
    print("--------------------------------------------------------------")
    print()


def print_intro():
    print("Q-Learning으로 미로를 학습합니다.")
    print("Q(s,a) <- Q(s,a) + alpha * (r + gamma * max Q(s',a') - Q(s,a))")
    print(
        f"alpha={ALPHA}, gamma={GAMMA}, 에피소드={EPISODES}, "
        f"엡실론 {EPSILON_START} -> {EPSILON_MIN}"
    )
    print(f"무작위 시드={RANDOM_SEED}")
    print()

def print_action_choice(episode, epsilon, action, explored):
    print()
    print(f"----- 에피소드 {episode}의 첫 걸음: 행동을 어떻게 골랐는가 -----")
    if explored:
        how = "탐험입니다. 엡실론 확률로 네 방향 중 하나를 무작위로 골랐습니다."
    else:
        how = "활용입니다. 지금 칸에서 Q가 가장 큰 행동을 골랐습니다."
    print(f"행동: {ACTION_NAMES[action]}")
    print(how)
    print(f"이때 엡실론 = {epsilon:.3f}")


def print_log(episode, rewards, successes, epsilon):
    avg_reward, success_rate = recent_stats(rewards, successes, LOG_EVERY)
    print(
        f"에피소드 {episode:4d} | "
        f"최근 {LOG_EVERY}판 평균 보상 {avg_reward:7.1f} | "
        f"성공률 {success_rate * 100:5.1f}% | "
        f"이번 판 엡실론 {epsilon:.3f}"
    )

def recent_stats(rewards, successes, count):
    """마지막 count판의 평균 보상과 성공률."""
    reward_slice = rewards[-count:]
    success_slice = successes[-count:]
    size = len(reward_slice)
    avg_reward = sum(reward_slice) / size
    success_rate = sum(success_slice) / size
    return avg_reward, success_rate

# 탐욕 정책: 학습된 Q를 따라 최적 경로를 탐색하는 정책
def follow_policy(agent):
    """
        탐욕 정책으로 시작 칸에서 걸어보고 경로와 보상 합계, 도착 여부 돌려줌
    """
    env = MazeEnv()
    state = env.reset()
    path = [state]
    seen = {state}
    total_reward = 0

    # 탐욕 정책으로 미로를 탐색하며 경로를 기록하고 보상을 누적
    for _ in range(MAX_STEPS):
        action = agent.best_action(state, random_tie=False)
        next_state, reward, done = env.step(action)
        total_reward += reward
        path.append(next_state)

        # 완료되었다면
        if done:
            return path, total_reward, True
        # 완료되지 않았다면(다음 상태가 이전 상태이거나 이미 본 상태라면 실패로 간주)
        if next_state == state or next_state in seen:
            return path, total_reward, False

        seen.add(next_state)
        state = next_state

    return path, total_reward, False



def train():
    random.seed(RANDOM_SEED)
    env = MazeEnv()  # 환경 초기화
    agent = QLearningAgent(ALPHA, GAMMA)
    epsilon = EPSILON_START
    rewards = []
    successes = []
    snapshots = []
    found_goal_policy = False
    best_fail_steps = -1    # 안움직이면 0이니까 있을 수 없는 값 -1을 넣어서 비교할거임
    final_snapshot = None   # (추가) 마지막 에피소드의 Q-table 저장

    print_intro()
    # perf_counter: 정밀도 높은 타이머
    started = time.perf_counter()

    for episode in range(1, EPISODES+1):
        current_epsilon = epsilon
        state = env.reset()
        total_reward = 0.0
        reached_goal = False

        for step_index in range(1, MAX_STEPS+1):
            action, explored = agent.choose_action(state, current_epsilon)
            next_state, reward, done = env.step(action)

            explain_this = episode == EXPLAIN_EPISODE and step_index == 1
            if explain_this:
                print_action_choice(episode, current_epsilon, action, explored)

            agent.update(state, action, reward, next_state, done, explain=explain_this)
            total_reward += reward
            state = next_state
            if done:
                reached_goal = True
                break

        # 여기 빠져나오면 한 판 끝난거임
        rewards.append(total_reward)
        successes.append(1 if reached_goal else 0)

        # 탐험 비율 조절: 에피소드마다 epsilon 감소
        epsilon = max(EPSILON_MIN, current_epsilon * EPSILON_DECAY)

        if episode % LOG_EVERY == 0:
            print_log(episode, rewards, successes, current_epsilon)

        if (not found_goal_policy) or episode == EPISODES:
            path, _total, reached_now = follow_policy(agent)
            steps_now = len(path) - 1
            shot = {
                "episode": episode,
                "q": dict(agent.q),
                "reached": reached_now,
                "steps": steps_now
            }
            if episode == EPISODES:
                final_snapshot = shot
            elif not reached_now:
                fail_count = len(snapshots)
                grew_enough = steps_now >= best_fail_steps + 4
                if episode == 1 or (fail_count < 3 and grew_enough):
                    snapshots.append(shot)
                    best_fail_steps = steps_now
            if reached_now:
                found_goal_policy = True
    
    if final_snapshot is None:
        final_snapshot = {
            "episode": EPISODES,
            "q": dict(agent.q),
            "reached": False,
            "steps": 0
        }

    if not snapshots or snapshots[-1]["episode"] != final_snapshot["episode"]:
        snapshots.append(final_snapshot)

    # 이제 시간 계산할거임
    elapsed = time.perf_counter() - started
    print()
    print(f"학습 시간 {elapsed:.2f}초")
    print(f"Q표에 기록된 (상태, 행동) 쌍: {len(agent.q)}개")
    print("표에 없는 쌍은 한 번도 실행되지 않은 것이며, 읽을 때는 0으로 봅니다.")
    print("창에서 이어서 볼 시점:")
    for shot in snapshots:
        if shot["reached"]:
            state_text = f"목표까지 {shot['steps']}걸음"
        else:
            state_text = f"{shot['steps']}걸음 뒤 막힘"
        print(f"  에피소드 {shot['episode']}: {state_text}")
    return agent, rewards, successes, snapshots


def report(agent, rewards, successes):
    """학습된 표를 글자로 보여 주고, 화면 패널에 넣을 요약을 돌려준다."""
    env = MazeEnv()
    avg_reward, success_rate = recent_stats(rewards, successes, LOG_EVERY)

    print()
    print("시작 칸에서 각 행동의 Q값")
    start_values = []
    for action in range(NUM_ACTIONS):
        value = agent.q_get(env.start, action)
        start_values.append(value)
        print(f"  {ACTION_NAMES[action]:<4} {value:8.2f}")
    best_start = max(start_values)
    chosen = [
        ACTION_NAMES[action]
        for action, value in enumerate(start_values)
        if value == best_start
    ]
    print(f"시작 칸에서 고르는 행동: {', '.join(chosen)}")

    print()
    print("각 칸에서 실제로 고르는 행동 중 Q가 큰 것. 목표에 가까울수록 값이 큽니다.")
    ranked = []
    for cell in env.walkable_cells():
        if cell == env.goal:
            continue
        action = agent.best_action(cell, random_tie=False)
        ranked.append((agent.q_get(cell, action), cell, action))
    ranked.sort(reverse=True)
    for value, cell, action in ranked[:6]:
        print(f"  상태 {cell}, {ACTION_NAMES[action]} -> {value:.2f}")

    print()
    print("탐욕 정책. 각 칸의 기호가 그 칸에서 Q가 가장 큰 방향입니다.")
    for row in range(env.rows):
        symbols = []
        for col in range(env.cols):
            pos = (row, col)
            if pos == env.goal:
                symbols.append("G")
            elif env.grid[row][col] == 1:
                symbols.append("#")
            else:
                action = agent.best_action(pos, random_tie=False)
                symbols.append(POLICY_SYMBOL[action])
        print("".join(symbols))

    path, total_reward, reached = follow_policy(agent)
    print()
    if reached:
        print(f"이 정책은 목표까지 {len(path) - 1}걸음, 보상 합 {total_reward} 입니다.")
    else:
        print("이 정책은 목표에 도착하지 못했습니다.")
        print("EPISODES를 늘리거나 EPSILON_DECAY를 1에 더 가깝게 바꿔 탐험을 오래 남겨 보세요.")

    parts = [f"({row},{col})" for row, col in path]
    for index in range(0, len(parts), 6):
        print("  " + " -> ".join(parts[index:index + 6]))

    return {
        "episodes": EPISODES,
        "alpha": ALPHA,
        "gamma": GAMMA,
        "recent_success": success_rate,
        "recent_reward": avg_reward,
        "policy_steps": len(path) - 1,
        "policy_reached": reached,
    }


# 실행하는 메인함수
def main():
    agent, rewards, successes, snapshots = train()
    summary = report(agent, rewards, successes)
    print()
    print("창에서 학습 초반부터 끝난 뒤까지 여러 판을 이어서 재생합니다.")
    print("한 판이 끝나면 다음 시점으로 넘어가고, 마지막 뒤에는 처음부터 반복합니다.")
    print("스페이스로 멈추고, R로 지금 판을 다시 보고, ESC로 닫습니다.")

    try:
        from maze_game import watch_agent
    except ImportError:
        print("화면 재생을 열지 못했습니다.")
        return

    watch_agent(agent, successes, summary, snapshots)

if __name__ == "__main__":
    main()