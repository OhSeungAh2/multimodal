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
EXPLORATION_EPISODES = 1    # 탐험을 강제로 수행할 게임 판 수
RANDOM_SEED = 2026          # 랜덤 시드 설정


class QLearningAgent:
    # Q-Learning 에이전트 클래스
    def __init__(self, alpha, gamma, epsilon):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {} # Q값 저장할 딕셔너리

    # Q값 조회하기
    def get_q_value(self, state, action):
        # 현재 위치를 키로 사용함 = 현재위치, 행동
        key = (state, action)

        # 이때 학습된 Q값 있다면 그 값을 반환하고, 없으면 0 반환
        # ::아직 경험하지 않았으면 Q-table에 저장된게 없으니까 0처리하는거임
        return self.q_table.get(key, 0.0)

    # 행동 선택하기 - epsilon-greedy 방식
    def choose_action(self, state):
        # 탐험할지? - epsilon 확률로 무작위 행동 선택
        if random.random() < self.epsilon:
            return random.randrange(NUM_ACTIONS)

        # 활용할지? - Q값이 가장 큰 행동 선택(4가지 행동에 대한 Q값 비교)
        q_values = [
            self.get_q_value(state, action) for action in range(NUM_ACTIONS)
        ]

        # 그리고 Q값이 가장 큰 행동 선택함
        max_q = max(q_values)

        # 만약에 최대 Q값 가진 행동 여러개라면? 이때는 무작위 선택
        best_actions = [
            action for action, q in enumerate(q_values) if q == max_q
        ]
        return random.choice(best_actions)

    # Q값 업데이트하기 - 행동 후 받은 보상으로 업데이트함
    def update_q_value(self, state, action, reward, next_state, done):
        # 기존에 Q값 조회
        current_q = self.get_q_value(state, action)

        # 게임 종료 시 미래가치 = 0
        if done:
            next_max_q = 0.0
        else:
            next_max_q = max(
                self.get_q_value(next_state, action) for action in range(NUM_ACTIONS)
            )

        # Q-Learning 목표값 계산
        target = reward + self.gamma * next_max_q

        # TD 오차로 새로운 Q값도 계산함
        new_q = current_q + self.alpha * (target - current_q)

        # 이제 계산한 Q값을 딕셔너리에 저장함
        self.q_table[(state, action)] = new_q

        
# 학습코드
def train():
    random.seed(RANDOM_SEED)
    env = MazeEnv()
    agent = QLearningAgent(alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON_START)
    success_count = 0

    for episode in range(1, EPISODES + 1):
        state = env.reset()
        total_reward = 0
        done = False

        # 한 에피소드 동안 행동을 반복함
        for step in range(MAX_STEPS):
            # 1. 현재 상태에서 행동 선택
            action = agent.choose_action(state)

            # 2. 선택한 행동으로 환경에서 한 칸 이동
            next_state, reward, done = env.step(action)

            # 3. 경험한 결과로 Q값 업데이트
            agent.update_q_value(state, action, reward, next_state, done)

            # 4. 누적 보상 기록
            total_reward += reward

            # 5. 현재 상태 변경
            state = next_state

            # 목표 도착 시 종료
            if done:
                success_count += 1
                break

        # 탐험 확률 점점 줄어들게
        agent.epsilon = max(EPSILON_MIN, agent.epsilon * EPSILON_DECAY)

        if episode % LOG_EVERY == 0:
            print(
                f"{episode}판 | "
                f"총 보상: {total_reward} | "
                f"누적 성공: {success_count} | "
                f"epsilon: {agent.epsilon:.3f}"
            )

    return agent, env


# 미로 탈출하기 - 학습 정책 테스트
def test_agent(agent, env):
    # 새로운 게임 시작 - 에이전트 위치 S로 초기화
    state = env.reset()

    # 탐험 확률 0 설정 
    # :: 무작위 행동 없이 학습된 Q의 정책값만 보고 행동 선택하도록 함
    agent.epsilon = 0.0

    print("\n========== 테스트 시작 ==========\n")

    # 최대 이동은 40번으로 제한하고 목표 도착 여부 확인함
    for step in range(1, MAX_STEPS + 1):
        # 현재 위치에서 Q값이 가장 높은 행동 선택
        action = agent.choose_action(state)

        # 선택한 행동으로 실제 미로에서 이동
        next_state, reward, done = env.step(action)

        # 이동 결과 출력
        print(
            f"{step}번째 | "
            f"{ACTION_NAMES[action]} | "
            f"{state} → {next_state} | "
            f"보상: {reward}"
        )

        # 이동한 위치를 현재 위치로 설정
        state = next_state

        # 목표 G에 도착했다면 테스트 종료!
        if done:
            print("\n========== 테스트 종료 ==========\n")
            print(f"총 이동 횟수: {step}")
            return

    # 40번 다 이동했는데 도착못하면 실패
    print("\n========== 목표 G 도착 실패 ==========\n")


"""
# 이건 미로 탈출 학습 함수 없을때 쓴거임
if __name__ == "__main__":
    agent, env = train()
"""

# 이거는 학습 끝난거로 한 판 테스트 헤보는거임
if __name__ == "__main__":
    # 1. Q-Learning 학습
    agent, env = train()
    # 2. 학습한 AI로 미로 탈출 테스트
    test_agent(agent, env)