
"""
같은 미로를 신경망으로 학습(DQN)

Q_Learning.py의 Q표와 비교하며 읽기
환경, 보상, 행동 번호, 앱실론-그리기, 목표값 식은 그대로!!
바뀌는건 Q값을 딕셔너리에 저장하지 않고, 신경망이 계산함!

"""

import argparse
import random
import time

from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from maze_env import (
    ACTION_NAMES,
    DOWN,
    LEFT,
    MazeEnv,
    NUM_ACTIONS,
    RIGHT,
    UP,
)


# [1. 학습 설정 - 하이퍼 파라미터] 
EPISODES = 2500         # 전체 학습 에피소드 수    
MAX_STEPS = 60          # 한 에피소드에서 최대 이동할 수 있는 횟수
GAMMA = 0.95            # 할인율 gamma
LEARNING_RATE = 0.001   # 신경망 학습률 : 신경망의 가중치를 얼마나 수정할지 결정
BATCH_SIZE = 64         # 한 번 학습할 때 Replay Buffer에서 꺼낼 경험 개수
REPLAY_SIZE = 15000     # Replay Buffer에 저장할 수 있는 최대 경험 개수
LEARN_EVERY = 4         # 환경에서 4번 행동할 때마다 신경망을 한 번 학습
TARGET_SYNC_EVERY = 200 # 신경망 학습 200회마다 Policy Network의 가중치 Target Network에 복사
EPSILON_START = 1.0     # 초기 탐험 확률 : 처음에는 100% 무작위 행동으로 탐험함
EPSILON_MIN = 0.05       
EPSILON_DECAY_EPISODES = 1800   # 탐험 확률을 점차 감소시킬 에피소드 수

RANDOM_SEED = 2026
LOG_EVERY = 250         # 학습 결과를 화면에 출력할 에피소드 간격
SNAPSHOT_EPISODES = (1, 250, 1000, 2500)   # 학습 시점에서 정책을 화면에 출력할 에피소드 번호

# 학습 에피소드의 70%는 임의의 이동 가능한 칸에서 시작
# 목표 근처에서도 학습 경험을 확보하기 위한 설정
# 단, 최종 성능 평가는 항상 원래 시작점 S에서 진행함
RANDOM_START_PROB = 0.7

SYMBOLS = {
    UP: "^",
    DOWN: "v",
    LEFT: "<",
    RIGHT: ">",
}

# seed 설정 함수
def set_seed(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# [3. 상태를 신경망 입력값으로 변환]
def state_tensor(states, rows, cols):
    """
    미로 위치 (행, 열)을 One-hot Encoding으로 변환함

    기존 Q-Learning
    - 상태를 (1, 1)과 같은 튜플로 사용
    - 딕셔너리에서 Q값을 조회함

    DQN
    - 상태를 숫자 벡터로 변환해 신경망에 입력함

    예) 10 x 10 미로
    - 전체 칸 100개
    - 하나의 위치를 길이 100인 벡터로 표현

    현재 위치에 해당하는 원소만 1
    나머지 원소는 모두 0
    """

    # (row, col)을 하나의 정수 인덱스로 변환
    # 예) (1, 1) -> 1 * 10 + 1 = 11
    indices = torch.tensor(
        [r * cols + c for r, c in states],
        dtype=torch.long
    )

    # One-hot Encoding 적용
    # 100개 값 중 현재 위치의 값만 1로 설정
    # .float()는 신경망 입력에 사용할 실수형 변환
    return F.one_hot(
        indices,
        num_classes=rows * cols
    ).float()


# [4. Q-Network (신경망)]
class QNetwork(nn.Module):
    """
    현재 상태를 입력받아 각 행동의 Q값을 예측하는 신경망

    입력
    - 현재 상태의 One-hot 벡터

    출력
    - 위, 아래, 왼쪽, 오른쪽의 Q값 4개

    기존 Q-Learning
    - self.q[(state, action)] 으로 Q값 조회

    DQN
    - 신경망에 상태를 입력해 Q값을 직접 예측함
    """

    def __init__(self, input_dim):
        super().__init__()

        # 입력층 -> 은닉층 -> 은닉층 -> 출력층
        self.layers = nn.Sequential(

            # 입력 상태를 64개 뉴런으로 전달
            nn.Linear(input_dim, 64),

            # 음수는 0으로, 양수는 그대로 전달하는 활성화 함수
            nn.ReLU(),

            # 두 번째 은닉층
            nn.Linear(64, 64),
            nn.ReLU(),

            # 각 행동에 대한 Q값 4개 출력
            # 출력값은 확률이 아니므로 Softmax를 사용하지 않음
            nn.Linear(64, NUM_ACTIONS),
        )

    def forward(self, x):
        # 입력 상태를 신경망에 통과시켜 Q값을 반환
        return self.layers(x)


# [5. Replay Buffer]
class ReplayBuffer:
    """
    에이전트가 경험한 정보를 저장하는 공간

    저장하는 정보
    (state, action, reward, next_state, done)

    state      : 행동하기 전 상태
    action     : 선택한 행동
    reward     : 행동 후 받은 보상
    next_state : 행동 후 도착한 상태
    done       : 목표에 도착했는지 여부

    DQN에서는 과거 경험을 무작위로 꺼내 학습함
    """

    def __init__(self, capacity):

        # 최대 capacity개까지 경험 저장
        # 용량을 초과하면 가장 오래된 경험부터 삭제
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):

        # 새로운 경험을 Replay Buffer에 저장
        self.memory.append(
            (state, action, reward, next_state, done)
        )

    def sample(self, batch_size):

        # 저장된 경험 중 batch_size개를 무작위로 추출
        # 시간순으로 연속된 경험만 학습하는 것을 피하기 위함
        return random.sample(self.memory, batch_size)

    def __len__(self):

        # 현재 저장된 경험 개수 반환
        return len(self.memory)


# [6. DQN Agent]
class DQNAgent:

    def __init__(self, rows, cols):

        self.rows = rows
        self.cols = cols

        # ------------------------------------------
        # Policy Network
        # ------------------------------------------

        # 실제 Q값을 예측하고 학습하는 신경망
        self.policy_net = QNetwork(rows * cols)

        # ------------------------------------------
        # Target Network
        # ------------------------------------------

        # TD Target 계산에 사용하는 별도의 신경망
        self.target_net = QNetwork(rows * cols)

        # 처음에는 두 신경망이 같은 가중치를 갖도록 설정
        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        # Target Network는 목표값 계산용으로 사용
        self.target_net.eval()

        # ------------------------------------------
        # Optimizer
        # ------------------------------------------

        # Adam Optimizer 사용
        # Loss를 줄이는 방향으로 신경망 가중치를 업데이트함
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=LEARNING_RATE
        )

        # 과거 경험을 저장할 Replay Buffer 생성
        self.replay = ReplayBuffer(REPLAY_SIZE)

        # 신경망 학습 업데이트 횟수
        self.learn_updates = 0


    # [6-1. 현재 상태의 Q값 예측]
    @torch.no_grad()
    def q_values(self, state):
        """
        현재 상태에서 네 방향의 Q값을 예측함

        예)
        위: 10.2
        아래: 15.8
        왼쪽: 8.1
        오른쪽: 12.4

        이 경우 Q값이 가장 큰 '아래'가 유리하다고 판단

        @torch.no_grad()
        - Q값 조회만 할 때는 역전파 계산을 하지 않음
        """

        # 현재 상태를 신경망 입력 형태로 변환
        x = state_tensor(
            [state],
            self.rows,
            self.cols
        )

        # 신경망을 통해 Q값 4개 예측
        q = self.policy_net(x)

        # 첫 번째 상태의 Q값을 Python 리스트로 반환
        return q[0].tolist()


    # [6-2. Q값이 가장 큰 행동 선택]
    def best_action(self, state, random_tie=False):

        # 현재 상태의 Q값 예측
        qs = self.q_values(state)

        # 네 방향 중 가장 큰 Q값
        max_value = max(qs)

        # 최대 Q값을 가진 행동 번호 찾기
        candidates = [
            action
            for action, q in enumerate(qs)
            if q == max_value
        ]

        # 최대 Q값이 같은 행동이 여러 개일 경우
        # random_tie=True면 그중 무작위 선택
        if random_tie:
            return random.choice(candidates)

        # 아니면 가장 앞에 있는 행동 선택
        return candidates[0]


    # [6-3. Epsilon-greedy 행동 선택]
    def choose_action(self, state, epsilon):
        """
        epsilon 확률로 무작위 행동을 선택함 (탐험)

        나머지 확률로 Q값이 가장 높은 행동을 선택함 (활용)

        기존 Q-Learning과 같은 행동 선택 방식
        """

        # 탐험
        if random.random() < epsilon:
            return random.randrange(NUM_ACTIONS), True

        # 활용
        return self.best_action(
            state,
            random_tie=True
        ), False


    # [6-4. 신경망 학습 (DQN 핵심)]
    def learn(self):
        """
        Replay Buffer에 저장된 경험으로 신경망 학습

        학습 과정

        1. 과거 경험을 무작위 추출
        2. 현재 Q(s,a) 계산
        3. Target Q값 계산
        4. Loss 계산
        5. 역전파로 가중치 수정
        6. 일정 횟수마다 Target Network 동기화
        """

        # 경험이 BATCH_SIZE개보다 적으면 학습하지 않음
        if len(self.replay) < BATCH_SIZE:
            return None


        # [STEP 1. 경험 무작위 추출]
        batch = self.replay.sample(BATCH_SIZE)

        # (s, a, r, s', done)으로 분리
        states, actions, rewards, next_states, dones = zip(*batch)

        # 상태와 다음 상태를 One-hot 벡터로 변환
        state_batch = state_tensor(
            states,
            self.rows,
            self.cols
        )

        next_state_batch = state_tensor(
            next_states,
            self.rows,
            self.cols
        )

        # 행동 번호를 정수형 Tensor로 변환
        # gather()에서 사용할 수 있도록 (64, 1) 형태로 변경
        action_batch = torch.tensor(
            actions,
            dtype=torch.long
        ).unsqueeze(1)

        # 보상과 종료 여부를 Tensor로 변환
        reward_batch = torch.tensor(
            rewards,
            dtype=torch.float32
        )

        done_batch = torch.tensor(
            dones,
            dtype=torch.float32
        )


        # [STEP 2. 현재 Q값 계산]
        # 각 상태에서 네 방향의 Q값을 모두 예측
        # 결과 형태: (64, 4)
        all_q = self.policy_net(state_batch)

        # 실제로 선택했던 행동의 Q값만 가져옴
        # gather(1, action_batch)는 각 행에서 선택한 행동의 Q를 조회
        # squeeze(1)은 결과를 (64,) 형태로 변경
        current_q = all_q.gather(
            1,
            action_batch
        ).squeeze(1)


        # [STEP 3. Target Q값 계산]
        # DQN의 Bellman Target
        #
        # target = r + gamma * max Q_target(s', a')
        #
        # 단, 목표에 도착한 경우(done=True)
        # 미래 보상이 없으므로 target = r

        # Target Network로 계산할 때는 역전파하지 않음
        with torch.no_grad():

            # 다음 상태에서의 각 행동 Q값 계산
            next_q = self.target_net(next_state_batch)

            # 다음 상태의 네 방향 중 최대 Q값 선택
            next_max_q = next_q.max(dim=1).values

            # TD Target 계산
            # done=True라면 (1 - done) = 0
            # 따라서 미래 Q값을 더하지 않음
            targets = (
                reward_batch
                + GAMMA * next_max_q * (1.0 - done_batch)
            )


        # [STEP 4. Loss 계산]
        # 현재 Q값과 Target Q값이 얼마나 다른지 계산
        #
        # 기존 Q-Learning
        # TD 오차를 이용해 Q-table 값을 직접 수정
        #
        # DQN
        # TD 오차를 바탕으로 Loss를 계산한 후
        # 신경망의 가중치를 수정함

        # Huber Loss(Smooth L1 Loss) 사용
        loss = F.smooth_l1_loss(
            current_q,
            targets
        )


        # [STEP 5. 역전파 및 가중치 수정]
        # 이전에 계산된 기울기 초기화
        self.optimizer.zero_grad()

        # Loss를 기준으로 각 가중치의 기울기 계산
        loss.backward()

        # 계산된 기울기를 이용해 가중치 수정
        self.optimizer.step()

        # 신경망 학습 횟수 증가
        self.learn_updates += 1


        # [STEP 6. Target Network 동기화]
        # 200번 학습할 때마다 가중치를 복사
        # Target Network를 매번 변경하지 않아
        # 목표값이 지나치게 빠르게 변하는 것을 줄임
        if self.learn_updates % TARGET_SYNC_EVERY == 0:

            self.target_net.load_state_dict(
                self.policy_net.state_dict()
            )

        # 이번 학습의 Loss 반환
        return float(loss.item())



# [7. Pygame 화면 연결용 클래스]
class DisplayAgent:
    """
    기존 maze_game.py의 watch_agent()와 연결하기 위한 클래스

    기존 화면은 agent.q 딕셔너리를 사용함

    DQN은 실제 학습 과정에서 Q-table을 사용하지 않지만
    화면 출력용으로만 신경망이 예측한 Q값을 딕셔너리에 담음

    즉, 여기서 사용하는 q는 학습용 Q-table이 아님
    """

    def __init__(self):
        self.q = {}

    def q_get(self, state, action):

        # 저장된 화면 표시용 Q값 조회
        return self.q.get(
            (state, action),
            0.0
        )

    def best_action(self, state, random_tie=False):

        # 화면에 저장된 Q값에서 가장 큰 행동 선택
        values = [
            self.q_get(state, action)
            for action in range(NUM_ACTIONS)
        ]

        best = max(values)

        candidates = [
            action
            for action, value in enumerate(values)
            if value == best
        ]

        if random_tie:
            return random.choice(candidates)

        return candidates[0]


# [8. 학습한 정책 평가]
def evaluate(agent):
    """
    학습된 신경망으로 실제 미로를 탈출할 수 있는지 평가함

    학습과 달리 epsilon 탐험은 사용하지 않음
    항상 Q값이 가장 높은 행동만 선택

    원래 시작점 S에서 출발하여 평가함
    """

    env = MazeEnv()

    # 시작점으로 환경 초기화
    state = env.reset()

    # 이동 경로 저장
    path = [state]

    # 이미 방문한 위치 저장
    visited = {state}

    total_reward = 0

    for _ in range(MAX_STEPS):

        # Q값이 가장 높은 행동 선택
        action = agent.best_action(state)

        # 환경에서 행동 실행
        next_state, reward, done = env.step(action)

        # 보상과 이동 경로 기록
        total_reward += reward
        path.append(next_state)

        # 목표에 도착했다면 성공
        if done:
            return path, total_reward, True

        # 같은 칸을 다시 방문했다면
        # 현재 탐욕 정책에서 반복될 수 있으므로 실패 처리
        # 벽에 부딪혀 같은 칸에 머무는 경우도 포함
        if next_state in visited:
            return path, total_reward, False

        visited.add(next_state)

        # 다음 상태로 이동
        state = next_state

    # 최대 이동 횟수 초과 시 실패
    return path, total_reward, False


# [9. 학습 시점별 정책 저장]
def make_snapshot(agent, episode, env):
    """
    학습 도중 특정 에피소드의 정책을 저장함

    나중에 Pygame에서
    학습 초기와 학습 완료 후의 경로를 비교하기 위함
    """

    shown_q = {}

    # 미로에서 이동 가능한 모든 칸 확인
    for cell in env.walkable_cells():

        # 각 칸에서 신경망이 예측한 Q값 4개 저장
        for action, value in enumerate(agent.q_values(cell)):

            shown_q[(cell, action)] = value

    # 현재까지 학습한 정책의 성능 평가
    path, total_reward, reached = evaluate(agent)

    return {
        "episode": episode,
        "q": shown_q,
        "reached": reached,
        "steps": len(path) - 1,
        "reward": total_reward,
    }


# [10. DQN 학습 실행]
def train():
    """
    전체 에피소드를 반복하며 DQN을 학습시키는 함수

    기존 Q-Learning의 train()과 역할은 같음

    가장 큰 차이
    - Q-table 업데이트 대신
    - 경험 저장 -> 신경망 학습 순서로 진행함
    """

    # 난수 고정
    set_seed(RANDOM_SEED)

    # 미로 환경 생성
    env = MazeEnv()

    # DQN 에이전트 생성
    agent = DQNAgent(
        env.rows,
        env.cols
    )

    # 출발 가능한 위치 목록
    # 목표 칸은 제외함
    possible_starts = [
        cell
        for cell in env.walkable_cells()
        if cell != env.goal
    ]

    # 학습 결과 기록
    rewards = []
    successes = []
    losses = []
    snapshots = []

    # 전체 환경 이동 횟수
    total_steps = 0

    # 학습 시간 측정 시작
    begin = time.perf_counter()

    print("DQN 미로 학습 시작")
    print(f"에피소드: {EPISODES}")
    print(f"할인율: {GAMMA}")
    print(f"학습률: {LEARNING_RATE}")
    print(f"배치 크기: {BATCH_SIZE}")
    print(f"무작위 시작 비율: {RANDOM_START_PROB:.0%}")
    print()

    # --------------------------------------------------------
    # 에피소드 반복
    # --------------------------------------------------------

    for episode in range(1, EPISODES + 1):

        # epsilon을 선형적으로 감소시킴
        # 초기: 1.0
        # 최종: 0.05
        decay_ratio = min(
            episode - 1,
            EPSILON_DECAY_EPISODES
        ) / EPSILON_DECAY_EPISODES

        epsilon = max(
            EPSILON_MIN,
            EPSILON_START
            - (EPSILON_START - EPSILON_MIN) * decay_ratio
        )

        # 미로 초기화
        state = env.reset()

        # 학습 중 일부 에피소드는 무작위 위치에서 시작
        # 목표 근처의 경험도 학습할 수 있도록 설정한 것임
        # 기존 환경의 이동/보상 규칙은 변경하지 않음
        if random.random() < RANDOM_START_PROB:

            state = random.choice(possible_starts)
            env.pos = state

        total_reward = 0
        reached_goal = False

        # ----------------------------------------------------
        # 한 에피소드 내 행동 반복
        # ----------------------------------------------------

        for step in range(MAX_STEPS):

            # 1. epsilon-greedy로 행동 선택
            action, explored = agent.choose_action(
                state,
                epsilon
            )

            # 2. 실제 미로에서 행동 실행
            next_state, reward, done = env.step(action)

            # 3. 이번 경험을 Replay Buffer에 저장
            agent.replay.push(
                state,
                action,
                reward,
                next_state,
                done
            )

            # 전체 이동 횟수 증가
            total_steps += 1

            # 4. 일정 횟수마다 신경망 학습
            if total_steps % LEARN_EVERY == 0:

                loss = agent.learn()

                # 실제 학습이 진행되었다면 Loss 기록
                if loss is not None:
                    losses.append(loss)

            # 5. 보상 합산
            total_reward += reward

            # 6. 다음 상태로 이동
            state = next_state

            # 7. 목표에 도착했다면 현재 에피소드 종료
            if done:
                reached_goal = True
                break

        # ----------------------------------------------------
        # 에피소드 종료 후 결과 기록
        # ----------------------------------------------------

        rewards.append(total_reward)

        # 성공하면 1, 실패하면 0 저장
        successes.append(
            1 if reached_goal else 0
        )

        # ----------------------------------------------------
        # 학습 로그 출력
        # ----------------------------------------------------

        if episode % LOG_EVERY == 0:

            # 최근 250판 성공률
            recent_successes = successes[-LOG_EVERY:]

            success_rate = (
                sum(recent_successes)
                / len(recent_successes)
                * 100
            )

            # 최근 250판 평균 보상
            recent_rewards = rewards[-LOG_EVERY:]

            avg_reward = (
                sum(recent_rewards)
                / len(recent_rewards)
            )

            # 원래 시작점에서 학습 결과 평가
            path, _, greedy_success = evaluate(agent)

            print(
                f"에피소드 {episode:4d}/{EPISODES} | "
                f"평균 보상 {avg_reward:6.1f} | "
                f"학습 성공률 {success_rate:5.1f}% | "
                f"epsilon={epsilon:.3f} | "
                f"S에서 평가: "
                f"{'성공' if greedy_success else '미도착'} "
                f"({len(path)-1}걸음)"
            )

        # ----------------------------------------------------
        # 특정 학습 시점 저장
        # ----------------------------------------------------

        if episode in SNAPSHOT_EPISODES or episode == EPISODES:

            snapshots.append(
                make_snapshot(agent, episode, env)
            )

    # 학습 시간 계산
    elapsed = time.perf_counter() - begin

    print()
    print(f"학습 시간: {elapsed:.2f}초")
    print(f"신경망 업데이트: {agent.learn_updates}회")

    return agent, rewards, successes, losses, snapshots


# [11. 최종 학습 결과 출력]
def report(agent):
    """
    학습 완료 후 결과를 출력하는 함수

    - 시작점에서의 Q값
    - 각 칸에서 선택할 행동
    - 최종 경로
    - 누적 보상
    """

    env = MazeEnv()

    # 최종 정책 평가
    path, reward, success = evaluate(agent)

    print()
    print("시작점에서 신경망이 예측한 Q값")

    # 네 방향의 Q값 출력
    for action, value in enumerate(agent.q_values(env.start)):

        print(
            f"{ACTION_NAMES[action]}: {value:.3f}"
        )

    print()
    print("학습된 최종 정책")
    print("S=시작점, G=목표, #=벽")
    print()

    # 미로의 모든 칸 순회
    for row in range(env.rows):

        symbols = []

        for col in range(env.cols):

            pos = (row, col)

            if pos == env.start:
                symbols.append("S")

            elif pos == env.goal:
                symbols.append("G")

            elif env.grid[row][col] == 1:
                symbols.append("#")

            else:
                # 해당 칸에서 Q값이 가장 높은 행동 표시
                action = agent.best_action(pos)
                symbols.append(SYMBOLS[action])

        print("".join(symbols))

    print()

    if success:
        print("목표 도착 성공")
    else:
        print("목표 도착 실패")

    print(f"이동 횟수: {len(path) - 1}")
    print(f"누적 보상: {reward}")

    # 실제 이동 경로 출력
    print()
    print("이동 경로")

    print(
        " -> ".join(
            f"({row},{col})"
            for row, col in path
        )
    )

    # 기존 maze_game.py에 전달할 요약 정보
    return {
        "alpha": LEARNING_RATE,
        "gamma": GAMMA,
        "policy_steps": len(path) - 1,
        "policy_reached": success,
    }


# [12. 메인 함수]
def main():

    # 실행 옵션 설정
    parser = argparse.ArgumentParser(
        description="미로 DQN 학습"
    )

    # --no-gui 옵션을 입력하면 Pygame 창을 열지 않음
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Pygame 화면 없이 학습"
    )

    args = parser.parse_args()

    # ------------------------------------------
    # DQN 학습
    # ------------------------------------------

    agent, rewards, successes, losses, snapshots = train()

    # 학습 결과 출력
    summary = report(agent)

    # ------------------------------------------
    # 학습된 신경망 저장
    # ------------------------------------------

    # DQN.py와 같은 폴더에 저장
    model_path = (
        Path(__file__).resolve().parent
        / "maze_dqn_model.pt"
    )

    # 신경망 가중치를 파일로 저장
    torch.save(
        agent.policy_net.state_dict(),
        model_path
    )

    print()
    print(f"학습 모델 저장: {model_path.name}")

    # ------------------------------------------
    # Pygame으로 학습 결과 재생
    # ------------------------------------------

    if not args.no_gui:

        try:
            from maze_game import watch_agent

        except ImportError as exc:
            print(f"Pygame 화면을 열 수 없습니다: {exc}")
            print("설치 명령어: pip install pygame")
            return

        print()
        print("학습 과정 재생")
        print("SPACE: 일시정지")
        print("R: 다시 보기")
        print("ESC: 종료")

        # 신경망으로 계산한 Q값을 화면 출력용으로 변환하여
        # 기존 Pygame 화면에서 학습 시점별로 재생
        watch_agent(
            DisplayAgent(),
            successes,
            summary,
            snapshots
        )


# [13. 프로그램 실행]
if __name__ == "__main__":
    main()
