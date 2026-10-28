
# ★★★★★ Windows spawn 직렬화 오류 대응
import torch.multiprocessing as mp
import random, time, math, copy
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical, Normal
import gymnasium as gym


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU()
        )
        self.actor = nn.Linear(64, 2)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        h = self.body(x)
        return self.actor(h), self.critic(h).squeeze(-1)


# ★★★★ Adam의 내부 상태를 프로세스 간 공유
class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=0.0003):
        super().__init__(params, lr=lr)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]

                state["step"] = torch.zeros(
                    (), dtype=torch.float32
                ).share_memory_()

                state["exp_avg"] = torch.zeros_like(
                    p
                ).share_memory_()

                state["exp_avg_sq"] = torch.zeros_like(
                    p
                ).share_memory_()


# ★ episodes 300 → 1000
# ★★ n_steps 5 → 20
# ★★★ episodes 1000 → 1500
# ★★★ n_steps 20 → 10
# ★★★★ n_steps 10 → 20
# ★★★★ optimizer를 Worker마다 만들지 않고 공유하도록 변경
# 마지막 확인(강사님과 통일용): episodes=2000
def worker(rank, global_net, optimizer, lock, queue, episodes=2000, gamma=0.99, n_steps=20):

    torch.set_num_threads(1)
    torch.manual_seed(100 + rank)
    random.seed(100 + rank)

    env = gym.make('CartPole-v1')
    # 마지막 확인(강사님과 통일용): manual_seed 추가
    torch.manual_seed(2026)
    local = Net()

    # ★ lr 0.001 → 0.01
    # ★★ lr 0.01 → 0.0003
    # ★★ optimizer SGD → Adam
    # ★★★ lr 0.0003 → 0.0001
    # ★★★★ lr 0.0001 → 0.0003
    # ★★★★ SharedAdam은 main()에서 생성하여 전달받음

    for ep in range(episodes):
        with lock:
            local.load_state_dict(global_net.state_dict())

        obs, _ = env.reset(seed=1000+rank*100+ep)
        done = False
        score = 0

        while not done:
            logps=[]
            values = []
            rewards = []
            entropies = []

            for _ in range(n_steps):
                x = torch.tensor(obs, dtype=torch.float32)
                logits, v = local(x)
                dist = Categorical(logits=logits)
                a = dist.sample()

                nxt, r, terminated, truncated, _ = env.step(a.item())

                logps.append(dist.log_prob(a))
                values.append(v)
                entropies.append(dist.entropy())
                rewards.append(r)

                score += r
                obs = nxt
                done = terminated or truncated

                if done:
                    break

            with torch.no_grad():
                bootstrap = (
                    0.0 if terminated
                    else local(
                        torch.tensor(obs, dtype=torch.float32)
                    )[1].item()
                )

            targets = []
            R = bootstrap

            for r in reversed(rewards):
                R = r + gamma * R
                targets.insert(0, R)

            targets = torch.tensor(targets, dtype=torch.float32)
            values = torch.stack(values)
            logps = torch.stack(logps)

            advantages = targets - values

            loss = (
                -(logps * advantages.detach()).mean()
                + 0.5 * advantages.square().mean()
                - 0.01 * torch.stack(entropies).mean()
            )

            local.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                local.parameters(), 5
            )

            with lock:
                optimizer.zero_grad()

                for gp, lp in zip(
                    global_net.parameters(),
                    local.parameters()
                ):
                    gp.grad = (
                        lp.grad.detach().clone()
                        if lp.grad is not None else None
                    )

                optimizer.step()
                local.load_state_dict(global_net.state_dict())

        queue.put((rank, ep+1, score))

    env.close()


def main():
    mp.set_start_method('spawn', force=True)
    torch.set_num_threads(1)

    global_net = Net()
    global_net.share_memory()

    lock = mp.Lock()
    queue = mp.Queue()

    # ★★★★ SharedAdam을 main()에서 한 번 생성
    # ★★★★ lr 0.0001 → 0.0003
    optimizer = SharedAdam(
        global_net.parameters(), lr=0.0003
    )

    # ★★★★ 각 Worker에 공유 Optimizer 전달
    workers = [
        mp.Process(
            target=worker,
            args=(i, global_net, optimizer, lock, queue)
        )
        for i in range(2)
    ]

    for p in workers:
        p.start()

    # ★ range 600 → 2000
    # ★★★ range 2000 → 3000
    # Worker 2개 × 각 1500 Episodes = 3000
    records = [queue.get(timeout=180) for _ in range(3000)]

    for p in workers:
        p.join(timeout=30)

    for p in workers:
        if p.is_alive():
            p.terminate()

    np.save('a3c_scores.npy', np.array(records, dtype=float))
    torch.save(global_net.state_dict(), 'a3c_cartpole.pt')

    print(
        'A3C finished, mean of last 10 records (마지막 10개 에피소드 평균): ',
        np.mean([x[2] for x in records[-10:]])
    )


if __name__ == '__main__':
    main()
