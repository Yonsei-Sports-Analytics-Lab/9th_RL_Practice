import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# Actor & Critic 클래스
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        
        # Actor: 상태를 받아 각 행동의 확률을 계산
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),  # PPO에서는 주로 ReLU보다 Tanh를 선호합니다
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic: 현재 상태가 얼마나 좋은지(Value)를 평가
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def act(self, state, deterministic=False):
        action_probs = self.actor(state)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
            action_logprob = None
        else:
            dist = Categorical(action_probs)
            action = dist.sample()           # 확률에 따라 행동 샘플링
            action_logprob = dist.log_prob(action)
            
        state_val = self.critic(state)
        
        if deterministic:
            return action.detach().cpu().numpy(), None, state_val.detach().cpu().numpy()
        else:
            return action.detach().cpu().numpy(), action_logprob.detach().cpu().numpy(), state_val.detach().cpu().numpy()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()    # 탐험(Exploration)을 장려하기 위한 엔트로피
        state_values = self.critic(state)
        
        return action_logprobs, state_values.squeeze(-1), dist_entropy
    
# PPO 클래스
class PPO:
    def __init__(self, state_dim, action_dim, lr, gamma, epochs, eps_clip):
        self.gamma = gamma          # 할인율 (Discount Factor)
        self.eps_clip = eps_clip    # Clipping 범위 (보통 0.1 ~ 0.2)
        self.epochs = epochs        # 업데이트 반복 횟수

        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # 수식의 pi_old를 계산하기 위해 과거 정책을 저장할 복사본
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # 1. 버퍼에서 넘파이 배열로 변환 (shape: steps, num_envs)
        rewards_arr = np.array(memory.rewards)
        dones_arr = np.array(memory.dones)

        # 2. 누적 보상(Return) 계산 (환경별로 병렬 계산)
        returns = []
        # num_envs 크기의 배열로 초기화
        discounted_reward = np.zeros(rewards_arr.shape[1]) 
        
        for reward, done in zip(reversed(rewards_arr), reversed(dones_arr)):
            # 게임이 끝난 환경(done==True)은 할인된 보상을 0으로 리셋
            discounted_reward = reward + (self.gamma * discounted_reward * (1 - done.astype(int)))
            returns.insert(0, discounted_reward.copy())
            
        # 3. 모델에 넣기 위해 모든 데이터를 1차원으로 쫙 펴기(Flatten) 및 텐서 변환
        # states: (steps, num_envs, state_dim) -> (steps * num_envs, state_dim)
        states = torch.FloatTensor(np.array(memory.states).reshape(-1, np.array(memory.states).shape[-1]))
        actions = torch.LongTensor(np.array(memory.actions).reshape(-1))
        logprobs = torch.FloatTensor(np.array(memory.logprobs).reshape(-1))
        returns = torch.FloatTensor(np.array(returns).reshape(-1))

        returns = (returns - returns.mean()) / (returns.std() + 1e-7) # 정규화

        # 평균 Loss를 계산하기 위한 변수 초기화
        avg_actor_loss = 0
        avg_critic_loss = 0

        # 3. K Epoch 동안 네트워크 업데이트 (클리핑 로직 적용)
        for _ in range(self.epochs):
            # 현재 정책이 평가한 로그 확률과 상태 가치
            curr_logprobs, state_values, entropy = self.policy.evaluate(states, actions)
            
            # Advantage (어떤 행동이 평균적인 가치보다 얼마나 더 좋았는가)
            advantages = returns - state_values.detach()

            # 비율 (Ratio) = pi_theta / pi_old
            ratios = torch.exp(curr_logprobs - logprobs)

            # Surrogate Loss 계산 (Clipping)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # 각 Loss를 스칼라(단일 숫자) 값으로 만들기 위해 .mean()을 취해줍니다.
            actor_loss = -torch.min(surr1, surr2).mean() 
            critic_loss = self.MseLoss(state_values, returns)
            entropy_bonus = entropy.mean()

            # 역전파(Backprop)를 위해 전체 Loss 합산
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_bonus

            # 실수값만 반환해 avg에 추가
            avg_actor_loss += actor_loss.item()
            avg_critic_loss += critic_loss.item()

            # 역전파 및 가중치 업데이트
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # 4. 업데이트가 끝나면 과거 정책(old)을 현재 정책으로 동기화
        self.policy_old.load_state_dict(self.policy.state_dict())
        memory.clear() # On-Policy 알고리즘이므로 학습된 데이터는 즉시 버림

        # 평균 Loss 반환
        return avg_actor_loss / self.epochs, avg_critic_loss / self.epochs

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = [] # 게임 종료 여부 (Terminal state)

    # 버퍼 지우는 메서드
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.dones.clear()

