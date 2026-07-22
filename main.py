import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from PPO_code import PPO, RolloutBuffer
from snake_code import SnakeGame

def train(policy_path = "", episodes = 10000):
    # TensorBoard 로그를 저장할 디렉토리 지정
    writer = SummaryWriter('runs/snake_ppo_experiment_1')

    # 초기화
    env = SnakeGame()
    state_dim = 11  # 예: 뱀의 상태 데이터 크기
    action_dim = 3  # 예: 직진, 좌, 우
    ppo_agent = PPO(state_dim, action_dim, lr=0.0003, gamma=0.99, epochs=4, eps_clip=0.2)
    memory = RolloutBuffer()

    max_episodes = episodes # 최대 돌리는 에피소드 수
    update_timestep = 2000 # 2000 스텝마다 모아서 한 번에 학습
    time_step = 0
    save_interval = 500 # 모델 저장 주기

    print("🚀 PPO 에이전트 학습을 시작합니다...")
    print("TensorBoard 실행: tensorboard --logdir=runs")
    print("종료하려면 Ctrl+C를 누르세요. (현재까지의 모델이 자동 저장됩니다)")

    load_path = policy_path
    
    if os.path.exists(load_path):
        # 현재 정책에 가중치 로드
        ppo_agent.policy.load_state_dict(torch.load(load_path))
        
        # 과거 정책에도 동일하게 덮어씌우기
        ppo_agent.policy_old.load_state_dict(ppo_agent.policy.state_dict())
        
        # 모델을 학습 모드로 설정
        ppo_agent.policy.train()
        
        print(f"✅ 기존 모델({load_path})을 성공적으로 불러왔습니다. 이어서 학습합니다.")
    else:
        print("⚠️ 불러올 모델이 없습니다. 처음부터 새로 학습을 시작합니다.")

    try:
        for episode in range(1, max_episodes + 1): # 1만 판 반복
            state = env.reset()
            episode_reward = 0  # 한 에피소드의 총 보상
            step_count = 0      # 생존 시간(스텝)

            while True:
                time_step += 1
                step_count += 1
                
                # 1. 상태를 PyTorch 텐서로 변환하여 에이전트에 전달
                state_tensor = torch.FloatTensor(state)
                action, logprob, _ = ppo_agent.policy.act(state_tensor)
                
                # 2. 선택한 행동으로 환경(게임) 1스텝 진행
                next_state, reward, done = env.step(action)

                # 3. 버퍼에 현재 스텝의 궤적(Trajectory) 저장
                memory.states.append(state)
                memory.actions.append(action)
                memory.logprobs.append(logprob)
                memory.rewards.append(reward)
                memory.dones.append(done)
                            
                episode_reward += reward
                state = next_state

                if time_step % update_timestep == 0:
                    actor_loss, critic_loss = ppo_agent.update(memory)
                    writer.add_scalar('Loss/Actor', actor_loss, time_step)
                    writer.add_scalar('Loss/Critic', critic_loss, time_step)
                
                # 게임 오버 시 루프 탈출
                if done:
                    break
            
            writer.add_scalar('Performance/Episode_Reward', episode_reward, episode)
            writer.add_scalar('Performance/Score (Apples)', env.score, episode)
            writer.add_scalar('Performance/Survival_Steps', step_count, episode)

            if episode % 50 == 0:
                print(f"Episode: {episode:4d} | Score: {env.score:2d} | Reward: {episode_reward:6.2f} | Steps: {step_count:3d}")

            if episode % save_interval == 0:
                torch.save(ppo_agent.policy.state_dict(), f"saved_models/ppo_snake_ep{episode}.pth")
                
    except KeyboardInterrupt:
        print("\n🛑 학습이 사용자에 의해 중단되었습니다.")

    finally:
        # 안전한 종료 처리 (강제 종료되더라도 마지막 모델 저장)
        os.makedirs("saved_models", exist_ok=True)
        torch.save(ppo_agent.policy.state_dict(), "saved_models/ppo_snake_final.pth")
        print("현재까지 학습된 모델이 'saved_models/ppo_snake_final.pth'에 저장되었습니다.")
        writer.close()


if __name__ == '__main__':
    # 이미 학습된 정책의 추가 학습을 원한다면 아래 string에 경로를 적으면 됩니다
    policy_path = 'saved_models/260722_0.pth'
    train(policy_path)