import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from PPO_code import PPO, RolloutBuffer
from snake_code import SnakeGame
from datetime import datetime
from multi_env_processor import MultiEnvWrapper
import numpy as np

def train(policy_path = "", episodes = 10000):
    # TensorBoard 로그를 저장할 디렉토리 지정
    writer = SummaryWriter('runs/snake_ppo_experiment_1')

    # TODO: 동시에 돌릴 게임 개수 (CPU 코어 수에 맞춰야 함)
    num_envs = 8 
    multi_env = MultiEnvWrapper(num_envs)

    # 초기화
    states = multi_env.reset()
    state_dim = 13  # 예: 뱀의 상태 데이터 크기
    action_dim = 3  # 예: 직진, 좌, 우
    ppo_agent = PPO(state_dim, action_dim, lr=0.0003, gamma=0.99, epochs=4, eps_clip=0.2)
    memory = RolloutBuffer()
    episode_rewards = np.zeros(num_envs)  # 한 에피소드의 총 보상
    step_counts = np.zeros(num_envs)      # 생존 시간(스텝)

    max_episodes = episodes # 최대 돌리는 에피소드 수
    update_timestep = 2000 # 2000 스텝마다 모아서 한 번에 학습
    time_step = 0
    global_episodes = 0 # 전체 완료된 에피소드 수
    save_interval = 500 # 모델 저장 주기

    print("🚀 PPO 에이전트 학습을 시작합니다...")
    print("TensorBoard 실행: tensorboard --logdir=runs")
    print("종료하려면 Ctrl+C를 누르세요. (현재까지의 모델이 자동 저장됩니다)")

    load_path = policy_path
    
    if load_path != '' and os.path.exists(load_path):
        # 현재 정책에 가중치 로드
        ppo_agent.policy.load_state_dict(torch.load(load_path))
        
        # 과거 정책에도 동일하게 덮어씌우기
        ppo_agent.policy_old.load_state_dict(ppo_agent.policy.state_dict())
        
        # 모델을 학습 모드로 설정
        ppo_agent.policy.train()
        
        print(f"✅ 기존 모델({load_path})을 성공적으로 불러왔습니다. 이어서 학습합니다.")
    elif load_path == '':
        print("처음부터 새로 학습을 시작합니다.")
    else:
        print("⚠️ 불러올 모델이 없습니다. 처음부터 새로 학습을 시작합니다.")

    try:
        while global_episodes < max_episodes:
            time_step += num_envs
            step_counts += 1
            
            # 1. 상태를 PyTorch 텐서로 변환하여 에이전트에 전달
            state_tensor = torch.FloatTensor(states)
            actions, logprobs, _ = ppo_agent.policy.act(state_tensor)
            
            # 2. 선택한 행동으로 환경(게임) 1스텝 진행
            next_states, rewards, dones, infos = multi_env.step(actions)

            # 3. 버퍼에 현재 스텝의 궤적(Trajectory) 배열을 통째로 저장
            memory.states.append(states)
            memory.actions.append(actions)
            memory.logprobs.append(logprobs)
            memory.rewards.append(rewards)
            memory.dones.append(dones)

            for i in range(num_envs):
                episode_rewards[i] += rewards[i]

                # 특정 환경이 게임 오버 되었을 때 해당 환경만 로깅 및 초기화
                if dones[i]:
                    global_episodes += 1
                    
                    # 로깅 (env 대신 multi_env.envs[i]에 접근하여 점수 확인)
                    score = infos[i]['score']
                    writer.add_scalar('Performance/Episode_Reward', episode_rewards[i], global_episodes)
                    writer.add_scalar('Performance/Score (Apples)', score, global_episodes)
                    writer.add_scalar('Performance/Survival_Steps', step_counts[i], global_episodes)

                    # 콘솔 출력 로직
                    if global_episodes % 100 == 0:
                        now = datetime.now()
                        formatted_time = now.strftime('%Y-%m-%d %H:%M:%S')
                        print(f"Episode: {global_episodes:4d} | Score: {score:2d} | Reward: {episode_rewards[i]:.2f} | Steps: {int(step_counts[i]):3d} | Time: {formatted_time}")

                    # 모델 저장 로직
                    if global_episodes % save_interval == 0:
                        torch.save(ppo_agent.policy.state_dict(), f"saved_models/ppo_snake_ep{global_episodes}_score{score}.pth")

                    # 해당 환경의 추적 변수만 초기화 (환경 자체는 MultiEnvWrapper에서 자동 reset됨)
                    episode_rewards[i] = 0
                    step_counts[i] = 0

            # 상태 업데이트 (반드시 루프 밖에서 처리!)
            states = next_states

            # PPO 업데이트 (num_envs 개씩 데이터가 쌓이므로 곱해서 계산)
            if len(memory.states) * num_envs >= update_timestep:
                actor_loss, critic_loss = ppo_agent.update(memory)
                writer.add_scalar('Loss/Actor', actor_loss, time_step)
                writer.add_scalar('Loss/Critic', critic_loss, time_step)

                # 모델을 저장하기 직전의 게임 상태에 대해서 렌더링 수행
                #if episode % save_interval == 0:
                #    env.render()


    except KeyboardInterrupt:
        print("\n🛑 학습이 사용자에 의해 중단되었습니다.")

    finally:
        # 안전한 종료 처리 (강제 종료되더라도 마지막 모델 저장)
        os.makedirs("saved_models", exist_ok=True)
        torch.save(ppo_agent.policy.state_dict(), "saved_models/ppo_snake_final.pth")
        print(f"총 {global_episodes} 에피소드만큼 학습되었습니다.")
        print("현재까지 학습된 모델이 'saved_models/ppo_snake_final.pth'에 저장되었습니다.")
        writer.close()


if __name__ == '__main__':
    # 이미 학습된 정책의 추가 학습을 원한다면 아래 from_scratch 값에 False를 넣고 policy_path에 string에 경로를 적으면 됩니다
    from_scratch = True
    policy_path = '' if from_scratch else 'saved_models/ppo_snake_poo_boo.pth'
    train(policy_path, 30000)