import torch
import time
from snake_code import SnakeGame
from PPO_code import PPO

def play_saved_model():
    # 1. 뼈대 준비 (학습할 때와 차원이 정확히 일치해야 합니다)
    state_dim = 11
    action_dim = 3
    env = SnakeGame()
    
    # 평가만 할 것이므로 lr, gamma 등 학습용 파라미터는 아무 값이나 넣어도 무방합니다.
    ppo_agent = PPO(state_dim, action_dim, lr=0.001, gamma=0.99, epochs=1, eps_clip=0.2)
    
    # 2. 가중치 불러오기 (경로 지정)
    model_path = "saved_models/ppo_snake_final.pth"
    
    # torch.load로 딕셔너리를 읽고, load_state_dict로 모델에 덮어씌웁니다.
    ppo_agent.policy.load_state_dict(torch.load(model_path))
    
    # 3. 평가 모드 전환 (Best Practice)
    ppo_agent.policy.eval()
    
    print(f"🎉 {model_path} 로드 완료! 플레이를 시작합니다.")

    state = env.reset()
    
    # 4. 연산량 절약을 위한 no_grad (기울기 추적 비활성화)
    with torch.no_grad():
        while True:
            env.render()
            
            # 뱀이 너무 빨리 움직이면 사람 눈에 안 보이므로 의도적인 지연 추가
            time.sleep(0.05) 
            
            state_tensor = torch.FloatTensor(state)
            
            # 행동 결정 (Actor 신경망만 사용됨)
            action, _, _ = ppo_agent.policy.act(state_tensor)
            
            state, reward, done = env.step(action)
            
            if done:
                print(f"게임 오버! 뱀 길이(점수): {env.score}")
                break

if __name__ == '__main__':
    play_saved_model()