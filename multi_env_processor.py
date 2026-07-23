import numpy as np
import multiprocessing as mp
from snake_code import SnakeGame

def worker(remote, parent_remote):
    """
    독립된 프로세스에서 돌아가는 게임 워커입니다.
    메인 프로세스(신경망)로부터 명령을 받아 게임을 진행하고 결과를 반환합니다.
    """
    parent_remote.close()
    env = SnakeGame()
    
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                ns, r, d = env.step(data)
                info = {'score': env.score}
                
                # 게임이 끝났다면 자동으로 리셋하여 다음 상태 전달
                if d:
                    ns = env.reset()
                    
                remote.send((ns, r, d, info))
            elif cmd == 'reset':
                ns = env.reset()
                remote.send(ns)
            elif cmd == 'close':
                remote.close()
                break
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        # 사용자가 Ctrl+C로 종료할 때 워커 프로세스에서 불필요한 에러 메시지가 뜨지 않도록 처리
        pass
    except EOFError:
        pass

class MultiEnvWrapper:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        
        # 데몬 프로세스로 워커들을 생성하여 백그라운드에서 실행
        self.ps = [mp.Process(target=worker, args=(work_remote, remote))
                   for (work_remote, remote) in zip(self.work_remotes, self.remotes)]
        
        for p in self.ps:
            p.daemon = True # 메인 프로세스 종료 시 워커들도 함께 종료되도록 설정
            p.start()
            
        for remote in self.work_remotes:
            remote.close()

    def reset(self):
        # 모든 워커에게 리셋 명령 전송
        for remote in self.remotes:
            remote.send(('reset', None))
            
        # 모든 워커로부터 리셋된 첫 상태(State) 수신
        results = [remote.recv() for remote in self.remotes]
        return np.array(results)

    def step(self, actions):
        # 각 워커에게 부여된 행동(Action)을 전송하여 게임 1스텝 진행
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
            
        # 모든 워커의 스텝 계산이 끝날 때까지 기다린 후 결과 수신 (여기서 병렬 처리의 진가가 발휘됨)
        results = [remote.recv() for remote in self.remotes]
        
        next_states, rewards, dones, infos = zip(*results)
        return np.array(next_states), np.array(rewards), np.array(dones), list(infos)

    def close(self):
        # 모든 워커에게 종료 명령 전송
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()