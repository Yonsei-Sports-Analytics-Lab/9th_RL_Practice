## 강화학습(RL) 실습 코드
Snake Game(랜덤하게 등장하는 사과를 먹는 게임)을 PPO 알고리즘을 통해 학습시킨 모델로 플레이하기

### **설치 방법**
1. fork 및 clone
2. 가상 환경 설치 및 활성화 (python -m venv .venv & .venv/Scripts/Activate.ps1) (Lecture Session 1 강의자료 참고)
3. requirements.txt 설치 (python -m pip install -r requirements.txt)

### **학습 방법**
1. main.py 가장 아래쪽에 policy_path에 값을 비우면 from scratch 학습(처음부터 학습), 이미 존재하는 정책 파일 경로를 넣으면 해당 정책을 checkpoint로 사용해 추가 학습 진행
2. 1번을 진행한 후 main.py 저장
3. 터미널 2개 생성(둘 다 가상환경 표시 초록색 (.venv) 있는지 확인!!!!) 
4. 첫 번째 터미널에 tensorboard --logdir=runs 실행 (대시보드 생성)
5. 두 번째 터미널에 'python -m main' 실행
6. [웹 대시보드](http://localhost:6006/)로 들어가 학습 진행상황 확인
7. 어느 정도 지난 후(시간 꽤 걸림), saved_models 폴더에 'ppo_snake_ep~.pth' & 'ppo_snake_final.pth' 파일이 생겼는지 확인

### **플레이 방법**
1. play.py 속 model.path에서 확인하고 싶은 모델의 파일 경로 적기
2. 'python -m play' 실행

### **과제**
1. 총 학습 episode 수, reward 함수(snake_code.py의 step 함수에 존재) 등의 하이퍼 파라미터를 수정하면서 학습을 진행한다.
2. 가장 평균 성능이 좋았던 모델을 github에 commit & push한 후 pull request를 진행한다.
3. 대시보드 스크린샷과 1번에 설정했던 하이퍼 파라미터를 담은 사진, play.py 실행 결과(영상 또는 점수를 담은 사진)를 포함한 보고서를 작성해 yonseiysal@gmail.com으로 제출
4. 제출 양식: 제목은 [Lecture1]8/9기_XXX, 파일은 pdf 형식으로 제출
