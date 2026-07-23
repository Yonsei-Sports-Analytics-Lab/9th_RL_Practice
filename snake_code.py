import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
from collections import deque
import os

# 학습 시 아래 주석 빼야함
# os.environ["SDL_VIDEODRIVER"] = "dummy"

# 색상 RGB 정의
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)      # 뱀의 테두리 색상
BLUE2 = (0, 100, 255)    # 뱀의 안쪽 색상
BLACK = (0, 0, 0)        # 배경 색상

pygame.init()

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# 게임 설정값
BLOCK_SIZE = 20
SPEED = 40 # 학습 화면을 볼 때의 속도

class SnakeGame:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = None
        pygame.display.set_caption('Snake RL')
        self.clock = None
        self.reset()

    def reset(self):
        # 게임 초기 상태 설정
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        
        # 보상 설계를 위한 변수 초기화
        self.frame_iteration = 0
        self.prev_distance = abs(self.food.x - self.head.x) + abs(self.food.y - self.head.y)
        
        return self.get_state()

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def step(self, action):
        # pygame 이벤트 처리 루프
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # 우측 상단의 X 버튼을 눌렀을 때 안전하게 종료되도록 처리
                pygame.quit()
                quit()

        self.frame_iteration += 1

        if len(self.snake) >= 20:
            # 행동에 따른 다음 머리 위치 예측을 위해 미리 계산
            clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
            idx = clock_wise.index(self.direction)
            if action == 0: chosen_dir = clock_wise[idx]
            elif action == 1: chosen_dir = clock_wise[(idx + 1) % 4]
            else: chosen_dir = clock_wise[(idx - 1) % 4]
            
            next_head = self._get_next_point(self.head, chosen_dir)
            
            # 이동할 공간의 생존 가능 칸 수 확인
            accessible_space = self._get_accessible_space(next_head)
            
            # 뒤주에 갇힌 것인지 확인
            if accessible_space < len(self.snake):
                game_over = True
                reward = -1.5 # 죽음보다 조금 약하거나 같은 수준의 강력한 패널티
                return self.get_state(), reward, game_over
        
        # 에이전트의 행동(Action)에 따라 이동
        self._move(action)
        self.snake.insert(0, self.head)
        
        reward = -0.01 # 스텝 패널티 (지연 방지)
        game_over = False

        # 게임 종료 조건 확인 (충돌 또는 무한 루프 아사)
        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            if self._is_collision():
                reward = -1.5
            return self.get_state(), reward, game_over

        # 조밀한 보상 (Dense Reward) 적용
        curr_distance = abs(self.food.x - self.head.x) + abs(self.food.y - self.head.y)
        if curr_distance < self.prev_distance:
            reward += 0.01
        else:
            reward -= 0.01
        self.prev_distance = curr_distance

        # 사과 획득 확인
        if self.head == self.food:
            self.score += 1
            reward += 1  # 기존 보상에 추가
            self._place_food()
            self.frame_iteration = 0 # 굶주림 초기화
            # 사과를 먹었으므로 꼬리를 자르지 않음 (길어짐)
            
            # 새 사과가 배치되었으므로 prev_distance 다시 계산
            self.prev_distance = abs(self.food.x - self.head.x) + abs(self.food.y - self.head.y)
        else:
            self.snake.pop() # 사과를 못 먹었으면 꼬리 한 칸 축소시켜 이동 구현
            
        return self.get_state(), reward, game_over

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # 벽에 부딪힘
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # 자기 몸통에 부딪힘
        if pt in self.snake[1:]:
            return True
        return False

    def _get_accessible_space(self, start_pt):
        # start_pt에서 출발해 벽과 자기 몸을 피해 갈 수 있는 빈 칸의 개수를 BFS로 측정
        
        # 이미 뱀 몸통이 차지하고 있는 좌표들을 Set으로 만들어두면 검색이 O(1)로 빨라집니다.
        snake_body_set = set(self.snake)
        
        # 시작 지점이 이미 벽이거나 몸통이면 갈 수 있는 칸은 0개
        if (start_pt.x < 0 or start_pt.x >= self.w or 
            start_pt.y < 0 or start_pt.y >= self.h or 
            start_pt in snake_body_set):
            return 0

        visited = set()
        queue = deque([start_pt])
        visited.add(start_pt)
        
        space_count = 0
        
        # BFS 탐색
        while queue:
            current = queue.popleft()
            space_count += 1
            
            # 상하좌우 인접 칸 확인
            neighbors = [
                Point(current.x + BLOCK_SIZE, current.y),
                Point(current.x - BLOCK_SIZE, current.y),
                Point(current.x, current.y + BLOCK_SIZE),
                Point(current.x, current.y - BLOCK_SIZE)
            ]
            
            for next_pt in neighbors:
                # 맵 내부이고, 뱀 몸통이 아니며, 아직 방문하지 않았다면 큐에 추가
                if (0 <= next_pt.x < self.w and 
                    0 <= next_pt.y < self.h and 
                    next_pt not in snake_body_set and 
                    next_pt not in visited):
                    
                    visited.add(next_pt)
                    queue.append(next_pt)
                    
        return space_count
    
    def _move(self, action):
        # action [직진, 우회전, 좌회전]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if action == 0:
            new_dir = clock_wise[idx] # 직진
        elif action == 1:
            new_dir = clock_wise[(idx + 1) % 4] # 우회전
        else:
            new_dir = clock_wise[(idx - 1) % 4] # 좌회전

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT: x += BLOCK_SIZE
        elif self.direction == Direction.LEFT: x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN: y += BLOCK_SIZE
        elif self.direction == Direction.UP: y -= BLOCK_SIZE

        self.head = Point(x, y)

    def get_state(self):
        head = self.snake[0]
        # 머리 기준 상하좌우 + 대각선 한 칸의 좌표
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        point_ul = Point(head.x - BLOCK_SIZE, head.y - BLOCK_SIZE)
        point_ur = Point(head.x + BLOCK_SIZE, head.y - BLOCK_SIZE)
        point_dl = Point(head.x - BLOCK_SIZE, head.y + BLOCK_SIZE)
        point_dr = Point(head.x + BLOCK_SIZE, head.y + BLOCK_SIZE)
        
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        # 현재 방향 기준으로 [직진, 우회전, 좌회전]에 해당하는 다음 칸 좌표
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        dir_straight = clock_wise[idx]
        dir_right = clock_wise[(idx + 1) % 4]
        dir_left = clock_wise[(idx - 1) % 4]
        
        next_pts = {
            'straight': self._get_next_point(head, dir_straight),
            'right': self._get_next_point(head, dir_right),
            'left': self._get_next_point(head, dir_left)
        }

        # 앞이 벽이나 자기 몸인지 확인 (직진, 우회전, 좌회전, 대각선)
        straight_collision = (dir_r and self._is_collision(point_r)) or (dir_l and self._is_collision(point_l)) or (dir_u and self._is_collision(point_u)) or (dir_d and self._is_collision(point_d))
        right_collision = (dir_u and self._is_collision(point_r)) or (dir_d and self._is_collision(point_l)) or (dir_l and self._is_collision(point_u)) or (dir_r and self._is_collision(point_d))
        left_collision = (dir_d and self._is_collision(point_r)) or (dir_u and self._is_collision(point_l)) or (dir_r and self._is_collision(point_u)) or (dir_l and self._is_collision(point_d))
        straight_right_collision = (dir_u and self._is_collision(point_ur)) or (dir_d and self._is_collision(point_dl)) or (dir_l and self._is_collision(point_ul)) or (dir_r and self._is_collision(point_dr))
        straight_left_collision = (dir_d and self._is_collision(point_dr)) or (dir_u and self._is_collision(point_ul)) or (dir_r and self._is_collision(point_ur)) or (dir_l and self._is_collision(point_dl))

        # 각 방향으로 갔을 때의 생존 공간 계산 (전체 맵 칸 수로 나눠서 정규화)
        total_cells = (self.w // BLOCK_SIZE) * (self.h // BLOCK_SIZE)
        space_straight = self._get_accessible_space(next_pts['straight']) / total_cells
        space_right = self._get_accessible_space(next_pts['right']) / total_cells if (straight_right_collision or straight_collision) else space_straight
        space_left = self._get_accessible_space(next_pts['left']) / total_cells if (straight_left_collision or straight_collision) else space_straight

        state = [
            # 위험 감지 (직진, 우회전, 좌회전)
            straight_collision,
            right_collision,
            left_collision,
            
            # 이동 방향
            dir_l, dir_r, dir_u, dir_d,
            
            # 사과 위치
            self.food.x < self.head.x,  # Food left
            self.food.x > self.head.x,  # Food right
            self.food.y < self.head.y,  # Food up
            self.food.y > self.head.y,   # Food down

            # 공간 감지 센서
            space_straight,
            space_right,
            space_left
        ]
        return np.array(state, dtype=float)

    def _get_next_point(self, current_head, direction):
        x, y = current_head.x, current_head.y
        if direction == Direction.RIGHT: x += BLOCK_SIZE
        elif direction == Direction.LEFT: x -= BLOCK_SIZE
        elif direction == Direction.DOWN: y += BLOCK_SIZE
        elif direction == Direction.UP: y -= BLOCK_SIZE
        return Point(x, y)

    def render(self):
        # init에서 창을 생성하지 않았기 때문에 여기서 생성
        if self.display is None:
            pygame.init()
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake RL')
            self.clock = pygame.time.Clock()

        # 배경을 검은색으로 지우기
        self.display.fill(BLACK)
        
        # 뱀 그리기
        for pt in self.snake:
            # 뱀의 몸통 (바깥쪽 꽉 찬 사각형)
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            # 뱀의 몸통 안쪽 (입체감을 위해 살짝 작은 사각형을 덧그림)
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
            
        # 사과 그리기
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # 좌측 상단에 현재 점수 표시
        font = pygame.font.SysFont('arial', 25)
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        
        # 화면 업데이트 및 재생 속도 조절
        pygame.display.flip()
        self.clock.tick(SPEED) # __init__ 외부에서 정의한 SPEED(예: 40)에 맞춰 프레임 고정    
