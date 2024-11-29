from collections import deque
from typing import Union, Tuple, List, Optional
import random
import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from models import Linear_QNet, QTrainer
from game_environment import GameEnvironment
from snake_agent.game import SnakeGame, Direction

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

def plot(scores, mean_scores):
    # display.clear_output(wait=True)
    # display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

class QAgent:
    def __init__(self, env: GameEnvironment, path: str = "game_model", snapshot_path: str = "snapshots"):
        self.save_path = path
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(len(env.current_state), 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.env = env
        self.steps = 0
        self.snapshot_path = snapshot_path
        self.recorded_actions = []
    
    def _remember(
        self,
        state: np.ndarray,
        action: Union[np.ndarray, List[int]],
        reward: Union[np.ndarray, float],
        next_state: np.ndarray,
        done: Union[np.ndarray, bool]
    ):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def _train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def _train_short_memory(
        self,
        state: np.ndarray,
        action: Union[np.ndarray, List[int]],
        reward: Union[np.ndarray, float],
        next_state: np.ndarray,
        done: Union[np.ndarray, bool]
    ):
        self.trainer.train_step(state, action, reward, next_state, done)

    def _get_action(self, state: np.ndarray) -> List[int]:
        final_move = [0,0,0]
        self.epsilon = 80 - self.env.count_games
        state = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state)
        move = torch.argmax(prediction).item()
        # return move
        final_move[move] = 1
        return final_move
    
    def _save_snapshot(self, step: int):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path = os.path.join(dir_path, self.snapshot_path)
        plt.imsave(os.path.join(dir_path, f'{step}.jpg'), self.env.game.get_snapshot().transpose(1,0,2))
    
    def _save_actions(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(dir_path, "actions")
        with open(file_path, mode="w") as file:
            file.write("\n".join([str(action) for action in self.recorded_actions]))
    
    def play_step(
        self,
        record: bool = False,
        step: Optional[int] = None
    ) -> Tuple[np.ndarray, List[int], int, np.ndarray, bool]:
        old_state = self.env.current_state
        action = self._get_action(self.env.current_state)
        self.steps += 1
        if step is None:
            step = self.steps
        new_state, reward, done = self.env.step(action)
        if record:
            self._save_snapshot(step)
            if np.array_equal(action, [0, 1, 0]):
                self.recorded_actions.append(1)
            elif np.array_equal(action, [0, 0, 1]):
                self.recorded_actions.append(2)
            else:
                self.recorded_actions.append(4)
        return old_state, action, reward, new_state, done
    
    # enter - 3
    # change dir - 1 and 2
    # 4 - idle direction
    # 0 - no action

    def train(self, iterations: int, show_plot: bool = False, record: bool = False):
        top_result = 0
        total_score = 0
        plot_scores = []
        plot_mean_scores = []
        self.steps = 0
        self.recorded_actions = []
        for _ in range(iterations):
            old_state, action, reward, new_state, done = self.play_step(
                record=record and self.env.count_games >= 100
            )
            self._train_short_memory(old_state, action, reward, new_state, done)
            self._remember(old_state, action, reward, new_state, done)

            if done:
                score = self.env.game.score
                self.env.reset()
                if record and self.env.count_games > 100:
                    self.steps += 1
                    self.recorded_actions.append(3)
                    self._save_snapshot(self.steps)
                self._save_actions()

                self._train_long_memory()

                if score > top_result:
                    top_result = score
                    self.save_agent()

                print('Game', self.env.count_games, 'Score', score, 'Record:', top_result)
                if show_plot:
                    plot_scores.append(score)
                    total_score += score
                    mean_score = total_score / self.env.count_games
                    plot_mean_scores.append(mean_score)
                    plot(plot_scores, plot_mean_scores)
            if self.env.game.score > top_result:
                self.save_agent()
        self._save_actions()

    def save_agent(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.model.save(os.path.join(dir_path,self.save_path))

    def load_agent(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.model.load_state_dict(torch.load(os.path.join(dir_path,self.save_path)))

def play():
    agent.load_agent()
    for episode in range(10):
        env.reset()
        done = False
        total_reward = 0
        up_count = 0
        while not done:
            _, _, _, _, done = agent.play_step()

if __name__ == '__main__':
    game = SnakeGame(w=60, h=60, speed=10, block_size=5)
    env = GameEnvironment(game)
    agent = QAgent(env, path="model-internal/60_model")
    # agent.train(100000, False, True)
    play()  
    # print(f"Episode {episode + 1}: Score = {total_reward}")