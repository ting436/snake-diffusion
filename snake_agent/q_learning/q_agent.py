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
    
    def play_step(
        self,
        record_snapshot: bool = False,
        step: Optional[int] = None
    ) -> Tuple[np.ndarray, List[int], int, np.ndarray, bool]:
        old_state = self.env.current_state
        action = self._get_action(self.env.current_state)
        self.steps += 1
        if step is None:
            step = self.steps
        if record_snapshot:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            dir_path = os.path.join(dir_path, self.snapshot_path)
            plt.imsave(os.path.join(dir_path, f'snapshot_{step}.jpg'), self.env.game.get_snapshot())
        new_state, reward, done = self.env.step(action)
        return old_state, action, reward, new_state, done

    def train(self, iterations: int, show_plot: bool = False):
        record = 0
        total_score = 0
        plot_scores = []
        plot_mean_scores = []
        self.steps = 0
        for _ in range(iterations):
            old_state, action, reward, new_state, done = self.play_step()
            self._train_short_memory(old_state, action, reward, new_state, done)
            self._remember(old_state, action, reward, new_state, done)

            if done:
                score = self.env.game.score
                self.env.reset()
                self._train_long_memory()

                if score > record:
                    record = score
                    self.save_agent()

                print('Game', self.env.count_games, 'Score', score, 'Record:', record)
                if show_plot:
                    plot_scores.append(score)
                    total_score += score
                    mean_score = total_score / self.env.count_games
                    plot_mean_scores.append(mean_score)
                    plot(plot_scores, plot_mean_scores)
            if self.env.game.score > record:
                self.save_agent()

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
            _, _, _, _, done = agent.play_step(record_snapshot=True)

if __name__ == '__main__':
    game = SnakeGame(w=160, h=120, speed=10000, block_size=8)
    env = GameEnvironment(game)
    agent = QAgent(env, path="model/int_model")
    # agent.train(100000, True)
    play()
        
    # print(f"Episode {episode + 1}: Score = {total_reward}")