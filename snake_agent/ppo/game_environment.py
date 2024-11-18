import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple
from snake_agent.game import SnakeGame, Point, Direction
import cv2
from PIL import Image

class GameEnvironment(gym.Env):
    """
    Custom Game Environment that follows gym interface.
    This provides the interface between your game and the RL agent.
    """
    def __init__(self, game: SnakeGame, frame_size=(64, 48)):
        super().__init__()
        
        # Define action and observation spaces
        self.frame_size = frame_size
        
        # Example: Discrete actions (e.g., 0=right, 1=left, 2=up, 3=down)
        self.action_space = spaces.Discrete(3)

        self.game = game
        game.reset()
        
        # Observation space: Game frame (RGB)
        # Shape: (3, height, width) - 3 channels for RGB
        # self.observation_space = spaces.Box(
        #     low=0,
        #     high=255,
        #     shape=(3, *frame_size),
        #     dtype=np.uint8
        # )
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(11,),
            dtype=np.int32
        )
        # self.reward_range = (-1000, 1000)
        
        # Initialize state from game
        self.current_state = self._get_state()
        self.steps_taken = 0
        self.count_games = 0
        self.max_steps = 1000  # Maximum steps per episode

    def reset(self, seed=None):
        """
        Reset the environment to initial state.
        Required by gym interface.
        """
        super().reset(seed=seed)
        self.game.reset()
        # Reset game state
        self.steps_taken = 0
        self.count_games += 1
        
        # Initialize state from game
        self.current_state = self._get_state()
        
        # Return initial observation and info
        return self.current_state, {}

    def step(self, action: int):
        """
        Execute one time step within the environment.
        Required by gym interface.
        
        Args:
            action: int - The action to take (0-3)
        Returns:
            observation: Current game state
            reward: Reward for the action
            terminated: Whether the episode is done
            truncated: Whether the episode was artificially terminated
            info: Additional information
        """
        self.steps_taken += 1
        
        # Execute action in game
        reward, terminated = self._take_action(action)
        self.current_state = self._get_state()
        
        # Check if episode is done
        truncated = self.steps_taken >= self.max_steps
        
        # Return step information
        return self.current_state, reward, terminated, truncated, { "count_games": self.count_games }
    
    def _get_state(self) -> np.ndarray:
        # return np.array([self.game.head.x, self.game.head.y, self.game.food.x, self.game.food.y, self.game.head.distance(self.game.food)])
        # state = Image.fromarray(self.game.get_snapshot().transpose(1,0,2))
        
        # state = np.array(state.resize(self.frame_size, Image.Resampling.LANCZOS))
        # return state.transpose(2,1,0)

        head = self.game.head
        point_l = Point(head.x - self.game.block_size, head.y)
        point_r = Point(head.x + self.game.block_size, head.y)
        point_u = Point(head.x, head.y - self.game.block_size)
        point_d = Point(head.x, head.y + self.game.block_size)
        
        dir_l = self.game.direction == Direction.LEFT
        dir_r = self.game.direction == Direction.RIGHT
        dir_u = self.game.direction == Direction.UP
        dir_d = self.game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and self.game.is_collision(point_r)) or 
            (dir_l and self.game.is_collision(point_l)) or 
            (dir_u and self.game.is_collision(point_u)) or 
            (dir_d and self.game.is_collision(point_d)),

            # Danger right
            (dir_u and self.game.is_collision(point_r)) or 
            (dir_d and self.game.is_collision(point_l)) or 
            (dir_l and self.game.is_collision(point_u)) or 
            (dir_r and self.game.is_collision(point_d)),

            # Danger left
            (dir_d and self.game.is_collision(point_r)) or 
            (dir_u and self.game.is_collision(point_l)) or 
            (dir_r and self.game.is_collision(point_u)) or 
            (dir_l and self.game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            self.game.food.x < head.x,  # food left
            self.game.food.x > head.x,  # food right
            self.game.food.y < head.y,  # food up
            self.game.food.y > head.y  # food down
        ]

        return np.array(state, dtype=int)

    def _take_action(self, action: int) -> Tuple[int, bool]:
        prev_score = self.game.score
        prev_direction = self.game.direction
        prev_head = self.game.head
        game_over, score = self.game.play_step(action)
        direction = self.game.direction
        head = self.game.head
        reward = 0
        if game_over:
            reward = -10
        elif score > prev_score:
            reward = 10
        # elif head.distance(self.game.food) < prev_head.distance(self.game.food):
        #     reward = 0.01
        # elif head.distance(self.game.food) > prev_head.distance(self.game.food):
        #     reward += -0.01
        if self.steps_taken >= 100 * len(self.game.snake):
            game_over = True
            reward = -10
        return reward, game_over

if __name__ == '__main__':
    game = SnakeGame()
    env = GameEnvironment(game)
    state = env._get_state()
    print(state.shape)
    print(env.observation_space)
    print(env.action_space)