import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn

from snake_agent.game import SnakeGame, Direction
from game_environment import GameEnvironment

class GameCNN(BaseFeaturesExtractor):
    """
    CNN feature extractor for the game environment following Mnih et al. (2015)
    architecture but adapted for the 160x120 input resolution.
    """
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        in_channels: int,
        features_dim: int = 512
    ):
        super().__init__(observation_space, features_dim)
        
        # Initialize CNN layers
        self.cnn = nn.Sequential(
            # First conv layer
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            
            # Second conv layer
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            
            # Third conv layer
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            
            # Flatten layer
            nn.Flatten()
        )
        
        # Calculate the size of flattened features
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, *observation_space.shape)).shape[1]
        
        # Final fully connected layer
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Process the observations through the CNN.
        
        Args:
            observations: Tensor of shape (batch_size, channels, height, width)
        Returns:
            Tensor of shape (batch_size, features_dim)
        """
        return self.linear(self.cnn(observations))

class GameAgent:
    """
    Game agent using PPO algorithm with CNN feature extraction.
    """
    def __init__(self, env, save_path="./game_model"):
        self.env = env
        self.save_path = save_path
        
        # Create PPO model with custom CNN feature extractor
        self.model = PPO(
            "MlpPolicy",
            # "CnnPolicy",
            env,
            # policy_kwargs=dict(
            #     features_extractor_class=GameCNN,
            #     features_extractor_kwargs=dict(features_dim=512, in_channels=3),
            # ),
            # learning_rate=2.5e-4,
            learning_rate=1e-3,
            n_steps=2048,
            batch_size=32,
            n_epochs=4,
            gamma=0.9,
            verbose=1
            # gae_lambda=0.95,
            # clip_range=0.2
        )

    def train(self, total_timesteps=1000000):
        """Train the agent for the specified number of timesteps."""
        self.model.learn(total_timesteps=total_timesteps)
        self.model.save(self.save_path)

    def load(self):
        """Load a pre-trained model."""
        self.model = PPO.load(self.save_path)

    def predict(self, observation):
        """
        Make a prediction for a given observation.
        
        Args:
            observation: Game state observation
        Returns:
            action: Predicted action
            state: Internal state (used in recurrent policies)
        """
        action, state = self.model.predict(observation, deterministic=True)
        return action, state
    
if __name__ == '__main__':
    game = SnakeGame(speed=1000)
    env = GameEnvironment(game, frame_size=(64, 48))
    # cnn = GameCNN(observation_space=gym.spaces.Box(
    #     low=0,
    #     high=255,
    #     shape=(3, 64, 48),
    #     dtype=np.uint8
    # ), in_channels=3, features_dim=512)
    # res = cnn.forward(env.current_state)
    # print(res.shape)

    agent = GameAgent(env)
    agent.train(2600)
    # agent.load()

    for episode in range(10):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        up_count = 0
        while not done:
            action, _ = agent.predict(obs)

            # print(Direction(action).name)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
        
        print(f"Episode {episode + 1}: Score = {total_reward}")