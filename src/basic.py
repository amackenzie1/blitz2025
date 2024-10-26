import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPO

# Custom environment
class SimpleEnv(gym.Env):
    def __init__(self, config=None):
        super().__init__()
        # Define action space: continuous action between -1 and 1
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,))
        
        # Define observation space: position and target position
        self.observation_space = Box(low=-10.0, high=10.0, shape=(2,))
        
        # Target position will be randomly set
        self.target_pos = 0
        self.current_pos = 0
        self.max_steps = 100
        self.current_step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Reset environment state with positions within bounds
        self.current_pos = np.random.uniform(-5, 5)  # Starting well within bounds
        self.target_pos = np.random.uniform(-5, 5)   # Target well within bounds
        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        # Return current position and target position as observation
        return np.array([self.current_pos, self.target_pos])

    def step(self, action):
        self.current_step += 1
        
        # Update position based on action and clamp to observation space bounds
        self.current_pos = np.clip(
            self.current_pos + action[0],
            self.observation_space.low[0],
            self.observation_space.high[0]
        )
        
        # Calculate reward based on distance to target
        distance = abs(self.current_pos - self.target_pos)
        reward = -distance  # Negative reward based on distance
        
        # Check if episode should end
        done = self.current_step >= self.max_steps
        truncated = False
        
        return self._get_obs(), reward, done, truncated, {}

# Initialize Ray
ray.init()

# Configure the algorithm
config = {
    "env": SimpleEnv,
    "num_workers": 2,
    "framework": "torch",
    "train_batch_size": 4000,
}

# Create PPO trainer
trainer = PPO(config=config)

# Training loop
for i in range(100):
    result = trainer.train()
    print(f"Training iteration {i}")
    print(f"Mean reward: {result['env_runners']['episode_reward_mean']}")
    print(f"Episode length mean: {result['env_runners']['episode_len_mean']}")
    print(f"Min/Max reward: {result['env_runners']['episode_reward_min']:.2f}/{result['env_runners']['episode_reward_max']:.2f}")

# Cleanup
ray.shutdown()

# Test the trained policy
env = SimpleEnv()
obs, _ = env.reset()

for _ in range(100):
    action = trainer.compute_single_action(obs)
    obs, reward, done, truncated, info = env.step(action)
    print(f"Position: {obs[0]:.2f}, Target: {obs[1]:.2f}, Reward: {reward:.2f}")
    
    if done:
        break
