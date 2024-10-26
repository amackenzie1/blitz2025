import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torch.cuda

SIZE = 8
# Custom CNN Model
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Projection shortcut if channels change
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return nn.functional.relu(out)

class MazeCNN(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        assert num_outputs == 4, "This model only works for 4 actions"
        
        input_size = obs_space.shape[0]  # Should be 10 for SIZE=10
        
        # Increased base channels but fewer pooling operations
        self.channels = [2, 64, 128, 256, 512]
        
        # Initial conv layer
        self.conv1 = nn.Conv2d(self.channels[0], self.channels[1], 3, padding=1)  # Smaller kernel
        self.bn1 = nn.BatchNorm2d(self.channels[1])
        
        # Create ResNet blocks
        self.res_blocks = nn.ModuleList([
            ResBlock(self.channels[1], self.channels[1]),
            ResBlock(self.channels[1], self.channels[2]),
            ResBlock(self.channels[2], self.channels[2]),
            ResBlock(self.channels[2], self.channels[3]),
            ResBlock(self.channels[3], self.channels[3]),
            ResBlock(self.channels[3], self.channels[4]),
            ResBlock(self.channels[4], self.channels[4]),
            ResBlock(self.channels[4], self.channels[4])
        ])
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2)
        
        # Calculate final feature map size
        # For 10x10 input, we can only do 2 pooling operations (10->5->2)
        num_pools = 2  
        final_size = input_size // (2 ** num_pools)  # Should be 2 for 10x10 input
        
        # Adjust FC layer sizes based on final feature map
        flat_size = self.channels[-1] * final_size * final_size
        self.fc1 = nn.Linear(flat_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_outputs)
        self.value_fc = nn.Linear(512, 1)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()
        x = x.permute(0, 3, 1, 2)
        
        # Initial conv
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        
        # ResNet blocks with reduced pooling
        for i, block in enumerate(self.res_blocks):
            x = block(x)
            if i in [1, 3]:  # Only pool twice
                x = self.pool(x)
        
        x = x.reshape(x.size(0), -1)
        
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        
        logits = self.fc3(x)
        self._value_out = self.value_fc(x)
        
        return logits, state

    def value_function(self):
        return self._value_out.squeeze(1)

# Register the custom model
ModelCatalog.register_custom_model("maze_cnn", MazeCNN)

# Custom environment
class MazeEnv(gym.Env):
    _shared_maze = None
    
    def __init__(self, config=None):
        super().__init__()
        self.size = SIZE
        self.action_space = Discrete(4)
        # Update observation space to have 2 channels
        self.observation_space = Box(low=0, high=1, shape=(SIZE, SIZE, 2))
        
        self.start_pos = (0, 0)
        self.goal_pos = (SIZE-2, SIZE-2)
        
        # Initialize maze first
        if MazeEnv._shared_maze is None:
            print("Generating maze (should only see this once)...")
            MazeEnv._shared_maze = np.ones((self.size, self.size))  # Initialize with walls
            self.maze = MazeEnv._shared_maze  # Set self.maze before generating
            self._generate_maze()
        else:
            self.maze = MazeEnv._shared_maze
        
        self.max_steps = 200
        self.current_step = 0
        self.current_pos = self.start_pos

    def _generate_maze(self):
        # No need to fill with 1s since we already initialized with ones
        self.maze[0, 0] = 0  # Now self.maze exists when this is called
        # Rest of the maze generation code remains the same...
        
        def carve_path(x, y, target=None):
            directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]  # Right, Down, Left, Up
            
            # If we have a target, prioritize directions that move towards it
            if target:
                tx, ty = target
                # Sort directions by which gets us closer to target
                directions.sort(key=lambda d: abs(x + d[0] - tx) + abs(y + d[1] - ty))
            else:
                np.random.shuffle(directions)
            
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < self.size and 
                    0 <= new_y < self.size and 
                    self.maze[new_x, new_y] == 1):  # If cell is unvisited
                    # Carve path by making cells empty (0)
                    self.maze[x + dx//2, y + dy//2] = 0  # Middle cell
                    self.maze[new_x, new_y] = 0          # Target cell
                    carve_path(new_x, new_y)
        
        # Start from (0,0) and make it a path
        self.maze[0, 0] = 0
        
        # Carve a guaranteed path to the goal
        carve_path(0, 0, target=self.goal_pos)
        
        # Then add additional random paths for variety
        carve_path(0, 0)
        
        # Ensure goal is accessible
        self.maze[self.goal_pos] = 0
        
        # Add a few random shortcuts
        for _ in range(2):
            x = np.random.randint(1, self.size-1)
            y = np.random.randint(1, self.size-1)
            self.maze[x, y] = 0
        
    def _get_obs(self):
        # Create two-channel observation
        obs = np.zeros((self.size, self.size, 2))
        obs[:, :, 0] = self.maze  # First channel: maze structure
        
        # Second channel: agent position
        agent_channel = np.zeros((self.size, self.size))
        agent_channel[self.current_pos] = 1.0
        obs[:, :, 1] = agent_channel
        
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_pos = self.start_pos
        self.current_step = 0
        # Currently the maze stays the same after env creation
        # If you want a new maze each episode, add:
        # self._generate_maze()  # Uncomment this line for new mazes each episode
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        
        # Calculate new position based on action
        x, y = self.current_pos
        if action == 0:   # Up
            new_pos = (max(0, x-1), y)
        elif action == 1: # Right
            new_pos = (x, min(self.size-1, y+1))
        elif action == 2: # Down
            new_pos = (min(self.size-1, x+1), y)
        else:            # Left
            new_pos = (x, max(0, y-1))
        
        # Check if new position is valid (not a wall)
        wall_penalty = 0
        if self.maze[new_pos] == 1:  # Hit a wall
            wall_penalty = -0.5  # Add negative reward for wall collision
            new_pos = self.current_pos  # Stay in current position
        self.current_pos = new_pos
        
        # Calculate normalized reward
        max_distance = 2 * self.size
        distance = abs(self.current_pos[0] - self.goal_pos[0]) + \
                  abs(self.current_pos[1] - self.goal_pos[1])
        # reward = -distance / max_distance + wall_penalty  # Add wall penalty to distance-based reward
        reward = wall_penalty - 1
        
        # Add smaller goal reward
        if self.current_pos == self.goal_pos:
            reward += 10.0
            done = True
        else:
            done = self.current_step >= self.max_steps
            
        return self._get_obs(), reward, done, False, {}

    def debug_maze(self):
        """Visualize the current maze."""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        plt.imshow(self.maze, cmap='binary')
        plt.plot(self.start_pos[1], self.start_pos[0], 'go', markersize=15, label='Start')
        plt.plot(self.goal_pos[1], self.goal_pos[0], 'ro', markersize=15, label='Goal')
        plt.grid(True)
        plt.legend()
        plt.title("Maze Layout")
        plt.savefig("maze_layout.png")

# You can call this after maze generation to verify:
env = MazeEnv()
env.debug_maze()

def print_model_summary(model):
    print("\nModel Architecture:")
    print("=" * 50)
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\nParameter Count:")
    print("=" * 50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Print parameter count for each layer
    print("\nParameters per layer:")
    print("=" * 50)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel():,}")

dummy_model = MazeCNN(
    env.observation_space,
    env.action_space,
    4,  # num_outputs
    {},  # empty model config
    "dummy_model"
)

print_model_summary(dummy_model)
# Initialize Ray
ray.init()

# Add after ray.init()
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")

# Add after ray.init()
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Configure the algorithm
config = {
    "env": MazeEnv,
    # Parallelization settings
    "num_workers": 15,
    "num_envs_per_worker": 4,
    "num_gpus": 0.8,
    "num_gpus_per_worker": 0.01,
    
    # Framework settings
    "framework": "torch",
    
    # Batch settings - adjusted for more frequent updates
    "rollout_fragment_length": 128,  # Reduced from 256
    "train_batch_size": 7680,  # = 15 * 4 * 128 (must be num_workers * num_envs_per_worker * rollout_fragment_length)
    "sgd_minibatch_size": 256,  # Reduced from 1024 for more frequent updates
    
    # Model settings
    "model": {
        "custom_model": "maze_cnn",
    },
    
    # Training settings
    "lr": 3e-4,
    "gamma": 0.99,
    "lambda": 0.95,
    "entropy_coeff": 0.01,
    "vf_loss_coeff": 0.5,
    
    # GPU memory optimization
    "_enable_learner_api": True,
    "torch_compile": True,
}

# Create PPO trainer
trainer = PPO(config=config)

def visualize_path(env, trainer, i):
    obs, _ = env.reset()
    path = [env.current_pos]
    
    plt.figure(figsize=(10, 10))
    
    # Create the base maze visualization
    plt.imshow(env.maze, cmap='binary')
    
    # Run the agent and collect the path
    done = False
    while not done:
        action = trainer.compute_single_action(obs)
        obs, _, done, _, _ = env.step(action)
        path.append(env.current_pos)
    
    # Convert path to plottable format
    path = np.array(path)
    
    # Plot start and goal
    plt.plot(env.start_pos[1], env.start_pos[0], 'go', markersize=15, label='Start')
    plt.plot(env.goal_pos[1], env.goal_pos[0], 'ro', markersize=15, label='Goal')
    
    # Plot the path
    plt.plot(path[:, 1], path[:, 0], 'b-', linewidth=2, label='Agent Path')
    
    plt.grid(True)
    plt.legend()
    plt.title("Maze Solution")
    # save the plot to a file
    plt.savefig(f"maze_solution-{i}.png")
    plt.show()

# Training loop
for i in range(10**3):
    result = trainer.train()
    print(f"Training iteration {i}")
    print(f"Mean reward: {result['env_runners']['episode_reward_mean']}")
    print(f"Episode length mean: {result['env_runners']['episode_len_mean']}")
    print(f"Min/Max reward: {result['env_runners']['episode_reward_min']:.2f}/{result['env_runners']['episode_reward_max']:.2f}")
    if i % 10 == 0:
        visualize_path(env, trainer, i)

# Cleanup
ray.shutdown()
