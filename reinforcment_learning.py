# from constants import *
# import numpy as np
# from constants import ACTION_MAP
# class ReinforcementLearning():
#     def __init__(self, car, grass_track_rl, max_steps=1000):
#         self.car = car
#         self.grass_track_rl = grass_track_rl
#         self.current_step = 0
#         self.max_episode_steps = max_steps  # Define the maximum number of steps for training/testing
#         self.current_waypoint_index = 0
#         self.waypoints = grass_track_rl.checkpoints
#         self.prev_dist = self.car.get_sensor_distances(0)  # Initialize previous distance to None
#         self.observation_size = 8  # Define the size of the observation space
#         self.num_actions = len(ACTION_MAP)  # Number of discrete actions available
#     def train(self):
#         pass

#     def test(self):
#         # Implement the testing logic here
#         pass

#     #verjetno ne rabim
#     def _get_observation(self):
#         # This function is identical to the Gymnasium version
#         # Collect all state information from your CarRL object and environment
#         if self.car:
#             car_pos = self.car.position
#             car_vel = self.car.speed
#             car_rot = self.car.rotation

#             dist_arr = self.car.get_sensor_distances(self.current_waypoint_index)
#             dist = sum(dist_arr)

#             obs = np.array([
#                 car_pos.x, car_pos.y, car_pos.z,
#                 car_vel,
#                 car_rot.x, car_rot.y, car_rot.z,
#                 dist
#             ], dtype=np.float32)
#             return obs
#         return np.zeros(self.observation_size, dtype=np.float32)

#     def _get_reward(self, old_dist):
#         # This function is identical to the Gymnasium version
#         reward = 0.0
#         if self.car and self.waypoints and self.current_waypoint_index < len(self.waypoints):
#             current_waypoint = self.waypoints[self.current_waypoint_index]
#             # Assume self.car.prev_position is updated in CarRL's update method
#             if hasattr(self.car, 'prev_position') and self.car.prev_position is not None:
#                 new_dist_arr = self.car.get_sensor_distances(current_waypoint)
#                 new_dist = sum(new_dist_arr)
#                 reward += (old_dist - new_dist) * 0.1 # Reward for getting closer
#                 self.prev_dist = new_dist
#                 for dist in new_dist_arr:
#                     if dist < 0.1:
#                         reward += 10 
#                         #tuki je problem, ker naslednji reward bo pol ful slabsi
#                         self.current_waypoint_index += 1
#                         print(f"Passed waypoint {self.current_waypoint_index}")

#         reward -= 0.04 # Time penalty

#         return reward

#     def _is_done(self):
#         # This function is identical to the Gymnasium version, but without `truncated`
#         done = False

#         # If max steps reached
#         if self.current_step >= self.max_episode_steps:
#             done = True

#         # If car completes lap
#         if self.current_waypoint_index >= len(self.waypoints):
#             print("All waypoints completed!")
#             done = True
#             # Final large reward can be added in _get_reward or here

#         return done

#     def reset(self):
#         # Reset car and environment state
            
#         self.car.position = (-80, -30, 18.5)
#         self.car.rotation = (0, 90, 0)
#         self.car.visible = True
#         self.car.collision = False 
#         self.car.camera_follow = False         
#         self.car.speed = 0
#         self.car.velocity_y = 0
#         self.car.anti_cheat = 1
#         self.car.timer_running = True
#         self.car.count = 0.0
#         self.car.reset_count = 0.0
#         self.car.total_reward = 0

#         self.current_waypoint_index = 0
#         self.current_step = 0

#         # Return initial observation
#         return self._get_observation()

#     def step(self, action):
#         # Execute the action
#         self.car.execute_action(action)

#         # Increment step count
#         self.current_step += 1

#         # Get next observation, reward, and done status
#         observation = self._get_observation()
#         reward = self._get_reward(self.prev_dist)
#         done = self._is_done()

#         # Return them
#         return observation, reward, done

#     def render(self):
#         # Ursina handles this automatically via app.run()
#         pass


'''2. The RL Agent Class (Conceptual)
You'll need an RL agent. For discrete actions like yours, a Deep Q-Network (DQN) is a common choice. This would involve a neural network (using PyTorch or TensorFlow) that takes the state as input and outputs Q-values for each action.
'''

import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import time
import numpy as np

from constants import ACTION_MAP

# Define your Neural Network
class QNetwork(nn.Module):
    def __init__(self, observation_size, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(observation_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, observation_size, num_actions, learning_rate=0.001, gamma=0.99,
                 epsilon_start=1, epsilon_end=0.01, epsilon_decay = 0.99995,
                 batch_size=32, buffer_size=300000, load_path=None):
    
        
        self.observation_size = observation_size
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.q_network = QNetwork(observation_size, num_actions)
        self.target_q_network = QNetwork(observation_size, num_actions)
        self.target_q_network.load_state_dict(self.q_network.state_dict()) # Copy weights
        self.target_q_network.eval() # Set target network to evaluation mode

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = deque(maxlen=buffer_size)

        self.current_step = 0

        if load_path is not None:
            try:
                self.load_model(load_path)
                print(f"✅ Loaded model from {load_path}")
            except Exception as e:
                print(f"⚠️ Could not load model from {load_path}: {e}")

    def choose_action(self, state):
        """
        Chooses an action using an epsilon-greedy policy based on the provided state dictionary.
        The state dict is converted to a flat numpy array for the neural network.
        Args:
            state: A dictionary containing keys like 'speed', 'distances', 'total_reward', 'next_checkpoint', 'rotation_speed'.
        Returns:
            action (int): The chosen action index.
        """

        state_vec = state # The input 'state' is already the np.array from get_state

        if random.random() < self.epsilon:
            # Explore
            return random.randint(0, len(ACTION_MAP) - 1)
        else:
            # Exploit
            with torch.no_grad():
                # Add a batch dimension (unsqueeze(0)) because the network expects a batch
                obs_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
                q_values = self.q_network(obs_tensor)
                return q_values.argmax().item()

    def store_experience(self, state, action, reward, next_state, done):
        """
        Stores a single experience tuple in the replay buffer.

        Args:
            state: The current state of the environment before taking the action.
            action: The action taken by the agent in the current state.
            reward: The reward received after taking the action.
            next_state: The state of the environment after taking the action.
            done: A boolean indicating whether the episode has ended after this step.
        """
        self.replay_buffer.append((state, action, reward, next_state, done))

    def learn(self):
        if self.current_step % 100 == 0:
            print(f"[Step {self.current_step}] Epsilon: {self.epsilon:.4f}")
        self.current_step += 1
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = random.sample(self.replay_buffer, self.batch_size)

        # Filter out any tuples where state or next_state is None
        filtered_batch = [exp for exp in batch if exp[0] is not None and exp[3] is not None]

        if len(filtered_batch) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = zip(*filtered_batch)
        states = [np.asarray(s, dtype=np.float32).reshape(self.observation_size) for s in states]
        next_states = [np.asarray(ns, dtype=np.float32).reshape(self.observation_size) for ns in next_states]
        states_t = torch.tensor(np.stack(states), dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.long).unsqueeze(-1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)
        next_states_t = torch.tensor(np.stack(next_states), dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1)

        # Compute Q-values for current states
        q_values = self.q_network(states_t).gather(1, actions_t)

        # Compute target Q-values (using target network)
        next_q_values = self.target_q_network(next_states_t).max(1)[0].unsqueeze(-1)
        target_q_values = rewards_t + self.gamma * next_q_values * (1 - dones_t)

        # Compute loss and update Q-network
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Periodically update target network
        if self.current_step % 100 == 0: # Update every 100 steps (or episodes)
            self.target_q_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, path):
        torch.save(self.q_network.state_dict(), path)

    def load_model(self, path):
        self.q_network.load_state_dict(torch.load(path))
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.q_network.eval() # Set to eval mode if just for inference
        self.target_q_network.eval()