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
                 epsilon_start=1, epsilon_end=0.15, epsilon_decay=0.99995,
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