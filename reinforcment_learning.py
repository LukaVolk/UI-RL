import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import time
import numpy as np

from constants import NUM_ACTIONS

# Define your Neural Network
class QNetwork(nn.Module):
    def __init__(self, observation_size, num_actions):
        super().__init__()
        # Improved architecture with dropout and batch normalization
        self.fc1 = nn.Linear(observation_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.1)
        
        self.fc4 = nn.Linear(128, num_actions)
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        is_single_input = False

        if x.dim() == 1:
            x = x.unsqueeze(0)
            is_single_input = True

        # Safe forward pass: temporarily disable BatchNorm stats update if batch size is 1
        if self.training and x.size(0) == 1:
            # Forward pass using eval mode for batch norm layers
            self.eval()
            with torch.no_grad():
                out = torch.relu(self.bn1(self.fc1(x)))
                out = self.dropout1(out)

                out = torch.relu(self.bn2(self.fc2(out)))
                out = self.dropout2(out)

                out = torch.relu(self.bn3(self.fc3(out)))
                out = self.dropout3(out)

                out = self.fc4(out)
            self.train()
        else:
            out = torch.relu(self.bn1(self.fc1(x)))
            out = self.dropout1(out)

            out = torch.relu(self.bn2(self.fc2(out)))
            out = self.dropout2(out)

            out = torch.relu(self.bn3(self.fc3(out)))
            out = self.dropout3(out)

            out = self.fc4(out)

        return out.squeeze(0) if is_single_input else out

class DQNAgent:
    def __init__(self, observation_size, num_actions, learning_rate=0.0003, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.15, epsilon_decay=0.9999,
                 batch_size=64, buffer_size=100000, target_update_freq=1000, load_path=None, epsilon_decay_steps=100000):
    
        
        self.observation_size = observation_size
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.q_network = QNetwork(observation_size, num_actions).to(self.device)
        self.target_q_network = QNetwork(observation_size, num_actions).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

        # Use AdamW optimizer with weight decay for better generalization
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=learning_rate, weight_decay=1e-4)
        # Use Huber loss for more stable training
        self.loss_fn = nn.SmoothL1Loss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.9)

        self.replay_buffer = deque(maxlen=buffer_size)
        self.current_step = 0
        self.total_steps = 0
        
        # Performance tracking
        self.episode_rewards = []
        self.recent_losses = deque(maxlen=100)

        if load_path is not None:
            self.load_model(load_path)

    def choose_action(self, state, training=True, override_epsilon=None):
        """
        Chooses an action using an epsilon-greedy policy during training,
        or pure exploitation during evaluation.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)

        if training:
            if random.random() < (override_epsilon if override_epsilon is not None else self.epsilon):
                # Explore: choose random action
                return random.randint(0, self.num_actions - 1)
            else:
                # Exploit with training mode (use dropout/batchnorm)
                self.q_network.train()
        else:
            # Exploit with evaluation mode (disable dropout/batchnorm)
            self.q_network.eval()

        with torch.no_grad():
            q_values = self.q_network(state_tensor)
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
        self.total_steps += 1
        
        # Only start learning when we have enough experiences
        if len(self.replay_buffer) < self.batch_size * 2:
            return None
        
        # Sample a batch of experiences
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        # Filter out invalid experiences
        filtered_batch = [exp for exp in batch if exp[0] is not None and exp[3] is not None]
        if len(filtered_batch) < self.batch_size // 2:
            return None
        
        # Prepare batch data
        states, actions, rewards, next_states, dones = zip(*filtered_batch)
        
        # Convert to tensors and move to device
        states_t = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # Compute current Q-values
        current_q_values = self.q_network(states_t).gather(1, actions_t.unsqueeze(1))
        
        # Compute next Q-values using Double DQN
        with torch.no_grad():
            # Use main network to select actions
            next_actions = self.q_network(next_states_t).argmax(1)
            # Use target network to evaluate those actions
            next_q_values = self.target_q_network(next_states_t).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards_t.unsqueeze(1) + (self.gamma * next_q_values * (1 - dones_t.unsqueeze(1)))
        
        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Track loss for monitoring
        self.recent_losses.append(loss.item())
        
        # Decay epsilon
        self.epsilon = max(
            self.epsilon_end, 
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) * self.total_steps / 100_000
        )
        
        # Update target network
        if self.total_steps % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
            print(f"Target network updated at step {self.total_steps}")
        
        # Periodic logging
        if self.total_steps % 1000 == 0:
            avg_loss = sum(self.recent_losses) / len(self.recent_losses) if self.recent_losses else 0
            print(f"Step {self.total_steps}: Epsilon={self.epsilon:.4f}, Avg Loss={avg_loss:.4f}, LR={self.scheduler.get_last_lr()[0]:.6f}")
        
        return loss.item()

    def save_model(self, path):
        """Save the complete model state including optimizer and training progress"""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_q_network_state_dict': self.target_q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'episode_rewards': self.episode_rewards,
            'observation_size': self.observation_size,
            'num_actions': self.num_actions
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load_model(self, path, evaluation_mode=False):
        """Load the complete model state"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
            
            if not evaluation_mode:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.epsilon = checkpoint.get('epsilon', self.epsilon)
                self.total_steps = checkpoint.get('total_steps', 0)
                self.episode_rewards = checkpoint.get('episode_rewards', [])
            
            if evaluation_mode:
                self.q_network.eval()
                self.target_q_network.eval()
            
            print(f"Model loaded from {path} (evaluation_mode={evaluation_mode})")
            return True
        except Exception as e:
            print(f"Failed to load model from {path}: {e}")
            return False
    
    def get_stats(self):
        """Get training statistics"""
        avg_loss = sum(self.recent_losses) / len(self.recent_losses) if self.recent_losses else 0
        return {
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'avg_recent_loss': avg_loss,
            'buffer_size': len(self.replay_buffer),
            'learning_rate': self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else 'N/A'
        }