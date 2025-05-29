"""
Utility functions for improved reinforcement learning process
"""
import time
from constants import *
import numpy as np

def create_learning_methods(track_instance):
    """Add improved learning methods to the track instance"""
    
    def _end_episode(self):
        """Handle end of episode logic"""
        self.print_timer = 0
        self.current_episode += 1
        
        # Calculate episode statistics
        car_rewards = {car: car.total_reward for car in self.cars}
        sorted_cars = sorted(self.cars, key=lambda c: car_rewards[c], reverse=True)
        self.top_k_cars = set(sorted_cars[:TOP_K])
        
        # Track best episode
        max_reward = max(car_rewards.values()) if car_rewards else 0
        if max_reward > self.best_episode_reward:
            self.best_episode_reward = max_reward
            # Save best model
            self.DQNAgent.save_model("models/dqn_best.pth")
            
        self.episode_rewards.append(max_reward)
        
        # Print episode statistics
        if self.current_episode % 10 == 0:
            self._print_episode_stats(car_rewards)
        
        # Reset cars and episode data
        for car in self.cars:
            if hasattr(car, 'reset'):
                car.reset()
            self.car_experiences[car] = {'state': None, 'action': None, 'reward': 0}

        print(f"\n🎉 Episode {self.current_episode} ended. Max reward: {max_reward:.1f}")

        total_actions = sum(self.action_counts.values())
        if total_actions > 0:
            print("\n📊 Action distribution this episode:")
            for action, count in sorted(self.action_counts.items()):
                percentage = 100 * count / total_actions
                print(f"Action {action}: {count} ({percentage:.2f}%)")
        
        self.action_counts = {action: 0 for action in range(NUM_ACTIONS)}
        
        # Save periodic checkpoints
        if self.current_episode % 50 == 0:
            self.DQNAgent.save_model(f"models/dqn_ep{self.current_episode}.pth")
            
    def _print_episode_stats(self, car_rewards):
        """Print detailed episode statistics"""
        avg_reward = sum(self.episode_rewards[-10:]) / min(10, len(self.episode_rewards))
        max_reward = max(car_rewards.values())
        
        print(f"\n📈 Episode {self.current_episode}:")
        print(f"  Max Reward: {max_reward:.1f}")
        print(f"  Avg Last 10: {avg_reward:.1f}")
        print(f"  Best Ever: {self.best_episode_reward:.1f}")
        
        # Print DQN stats
        stats = self.DQNAgent.get_stats()
        print(f"  Epsilon: {stats['epsilon']:.4f}")
        print(f"  Buffer: {stats['buffer_size']}/100000")
        if stats['avg_recent_loss'] > 0:
            print(f"  Avg Loss: {stats['avg_recent_loss']:.4f}")
            
    def _process_car_interactions(self):
        """Process all car interactions with environment"""
        active_cars = []
        
        for car in self.cars:
            if not (car.rl and self.enabled and car.visible):
                continue
                
            # Track car for learning
            active_cars.append(car)
            
            # Handle environment interactions
            self._handle_environment_interactions(car)
            
        return active_cars
    
    def _handle_environment_interactions(self, car):
        """Handle all environment interactions for a car"""
        # Finish line
        if car.simple_intersects(self.finish_line):
            if car.anti_cheat == 1 and all(self.car_checkpoints[car]):
                car.give_reward(FINISH_LINE_REWARD)
                print(f"🏁 Lap complete! Reward: {FINISH_LINE_REWARD}")
            car.anti_cheat = 0
            car.next_checkpoint_index = 0
            self.car_checkpoints[car] = [False] * len(self.checkpoints)
            
            self.wall1.enable()
            self.wall2.enable()
            self.wall3.disable()
            self.wall4.disable()
            
        # Wall triggers
        if car.simple_intersects(self.wall_trigger):
            self.wall1.disable()
            self.wall2.disable()
            self.wall3.enable()
            self.wall4.enable()
            car.anti_cheat = 0.5
            
        if car.simple_intersects(self.wall_trigger_ramp):
            if car.anti_cheat == 0.5:
                car.anti_cheat = 1
                
        # Continuous rewards
        self._apply_continuous_rewards(car)
        
        # Checkpoint handling
        self._handle_checkpoints(car)
        
    def _apply_continuous_rewards(self, car):
        """Apply all continuous rewards"""
        if not (hasattr(car, 'give_reward') and self.timer_running):
            return
            
        # Time penalty
        car.give_reward(TIME_PENALTY * time.dt)
        
        # Speed-based rewards
        if car.speed > MIN_SPEED_THRESHOLD:
            reward = car.speed * SPEED_REWARD * time.dt
            car.give_reward(reward)
            
            if car.speed > car.topspeed * 0.7:
                car.give_reward(reward * 1.5)
                
        elif car.speed < -MIN_FORWARD_SPEED:
            penalty = BACKWARD_PENALTY * abs(car.speed) * time.dt
            car.give_reward(penalty)
            
        elif abs(car.speed) < MIN_SPEED_THRESHOLD:
            car.give_reward(SPEED_PENALTY * time.dt)
            
        # Progress reward
        if car.next_checkpoint_index < len(self.checkpoints):
            checkpoint = self.checkpoints[car.next_checkpoint_index]
            distance = (checkpoint.position - car.position).length()
            
            if car.last_checkpoint_distance is not None:
                progress = car.last_checkpoint_distance - distance
                if progress > 0:
                    car.give_reward(progress * PROGRESS_REWARD * time.dt)
                    
            car.last_checkpoint_distance = distance
            
        # Wall collision
        if car.wall_hit:
            car.give_reward(WALL_PENALTY)
            
    def _handle_checkpoints(self, car):
        """Handle checkpoint interactions"""
        for i, cp in enumerate(self.checkpoints):
            if car.simple_intersects(cp):
                result = self.handle_checkpoint(car, i)
                if result is True:
                    car.give_reward(CHECKPOINT_REWARD)
                elif result is False:
                    car.give_reward(WRONG_CHECKPOINT_PENALTY)
                    
    def _handle_car_learning(self, car):
        """Handle RL learning for a single car"""
        # Get current state
        current_state = car.get_state2(self.checkpoints)
        current_state = np.append(current_state, self.step_number)


        
        # Check if it's time for a new action
        self.car_action_timers[car] += time.dt
        
        if self.car_action_timers[car] >= self.action_interval:
            self.car_action_timers[car] = 0
            
            # Store previous experience if available
            if self.car_experiences[car]['state'] is not None:
                self._store_experience(car, current_state)
            
            # Choose and execute new action
            override_eps = 0.0 if car.is_exploitation_car else 1
            action = self.DQNAgent.choose_action(current_state, training=True, override_epsilon=override_eps)
            self.action_counts[action] += 1
            car.execute_action(action)
            
            # Update car's experience
            self.car_experiences[car] = {
                'state': current_state.copy(),
                'action': action,
                'reward': car.total_reward
            }
            
            # Learn from experience
            if car in self.top_k_cars:
                loss = self.DQNAgent.learn()
                if loss is not None:
                    self.learning_step_count += 1
            
            self.step_number += 1
                    
    def _store_experience(self, car, next_state):
        """Store experience in replay buffer"""
        if car not in self.top_k_cars:
            return
            
        exp = self.car_experiences[car]
        if exp['state'] is None:
            return
            
        # Calculate reward difference
        reward_diff = car.total_reward - exp['reward']
        done = self._is_done(car)
        
        self.DQNAgent.store_experience(
            state=exp['state'],
            action=exp['action'],
            reward=reward_diff,
            next_state=next_state,
            done=done
        )
    
    # Bind methods to the track instance
    track_instance._end_episode = _end_episode.__get__(track_instance, track_instance.__class__)
    track_instance._print_episode_stats = _print_episode_stats.__get__(track_instance, track_instance.__class__)
    track_instance._process_car_interactions = _process_car_interactions.__get__(track_instance, track_instance.__class__)
    track_instance._handle_environment_interactions = _handle_environment_interactions.__get__(track_instance, track_instance.__class__)
    track_instance._apply_continuous_rewards = _apply_continuous_rewards.__get__(track_instance, track_instance.__class__)
    track_instance._handle_checkpoints = _handle_checkpoints.__get__(track_instance, track_instance.__class__)
    track_instance._handle_car_learning = _handle_car_learning.__get__(track_instance, track_instance.__class__)
    track_instance._store_experience = _store_experience.__get__(track_instance, track_instance.__class__)