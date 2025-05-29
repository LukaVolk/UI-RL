from ursina import *
import random
import os
from constants import *
from reinforcment_learning import DQNAgent
from learning_utils import create_learning_methods
import os
import numpy as np

class GrassTrackRL(Entity):
    def __init__(self, cars):
        application.audio = False
        super().__init__(
            model = "grass_track.obj", 
            texture = "grass_track.png", 
            position = (0, -50, 0), 
            rotation = (0, 270, 0), 
            scale = (25, 25, 25), 
            collider = "mesh"
        )

        self.cars = cars if isinstance(cars, list) else [cars]
        self.car_action_timers = {car: 0.0 for car in self.cars}
        self.car_seeds = {car: random.randint(1, 10000) for car in self.cars}

        self.eval_car = cars[0] if cars else None

        self.action_interval = ACTION_INTERVAL
        self.step_number = 0

        # Initialize DQN agent with improved hyperparameters
        model_path = "models/dqn_latest.pth"
        self.DQNAgent = DQNAgent(
            observation_size=OBESERVATION_SIZE, 
            num_actions=NUM_ACTIONS,
            learning_rate=0.0003,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.15,
            epsilon_decay=0.9999,
            batch_size=64,
            buffer_size=100000,
            target_update_freq=1000,
            load_path=model_path if os.path.exists(model_path) else None
        )
        
        self.num_of_episodes = EPISODE_NUMBERS
        self.current_episode = 0
        self.episode_length = EPISODE_LENGTH
        self.car_action_queues = []
        self.print_timer = 0
        self.is_learning = False
        self.is_reinforcement_learning = False
        
        # Learning tracking
        self.top_k_cars = set()  # Set to store top K cars
        self.action_counts = {action: 0 for action in range(NUM_ACTIONS)}
        self.episode_rewards = []
        self.best_episode_reward = float('-inf')
        self.learning_step_count = 0
        
        # Experience storage for each car
        self.car_experiences = {car: {'state': None, 'action': None, 'reward': 0} for car in self.cars}
        
        # Add improved learning methods
        create_learning_methods(self)

        #timer
        self.timer_running = False
        self.timer = Text(text = "", origin = (0, 0), size = 0.05, scale = (1, 1), position = (-0.7, 0.43))
        self.episode = Text(text = "", origin = (0, 0), size = 0.05, scale = (0.6, 0.6), position = (-0.65, 0.38))
        self.phase = Text(text = "Learning Phase", origin = (0, 0), size = 0.05, scale = (0.6, 0.6), position = (-0.65, 0.33))
        self.timer.disable()
        self.episode.disable()
        self.phase.disable()

        self.finish_line = Entity(model = "cube", position = (-62, -40, 15), rotation = (0, 0, 0), scale = (3, 8, 30), visible = False)
        self.boundaries = Entity(model = "grass_track_bounds.obj", collider = "mesh", position = (0, -50, 0), rotation = (0, 270, 0), scale = (25, 25, 25), visible = SHOW_BOUNDRIES)

        self.wall1 = Entity(model = "cube", position = (-5, -40, 35), rotation = (0, 90, 0), collider = "box", scale = (5, 30, 50), visible = SHOW_WALLS)
        self.wall2 = Entity(model = "cube", position = (20, -40, 1), rotation = (0, 90, 0), collider = "box", scale = (5, 30, 150), visible = SHOW_WALLS)
        self.wall3 = Entity(model = "cube", position = (-21, -40, 15), rotation = (0, 0, 0), collider = "box", scale = (5, 30, 50), visible = SHOW_WALLS)
        self.wall4 = Entity(model = "cube", position = (9, -40, 14), rotation = (0, 0, 0), collider = "box", scale = (5, 30, 50), visible = SHOW_WALLS)

        self.wall_trigger = Entity(model = "cube", position = (25, -40.2, 65), rotation = (0, 0, 0), scale = (3, 20, 50), visible = False)
        self.wall_trigger_ramp = Entity(model = "cube", position = (-82, -34, -64), rotation = (0, 0, 0), scale = (3, 20, 50), visible = False)
        
        self.trees = Entity(model = "trees-grass.obj", texture = "tree-grass.png", position = (0, -50, 0), rotation_y = 270, scale = 25)
        self.rocks = Entity(model = "rocks-grass.obj", texture = "rock-grass.png", position = (0, -50, 0), rotation_y = 270, scale = 25)
        self.grass = Entity(model = "grass-grass_track.obj", texture = "grass-grass_track.png", position = (0, -50, 0), rotation_y = 270, scale = 25)
        self.thin_trees = Entity(model = "thintrees-grass.obj", texture = "thintree-grass.png", position = (0, -50, 0), rotation_y = 270, scale = 25)


        self.checkpoints = [
            Entity(
                model="cube", 
                position=(-31.180683, -34.7023, 18.503328), 
                scale=(1, 8, CHECKPOINT_WIDTH), 
                visible=SHOW_CHECKPOINTS,
                collision=False  # Disable physical collision
            ),
            Entity(
                model="cube", 
                position=(12, -42.648094, 15.503328), 
                scale=(1, 8, CHECKPOINT_WIDTH), 
                visible=SHOW_CHECKPOINTS,
                collision=False  # Disable physical collision
            ),
            Entity(
                model="cube", 
                position=(42.992534, -42.648094, 18.128223), 
                scale=(1, 8, CHECKPOINT_WIDTH), 
                visible=SHOW_CHECKPOINTS,
                collision=False,
                rotation=(0, -45, 0)
            ),
            Entity(
                model="cube", 
                position=(41.169925, -42.723522, 61.554302), 
                scale=(1, 8, CHECKPOINT_WIDTH), 
                visible=SHOW_CHECKPOINTS,
                collision=False,
                rotation=(0, 45, 0)
            ),
            Entity(
                model="cube", 
                position=(0, -42.723522, 61.554302), 
                scale=(1, 8, CHECKPOINT_WIDTH), 
                visible=SHOW_CHECKPOINTS,
                collision=False,
                rotation=(0, -45, 0)
            ),
            Entity(
                model="cube", 
                position=(0, -42.50573, 34.137355), 
                scale=(1, 8, 30), 
                visible=SHOW_CHECKPOINTS,
                collision=False,
                rotation=(0, 90, 0)
            ),
            Entity(
                model="cube", 
                position=(0, -42.50573, 0), 
                scale=(1, 8, 30), 
                visible=SHOW_CHECKPOINTS,
                collision=False,
                rotation=(0, 90, 0)
            ),
            Entity(
                model="cube", 
                position=(-4.456625, -42.379024, -53.188812), 
                scale=(1, 8, CHECKPOINT_WIDTH), 
                visible=SHOW_CHECKPOINTS,
                collision=False,
                rotation=(0, -45, 0)
            ),
            Entity(
                model="cube", 
                position=(-40, -42.50573, -63.188812), 
                scale=(1, 8, 30), 
                visible=SHOW_CHECKPOINTS,
                collision=False,
            ),
            Entity(
                model="cube", 
                position=(-80, -34.50573, -63.188812), 
                scale=(1, 8, 30), 
                visible=SHOW_CHECKPOINTS,
                collision=False,
            ),
            Entity(
                model="cube", 
                position=(-107.281555, -42.50573, -34.137355), 
                scale=(1, 8, 30), 
                visible=SHOW_CHECKPOINTS,
                collision=False,
                rotation=(0, 90, 0)
            ),
            Entity(
                model="cube", 
                position=(-107.281555, -42.50573, 0), 
                scale=(1, 8, 30), 
                visible=SHOW_CHECKPOINTS,
                collision=False,
                rotation=(0, 90, 0)
            ),	

        ]
        self.car_checkpoints = {car: [False] * len(self.checkpoints) for car in self.cars}

        self.track = [
            self.finish_line, self.boundaries, self.wall1, self.wall2, self.wall3, 
            self.wall4, self.wall_trigger, self.wall_trigger_ramp, self.checkpoints[0],
            self.checkpoints[1], self.checkpoints[2], self.checkpoints[3], self.checkpoints[4]
        ]

        self.details = [
            self.trees, self.rocks, self.grass, self.thin_trees
        ]

        
        for i in self.track:
            i.enable()
        for i in self.details:
            i.disable()

        

        self.played = False
        self.unlocked = False

    def handle_checkpoint(self, car, checkpoint_index):
        """Handle checkpoint collision with proper sequence validation
        
        Returns:
            True: Correct checkpoint hit
            False: Wrong checkpoint hit
            None: Previous checkpoint hit - no action needed
        """
        if not self.car_checkpoints[car][checkpoint_index]:
            # Previous checkpoint check
            prev_checkpoint = (car.next_checkpoint_index - 1) % len(self.checkpoints)
            if checkpoint_index == prev_checkpoint:
                #print(f"Previous checkpoint {checkpoint_index} hit - ignoring")
                return None
                
            # Expected checkpoint check    
            if checkpoint_index == car.next_checkpoint_index:
                self.car_checkpoints[car][checkpoint_index] = True
                car.next_checkpoint_index = (checkpoint_index + 1) % len(self.checkpoints)
                #print(f"Car {car} hit checkpoint {checkpoint_index}. Next: {car.next_checkpoint_index}")
                return True
                
            # Wrong checkpoint
            #print(f"Wrong checkpoint! Expected {car.next_checkpoint_index}, got {checkpoint_index}")
            return False
    
    def get_random_action(self, car):
        # Use car-specific seed for randomization
        #tuki dodaj dodatne utezi glede na avto
        random.seed(self.car_seeds[car] + int(time.time() * 1000))
        action = random.randint(0, NUM_ACTIONS - 1)  # Random action index
        self.car_seeds[car] = random.randint(1, 10000)  # Update seed
        return action
    
    def _is_done(self, car):
        # This function is identical to the Gymnasium version, but without `truncated`
        done = False

        # If max steps reached
        if self.current_episode >= self.num_of_episodes:
            print("Maximum number of episodes reached!")
            done = True

        # If car completes lap
        if car.next_checkpoint_index >= len(self.checkpoints):
            print("All waypoints completed!")
            done = True
            # Final large reward can be added in _get_reward or here

        return done

    def learning_process(self):
        """Improved learning process using modular approach"""
        # Episode management
        if self.print_timer >= self.episode_length:
            self.step_number = 0
            self._end_episode()
            
        # Main learning loop for each car
        for car in self._process_car_interactions():
            self._handle_car_learning(car)
            

    def exploitation_process(self):
        """Exploitation phase - use trained model without learning"""
        if self.print_timer >= self.episode_length and REINFORCEMENT_LEARNING:
            self.print_timer = 0
            self.current_episode += 1
            self.step_number = 0
            self.DQNAgent.learn()
            for car in self.cars:
                if hasattr(car, 'reset'):
                    car.reset()

        # Process car interactions (rewards still apply for monitoring)
        for car in self._process_car_interactions():
            # Use trained model for action selection (no learning)
            if car.rl and self.enabled and car.visible:
                self.car_action_timers[car] += time.dt
                if self.car_action_timers[car] >= self.action_interval:
                    self.car_action_timers[car] = 0
                    state = car.get_state2(self.checkpoints)
                    state = np.append(state, self.step_number)
                    # Use exploitation mode (training=False)
                    action = self.DQNAgent.choose_action(state, training=True)
                    car.execute_action(action)
        
                next_state = car.get_state2(self.checkpoints)
                next_state = np.append(next_state, self.step_number+1)
                # Store experience with learning
                if car in self.top_k_cars:
                    self._store_experience(car, next_state)
                
        self.step_number += 1


    def update(self):
        if self.current_episode >= self.num_of_episodes:
            self.episode.disable()
            self.timer.disable()
            self.timer_running = False
            self.print_timer = 0
            self.current_episode = 0
            if self.phase.text == "Learning Phase":
                self.phase.text = "Exploitation Phase"
                self.is_learning = False
                self.is_reinforcement_learning = True
            else:
                self.is_learning = False
                self.is_reinforcement_learning = False



        if self.is_learning:
            self.phase.enable()
            self.timer_running = True
            self.timer.enable()
            self.episode.enable()
            self.print_timer += time.dt  # Add elapsed time
            self.timer.text = str(round(self.print_timer, 1)) + "s"
            self.episode.text = f"Episode: {self.current_episode + 1}/{self.num_of_episodes}"
            # if REINFORCEMENT_LEARNING:
            #     self.learning_process()
            self.learning_process()

        if self.is_reinforcement_learning:
            self.timer_running = True
            self.timer.enable()
            self.episode.enable()
            self.print_timer += time.dt  # Add elapsed time
            self.timer.text = str(round(self.print_timer, 1)) + "s"
            self.episode.text = f"Episode: {self.current_episode + 1}/{self.num_of_episodes}"
            # if REINFORCEMENT_LEARNING:
            #     self.exploitation_process()
            self.exploitation_process()

        if not self.is_learning and not self.is_reinforcement_learning:
            self.timer.disable()
            self.episode.disable()
            self.phase.disable()
            self.timer_running = False
            