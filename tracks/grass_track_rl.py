from ursina import *
import random
from constants import *
from reinforcment_learning import DQNAgent
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

        self.action_interval = ACTION_INTERVAL

        self.DQNAgent = DQNAgent(observation_size=OBESERVATION_SIZE, num_actions=len(ACTION_MAP))
        self.num_of_episodes = EPISODE_NUMBERS
        self.current_episode = 0
        self.episode_length = EPISODE_LENGTH
        self.car_action_queues = []

        self.print_timer = 0
        self.is_learning = False
        self.is_reinforcement_learning = False

        self.top_k_cars = set()  # Set to store top K cars

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
                position=(-4.456625, -42.379024, -53.188812), 
                scale=(1, 8, CHECKPOINT_WIDTH), 
                visible=SHOW_CHECKPOINTS,
                collision=False,
                rotation=(0, -45, 0)
            ),
            Entity(
                model="cube", 
                position=(-107.281555, -42.50573, -34.137355), 
                scale=(1, 8, 30), 
                visible=SHOW_CHECKPOINTS,
                collision=False,
                rotation=(0, 90, 0)
            )
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
        action = random.randint(0, len(ACTION_MAP) - 1)  # Random action index
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
        #print(f"Episode: {self.current_episode}, Timer: {self.print_timer:.2f}")
        if self.print_timer >= self.episode_length:
            self.print_timer = 0
            self.current_episode += 1
            # Store current episode rewards
            car_rewards = {car: car.total_reward for car in self.cars}
            sorted_cars = sorted(self.cars, key=lambda c: car_rewards[c], reverse=True)
            self.top_k_cars = set(sorted_cars[:TOP_K])

            print(f"Top {TOP_K} cars this episode:")
            for c in self.top_k_cars:
                print(f" - {c} with reward {car_rewards[c]:.2f}")
            for car in self.cars:
                self.car_action_queues.append([car.action_queue.copy(), car.total_reward])
                if hasattr(car, 'reset'):
                    car.reset()
                self.current_waypoint_index = 0
            # reset learning
            
        #learning
        for car in self.cars:
            if car.simple_intersects(self.finish_line):
                if car.anti_cheat == 1:
                    if all(self.car_checkpoints[car]):
                        # FINISH LINE REWARD
                        if hasattr(car, 'give_reward'):
                            car.give_reward(FINISH_LINE_REWARD)
                            print(f"Lap complete! Giving finish line reward")
                    car.anti_cheat = 0
                    car.next_checkpoint_index = 0

                    self.car_checkpoints[car] = [False] * len(self.checkpoints)

                self.wall1.enable()
                self.wall2.enable()
                self.wall3.disable()
                self.wall4.disable()

            if car.simple_intersects(self.wall_trigger):
                self.wall1.disable()
                self.wall2.disable()
                self.wall3.enable()
                self.wall4.enable()
                car.anti_cheat = 0.5

            if car.simple_intersects(self.wall_trigger_ramp):
                if car.anti_cheat == 0.5:
                    car.anti_cheat = 1

            # TIME PENALTY
            if hasattr(car, 'give_reward') and self.timer_running:
                # Time penalty - encourage faster lap times
                penalty = TIME_PENALTY * time.dt
                car.give_reward(penalty)
                #print(f"Time penalty: {penalty:.2f}")
             
            # SPEED REWARD
            if hasattr(car, 'give_reward') and self.timer_running:
                # Speed reward - encourage maintaining good speed
                if abs(car.speed) > MIN_SPEED_THRESHOLD:
                    # Scale reward with speed and time
                    reward = abs(car.speed) * SPEED_REWARD * time.dt
                    car.give_reward(reward)
                    
                    # Extra reward for near max speed
                    if car.speed > car.topspeed * 0.8:
                        car.give_reward(reward * 2)
                        #print(f"Speed reward: {reward:.2f} + bonus: {reward * 2:.2f}")
                        
                elif abs(car.speed) < MIN_SPEED_THRESHOLD:
                    # Small penalty for very low speed
                    penalty = -SPEED_REWARD * time.dt
                    car.give_reward(penalty)
                    #print(f"Speed penalty: {penalty:.2f}")

            
            # Check for collisions with walls
            if car.wall_hit:
                if hasattr(car, 'give_reward'):
                    car.give_reward(WALL_PENALTY)
                    #print(f"Car {car} hit a wall! Giving penalty")

            # CHECKPOINT REWARD
            for i, cp in enumerate(self.checkpoints):
                if car.simple_intersects(cp):
                    result = self.handle_checkpoint(car, i)
                    if result is True:
                        if hasattr(car, 'give_reward'):
                            car.give_reward(CHECKPOINT_REWARD)
                    elif result is False:
                        if hasattr(car, 'give_reward'):
                            car.give_reward(WRONG_CHECKPOINT_PENALTY)

                    # Give bonus for reinforcement learning
                    # if hasattr(self.car, "give_bonus_reward"):
                    #     self.car.give_bonus_reward(i)
        
            state = car.get_state2(self.checkpoints)
            action = None
            # Random action for reinforcement learning
            if car.rl and self.enabled and car.visible:
                self.car_action_timers[car] += time.dt
                if self.car_action_timers[car] >= self.action_interval:
                    self.car_action_timers[car] = 0
                    action = self.DQNAgent.choose_action(state)
                    car.execute_action(action)

            
            reward = car.total_reward if hasattr(car, 'total_reward') else 0
            next_state = car.get_state2(self.checkpoints)
            done = self._is_done(car)

            #print(f"Action: {action}, Reward: {reward}, State: {state}")
            #store experience
            if car in self.top_k_cars and action is not None:
                self.DQNAgent.store_experience(
                    state=state, 
                    action=action, 
                    reward=reward, 
                    next_state=next_state, 
                    done=done
                )
        self.DQNAgent.learn()
        if self.current_episode % 10 == 0:
            self.DQNAgent.save_model(f"models/dqn_model_ep{self.current_episode}.pth")

    def exploitation_process(self):
        if self.print_timer >= self.episode_length and REINFORCEMENT_LEARNING:
            self.print_timer = 0
            self.current_episode += 1
            for car in self.cars:
                if hasattr(car, 'reset'):
                    car.reset()
                self.current_waypoint_index = 0

        for car in self.cars:
            if car.simple_intersects(self.finish_line):
                if car.anti_cheat == 1:
                    if all(self.car_checkpoints[car]):
                        print(f"Lap complete!")
                    car.anti_cheat = 0
                    car.next_checkpoint_index = 0
                    self.car_checkpoints[car] = [False] * len(self.checkpoints)

                self.wall1.enable()
                self.wall2.enable()
                self.wall3.disable()
                self.wall4.disable()

            if car.simple_intersects(self.wall_trigger):
                self.wall1.disable()
                self.wall2.disable()
                self.wall3.enable()
                self.wall4.enable()
                car.anti_cheat = 0.5

            if car.simple_intersects(self.wall_trigger_ramp):
                if car.anti_cheat == 0.5:
                    car.anti_cheat = 1

            # Skip car control if not enabled
            if car.rl and self.enabled and car.visible:
                self.car_action_timers[car] += time.dt
                if self.car_action_timers[car] >= self.action_interval:
                    self.car_action_timers[car] = 0
                    state = car.get_state2(self.checkpoints)
                    action = self.DQNAgent.choose_action(state)
                    car.execute_action(action)


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
            