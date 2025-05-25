from ursina import *
from ursina import curve
import os
from reinforcment_learning import ReinforcementLearning, DQNAgent



Text.default_resolution = 1080 * Text.size

class MainMenuRL(Entity):
    def __init__(self, grass_track_rl, cars):
        super().__init__(
            parent = camera.ui
        )

        # The different menus
        self.start_menu = Entity(parent = self, enabled = True)
        self.host_menu = Entity(parent = self, enabled = False)
        self.created_server_menu = Entity(parent = self, enabled = False)
        self.server_menu = Entity(parent = self, enabled = False)
        self.main_menu = Entity(parent = self, enabled = False)
        self.race_menu = Entity(parent = self, enabled = False)
        self.maps_menu = Entity(parent = self, enabled = False)
        self.settings_menu = Entity(parent = self, enabled = False)
        self.video_menu = Entity(parent = self, enabled = False)
        self.gameplay_menu = Entity(parent = self, enabled = False)
        self.audio_menu = Entity(parent = self, enabled = False)
        self.controls_menu = Entity(parent = self, enabled = False)
        self.garage_menu = Entity(parent = self, enabled = False)
        self.cars_menu = Entity(parent = self.garage_menu, enabled = False)
        self.colours_menu = Entity(parent = self.garage_menu, enabled = False)
        self.cosmetics_menu = Entity(parent = self.garage_menu, enabled = False)
        self.pause_menu = Entity(parent = self, enabled = False)
        self.quit_menu = Entity(parent = self, enabled = False)

        self.menus = [
            self.start_menu,
            self.main_menu, self.race_menu, self.maps_menu, self.settings_menu, self.video_menu, self.gameplay_menu,
            self.audio_menu, self.pause_menu, self.quit_menu
        ]
        
        #RL cars
        self.cars = cars

        self.sun = None

        self.click = Audio("click.wav", False, False, volume = 10)

        self.grass_track_rl = grass_track_rl

        self.tracks = [
            self.grass_track_rl
        ]

        # Animate the menu
        for menu in (self.start_menu, self.main_menu, self.race_menu, self.maps_menu, self.settings_menu, self.video_menu, self.gameplay_menu, self.audio_menu, self.controls_menu, self.pause_menu, self.quit_menu, self.garage_menu):
            def animate_in_menu(menu = menu):
                for i, e in enumerate(menu.children):
                    e.original_scale = e.scale
                    e.scale -= 0.01
                    e.animate_scale(e.original_scale, delay = i * 0.05, duration = 0.1, curve = curve.out_quad)

                    e.alpha = 0
                    e.animate("alpha", 0.7, delay = i * 0.05, duration = 0.1, curve = curve.out_quad)

                    if hasattr(e, "text_entity"):
                        e.text_entity.alpha = 0
                        e.text_entity.animate("alpha", 1, delay = i * 0.05, duration = 0.1)

            menu.on_enable = animate_in_menu

        # Start Menu

        self.cars[0].position = (-80, -42, 18.8)
        self.cars[0].visible = False
        self.grass_track_rl.enable()
        for track in self.grass_track_rl.track:
            track.enable()
        for detail in self.grass_track_rl.details:
            detail.enable()

        def quit():
            application.quit()
            os._exit(0)

        start_title = Entity(model = "quad", scale = (0.5, 0.2, 0.2), texture = "rally-logo", parent = self.start_menu, y = 0.3)

        # Quit Menu

        def quit():
            application.quit()
            os._exit(0)

        def dont_quit():
            self.quit_menu.disable()
            self.start_menu.enable()

        quit_text = Text("Are you sure you want to quit?", scale = 1.5, line_height = 2, x = 0, origin = 0, y = 0.2, parent = self.quit_menu)
        quit_yes = Button(text = "Yes", color = color.black, scale_y = 0.1, scale_x = 0.3, y = 0.05, parent = self.quit_menu)
        quit_no = Button(text = "No", color = color.black, scale_y = 0.1, scale_x = 0.3, y = -0.07, parent = self.quit_menu)

        quit_yes.on_click = Func(quit)
        quit_no.on_click = Func(dont_quit)

        def grass_track_func_ai():
            print("reinforcement learning")

            self.start_menu.disable()
            for track in self.tracks:
                    track.disable()
                    for i in track.track:
                        i.disable()
                    for i in track.details:
                        i.disable()
            grass_track_rl.enable()
            mouse.locked = True
            #set camera
            # 1. Position the camera even higher and to the side
            # Experiment with these values!
            sky_height = 300  # Much higher in the sky
            side_offset_x = 0 # Move it 30 units on the X-axis (e.g., to the "right")
            side_offset_z = 200 # Move it 40 units on the Z-axis (e.g., "behind" the center)

            camera.position = (side_offset_x, sky_height, side_offset_z)

            # 2. Make the camera look at the center of your map
            # If your map isn't centered at (0,0,0), change the target coordinates here
            camera.look_at((0, 0, 0)) # Look at the origin, which we assume is your map's center
            camera_roll_angle = 0 # Example: tilt 15 degrees to the left
            camera.rotation_z = camera_roll_angle
            camera.position = (side_offset_x, sky_height, side_offset_z+50)
            learn = None
            cars = []
            
            for i, rl_car in enumerate(self.cars):
                rl_car.position = (-80, -30, 18.5)
                rl_car.rotation = (0, 90, 0)
                rl_car.visible = True
                rl_car.collision = False 
                rl_car.camera_follow = False         
                cars.append(rl_car)
                rl_car.speed = 0
                rl_car.velocity_y = 0
                rl_car.anti_cheat = 1
                rl_car.timer_running = True
                rl_car.count = 0.0
                rl_car.reset_count = 0.0
                rl_car.total_reward = 0
                rl_car.grass_track_rl = grass_track_rl
            
            for rl_car in cars:
                rl_car.update()

            grass_track_rl.played = True
            for g in grass_track_rl.track:
                g.enable()
                g.alpha = 255

            time.sleep(3) 
            for _ in range(5):

                env = ReinforcementLearning(cars[0], grass_track_rl)
                agent = DQNAgent(observation_size=env.observation_size, num_actions=env.num_actions)

                num_episodes = 1000 # Number of training episodes

                # Ursina's update function to handle game logic and RL steps
                # We'll put the core RL step logic here, to synchronize with Ursina's framerate
                # Or, for fixed steps, you'd call application.step() multiple times inside your RL loop
                global_episode_reward = 0
                global_episode_steps = 0
                current_obs = None # Store the current observation globally
                episode_ended = False # Flag to manage episode resets
                for i in range(num_episodes):

                    if episode_ended: # If last episode ended, reset
                        current_obs = env.reset()
                        global_episode_reward = 0
                        global_episode_steps = 0
                        episode_ended = False

                    # Agent chooses an action
                    action = agent.choose_action(current_obs)

                    # Environment steps based on action
                    next_obs, reward, done = env.step(action)

                    # Store experience
                    agent.store_experience(current_obs, action, reward, next_obs, done)

                    # Agent learns
                    agent.learn()

                    global_episode_reward += reward
                    global_episode_steps += 1
                    current_obs = next_obs # Update current observation

                    if done:
                        print(f"Episode {env.current_step}: Total Reward = {global_episode_reward:.2f}, Steps = {global_episode_steps}")
                        episode_ended = True # Set flag to reset on next update_rl_step

                # Start the first episode
                current_obs = env.reset()
                env.current_episode = 0 # Track episodes manually

        #Reinforcement Learning Menu
        reinforcment_learning_button = Button(text = "Reinforcement Learning", color = color.gray, highlight_color = color.light_gray, scale_y = 0.1, scale_x = 0.3, y = -0.21, parent = self.start_menu)
        reinforcment_learning_button.on_click = Func(grass_track_func_ai)

        def main_menu():
            for car in self.cars:
                car.visible = False
                car.position = (0, 0, 4)
                car.rotation = (0, 65, 0)
                car.speed = 0
                car.velocity_y = 0
                car.count = 0.0
                car.last_count = 0.0
                car.reset_count = 0.0
                car.laps = 0
            self.start_menu.enable()
            self.pause_menu.disable()
            for track in self.tracks:
                track.disable()
                track.alpha = 255
                for i in track.track:
                    if track != self.grass_track_rl:
                        i.disable()
                    else:
                        i.enable()
            grass_track_rl.enable()
        
        p_mainmenu_button = Button(text = "Main Menu", color = color.black, scale_y = 0.1, scale_x = 0.3, y = -0.13, parent = self.pause_menu)
        p_mainmenu_button.on_click = Func(main_menu)

        # Video Menu

        def fullscreen():
            window.fullscreen = not window.fullscreen
            if window.fullscreen:
                fullscreen_button.text = "Fullscreen: On"
            elif window.fullscreen == False:
                fullscreen_button.text = "Fullscreen: Off"

        def borderless():
            window.borderless = not window.borderless
            if window.borderless:
                borderless_button.text = "Borderless: On"
            elif window.borderless == False:
                borderless_button.text = "Borderless: Off"
            window.exit_button.enable()

        def fps():
            window.fps_counter.enabled = not window.fps_counter.enabled
            if window.fps_counter.enabled:
                fps_button.text = "FPS: On"
            elif window.fps_counter.enabled == False:
                fps_button.text = "FPS: Off"

        def exit_button_func():
            window.exit_button.enabled = not window.exit_button.enabled
            if window.exit_button.enabled:
                exit_button.text = "Exit Button: On"
            elif window.exit_button.enabled == False:
                exit_button.text = "Exit Button: Off"

        def back_video():
            self.video_menu.disable()
            self.settings_menu.enable()

        fullscreen_button = Button("Fullscreen: On", color = color.black, scale_y = 0.1, scale_x = 0.3, y = 0.24, parent = self.video_menu)
        borderless_button = Button("Borderless: On", color = color.black, scale_y = 0.1, scale_x = 0.3, y = 0.12, parent = self.video_menu)
        fps_button = Button("FPS: Off", color = color.black, scale_y = 0.1, scale_x = 0.3, y = 0, parent = self.video_menu)
        exit_button = Button("Exit Button: Off", color = color.black, scale_y = 0.1, scale_x = 0.3, y = -0.12, parent = self.video_menu)
        back_button_video = Button(text = "Back", color = color.black, scale_y = 0.1, scale_x = 0.3, y = -0.24, parent = self.video_menu)

        fullscreen_button.on_click = Func(fullscreen)
        borderless_button.on_click = Func(borderless)
        fps_button.on_click = Func(fps)
        exit_button.on_click = Func(exit_button_func)
        back_button_video.on_click = Func(back_video)



    def update(self):
        #if not self.start_menu.enabled and not self.main_menu.enabled and not self.settings_menu.enabled and not self.race_menu.enabled and not self.maps_menu.enabled and not self.settings_menu.enabled and not self.garage_menu.enabled and not self.controls_menu.enabled and not self.host_menu.enabled and not self.server_menu.enabled and not self.created_server_menu.enabled and not self.video_menu.enabled and not self.gameplay_menu.enabled and not self.audio_menu.enabled and not self.quit_menu.enabled:
        #    self.car.camera_follow = True
        #else:
        #    self.car.camera_follow = False
        
        for car in self.cars:
            car.camera_follow = False

        ## Set the camera's position and make the car rotate
        #if self.start_menu.enabled or self.host_menu.enabled or self.garage_menu.enabled or self.server_menu.enabled or self.quit_menu.enabled:
        #    if not held_keys["right mouse"]:
        #        if self.start_spin:
        #            self.car.rotation_y += 15 * time.dt
        #    else:
        #        self.car.rotation_y = mouse.x * 200
#
        #    camera.position = lerp(camera.position, self.car.position + self.car.camera_offset, time.dt * self.car.camera_speed)
#
        #    if self.start_menu.enabled or self.quit_menu.enabled:
        #        self.car.camera_offset = (-25, 4, 0)
        #        camera.rotation = (5, 90, 0)
        #    elif self.host_menu.enabled:
        #        self.car.camera_offset = (-25, 8, 0)
        #        camera.rotation = (14, 90, 0)
        #    else:
        #        self.car.camera_offset = (-25, 6, 5)
        #        camera.rotation = (10, 90, 0)
        #else:
        #    if not self.car.camera_follow:
        #        camera.rotation = (35, -20, 0)
        #        camera.position = lerp(camera.position, self.car.position + (20, 40, -50), time.dt * self.car.camera_speed)

            
    
    def input(self, key):
        # Pause menu
        if not self.start_menu.enabled and not self.main_menu.enabled and not self.server_menu.enabled and not self.settings_menu.enabled and not self.race_menu.enabled and not self.maps_menu.enabled and not self.settings_menu.enabled and not self.garage_menu.enabled and not self.audio_menu.enabled and not self.controls_menu.enabled and not self.host_menu.enabled and not self.created_server_menu.enabled and not self.video_menu.enabled and not self.gameplay_menu.enabled and not self.quit_menu.enabled:
            if key == "escape":
                self.pause_menu.enabled = not self.pause_menu.enabled
                mouse.locked = not mouse.locked

            self.start_spin = False
            
        else:
            #self.car.timer.disable()
            #self.car.reset_count_timer.disable()
            #self.car.highscore.disable()
            #self.car.laps_text.disable()
            #self.car.drift_text.disable()
            #self.car.drift_timer.disable()
            #self.car.camera_speed = 8
            self.start_spin = True

        # Quit Menu
        if self.start_menu.enabled or self.quit_menu.enabled:
            if key == "escape":
                self.quit_menu.enabled = not self.quit_menu.enabled
                self.start_menu.enabled = not self.start_menu.enabled

        # Settings Menu
        if key == "escape":
            if self.settings_menu.enabled:
                self.settings_menu.disable()
                self.main_menu.enable()
            elif self.video_menu.enabled:
                self.video_menu.disable()
                self.settings_menu.enable()
            elif self.controls_menu.enabled:
                self.controls_menu.disable()
                self.settings_menu.enable()
            elif self.gameplay_menu.enabled:
                self.gameplay_menu.disable()
                self.settings_menu.enable()
            elif self.audio_menu.enabled:
                self.audio_menu.disable()
                self.settings_menu.enable()