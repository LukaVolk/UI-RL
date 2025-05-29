from ursina import *
from ursina import curve
from particles import Particles, TrailRenderer
import json
from constants import ACTION_MAP, REINFORCEMENT_LEARNING,SHOW_SENSORS
import numpy as np

sign = lambda x: -1 if x < 0 else (1 if x > 0 else 0)
Text.default_resolution = 1080 * Text.size

class CarRL(Entity):
    def __init__(self, position = (0, 0, 4), rotation = (0, 0, 0), topspeed = 30, acceleration = 0.35, braking_strength = 30, friction = 0.6, drift_speed = 35, camera_speed = 8):
        super().__init__(
            model = "sports-car.obj",
            texture = "sports-red.png",
            collider = "box",
            position = position,
            rotation = rotation,
        )

        self.grass_track_rl = False

        # Rotation parent
        self.rotation_parent = Entity()

        # Controls
        self.controls = "wasd"

        #camera
        # Camera Follow
        self.camera_speed = camera_speed
        self.camera_angle = "top"
        self.camera_offset = (0, 0, 100)
        self.camera_rotation = 0
        self.camera_follow = False
        self.change_camera = False
        self.c_pivot = Entity()
        self.camera_pivot = Entity(parent = self.c_pivot, position = self.camera_offset)

        # Car's values
        self.speed = 0
        self.velocity_y = 0
        self.rotation_speed = 0
        self.max_rotation_speed = 2.6
        self.steering_amount = 8
        self.topspeed = topspeed
        self.braking_strenth = braking_strength
        self.acceleration = acceleration
        self.friction = friction
        self.drift_speed = drift_speed
        self.drift_amount = 4.5
        self.turning_speed = 5
        self.max_drift_speed = 40
        self.min_drift_speed = 20
        self.pivot_rotation_distance = 1

        # Pivot for drifting
        self.pivot = Entity()
        self.pivot.position = self.position
        self.pivot.rotation = self.rotation
        self.drifting = False

        # Car Type
        self.car_type = "sports"

        # Particles
        self.particle_time = 0
        self.particle_amount = 0.07 # The lower, the more
        self.particle_pivot = Entity(parent = self)
        self.particle_pivot.position = (0, -1, -2)

        # TrailRenderer
        self.trail_pivot = Entity(parent = self, position = (0, -1, 2))

        self.trail_renderer1 = TrailRenderer(parent = self.particle_pivot, position = (0.8, -0.2, 0), color = color.black, alpha = 0, thickness = 7, length = 200)
        self.trail_renderer2 = TrailRenderer(parent = self.particle_pivot, position = (-0.8, -0.2, 0), color = color.black, alpha = 0, thickness = 7, length = 200)
        self.trail_renderer3 = TrailRenderer(parent = self.trail_pivot, position = (0.8, -0.2, 0), color = color.black, alpha = 0, thickness = 7, length = 200)
        self.trail_renderer4 = TrailRenderer(parent = self.trail_pivot, position = (-0.8, -0.2, 0), color = color.black, alpha = 0, thickness = 7, length = 200)
        
        self.trails = [self.trail_renderer1, self.trail_renderer2, self.trail_renderer3, self.trail_renderer4]
        self.start_trail = True

        # Audio
        self.audio = False
        self.volume = 1
        self.start_sound = True
        self.start_fall = True
        self.drive_sound = Audio("rally.mp3", loop = True, autoplay = False, volume = 0.5)
        self.dirt_sound = Audio("dirt-skid.mp3", loop = True, autoplay = False, volume = 0.8)
        self.skid_sound = Audio("skid.mp3", loop = True, autoplay = False, volume = 0.5)
        self.hit_sound = Audio("hit.wav", autoplay = False, volume = 0.5)
        self.drift_swush = Audio("unlock.mp3", autoplay = False, volume = 0.8)

        # Collision
        self.copy_normals = False
        self.hitting_wall = False
        self.wall_hit = False

        # Cosmetics
        self.current_cosmetic = "none"

        # Graphics
        self.graphics = "fancy"
        
        self.gamemode = "race"
        self.start_time = False
        self.laps = 0
        self.laps_hs = 0
        self.anti_cheat = 1

        # Bools
        self.driving = False
        self.braking = False

        self.ai = False
        self.ai_list = []

        # Multiplayer
        self.multiplayer = False
        self.multiplayer_update = False
        self.server_running = False

        # Reinforcement learning
        self.rl = REINFORCEMENT_LEARNING
        self.ai_cars = []
        self.print_timer = 0
        self.next_checkpoint_index = 0
        self.prev_position = None
        self.ignore_entities = []
        self.max_track_distance_for_norm = None

        # Add reward tracking
        self.current_reward = 0
        self.total_reward = 0
        self.last_reward = 0

        self.input_states = {
            'forward': False,
            'left': False, 
            'back': False,
            'right': False,
            'handbrake': False
        }

        self.action_queue = []

        # Car sensors
        self.show_sensors = SHOW_SENSORS
        self.sensor_rays = []
        self.sensor_length = 100

        # Show sensors
        if self.show_sensors and self.gamemode == "race":
            for _ in range(5):
                ray = Entity(model='cube', scale=(0.1, 0.1, self.sensor_length), color=color.green)
                ray.disable()
                self.sensor_rays.append(ray)

        self.model_path = str(self.model).replace("render/scene/car/", "")

        invoke(self.update_model_path, delay = 3)

    def sports_car(self):
        self.car_type = "sports"
        self.model = "sports-car.obj"
        self.texture = "sports-red.png"
        self.drive_sound.clip = "sports.mp3"
        self.topspeed = 30
        self.acceleration = 0.38
        self.drift_amount = 5
        self.turning_speed = 5
        self.min_drift_speed = 18
        self.max_drift_speed = 38
        self.max_rotation_speed = 3
        self.steering_amount = 8
        self.particle_pivot.position = (0, -1, -1.5)
        self.trail_pivot.position = (0, -1, 1.5)
        for cosmetic in self.cosmetics:
            cosmetic.y = 0

    def muscle_car(self):
        self.car_type = "muscle"
        self.model = "muscle-car.obj"
        self.texture = "muscle-orange.png"
        self.drive_sound.clip = "muscle.mp3"
        self.topspeed = 38
        self.acceleration = 0.32
        self.drift_amount = 6
        self.turning_speed = 10
        self.min_drift_speed = 22
        self.max_drift_speed = 40
        self.max_rotation_speed = 3
        self.steering_amount = 8.5
        self.particle_pivot.position = (0, -1, -1.8)
        self.trail_pivot.position = (0, -1, 1.8)
        for cosmetic in self.cosmetics:
            cosmetic.y = 0

    def limo(self):
        self.car_type = "limo"
        self.model = "limousine.obj"
        self.texture = "limo-black.png"
        self.drive_sound.clip = "limo.mp3"
        self.topspeed = 30
        self.acceleration = 0.33
        self.drift_amount = 5.5
        self.turning_speed = 8
        self.min_drift_speed = 20
        self.max_drift_speed = 40
        self.max_rotation_speed = 3
        self.steering_amount = 8
        self.particle_pivot.position = (0, -1, -3.5)
        self.trail_pivot.position = (0, -1, 3.5)
        for cosmetic in self.cosmetics:
            cosmetic.y = 0.1

    def lorry(self):
        self.car_type = "lorry"
        self.model = "lorry.obj"
        self.texture = "lorry-white.png"
        self.drive_sound.clip = "lorry.mp3"
        self.topspeed = 30
        self.acceleration = 0.3
        self.drift_amount = 7
        self.turning_speed = 7
        self.min_drift_speed = 20
        self.max_drift_speed = 40
        self.max_rotation_speed = 3
        self.steering_amount = 7.5
        self.particle_pivot.position = (0, -1, -3.5)
        self.trail_pivot.position = (0, -1, 3.5)
        for cosmetic in self.cosmetics:
            cosmetic.y = 1.5

    def hatchback(self):
        self.car_type = "hatchback"
        self.model = "hatchback.obj"
        self.texture = "hatchback-green.png"
        self.drive_sound.clip = "hatchback.mp3"
        self.topspeed = 28
        self.acceleration = 0.43
        self.drift_amount = 6
        self.turning_speed = 15
        self.min_drift_speed = 20
        self.max_drift_speed = 40
        self.max_rotation_speed = 3
        self.steering_amount = 8.5
        self.particle_pivot.position = (0, -1, -1.5)
        self.trail_pivot.position = (0, -1, 1.5)
        for cosmetic in self.cosmetics:
            cosmetic.y = 0.4

    def rally_car(self):
        self.car_type = "rally"
        self.model = "rally-car.obj"
        self.texture = "rally-red.png"
        self.drive_sound.clip = "rally.mp3"
        self.topspeed = 34
        self.acceleration = 0.46
        self.drift_amount = 4
        self.turning_speed = 7
        self.min_drift_speed = 22
        self.max_drift_speed = 40
        self.max_rotation_speed = 3
        self.steering_amount = 8.5
        self.particle_pivot.position = (0, -1, -1.5)
        self.trail_pivot.position = (0, -1, 1.5)
        for cosmetic in self.cosmetics:
            cosmetic.y = 0.3

    def execute_action(self, action):
        """Execute action from RL agent
        
        Actions:
        0: No action
        1: Forward
        2: Forward + Left
        3: Forward + Right
        4: Brake
        5: Handbrakes
        6: Left
        7: Right
        """
        self.input_states = {k: False for k in self.input_states}
        for key in ACTION_MAP.get(action, []):
            self.input_states[key] = True
        
        

    def get_sensor_distances(self, waypoint):
        """Returns distances from sensors in all directions"""
        directions = [
            (0, 0, 1),      # Forward
            (0.7, 0, 0.7),  # Forward Right
            (1, 0, 0),      # Right
            (-1, 0, 0),     # Left
            (-0.7, 0, 0.7), # Forward Left
        ]
        
        distances = []

        for i, direction in enumerate(directions):
            # Rotate direction based on car's rotation
            rotated_direction = self.forward * direction[2] + self.right * direction[0]
            
            # Cast ray
            # if waypoint is not None and self.grass_track_rl:
            #     ray = raycast(
            #         origin=self.world_position,
            #         direction=rotated_direction,
            #         distance=self.sensor_length,
            #         traverse_target=self.grass_track_rl.checkpoints[waypoint] if waypoint is not None and self.grass_track_rl else None,
            #         ignore=[self]
            #     )
            # else:
            #     ray = raycast(
            #         origin=self.world_position,
            #         direction=rotated_direction,
            #         distance=self.sensor_length,
            #         ignore=[self]
            #     )

            ray = raycast(
                    origin=self.world_position,
                    direction=rotated_direction,
                    distance=self.sensor_length,
                    ignore=[self]
                )
            
            # Get distance or max length if no hit
            distance = ray.distance if ray.hit else self.sensor_length
            distances.append(distance)
            
            # Update sensor visualization
            if self.show_sensors and i < len(self.sensor_rays):
                ray_entity = self.sensor_rays[i]
                # Position the ray to start from car's position
                ray_entity.world_position = self.world_position
                ray_entity.look_at(self.world_position + rotated_direction * distance)
                ray_entity.world_position += rotated_direction * (distance/2)
                ray_entity.scale_z = distance
                ray_entity.enable()
        
        return distances
        
    def give_reward(self, reward):
        """Give reward to the car and update reward tracking
        
        Args:
            reward (float): Reward value to give (positive or negative)
        """
        self.current_reward = reward
        self.total_reward += reward
        self.last_reward = reward
        
        #print(f"Reward given: {reward:.1f} | Total: {self.total_reward:.1f}")

    def get_state(self):
        """Get current state for RL model
        
        Returns:
            tuple: (speed, sensor_distances, total_reward, next_checkpoint_index, rotation_speed)
        """
        # Get normalized speed (-1 to 1)
        normalized_speed = self.speed / self.topspeed
        
        # Get sensor distances
        sensor_distances = self.get_sensor_distances(self.next_checkpoint_index)
        
        # Normalize distances to 0-1 range
        normalized_distances = [d / self.sensor_length for d in sensor_distances]
        
        state = {
            'speed': normalized_speed,
            'distances': normalized_distances,
            'total_reward': self.total_reward,
            'next_checkpoint': self.next_checkpoint_index,
            'rotation_speed': self.rotation_speed
        }
        
        return state
    
    def calculate_max_track_distance_for_norm(self, checkpoints):
        if not checkpoints:
            return 1.0  # Default sensible minimum if no checkpoints

        max_dist = 0.0
        # Iterate through all pairs of checkpoints
        for i in range(len(checkpoints)):
            for j in range(i + 1, len(checkpoints)):
                pos1 = checkpoints[i].position
                pos2 = checkpoints[j].position
                # Assuming Vec3 objects support subtraction and .length()
                dist = (pos1 - pos2).length()
                if dist > max_dist:
                    max_dist = dist

        # Add a buffer: The car might be slightly off the direct line between checkpoints,
        # or even further away if it's reset to a start position far from the first checkpoint.
        # A buffer of 1.2 to 1.5 times the max_dist is often reasonable.
        return max_dist * 1.2
    
    def get_state2(self, checkpoints):
        """Get current state for RL model with improved features, adapted to car values.

        Returns:
            np.array: A flat NumPy array representing the state.
                      (speed, vel_x, vel_z, rotation_yaw, rotation_pitch, rotation_roll,
                       rotation_speed_y, dist_to_next_cp, angle_to_next_cp,
                       sensor_distance_1, sensor_distance_2, ...)
        """

        if self.max_track_distance_for_norm is None:
            # Calculate the maximum track distance for normalization
            self.max_track_distance_for_norm = self.calculate_max_track_distance_for_norm(checkpoints)
        # --- 1. Car Dynamics ---
        # Normalized speed (scalar)
        # self.speed can be negative if going backwards. Normalization to [-1, 1] is appropriate.
        normalized_speed = self.speed / self.topspeed 

        # Get normalized velocity components (x, z)
        # Derived from car's forward direction and its current speed
        # Using self.forward (Vec3) which is normalized, and self.speed (scalar)
        current_velocity_vec = self.forward * self.speed
        
        # Max velocity component can be topspeed
        max_vel_component_norm = self.topspeed 
        normalized_vel_x = current_velocity_vec.x / max_vel_component_norm
        normalized_vel_z = current_velocity_vec.z / max_vel_component_norm

        # Car's Euler rotation angles (yaw, pitch, roll)
        car_rotation_euler = self.rotation

        # Normalize yaw (y-axis rotation, heading) from -180 to 180 degrees, then to -1 to 1.
        normalized_yaw = car_rotation_euler.y / 180.0

        # Pitch (x-axis rotation, nose up/down) and Roll (z-axis rotation, tilt sideways)
        # Normalize by 90.0 as cars typically don't exceed +/- 90 degrees tilt/pitch in normal driving.
        normalized_pitch = car_rotation_euler.x / 90.0
        normalized_roll = car_rotation_euler.z / 90.0

        # Normalized rotation speed (angular velocity around Y-axis)
        # Using your defined self.max_rotation_speed for normalization.
        # self.rotation_speed can be negative if turning left.
        normalized_rotation_speed = self.rotation_speed / self.max_rotation_speed


        # --- 2. Environment Interaction ---
        normalized_dist_to_next_cp = 0.0
        normalized_angle_to_next_cp = 0.0

        if checkpoints and self.next_checkpoint_index < len(checkpoints):
            # Get the position attribute of the Entity object for the next checkpoint
            next_checkpoint_pos = checkpoints[self.next_checkpoint_index].position

            # Vector from car to checkpoint
            vec_to_cp = next_checkpoint_pos - self.position 

            dist_to_next_cp = vec_to_cp.length()

            # Normalize distance to checkpoint. Use a value that covers your largest track distances.
            # self.max_track_distance_for_norm should be defined in your class
            normalized_dist_to_next_cp = dist_to_next_cp / self.max_track_distance_for_norm

            # Calculate angle between car's forward vector and vector to checkpoint
            car_forward_vector = self.forward # self.forward is already a normalized Vec3

            # Ensure the vector to checkpoint is not zero length to avoid division by zero
            if vec_to_cp.length() > 0.01: # Small epsilon to handle being exactly on the checkpoint
                normalized_vec_to_cp = vec_to_cp.normalized()

                # Use atan2 for robust angle calculation (handles full 360 range)
                # This calculation assumes a 2D projection (ignoring the Y-axis, which is typically 'up' in 3D game engines)
                # It calculates the angle in the XZ plane.
                angle_rad = math.atan2(car_forward_vector.x * normalized_vec_to_cp.z - car_forward_vector.z * normalized_vec_to_cp.x, 
                                       car_forward_vector.x * normalized_vec_to_cp.x + car_forward_vector.z * normalized_vec_to_cp.z)

                # Normalize angle from -pi to pi to -1 to 1
                normalized_angle_to_next_cp = angle_rad / math.pi
            else:
                normalized_angle_to_next_cp = 0.0 # If at checkpoint, perfectly aligned
        else:
            # If no checkpoints left or end of track, provide default values (agent is "done" or lost)
            normalized_dist_to_next_cp = 0.0
            normalized_angle_to_next_cp = 0.0


        # Get normalized sensor distances
        sensor_distances = self.get_sensor_distances(self.next_checkpoint_index) 
        normalized_distances = [d / self.sensor_length for d in sensor_distances]

        #print(f"Sensor distances: {normalized_distances}")
        #print(f"Normalized distances to next checkpoint: {normalized_dist_to_next_cp}")
        # --- 3. Construct the flat state array ---
        # Flatten all relevant features into a single NumPy array
        # This order should be consistent for your neural network input layer.
        state_list = [
            normalized_speed,
            normalized_vel_x,
            normalized_vel_z,
            normalized_yaw,
            normalized_pitch,
            normalized_roll,
            normalized_rotation_speed,
            normalized_dist_to_next_cp,
            normalized_angle_to_next_cp,
        ]
        
        # Extend with sensor distances (assuming this returns a list of floats)
        state_list.extend(normalized_distances)

        return np.array(state_list, dtype=np.float32)
    
    def reset(self):
        # Reset car and environment state
            
        self.position = (-80, -30, 18.5)
        self.rotation = (0, 90, 0)
        self.visible = True
        self.collision = False 
        self.camera_follow = False         
        self.speed = 0
        self.velocity_y = 0
        self.anti_cheat = 1
        #self.timer_running = True
        self.count = 0.0
        self.reset_count = 0.0
        self.total_reward = 0

    def ignore_other_cars(self):
        """Ignore other cars in raycasts for collision detection"""
        # Disable collision with other Car entities
        # Find all Car instances and add them to the ignore list for raycasts
        self.ignore_entities = [self]
        for e in scene.entities:
            if isinstance(e, CarRL) and e is not self:
                self.ignore_entities.append(e)

    def update(self):
        # The y rotation distance between the car and the pivot
        self.pivot_rotation_distance = (self.rotation_y - self.pivot.rotation_y)

        # Drifting
        if self.pivot.rotation_y != self.rotation_y:
            if self.pivot.rotation_y > self.rotation_y:
                self.pivot.rotation_y -= (self.drift_speed * ((self.pivot.rotation_y - self.rotation_y) / 40)) * time.dt
                if self.speed > 1 or self.speed < -1:
                    self.speed += self.pivot_rotation_distance / self.drift_amount * time.dt
                self.rotation_speed -= 1 * time.dt
                if self.pivot_rotation_distance >= 50 or self.pivot_rotation_distance <= -50:
                    self.drift_speed += self.pivot_rotation_distance / 5 * time.dt
                else:
                    self.drift_speed -= self.pivot_rotation_distance / 5 * time.dt
            if self.pivot.rotation_y < self.rotation_y:
                self.pivot.rotation_y += (self.drift_speed * ((self.rotation_y - self.pivot.rotation_y) / 40)) * time.dt
                if self.speed > 1 or self.speed < -1:
                    self.speed -= self.pivot_rotation_distance / self.drift_amount * time.dt
                self.rotation_speed += 1 * time.dt
                if self.pivot_rotation_distance >= 50 or self.pivot_rotation_distance <= -50:
                    self.drift_speed -= self.pivot_rotation_distance / 5 * time.dt
                else:
                    self.drift_speed += self.pivot_rotation_distance / 5 * time.dt

        # Gravity
        movementY = self.velocity_y / 50
        direction = (0, sign(movementY), 0)

        # You can use ignore_entities in your raycasts like:
        # y_ray = raycast(origin=self.world_position, direction=(0, -1, 0), ignore=ignore_entities)
        # (To apply this, move the ignore_entities logic above the raycast if you want all raycasts to ignore Cars)
        ignore_entities = [self]
        for e in scene.entities:
            if isinstance(e, CarRL) and e is not self:
                ignore_entities.append(e)


        # Main raycast for collision
        y_ray = raycast(origin = self.world_position, direction = (0, -1, 0), ignore = ignore_entities)

        if self.rl:
            if y_ray.distance <= 5:
                # Driving
                if self.input_states['forward']:
                    self.speed += self.acceleration * 50 * time.dt
                    self.speed += -self.velocity_y * 4 * time.dt
                    self.driving = True
                    # ...existing particle and trail code...
                
                # Braking
                if self.input_states['back']:
                    self.speed -= self.braking_strenth * time.dt
                    self.drift_speed -= 20 * time.dt
                    self.braking = True
                else:
                    self.braking = False

                # Hand Braking
                if self.input_states['handbrake']:
                    if self.rotation_speed < 0:
                        self.rotation_speed -= 3 * time.dt
                    elif self.rotation_speed > 0:
                        self.rotation_speed += 3 * time.dt
                    self.drift_speed -= 40 * time.dt
                    self.speed -= 20 * time.dt
                    self.max_rotation_speed = 3.0

            # Steering
            if self.speed > 1 or self.speed < -1:
                if self.input_states['left']:
                    self.rotation_speed -= self.steering_amount * time.dt
                    self.drift_speed -= 5 * time.dt
                    if self.speed > 1:
                        self.speed -= self.turning_speed * time.dt
                    elif self.speed < 0:
                        self.speed += self.turning_speed / 5 * time.dt
                elif self.input_states['right']:
                    self.rotation_speed += self.steering_amount * time.dt
                    self.drift_speed -= 5 * time.dt
                    if self.speed > 1:
                        self.speed -= self.turning_speed * time.dt
                    elif self.speed < 0:
                        self.speed += self.turning_speed / 5 * time.dt
                else:
                    self.drift_speed += 15 * time.dt
                    if self.rotation_speed > 0:
                        self.rotation_speed -= 5 * time.dt
                    elif self.rotation_speed < 0:
                        self.rotation_speed += 5 * time.dt
        else:
            if y_ray.distance <= 5:
                # Driving
                if held_keys[self.controls[0]] or held_keys["up arrow"]: #
                    self.speed += self.acceleration * 50 * time.dt
                    self.speed += -self.velocity_y * 4 * time.dt

                    self.driving = True

                    # Particles
                    self.particle_time += time.dt
                    if self.particle_time >= self.particle_amount:
                        self.particle_time = 0
                        self.particles = Particles(self, self.particle_pivot.world_position - (0, 1, 0))
                        self.particles.destroy(1)
                
                    # TrailRenderer / Skid Marks
                    if self.graphics != "ultra fast":
                        if self.drift_speed <= self.min_drift_speed + 2 and self.start_trail:   
                            if self.pivot_rotation_distance > 60 or self.pivot_rotation_distance < -60 and self.speed > 10:
                                for trail in self.trails:
                                    trail.start_trail()
                                if self.audio:
                                    self.skid_sound.volume = self.volume / 2
                                    self.skid_sound.play()
                                self.start_trail = False
                                self.drifting = True
                            else:
                                self.drifting = False
                        elif self.drift_speed > self.min_drift_speed + 2 and not self.start_trail:
                            if self.pivot_rotation_distance < 60 or self.pivot_rotation_distance > -60:
                                for trail in self.trails:
                                    if trail.trailing:
                                        trail.end_trail()
                                if self.audio:
                                    self.skid_sound.stop(False)
                                self.start_trail = True
                                self.drifting = False
                            self.drifting = False
                        if self.speed < 10:
                            self.drifting = False
                else:
                    self.driving = False
                    if self.speed > 1:
                        self.speed -= self.friction * 5 * time.dt
                    elif self.speed < -1:
                        self.speed += self.friction * 5 * time.dt

                # Braking
                if held_keys[self.controls[2] or held_keys["down arrow"]]:
                    self.speed -= self.braking_strenth * time.dt
                    self.drift_speed -= 20 * time.dt
                    self.braking = True
                else:
                    self.braking = False

            # Audio
            if self.driving or self.braking:
                if self.start_sound and self.audio:
                    if not self.drive_sound.playing:
                        self.drive_sound.loop = True
                        self.drive_sound.play()
                    if not self.dirt_sound.playing:
                        self.drive_sound.loop = True
                        self.dirt_sound.play()
                    self.start_sound = False

                if self.speed > 0:
                    self.drive_sound.volume = self.speed / 80 * self.volume
                elif self.speed < 0:
                    self.drive_sound.volume = -self.speed / 80 * self.volume

                if self.pivot_rotation_distance > 0:
                    self.dirt_sound.volume = self.pivot_rotation_distance / 110 * self.volume
                elif self.pivot_rotation_distance < 0:
                    self.dirt_sound.volume = -self.pivot_rotation_distance / 110 * self.volume
            else:
                self.drive_sound.volume -= 0.5 * time.dt
                self.dirt_sound.volume -= 0.5 * time.dt
                if self.skid_sound.playing:
                    self.skid_sound.stop(False)

            # Hand Braking
            if held_keys["space"]:
                if self.rotation_speed < 0:
                    self.rotation_speed -= 3 * time.dt
                elif self.rotation_speed > 0:
                    self.rotation_speed += 3 * time.dt
                self.drift_speed -= 40 * time.dt
                self.speed -= 20 * time.dt
                self.max_rotation_speed = 3.0

        # If Car is not hitting the ground, stop the trail
        if self.graphics != "ultra fast":
            if y_ray.distance > 2.5:
                if self.trail_renderer1.trailing:
                    for trail in self.trails:
                        trail.end_trail()
                    self.start_trail = True

        # Steering
        self.rotation_y += self.rotation_speed * 50 * time.dt

        if self.rotation_speed > 0:
            self.rotation_speed -= self.speed / 6 * time.dt
        elif self.rotation_speed < 0:
            self.rotation_speed += self.speed / 6 * time.dt

        if self.speed > 1 or self.speed < -1:
            if held_keys[self.controls[1]] or held_keys["left arrow"]:
                self.rotation_speed -= self.steering_amount * time.dt
                self.drift_speed -= 5 * time.dt
                if self.speed > 1:
                    self.speed -= self.turning_speed * time.dt
                elif self.speed < 0:
                    self.speed += self.turning_speed / 5 * time.dt
            elif held_keys[self.controls[3]] or held_keys["right arrow"]:
                self.rotation_speed += self.steering_amount * time.dt
                self.drift_speed -= 5 * time.dt
                if self.speed > 1:
                    self.speed -= self.turning_speed * time.dt
                elif self.speed < 0:
                    self.speed += self.turning_speed / 5 * time.dt
            else:
                self.drift_speed += 15 * time.dt
                if self.rotation_speed > 0:
                    self.rotation_speed -= 5 * time.dt
                elif self.rotation_speed < 0:
                    self.rotation_speed += 5 * time.dt
        else:
            self.rotation_speed = 0

        # Cap the speed
        if self.speed >= self.topspeed:
            self.speed = self.topspeed
        if self.speed <= -15:
            self.speed = -15
        if self.speed <= 0:
            self.pivot.rotation_y = self.rotation_y

        # Cap the drifting
        if self.drift_speed <= self.min_drift_speed:
            self.drift_speed = self.min_drift_speed
        if self.drift_speed >= self.max_drift_speed:
            self.drift_speed = self.max_drift_speed

        # Cap the steering
        if self.rotation_speed >= self.max_rotation_speed:
            self.rotation_speed = self.max_rotation_speed
        if self.rotation_speed <= -self.max_rotation_speed:
            self.rotation_speed = -self.max_rotation_speed

        # Respawn
        if held_keys["g"]:
            self.reset_car()

        # Reset the car's position if y value is less than -100
        if self.y <= -100:
            self.reset_car()

        # Reset the car's position if y value is greater than 300
        if self.y >= 300:
            self.reset_car()

        # At the start of your update function
        max_dt = 1 / 30  # e.g., clamp to 30 FPS
        dt = min(time.dt, max_dt)
        # Use 'dt' instead of 'time.dt' in all calculations

        # Rotation
        self.rotation_parent.position = self.position

        factor = min(1, 20*time.dt)  # Clamp factor to avoid overshooting
        # Lerps the car's rotation to the rotation parent's rotation (Makes it smoother)
        self.rotation_x = lerp(self.rotation_x, self.rotation_parent.rotation_x, factor)
        self.rotation_z = lerp(self.rotation_z, self.rotation_parent.rotation_z, factor)

        # Check if car is hitting the ground
        if self.visible:
            wall_check_front = raycast(
                origin=self.world_position,
                direction=self.forward,
                distance=1,
                ignore=ignore_entities
            )
            wall_check_back = raycast(
                origin=self.world_position,
                direction=-self.forward,
                distance=1,
                ignore=ignore_entities
            )
            if wall_check_front.hit or wall_check_back.hit:
                wall_impact = 0
                if wall_check_front.hit:
                    wall_impact = abs(Vec3.dot(wall_check_front.world_normal, self.forward))
                if wall_check_back.hit:
                    wall_impact = abs(Vec3.dot(wall_check_back.world_normal, -self.forward))
                #print(f"Wall impact: {wall_impact:.2f}")
                if wall_impact > 0.5:  # Head-on collision
                    deceleration = 50 * time.dt * wall_impact
                    self.speed -= deceleration
                    self.speed = max(self.speed, 2)
                    #print(f"Speed after collision: {self.speed:.2f}")
                    self.wall_hit = True
                else:
                    self.wall_hit = False
            else:
                self.wall_hit = False

            #print(f"y_ray.world_normal.y = {y_ray.world_normal.y} | y_ray.world_point.y = {y_ray.world_point.y} | self.world_y = {self.world_y}")
            if y_ray.distance <= self.scale_y * 1.7 + abs(movementY):
                self.velocity_y = 0
                # Check if hitting a wall or steep slope
                if y_ray.world_normal.y > 0.7 and y_ray.world_point.y - self.world_y < 0.5:
                    # Set the y value to the ground's y value
                    self.y = y_ray.world_point.y + 1.4
                    self.hitting_wall = False
                else:
                    # Car is hitting a wall
                    self.hitting_wall = True

                if self.copy_normals:
                    self.ground_normal = self.position + y_ray.world_normal
                else:
                    self.ground_normal = self.position + (0, 180, 0)

                # Rotates the car according to the grounds normals
                if not self.hitting_wall:
                    self.rotation_parent.look_at(self.ground_normal, axis = "up")
                    self.rotation_parent.rotate((0, self.rotation_y + 180, 0))
                else:
                    self.rotation_parent.rotation = self.rotation

                if self.start_fall and self.audio:
                    self.hit_sound.volume = self.volume / 2
                    self.hit_sound.play()
                    self.start_fall = False
            else:
                self.y += movementY * 50 * time.dt
                self.velocity_y -= 50 * time.dt
                self.rotation_parent.rotation = self.rotation
                self.start_fall = True

        # Movement
        movementX = self.pivot.forward[0] * self.speed * time.dt
        movementZ = self.pivot.forward[2] * self.speed * time.dt

        # Collision Detection
        if movementX != 0:
            direction = (sign(movementX), 0, 0)
            x_ray = raycast(origin = self.world_position, direction = direction, ignore = ignore_entities)

            if x_ray.distance > self.scale_x / 2 + abs(movementX):
                self.x += movementX

        if movementZ != 0:
            direction = (0, 0, sign(movementZ))
            z_ray = raycast(origin = self.world_position, direction = direction, ignore = ignore_entities)

            if z_ray.distance > self.scale_z / 2 + abs(movementZ):
                self.z += movementZ

    def reset_car(self):
        """
        Resets the car
        """
        if self.rl:
            # For RL training, always use grass track position
            self.position = (-80, -30, 18.5)
            self.rotation = (0, 90, 0)
            self.total_reward = 0
        else:
            self.position = (-80, -30, 18.5)
            self.rotation = (0, 90, 0)
            # if self.sand_track.enabled:
            #     self.position = (-63, -40, -7)
            #     self.rotation = (0, 90, 0)
            # elif self.grass_track.enabled:
            #     self.position = (-80, -30, 18.5)
            #     self.rotation = (0, 90, 0)
            # elif self.snow_track.enabled:
            #     self.position = (-5, -35, 93)
            #     self.rotation = (0, 90, 0)
            # elif self.forest_track.enabled:
            #     self.position = (12, -35, 76)
            #     self.rotation = (0, 90, 0)
            # elif self.savannah_track.enabled:
            #     self.position = (-14, -35, 42)
            #     self.rotation = (0, 90, 0)
            # elif self.lake_track.enabled:
            #     self.position = (-121, -40, 158)
            #     self.rotation = (0, 90, 0)
        self.speed = 0
        self.velocity_y = 0
        self.anti_cheat = 1
        self.start_trail = True
        self.start_sound = True
        if self.audio:
            if self.skid_sound.playing:
                self.skid_sound.stop(False)
            if self.dirt_sound.playing:
                self.dirt_sound.stop(False)

    def simple_intersects(self, entity):
        """
        A faster AABB intersects for detecting collision with
        simple objects, doesn't take rotation into account
        """
        minXA = self.x - self.scale_x
        maxXA = self.x + self.scale_x
        minYA = self.y - self.scale_y + (self.scale_y / 2)
        maxYA = self.y + self.scale_y - (self.scale_y / 2)
        minZA = self.z - self.scale_z
        maxZA = self.z + self.scale_z

        minXB = entity.x - entity.scale_x + (entity.scale_x / 2)
        maxXB = entity.x + entity.scale_x - (entity.scale_x / 2)
        minYB = entity.y - entity.scale_y + (entity.scale_y / 2)
        maxYB = entity.y + entity.scale_y - (entity.scale_y / 2)
        minZB = entity.z - entity.scale_z + (entity.scale_z / 2)
        maxZB = entity.z + entity.scale_z - (entity.scale_z / 2)
        
        return (
            (minXA <= maxXB and maxXA >= minXB) and
            (minYA <= maxYB and maxYA >= minYB) and
            (minZA <= maxZB and maxZA >= minZB)
        )

    def reset_drift(self):
        """
        Resets the drift
        """
        self.animate_text(self.drift_text, 1.7, 1.1)
        invoke(self.drift_text.animate_position, (-0.8, 0.43), 0.3, curve = curve.out_expo, delay = 0.3)
        invoke(self.reset_drift_text, delay = 0.4)
        self.drift_swush.play()
        self.get_hundred = False
        self.get_thousand = False
        self.get_fivethousand = False

    def reset_drift_text(self):
        """
        Resets the drift text
        """
        self.drift_score += self.count
        self.drift_multiplier = 20
        self.count = 0
        self.drifting = False
        invoke(setattr, self.drift_text, "visible", False, delay = 0.1)
        invoke(setattr, self.drift_text, "position", (0, 0.43), delay = 0.3)

    def reset_drift_score(self):
        self.count = 0
        self.drift_score = 0
        self.drift_multiplier = 20
        self.drifting = False

        try:
            if self.sand_track.enabled:
                self.drift_time = 25.0
            elif self.grass_track.enabled:
                self.drift_time = 30.0
            elif self.snow_track.enabled:
                self.drift_time = 50.0
            elif self.forest_track.enabled:
                self.drift_time = 40.0
            elif self.savannah_track.enabled:
                self.drift_time = 25.0
            elif self.lake_track.enabled:
                self.drift_time = 75.0
        except AttributeError:
            print("Drift time not set, using default value of 25.0")

    def animate_text(self, text, top = 1.2, bottom = 0.6):
        """
        Animates the scale of text
        """
        if self.gamemode != "drift":
            if self.last_count > 1:
                text.animate_scale((top, top, top), curve = curve.out_expo)
                invoke(text.animate_scale, (bottom, bottom, bottom), delay = 0.2)
        else:
            text.animate_scale((top, top, top), curve = curve.out_expo)
            invoke(text.animate_scale, (bottom, bottom, bottom), delay = 0.2)

    def update_model_path(self):
        """
        Updates the model's file path for multiplayer
        """
        self.model_path = str(self.model).replace("render/scene/car/", "")
        invoke(self.update_model_path, delay = 3)

# Class for copying the car's position, rotation for multiplayer
class CarRepresentationRL(Entity):
    def __init__(self, car, position = (0, 0, 0), rotation = (0, 65, 0)):
        super().__init__(
            parent = scene,
            model = "sports-car.obj",
            texture = "sports-red.png",
            position = position,
            rotation = rotation,
            scale = (1, 1, 1)
        )

        self.model_path = str(self.model).replace("render/scene/car_representation/", "")


        self.text_object = None