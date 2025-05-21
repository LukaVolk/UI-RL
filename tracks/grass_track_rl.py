from ursina import *

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
        


        self.print_timer = 0

        self.finish_line = Entity(model = "cube", position = (-62, -40, 15), rotation = (0, 0, 0), scale = (3, 8, 30), visible = False)
        self.boundaries = Entity(model = "grass_track_bounds.obj", collider = "mesh", position = (0, -50, 0), rotation = (0, 270, 0), scale = (25, 25, 25), visible = False)

        self.wall1 = Entity(model = "cube", position = (-5, -40, 35), rotation = (0, 90, 0), collider = "box", scale = (5, 30, 50), visible = False)
        self.wall2 = Entity(model = "cube", position = (20, -40, 1), rotation = (0, 90, 0), collider = "box", scale = (5, 30, 150), visible = False)
        self.wall3 = Entity(model = "cube", position = (-21, -40, 15), rotation = (0, 0, 0), collider = "box", scale = (5, 30, 50), visible = False)
        self.wall4 = Entity(model = "cube", position = (9, -40, 14), rotation = (0, 0, 0), collider = "box", scale = (5, 30, 50), visible = False)

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
                scale=(1, 8, 30), 
                visible=True,
                collision=False  # Disable physical collision
            ),
            Entity(
                model="cube", 
                position=(42.992534, -42.648094, 18.128223), 
                scale=(1, 8, 30), 
                visible=True,
                collision=False,
                rotation=(0, -45, 0)
            ),
            Entity(
                model="cube", 
                position=(41.169925, -42.723522, 61.554302), 
                scale=(1, 8, 30), 
                visible=True,
                collision=False,
                rotation=(0, 45, 0)
            ),
            Entity(
                model="cube", 
                position=(-4.456625, -42.379024, -53.188812), 
                scale=(1, 8, 30), 
                visible=True,
                collision=False,
                rotation=(0, -45, 0)
            ),
            Entity(
                model="cube", 
                position=(-107.281555, -42.50573, -34.137355), 
                scale=(1, 8, 30), 
                visible=True,
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

    # Add method to get checkpoint data
    def get_checkpoint_data(self, car):
        if car not in self.car_checkpoints:
            raise ValueError(f"Car {car} not found in checkpoints.")
        return {
            'positions': [cp.position for cp in self.checkpoints],
            'status': self.car_checkpoints[car],
            'total': len(self.checkpoints)
        }
    
    def check_checkpoint(self, car):
        """Check if car has hit next checkpoint in sequence"""
        if car not in self.car_checkpoints:
            return False
            
        next_checkpoint = self.car_last_checkpoint[car] + 1
        if next_checkpoint >= len(self.checkpoints):
            return False
            
        # Check if car has hit the next checkpoint
        if car.simple_intersects(self.checkpoints[next_checkpoint]):
            if not self.car_checkpoints[car][next_checkpoint]:
                self.car_checkpoints[car][next_checkpoint] = True
                self.car_last_checkpoint[car] = next_checkpoint
                print(f"Checkpoint {next_checkpoint} hit! Reward earned!")
                return True
                
        return False

    def update(self):
        # Handle updates for each car
        for car in self.cars:
            # Check finish line
            if car.simple_intersects(self.finish_line):
                if car.anti_cheat == 1:
                    car.timer_running = True
                    car.anti_cheat = 0
                    if car.gamemode != "drift":
                        invoke(car.reset_timer, delay=3)

                    car.check_highscore()
                    self.car_checkpoints[car] = [False] * len(self.checkpoints)

                self.wall1.enable()
                self.wall2.enable()
                self.wall3.disable()
                self.wall4.disable()

            # Check wall triggers
            if car.simple_intersects(self.wall_trigger):
                self.wall1.disable()
                self.wall2.disable()
                self.wall3.enable()
                self.wall4.enable()
                car.anti_cheat = 0.5

            if car.simple_intersects(self.wall_trigger_ramp):
                if car.anti_cheat == 0.5:
                    car.anti_cheat = 1

            # Check checkpoints
            for i, cp in enumerate(self.checkpoints):
                if car.simple_intersects(cp) and not self.car_checkpoints[car][i]:
                    # Checkpoint passed
                    self.car_checkpoints[car][i] = True
                    print(f"Car {id(car)} passed checkpoint {i}! Status: {self.car_checkpoints[car]}")