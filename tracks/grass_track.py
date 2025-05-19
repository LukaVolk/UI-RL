from ursina import *

class GrassTrack(Entity):
    def __init__(self, car):
        application.audio = False
        super().__init__(
            model = "grass_track.obj", 
            texture = "grass_track.png", 
            position = (0, -50, 0), 
            rotation = (0, 270, 0), 
            scale = (25, 25, 25), 
            collider = "mesh"
        )

        self.car = car

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
        self.checkpoint_status = [False] * len(self.checkpoints)


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



    def update(self):
        # print car position every 5 seconds

        # self.print_timer += time.dt  # Add elapsed time
        # if self.print_timer >= 5:
        #     print(self.car.position)
        #     self.print_timer = 0
        if self.car.simple_intersects(self.finish_line):
            if self.car.anti_cheat == 1:
                self.car.timer_running = True
                self.car.anti_cheat = 0
                if self.car.gamemode != "drift":
                    invoke(self.car.reset_timer, delay = 3)

                self.car.check_highscore()
                self.checkpoint_status = [False] * len(self.checkpoints)

            self.wall1.enable()
            self.wall2.enable()
            self.wall3.disable()
            self.wall4.disable()

        if self.car.simple_intersects(self.wall_trigger):
            self.wall1.disable()
            self.wall2.disable()
            self.wall3.enable()
            self.wall4.enable()
            self.car.anti_cheat = 0.5

        if self.car.simple_intersects(self.wall_trigger_ramp):
            if self.car.anti_cheat == 0.5:
                self.car.anti_cheat = 1

        for i, cp in enumerate(self.checkpoints):
            if self.car.simple_intersects(cp) and not self.checkpoint_status[i]:
                self.checkpoint_status[i] = True
                print(f"Checkpoint {i} passed! Status: {self.checkpoint_status}")

                # Give bonus for reinforcement learning
                # if hasattr(self.car, "give_bonus_reward"):
                #     self.car.give_bonus_reward(i)