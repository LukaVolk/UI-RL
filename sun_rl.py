from panda3d.core import DirectionalLight
from ursina import Entity

class SunLightRL(Entity):
    def __init__(self, direction, resolution, cars):
        super().__init__()

        self.cars = cars
        self.resolution = resolution

        self.dlight = DirectionalLight("sun")
        self.dlight.setShadowCaster(True, self.resolution, self.resolution)

        lens = self.dlight.getLens()
        lens.setNearFar(-80, 200)
        lens.setFilmSize((100, 100))

        self.dlnp = render.attachNewNode(self.dlight)
        self.dlnp.lookAt(direction)
        render.setLight(self.dlnp)

    def update(self):
        for car in self.cars:
            if car.visible:
                self.dlnp.setPos(car.world_position)
                break

    def update_resolution(self):
        self.dlight.setShadowCaster(True, self.resolution, self.resolution)