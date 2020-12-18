import glob
import os
import sys
import time
import math
import cv2
import numpy as np

try:
    sys.path.append(glob.glob('../CARLA_0.9.9.4/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from GripPredModel.src.Grip import my_load_model, run_test

SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480

class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 0.5
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.prius = self.blueprint_library.filter("prius")[0]

    def reset(self):
        self.laneChangeHist = []
        self.collision_hist = []
        self.actor_list = []

        # self.transform = random.choice(self.world.get_map().get_spawn_points())
        # SPAWN POINTS - 20, 22, 23, 41, 42, 43
        self.transform = self.world.get_map().get_spawn_points()[41]
        self.ego_vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.ego_vehicle)

        self.npc_transform = self.world.get_map().get_spawn_points()[4]
        self.agent_vehicle = self.world.spawn_actor(self.prius, self.npc_transform)
        self.actor_list.append(self.agent_vehicle)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.ego_vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.ego_vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        # self.lane_invasion_sensor = LaneInvasionSensor(self.ego_vehicle, self.hud)

        laneChangeSensor = self.blueprint_library.find("sensor.other.lane_invasion")
        self.laneChangeSensor = self.world.spawn_actor(laneChangeSensor, transform, attach_to=self.ego_vehicle)
        self.actor_list.append(self.laneChangeSensor)
        self.laneChangeSensor.listen(lambda event: self.laneChange(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera
    def laneChange(self, event):
        lane_type = set(x.type for x in event.crossed_lane_markings)
        self.laneChangeHist.append(lane_type)

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)
        #print(i.shape)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def step(self, action, prediction):
        thr = 0.8
        steer = 0.4
        if action == 0:
            print("ACTION::::::       <------ LEFT")
            self.ego_vehicle.apply_control(carla.VehicleControl(throttle=thr, steer=-steer))

        elif action == 1:
            print("ACTION::::::               RIGHT ------> ")
            self.ego_vehicle.apply_control(carla.VehicleControl(throttle=thr, steer=+steer))

        elif action == 2:
            print("ACTION::::::            | STRAIGHT |")
            self.ego_vehicle.apply_control(carla.VehicleControl(throttle=thr, steer= 0))


        v = self.ego_vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        transform = self.ego_vehicle.get_transform()

        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        elif kmh < 50:
            done = False
            reward = -1
        else:
            done = False
            reward = 1

        # if self.episode_start + SECONDS_PER_EPISODE < time.time():
        #     done = True

        return self.front_camera, reward, done, None

    def predict(self,model, frames_list, graph_args):
        predictions = run_test(model, frames_list, graph_args)
        return predictions
