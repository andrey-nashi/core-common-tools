import numpy as np
import pybullet as p
import time
import pybullet_data
import math
import cv2

from core.simulator.sim_camera import Camera
from core.simulator.sim_library import MeshLibrary
from core.simulator.sim_levelmap import LevelMap
from core.simulator.robot import RobotController

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.8)
planeId = p.loadURDF("plane.urdf")



path_mesh_lib = "resources/mesh"
path_level = "resources/sim_level.yaml"
path_cfg = "resources/sim_cfg.yaml"

mesh_library = MeshLibrary(path_mesh_lib)
level = LevelMap(path_level, mesh_library)


ROBOT_BASE = 1
rc = RobotController(level.robot.ref_id, 14, [2, 3, 4, 6, 7, 8])
rc.set_pose([0.485, 0.027, ROBOT_BASE + 0.4], [math.pi, 0, 0])
rc.get_joint_info()


for i in range(0, 3):
    name = "can_" + str(i)
    mesh = "cola"
    origin = [1.041, -0.39, 0.94  + 0.01 * i]
    scale = 5
    level.spawn(name, mesh, origin, scale=scale)
    print(name, level._table[name].ref_id)

pycam = Camera(look_at=[1.041, -0.39, 0.94], distance=0.4)
pycam.open()

frame_counter = 0

pose_tote_1 = [1.041, -0.39, 1]
pose_tote_2 = [1.041, 0.39, 1]

action = 0

def detect(pycam):
    seg = pycam.get_mask()
    depth = pycam.get_depth()
    print(np.unique(seg))
    x = np.mean(np.argwhere(seg == 4), axis=0)
    y = np.mean(np.argwhere(seg == 4), axis=1)
    #z = depth[x, y]
    #print (x, y, z)
    out = np.copy(seg) * 64
    out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    cv2.imwrite("segmentation.png", out)


t = 0
while True:
    t += 1

    if t > 480:

        if action == 0:
            pycam.capture()
            detect(pycam)
            action = 1
        elif action == 1:
            flag = rc.move_to(pose_tote_1, [math.pi, 0, 0], 240)
            if not flag:
                print("SWITCHING TO NEXT")
                action = 2
        elif action == 2:
            flag = rc.move_to(pose_tote_2, [math.pi, 0, 0], 240)
            if not flag:
                print("SWITCHING TO NEXT")
                action = 1


    p.stepSimulation()





rgb_image = pycam.get_rgb()
depth_data = pycam.get_depth()
depth_data = (depth_data - np.min(depth_data)) / (np.max(depth_data)  - np.min(depth_data)) * 255
segmentation_mask = pycam.get_mask()
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
cv2.imwrite("rgb.png", rgb_image )
cv2.imwrite("depth.png", depth_data)
cv2.imwrite( "segmentation.png", segmentation_mask)

p.disconnect()
