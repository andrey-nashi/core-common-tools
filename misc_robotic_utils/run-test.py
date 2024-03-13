
import pybullet as p
import pybullet_data
import math

from core.simulator.camera import Camera
from core.simulator.meshlib import MeshLibrary
from core.simulator.levelmap import LevelMap
from core.simulator.robot import RobotController
from core.simulator.action import ActionSequence


physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.8)
planeId = p.loadURDF("plane.urdf")

path_mesh_lib = "resources/mesh"
path_level = "resources/sim_level.yaml"
path_cfg = "resources/sim_cfg.yaml"

mesh_library = MeshLibrary(path_mesh_lib)
level = LevelMap(path_level, mesh_library)


ROBOT_EFF_INDEX = 14
ROBOT_JOINT_INDICES = [2, 3, 4, 6, 7, 8]
ROBOT_POSE_HOME = [0.485, 0.027, 1 + 0.4]
ROBOT_POSE_TOTE1 = [1.041, -0.39, 1]
ROBOT_POSE_TOTE2 = [1.041, 0.39, 1]


ROBOT_BASE = 1
rc = RobotController(level.robot.ref_id, ROBOT_EFF_INDEX, ROBOT_JOINT_INDICES)
rc.set_pose(ROBOT_POSE_HOME, [math.pi, 0, 0])
rc.get_joint_info()


for i in range(0, 3):
    name = "can_" + str(i)
    mesh = "cola"
    origin = [1.041, -0.39, 0.94  + 0.005 * i]
    scale = 5
    level.spawn(name, mesh, origin, scale=scale)


camera_1 = Camera(look_at=[1.041, -0.39, 0.94], distance=0.4)
camera_1.open()

camera_2 = Camera(look_at=[1.041, 0.39, 0.94], distance=0.4)
camera_2.open()

sequence = ActionSequence(rc, {"camera_1": camera_1, "camera_2": camera_2})
sequence.add_action(ActionSequence.ACTION_WAIT, {"t": 50})
sequence.add_action(ActionSequence.ACTION_CAPTURE, {"camera_id": "camera_1"})
sequence.add_action(ActionSequence.ACTION_MOVE, {"pose": ROBOT_POSE_TOTE1,"orientation": [math.pi, 0, 0], "time_interval": 100})
sequence.add_action(ActionSequence.ACTION_MOVE, {"pose": "can", "orientation": [math.pi, 0, 0], "time_interval": 100})
sequence.add_action(ActionSequence.ACTION_WAIT, {"t": 50})
sequence.add_action(ActionSequence.ACTION_GRASP, {})
sequence.add_action(ActionSequence.ACTION_MOVE, {"pose": ROBOT_POSE_TOTE1,"orientation": [math.pi, 0, 0], "time_interval": 100 })
sequence.add_action(ActionSequence.ACTION_MOVE, {"pose": ROBOT_POSE_TOTE2,"orientation": [math.pi, 0, 0], "time_interval": 100 })
sequence.add_action(ActionSequence.ACTION_WAIT, {"t": 50})
sequence.add_action(ActionSequence.ACTION_RELEASE, {})
sequence.add_action(ActionSequence.ACTION_MOVE, {"pose": ROBOT_POSE_HOME, "orientation": [math.pi, 0, 0], "time_interval": 100})
sequence.add_action(ActionSequence.ACTION_WAIT, {"t": 50})
sequence.add_action(ActionSequence.ACTION_CAPTURE, {"camera_id": "camera_2"})
sequence.add_action(ActionSequence.ACTION_MOVE, {"pose": "can", "orientation": [math.pi, 0, 0], "time_interval": 100})
sequence.add_action(ActionSequence.ACTION_WAIT, {"t": 50})
sequence.add_action(ActionSequence.ACTION_GRASP, {})
sequence.add_action(ActionSequence.ACTION_MOVE, {"pose": ROBOT_POSE_TOTE2,"orientation": [math.pi, 0, 0], "time_interval": 100 })
sequence.add_action(ActionSequence.ACTION_MOVE, {"pose": ROBOT_POSE_TOTE1,"orientation": [math.pi, 0, 0], "time_interval": 100 })
sequence.add_action(ActionSequence.ACTION_WAIT, {"t": 50})
sequence.add_action(ActionSequence.ACTION_RELEASE, {})
sequence.add_action(ActionSequence.ACTION_WAIT, {"t": 50})
sequence.add_action(ActionSequence.ACTION_MOVE, {"pose": ROBOT_POSE_HOME, "orientation": [math.pi, 0, 0], "time_interval": 100})
sequence.add_action(ActionSequence.ACTION_RESET, {})

while True:
    sequence.update()
    p.stepSimulation()

p.disconnect()
