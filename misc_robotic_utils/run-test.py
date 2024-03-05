import pybullet as p
import time
import pybullet_data
import math

from core.simulator.sim_camera import Camera
from core.simulator.sim_library import MeshLibrary
from core.simulator.sim_levelmap import LevelMap


physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.8)
planeId = p.loadURDF("plane.urdf")


path_mesh_lib = "resources/mesh"
path_level = "resources/sim_level.yaml"
path_cfg = "resources/sim_cfg.yaml"

mesh_library = MeshLibrary(path_mesh_lib)
level = LevelMap(path_level, mesh_library)
x = p.getNumJoints(level.robot.ref_id)
print(">>>>", x)
for i in range(0, x):
    print(p.getJointInfo(level.robot.ref_id, i))

target_orn = p.getQuaternionFromEuler([0, - math.pi, 0])
target_pos = [1, 0, 1]
joint_poses = p.calculateInverseKinematics(level.robot.ref_id, 14, target_pos, target_orn)
print(joint_poses)
index = [2, 3, 4, 6, 7, 8]
for j in range(0, len(joint_poses)):
    p.setJointMotorControl2(bodyIndex=level.robot.ref_id, jointIndex=index[j], controlMode=p.POSITION_CONTROL,
                            targetPosition=joint_poses[j])

for i in range(0, 10):
    name = "can_" + str(i)
    mesh = "cola"
    origin = [2, 2, 1 + 0.01 * i]
    scale = 5
    level.spawn(name, mesh, origin, scale)

pycam = Camera()
pycam.open()

frame_counter = 0

while True:
    p.stepSimulation()
    time.sleep(1./240)

pycam.capture()

rgb_image = pycam.get_rgb()
depth_data = pycam.get_depth()
depth_data = (depth_data - np.min(depth_data)) / (np.max(depth_data)  - np.min(depth_data)) * 255
segmentation_mask = pycam.get_mask()
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
cv2.imwrite("rgb.png", rgb_image )
cv2.imwrite("depth.png", depth_data)
cv2.imwrite( "segmentation.png", segmentation_mask)

p.disconnect()
