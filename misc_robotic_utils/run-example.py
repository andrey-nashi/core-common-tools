import math
import pybullet as p
import pybullet_data
from core.simulator.robot import RobotController

p.connect(p.GUI)  # or p.GUI for graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)

plane_id = p.loadURDF("plane.urdf")
path = "kuka_iiwa/model_vr_limits.urdf"
#path = "./resources/mesh/robot/fanuc_robot.urdf"


kuka_id = p.loadURDF(path, 1.400000, -0.200000, 0.600000, 0.000000, 0.000000, 0.000000, 1.000000)
table_id = p.loadURDF("table/table.urdf", basePosition=[1.0, -0.2, 0.0], baseOrientation=[0, 0, 0.7071, 0.7071])
cube_id = p.loadURDF("cube.urdf", basePosition=[0.85, -0.2, 0.65], globalScaling=0.05)

rc = RobotController(kuka_id)


kuka_cid = None



rc.reset_joints()

num_joints = p.getNumJoints(kuka_id)
kuka_end_effector_idx = 6

# camera parameters
cam_target_pos = [.95, -0.2, 0.2]
cam_distance = 2.05
cam_yaw, cam_pitch, cam_roll = -50, -40, 0
cam_width, cam_height = 480, 360

cam_up, cam_up_axis_idx, cam_near_plane, cam_far_plane, cam_fov = [0, 0, 1], 2, 0.01, 100, 60



def move_to_location(pos_start, pos_end, time_start, time_interval, time_now):
    x_start, y_start, z_start = pos_start
    x_end, y_end, z_end = pos_end

    x = x_start + (time_now - time_start) * (x_end - x_start) / time_interval
    y = y_start + (time_now - time_start) * (y_end - y_start) / time_interval
    z = z_start + (time_now - time_start) * (z_end - z_start) / time_interval
    return [x, y, z]

pose_above_1 = [0.85, -0.2, 0.95]
pose_object_1 = [0.85, -0.2, 0.75]
pose_above_2 = [0.85, 0.2, 0.95]
pose_object_2 = [0.85, 0.2, 0.75]
move_time = 200

target_pos = None
is_grab = None

t = -1
while True:
    t += 1
    if t % 8 == 0:
        cam_view_matrix = p.computeViewMatrixFromYawPitchRoll(cam_target_pos, cam_distance, cam_yaw, cam_pitch,
                                                              cam_roll, cam_up_axis_idx)
        cam_projection_matrix = p.computeProjectionMatrixFOV(cam_fov, cam_width * 1. / cam_height, cam_near_plane,
                                                             cam_far_plane)
        image = p.getCameraImage(cam_width, cam_height, cam_view_matrix, cam_projection_matrix)[2][:, :, :3]

    if 0 * move_time < t <= 1 * move_time:
        target_pos = [0.85, -0.2, 0.95]
        is_grab = 0
    if 1 * move_time < t < 2 * move_time:
        target_pos = move_to_location(pose_above_1, pose_object_1, 1 * move_time, move_time, t)
        is_grab = 0
    if t  == 2 * move_time:
        is_grab = 1
    if 2 * move_time < t < 3*move_time:
        target_pos = move_to_location(pose_object_1, pose_above_1, 2 * move_time, move_time, t)
        is_grab = 1
    if 3 * move_time < t < 4 * move_time:
        target_pos = move_to_location(pose_above_1, pose_above_2, 3 * move_time, move_time, t)
        is_grab = 1
    if 4 * move_time < t < 5 * move_time:
        target_pos = move_to_location(pose_above_2, pose_object_2, 4 * move_time, move_time, t)
        is_grab = 1
    if t == 5 * move_time:
        is_grab = 0
    if 5 * move_time < t < 6 * move_time:
        target_pos = move_to_location(pose_object_2, pose_above_2, 5 * move_time, move_time, t)
        is_grab = 0
    if 6 * move_time < t < 7 * move_time:
        target_pos = move_to_location(pose_above_2, pose_object_2, 6 * move_time, move_time, t)
        is_grab = 0
    if t == 7 * move_time:
        is_grab = 1
    if 7 * move_time < t < 8 * move_time:
        target_pos = move_to_location(pose_object_2, pose_above_2, 7 * move_time, move_time, t)
        is_grab = 1
    if 8 * move_time < t < 9 * move_time:
        target_pos = move_to_location(pose_above_2, pose_above_1, 8 * move_time, move_time, t)
        is_grab = 1
    if 9 * move_time < t < 10 * move_time:
        target_pos = move_to_location(pose_above_1, pose_object_1, 9 * move_time, move_time, t)
        is_grab = 1
    if 10 * move_time == t:
        is_grab = 0
    if 10 * move_time < t < 11 * move_time:
        target_pos = move_to_location(pose_object_1, pose_above_1, 10 * move_time, move_time, t)
        is_grab = 0
    if t == 11 * move_time:
        t = 0

    if target_pos is None: continue


    rc.set_pose(target_pos, [0, 1.01 * math.pi, 0])


    if is_grab == 0 and kuka_cid != None:
        p.removeConstraint(kuka_cid)
        kuka_cid = None
    if is_grab == 1 and kuka_cid == None:
        cube_orn = p.getQuaternionFromEuler([0, math.pi, 0])
        kuka_cid = p.createConstraint(kuka_id, 6, cube_id, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0.05], [0, 0, 0],
                                      childFrameOrientation=cube_orn)

    p.stepSimulation()

p.disconnect()