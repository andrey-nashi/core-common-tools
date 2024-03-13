import math
import pybullet as p


class RobotController:

    def __init__(self, ref_id, eff_index = None, joint_indices = None):
        self._robot_id = ref_id
        self._joints_total = p.getNumJoints(self._robot_id)
        self._eff_index = eff_index

        if self._eff_index is None:
            self._eff_index = self._joints_total - 1

        self._joint_indices = [i for i in range(0, self._joints_total)]
        if joint_indices is not None:
            self._joint_indices = joint_indices.copy()

        self._grasp_descriptor = None

        self._eff_position = None

        self._is_moving = False
        self._move_position_start = None
        self._move_position_end = None
        self._move_timestamp = None

    def get_joint_info(self):
        for i in range(self._joints_total):
            print(p.getJointInfo(self._robot_id, i))

    def set_pose(self, position, orientation):
        self._eff_position = position
        target_orn = p.getQuaternionFromEuler(orientation)
        joint_poses = p.calculateInverseKinematics(self._robot_id, self._eff_index, position, target_orn)

        index = 0
        for joint_index in self._joint_indices:

            p.setJointMotorControl2(bodyIndex=self._robot_id, jointIndex=joint_index, controlMode=p.POSITION_CONTROL,
                                    targetPosition=joint_poses[index])
            index += 1


    def reset_joints(self):
        for i in range(self._joints_total):
            p.resetJointState(self._robot_id, i, 0)
            p.setJointMotorControl2(self._robot_id, i, p.POSITION_CONTROL, 0, 0)

    def grasp(self, object_index):
        if self._grasp_descriptor is None:
            cube_orn = p.getQuaternionFromEuler([0, math.pi, 0])
            self._grasp_descriptor = p.createConstraint(self._robot_id, self._eff_index, object_index, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, -0.01], [0, 0, 0],
                                         )

    def release(self):
        if self._grasp_descriptor is not None:
            p.removeConstraint(self._grasp_descriptor)
            self._grasp_descriptor = None


    def move_to(self, pose, orientation, time_interval):

        if not self._is_moving:
            self._move_position_start = self._eff_position.copy()
            self._move_position_end = pose.copy()

            self._move_timestamp = 0
            self._is_moving = True
            return True
        else:

            self._move_timestamp += 1

            x_start, y_start, z_start = self._move_position_start
            x_end, y_end, z_end = self._move_position_end

            x = x_start + self._move_timestamp * (x_end - x_start) / time_interval
            y = y_start + self._move_timestamp * (y_end - y_start) / time_interval
            z = z_start + self._move_timestamp * (z_end - z_start) / time_interval

            target_orn = p.getQuaternionFromEuler(orientation)
            joint_poses = p.calculateInverseKinematics(self._robot_id, self._eff_index, [x, y, z], target_orn)

            index = 0
            for joint_index in self._joint_indices:
                p.setJointMotorControl2(bodyIndex=self._robot_id, jointIndex=joint_index,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=joint_poses[index])
                index += 1

            if self._move_timestamp == time_interval:
                self._is_moving = False
                self._eff_position = self._move_position_end.copy()
                self._move_position_start = None
                self._move_position_end = None
                self._move_timestamp = None
                return False

            return True