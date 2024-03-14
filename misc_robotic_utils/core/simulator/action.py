import random
import pybullet as p

class ActionSequence:

    ACTION_WAIT = 0
    ACTION_MOVE = 1
    ACTION_GRASP = 2
    ACTION_RELEASE = 3
    ACTION_CAPTURE = 4
    ACTION_RESET = 5

    def __init__(self, robot_control, camera_table):
        self._action_id = 0
        self._action_sequence = []
        self._time = None
        self._robot_control = robot_control
        self._temp = {}
        self._camera_table = camera_table
        self._obj = 4
    def update(self):
        if self._action_id >= len(self._action_sequence):
            self._action_id = len(self._action_sequence) - 1

        action_cfg = self._action_sequence[self._action_id]
        action_id = action_cfg["id"]
        action_args = action_cfg["args"]

        if action_id == self.ACTION_WAIT:
            self._action_wait(action_args)
        elif action_id == self.ACTION_MOVE:
            self._action_move(action_args)
        elif action_id == self.ACTION_CAPTURE:
            self._action_capture(action_args)
        elif action_id == self.ACTION_GRASP:
            self._action_grasp(action_args)
        elif action_id == self.ACTION_RELEASE:
            self._action_release(action_args)
        elif action_id == self.ACTION_RESET:
            self._action_reset(action_args)

    def _action_wait(self, action_args):
        if self._time is None:
            self._time = 0
        else:
            time_target = action_args["t"]
            if self._time == time_target:
                self._action_id += 1
                self._time = None
                return
            self._time += 1

    def _action_move(self, action_args):
        args = action_args.copy()
        if isinstance(action_args["pose"], str):
            args["pose"] = self._temp[action_args["pose"]]
        is_moving = self._robot_control.move_to(**args)
        if not is_moving: self._action_id += 1

    def _action_grasp(self, action_args):
        self._robot_control.grasp(self._obj)
        self._action_id += 1

    def _action_release(self, action_args):
        self._robot_control.release()
        self._action_id += 1

    def _action_capture(self, action_args):
        camera_id = action_args["camera_id"]
        obj_ids = action_args["obj_ids"]
        self._camera_table[camera_id].capture()
        self._obj, obj_pose_cam = self._camera_table[camera_id].find_object(obj_ids)

        self._temp["can"] = list(obj_pose_cam)
        self._action_id += 1

    def _action_reset(self, action_args):
        self._action_id = 0

    def add_action(self, action_id, action_args):
        self._action_sequence.append({
            "id": action_id,
            "args": action_args
        })

