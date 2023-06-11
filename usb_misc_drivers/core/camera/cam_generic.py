import cv2
import json
import time
import threading
import numpy as np

from .cam_abstract import CameraAbstract

#-----------------------------------------------------------------------------------------
class CameraGeneric(CameraAbstract):

    def __init__(self, camera_name: str = None):
        super().__init__(camera_name)
        self.camera_index = 0
        self.camera_driver = None

        self.s_open = False
        self.s_callback_hook = None
        self.s_thread = None

    def setup_from_file(self, path_json: str):
        f = open(path_json, "r")
        data = json.load(f)
        f.close()

        args = {k: v for k, v in data.items() if k in self.setup_from_args.__code__.co_varnames}
        self.setup_from_args(**args)

    def setup_from_args(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.is_setup_ok = True


    def open(self) -> bool:
        if not self.is_setup_ok: return False
        self.camera_driver = cv2.VideoCapture(self.camera_index)
        self.is_open = True
        return True

    def close(self) -> bool:
        if not self.is_open: return False
        self.is_open = False
        return True

    def get_frame(self) -> np.ndarray:
        ret, frame = self.camera_driver.read()
        return frame

    def _thread_get_frame(self):
        while self.s_open:
            ret, frame = self.camera_driver.read()
            self.s_callback_hook(frame)

    def stream_start(self, func_hook: callable) -> bool:
        if not self.is_open: return False
        self.s_callback_hook = func_hook
        self.s_open = True
        self.s_thread = threading.Thread(target=self._thread_get_frame, args=())
        self.s_thread.start()
        return True


    def stream_stop(self):
        self.s_open = False
        self.s_thread.join()
        self.s_callback_hook = None