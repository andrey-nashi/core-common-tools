import os
import cv2
import json
import time
import threading
import numpy as np

from .cam_abstract import CameraAbstract

# -----------------------------------------------------------------------------------------

class CameraMock(CameraAbstract):

    def __init__(self, camera_name: str = None):
        """
        Implementation of a mock camera that would return pre-defined images with given FPS
        :param camera_name: unique name of the camera sensor
        """
        super().__init__(camera_name)
        self.images = []
        self.image_index = 0
        self.fps = 0

        self.s_open = False
        self.s_callback_hook = None
        self.s_thread = None

    def setup_from_file(self, path_json: str):
        """
        Load setting from JSON file
        :param path_json: path to JSON file that has the following fields
        - image_paths: [<path>, <path>] <- list of RGB image paths
        - camera_name: str <- name of the camera (will be set to None if not specified)
        - fps: int <- frames per second for camera simulation
        :return:
        """
        f = open(path_json, "r")
        data = json.load(f)
        f.close()

        args = {k: v for k, v in data.items() if k in self.setup_from_args.__code__.co_varnames}
        self.setup_from_args(**args)

    def setup_from_args(self, image_paths: list, camera_name: str = None, fps: int = 30):
        """
        Initialize camera parameters
        :param image_paths: list of image paths [<path>, <path>, ...]
        :param camera_name: camera name (ideally a unique identified)
        :param fps: frames per second for camera simulation
        :return:
        """
        self.camera_name = camera_name
        self.fps = fps

        for path in image_paths:
            if os.path.exists(path):
                image = cv2.imread(path)
                self.images.append(image)

        self.is_setup_ok = True

    def open(self) -> bool:
        """
        Open camera (setup has to be done in advance)
        :return: True if all is ok, False if failed
        """
        if not self.is_setup_ok: return False
        self.is_open = True
        return True

    def close(self) -> bool:
        """
        Close camera (if it was open before)
        :return: True if all is ok, False if failed
        """
        if not self.is_open: return False
        self.is_open = False
        return True

    def get_frame(self) -> np.ndarray:
        """
        Get one frame, for this mock camera will loop over defined images
        :return: numpy array - an opencv image
        """
        if not self.is_open: return None
        time.sleep(1 / self.fps)
        output = self.images[self.image_index]
        self.image_index = 0 if self.image_index == len(self.images) - 1 else self.image_index + 1
        return output

    def _thread_get_frame(self):
        while self.s_open:
            image = self.get_frame()
            self.s_callback_hook(image)

    def stream_start(self, func_hook: callable) -> bool:
        """
        Start a separate thread to stream images invoking the passed hook function and passing the image
        :param func_hook: call back function that would accept an image as a single argument
        :return: True if all is ok, False if failed
        """
        if not self.is_open: return False
        self.s_callback_hook = func_hook
        self.s_open = True
        self.s_thread = threading.Thread(target=self._thread_get_frame, args=())
        self.s_thread.start()
        return False

    def stream_stop(self) -> bool:
        """
        Stop streaming frames
        :return: True if all is ok, False if failed
        """
        if not self.s_open: return False
        self.s_open = False
        self.s_thread.join()
        self.s_callback_hook = None
        return True

