

class CameraAbstract:

    def __init__(self, camera_name: str):
        """
        Definition of the basic camera interface.
        :param camera_name: unique name of the camera sensor
        """
        self.camera_name = camera_name
        self.is_open = False
        self.is_setup_ok = False

    def setup_from_file(self, **kwargs):
        raise NotImplemented

    def setup_from_args(self, **kwargs):
        raise NotImplemented

    def open(self):
        raise NotImplemented

    def close(self):
        raise NotImplemented

    def get_frame(self):
        raise NotImplemented

    def stream_start(self, func_hook: callable):
        raise NotImplemented

    def stream_stop(self):
        raise NotImplemented