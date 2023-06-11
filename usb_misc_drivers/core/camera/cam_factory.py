from .cam_mock import CameraMock
from .cam_generic import CameraGeneric
from .cam_abstract import CameraAbstract

#-----------------------------------------------------------------------------------------

class CameraFactory:

    _LIST_CAMERA_CLASSES = [
        CameraMock,
        CameraGeneric
    ]

    _TABLE_CAMERA_CLASSES = {m.__name__:m for m in _LIST_CAMERA_CLASSES}

    @staticmethod
    def get_by_name(camera_class_name: str, sensor_name: str) -> CameraAbstract:
        if camera_class_name in CameraFactory._TABLE_CAMERA_CLASSES:
            return CameraFactory._TABLE_CAMERA_CLASSES[camera_class_name](sensor_name)




