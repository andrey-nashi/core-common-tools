from .ann_label import Label

class Box:

    XY_ABSOLUTE = 0
    XY_RELATIVE = 1
    XY_FORMAT_CORNER = 0
    XY_FORMAT_CENTER = 1

    def __init__(self, min_x: int = None, min_y: int = None, max_x: int = None, max_y: int = None, labels: list = []):
        self.xy_mode = self.XY_ABSOLUTE
        self.xy_format = self.XY_FORMAT_CORNER
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        if labels is not None:
            self.labels = labels.copy()
        else:
            self.labels = []


    def assign_coordinates(self, min_x: int, min_y: int, max_x: int, max_y: int):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y

    def assign_label(self, label: Label):
        self.labels.append(label)

    def xy_to_relative(self, width: int, height: int):
        """
        Con
        :param width:
        :param height:
        :return:
        """
        if self.xy_mode == self.XY_ABSOLUTE:
            self.min_x = self.min_x / width
            self.max_x = self.max_x / width
            self.min_y = self.min_y / height
            self.max_y = self.max_y / height
            self.xy_mode = self.XY_RELATIVE

    def xy_to_absolute(self, width: int, height: int):
        if self.xy_mode == self.XY_RELATIVE:
            self.min_x *= width
            self.min_y *= height
            self.max_x *= width
            self.max_y *= height
            self.xy_mode = self.XY_ABSOLUTE

    def xy_to_center(self):
        if self.xy_format == self.XY_FORMAT_CORNER:
            center_x = (self.min_x + self.max_x) / 2
            center_y = (self.min_y + self.max_y) / 2
            bbox_width = self.max_x - self.min_x
            bbox_height = self.max_y - self.min_y

            self.min_x = center_x
            self.min_y = center_y
            self.max_x = bbox_width
            self.max_y = bbox_height

    def xy_to_corner(self):
        if self.xy_format == self.XY_FORMAT_CENTER:
            center_x = self.min_x
            center_y = self.max_y
            bbox_width = self.max_x
            bbox_height = self.max_y

            self.min_x = int(center_x - bbox_width / 2)
            self.min_y = int(center_y - bbox_height / 2)
            self.max_x = int(center_x + bbox_width / 2)
            self.max_y = int(center_y + bbox_width / 2)
