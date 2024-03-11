

class BoundingBox2D:

    _MODE_XY_ABSOLUTE = 0
    _MODE_XY_RELATIVE = 1

    def __init__(self):
        self._x_min = None
        self._x_max = None
        self._y_min = None
        self._y_max = None
        self._width = None
        self._height = None
        self._mode = None

    def set_xy(self, x_min: int, y_min: int, x_max: int, y_max: int, mode: int, width: int = None, height: int = None):
        if mode == self._MODE_XY_ABSOLUTE:
            self.set_xy_absolute(x_min, y_min, x_max, y_max, width, height)
        if mode == self._MODE_XY_RELATIVE:
            self.set_xy_relative(x_min, y_min, x_max, y_max, width, height)

    def set_xy_absolute(self, x_min: int, y_min: int, x_max: int, y_max: int, width: int = None, height: int = None) -> None:
        self._x_min = x_min
        self._x_max = x_max
        self._y_min = y_min
        self._y_max = y_max
        self._width = width
        self._height = height
        self._mode = self._MODE_XY_ABSOLUTE

    def set_xy_relative(self, x_min: float, y_min: float, x_max: float, y_max: float, width: int = None, height: int = None) -> None:
        self._x_min = x_min
        self._x_max = x_max
        self._y_min = y_min
        self._y_max = y_max
        self._width = width
        self._height = height
        self._mode = self._MODE_XY_RELATIVE

    def get_xy_absolute(self) -> tuple:
        if self._mode == self._MODE_XY_ABSOLUTE:
            return (self._x_min, self._y_min, self._x_max, self._y_max)
        else:
            x_min = int(self._x_min * self._width)
            x_max = int(self._x_max * self._width)
            y_min = int(self._y_min * self._height)
            y_max = int(self._y_max * self._height)
            return (x_min, y_min, x_max, y_max)

    def get_xy_relative(self) -> tuple:
        if self._mode == self._MODE_XY_RELATIVE:
            return (self._x_min, self._y_min, self._x_max, self._y_max)
        else:
            x_min = self._x_min / self._width
            x_max = self._x_max / self._width
            y_min = self._y_min / self._height
            y_max = self._y_max / self._height
            return (x_min, y_min, x_max, y_max)