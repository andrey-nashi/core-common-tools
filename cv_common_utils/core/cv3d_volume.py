import os
import cv2
import numpy as np

class Volume:

    def __init__(self, array: np.ndarray = None):
        self.array = array

    def init_with_zeros(self, width: int, height: int, z: int):
        self.array = np.zeros((z, height, width))

    def init_with_ones(self, width: int, height: int, z: int):
        self.array = np.zeros((z, height, width))

    def __eq__(self, obj):
        if isinstance(obj, Volume): return self.array == obj.array
        if isinstance(obj, np.ndarray): return self.array == obj

    def __ne__(self, obj):
        if isinstance(obj, Volume): return self.array != obj.array
        if isinstance(obj, np.ndarray): return self.array != obj

    def __add__(self, obj):
        if self.array is None: return None
        if isinstance(obj, Volume):
            self.array = self.array + obj.array
            return self
        if isinstance(obj, np.ndarray):
            self.array = self.array + obj
            return self

    def init_zeros_like(self, v: np.ndarray):
        """
        Init with zeros, same shape as given 'v'
        :param v: volume
        """
        s = v.shape()
        self.array = np.zeros((s[0], s[1], s[2]))

    def init_ones_like(self, v):
        """
        Init with ones, same shape as given 'v'
        :param v: volume
        """
        s = v.shape
        self.array = np.ones((s[0], s[1], s[2]))

    @property
    def shape(self):
        """
        Get shape of the volume
        :return: [z, height, width]
        """
        if self.array is not None:
            z, h, w = self.array.shape
            return [z, h, w]
        else:
            return [0, 0, 0]

    def get_slice(self, index: int, axis: int = 0):
        if self.array is None or axis not in [0, 1, 2]:
            return None
        z, h, w = self.array.shape

        if 0 <= index < z:
            if axis == 0: return self.array[index, :, :]
            if axis == 1: return self.array[:, index, :]
            if axis == 2: return self.array[:, :, index]

        return None

    def set_slice(self, slice: np.ndarray, index: int, axis: int = 0):
        if self.array is None: return

        z, h, w = self.array.shape
        s_h, s_w = slice.shape

        if axis == 0:
            if (0 <= index < z) and (s_h == h and s_w == w):
                self.array[index, :, :] = slice
        if axis == 1:
            if (0 <= index < h) and (s_h == z and s_w == w):
                self.array[:, index, :] = slice
        if axis == 2:
            if (0 <= index < w) and (s_h == z and s_w == w):
                self.array[:, :, index] = slice

    def add_slice_nonzero(self, mask: np.ndarray, index: int):
        if self.array is None: return

        self.array[index, self.array[index, :, :] == 0] = mask[self.array[index, :, :] == 0]

    def add_slice(self, slice: np.ndarray, index: int, weight: float = 1.0):
        if self.array is None: return

        z, h, w = self.array.shape
        s_h, s_w = slice.shape

        if (0 <= index < z) and (s_h == h and s_w == w):
            self.array[index, :, :] += weight * slice

    def get_slices(self, axis: int = 0):
        output = []
        if self.array is None: return output

        z, h, w = self.array.shape

        if axis == 0:
            for i in range(0, z):
                output.append(self.array[i, :, :])
        if axis == 1:
            for i in range(0, h):
                output.append(self.array[:, i, :])
        if axis == 2:
            for i in range(0, w):
                output.append(self.array[:, :, i])

    def save(self, path_dir: str, axis: int = 0, m: list = (1, 1, 1), is_rgb = False):
        assert type(m) == list or type(m) == tuple
        assert len(m) == 3
        if self.array is None or axis not in [0, 1, 2]: return

        z, h, w = self.array.shape


        if axis == 0:
            for index in range(0, z):
                slice = self.array[index, :, :]
                if m[1] != 1 or m[2] != 1:
                    h = slice.shape[0]
                    w = slice.shape[1]
                    slice = cv2.resize(slice, (w * m[2], h * m[1]), cv2.INTER_LINEAR_EXACT)
                cv2.imwrite(os.path.join(path_dir, "sl-" + str(index) + ".png"), slice)
        if axis == 1:
            for index in range(0, h):
                slice = self.array[:, index, :]
                if m[0] != 1 or m[2] != 1:
                    z = slice.shape[0]
                    w = slice.shape[1]
                    slice = cv2.resize(slice, (w * m[2], z * m[0]), cv2.INTER_LINEAR_EXACT)
                cv2.imwrite(os.path.join(path_dir, "sl-" + str(index) + ".png"), slice)
        if axis == 2:
            for index in range(0, w):
                slice = self.array[:, :, index]
                if m[0] != 1 or m[1] != 1:
                    z = slice.shape[0]
                    h = slice.shape[1]
                    slice = cv2.resize(slice, (h * m[1], z * m[0]), cv2.INTER_LINEAR_EXACT)
                    print(slice.shape)
                cv2.imwrite(os.path.join(path_dir, "sl-" + str(index) + ".png"), slice)

    def load(self, path_dir: str):
        self.array = None

        fl = [f for f in os.listdir(path_dir) if os.path.isfile(os.path.join(path_dir, f)) and f.endswith(".png")]
        fl.sort()

        for index in range(0, len(fl)):
            file = fl[index]
            image = cv2.imread(os.path.join(path_dir, file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = image.shape

            if self.array is None:
                self.array = np.zeros((len(fl), h, w))
            if self.array.shape[1] != h or self.array.shape[2] != w:
                raise RuntimeError

            self.array[index, :, :] = image

    def load_partial(self, path_dir: str, depth: int = None):
        self.array = None

        fl = [f for f in os.listdir(path_dir) if os.path.isfile(os.path.join(path_dir, f)) and f.endswith(".png")]
        fl.sort()

        if depth is None:
            depth = 0
            for file in fl:
                try:
                    slice_id = file.replace(".png").split("_")[-1]
                    slice_id = int(slice_id)
                    depth = max(slice_id, depth)
                except:
                    continue

        for index in range(0, len(fl)):
            file = fl[index]
            image = cv2.imread(os.path.join(path_dir, file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = image.shape

            if self.array is None:
                self.array = np.zeros((depth, h, w))

            # ---- Get slice index from file name, here the assumptions is
            # ---- the file name ends with "_***"
            try:
                slice_id = file.replace(".png", "").split("_")[-1]
                slice_id = int(slice_id)
                if 0 <= slice_id < depth:
                    self.array[slice_id, :, :] = image
            except:
                continue

    def show(self):
        from matplotlib import pyplot as plt
        shape = self.array.shape
        r_volume = Volume()
        r_volume.init_zeros_like(self)

        for slice_index in range(0, shape[0]):
            slice = np.copy(self.get_slice(slice_index))
            slice[slice != 0] = 255
            slice = slice.astype(np.uint8)
            contours, hierarchy = cv2.findContours(slice, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            slice[:,:] = 0
            cv2.drawContours(slice, contours, -1, 255, 1)
            r_volume.add_slice_nonzero(slice, slice_index)


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        z, x, y = r_volume.array.nonzero()
        color = []
        for i in range(0, len(z)):
            color.append(self.array[z[i], x[i], y[i]])

        ax.scatter(x, y, z, c=color, alpha=1)
        plt.show()