import pybullet as p
import yaml

from .meshlib import *
# ----------------------------------------------------------------------------------------

class Object3D:

    def __init__(self, name: str, mesh: str, descriptor: MeshDescriptor, origin: tuple = (0, 0, 0), orientation: tuple = None, scale: int = 1,  is_static: bool = False):
        """
        Initialize a 3D object specified by a mesh descriptor
        :param name: unique name of the object in the level
        :param mesh: reference name of the mesh in the mesh library
        :param descriptor: mesh descriptor
        :param origin: [X,Y,Z] - origin coordinate to place the object at
        :param orientation: [X,Y,Z,W] - orientation of the object
        :param scale: scale of the mesh object
        :param is_static: if True the object is static
        """

        self.name = name
        self.mesh = mesh
        self.origin = origin
        self.orientation = orientation
        self.scale = scale
        self.is_static = is_static

        if orientation is None:
            self.ref_id = p.loadURDF(descriptor.urdf, basePosition=self.origin, globalScaling=self.scale,
                                    useFixedBase=self.is_static)
        else:
            self.ref_id = p.loadURDF(descriptor.urdf, basePosition=self.origin, globalScaling=self.scale,
                                    useFixedBase=self.is_static, baseOrientation=self.orientation)

# ----------------------------------------------------------------------------------------

class LevelMap:

    KEY_SCENE = "scene"
    KEY_NAME = "name"
    KEY_MESH = "mesh"
    KEY_ORIGIN = "origin"
    KEY_SCALE = "scale"
    KEY_STATIC = "is_static"
    KEY_ORIENTATION = "orientation"

    def __init__(self, path_level: str, mesh_library: MeshLibrary):
        """
        Initialize entire scene with object specified in the level file
        :param path_level: absolute path to the level file
        :param mesh_library: mesh library
        """

        self._library = mesh_library
        self._table = {}

        f = open(path_level, "r")
        data = yaml.safe_load(f)
        f.close()

        for mesh_cfg in data[self.KEY_SCENE]:
            name = mesh_cfg[self.KEY_NAME]
            mesh = mesh_cfg[self.KEY_MESH]
            origin = mesh_cfg.get(self.KEY_ORIGIN, [0, 0, 0])
            scale = mesh_cfg.get(self.KEY_SCALE, 1)
            is_static = mesh_cfg.get(self.KEY_STATIC, False)
            orientation = mesh_cfg.get(self.KEY_ORIENTATION, None)

            descriptor = self._library.get(mesh)

            m = Object3D(name=name, mesh=mesh, origin=origin, scale=scale,
                         is_static=is_static, orientation=orientation, descriptor=descriptor)

            self._table[name] = m
            setattr(self, name, m)

    def spawn(self, name: str, mesh: str, origin: tuple = (0, 0, 0), orientation: tuple = None, scale: int = 1,  is_static: bool = False):
        """
        Spawn an object with the specified parameters
        :param name: unique name of the object in the level
        :param mesh: reference name of the mesh in the mesh library
        :param origin: [X,Y,Z] - origin coordinate to place the object at
        :param orientation: [X,Y,Z,W] - orientation of the object
        :param scale: scale of the mesh object
        :param is_static: if True the object is static
        """

        descriptor = self._library.get(mesh)
        m = Object3D(name=name, mesh=mesh, origin=origin, scale=scale,
                     is_static=is_static, orientation=orientation, descriptor=descriptor)
        self._table[name] = m
        setattr(self, name, m)
