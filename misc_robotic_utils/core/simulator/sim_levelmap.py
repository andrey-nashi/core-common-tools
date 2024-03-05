import pybullet as p
import pybullet_data
import yaml

class Object3D:
    def __init__(self, name, mesh, origin, scale, library):
        self.name = name
        self.mesh = mesh
        self.origin = origin
        self.scale = scale

        record = library.get(self.mesh)
        self.ref_id = p.loadURDF(record.urdf, basePosition=self.origin, globalScaling=self.scale)




class LevelMap:

    KEY_SCENE = "scene"
    KEY_NAME = "name"
    KEY_MESH = "mesh"
    KEY_ORIGIN = "origin"
    KEY_SCALE = "scale"

    def __init__(self, path_level: str, mesh_library):
        self._library = mesh_library
        self._table = {}

        f = open(path_level, "r")
        data = yaml.safe_load(f)
        f.close()

        for mesh_cfg in data[self.KEY_SCENE]:
            name = mesh_cfg[self.KEY_NAME]
            mesh = mesh_cfg[self.KEY_MESH]
            origin = mesh_cfg[self.KEY_ORIGIN]
            scale = mesh_cfg[self.KEY_SCALE]
            m = Object3D(name, mesh, origin, scale, self._library)

            self._table[name] = m
            setattr(self, name, m)

    def spawn(self, name, mesh, origin, scale):
        m = Object3D(name, mesh, origin, scale, self._library)
        self._table[name] = m
        setattr(self, name, m)
