import os

class MeshDescriptor:

    def __init__(self, path_dir: str):
        self._all_files = [f for f in os.listdir(path_dir)]
        file_urdf = [f for f in os.listdir(path_dir) if f.endswith("urdf")][0]
        self.urdf = os.path.join(path_dir, file_urdf)

class MeshLibrary:

    DEFAULT_MESH_DIR = "./resources/mesh"
    def __init__(self, path_mesh_dir: str = DEFAULT_MESH_DIR):
        """
        Load mesh library automatically scanning the given directory
        :param path_mesh_dir: path to the library with meshes
        Each individual mesh should be inside a separate directory.
        """
        self._table = {}

        dl = [d for d in os.listdir(path_mesh_dir) if os.path.isdir(os.path.join(path_mesh_dir, d))]
        for d in dl:
            path_mesh = os.path.join(path_mesh_dir, d)

            descriptor = MeshDescriptor(path_mesh)
            self._table[d] = descriptor

        setattr(self, d, descriptor)

    def get(self, mesh_name: str):
        if mesh_name not in self._table:
            return None

        return self._table[mesh_name]