from .core.simulator.sim_obj2urdf import ObjectUrdfBuilder

path_model_dir = "./resources/model"
b = ObjectUrdfBuilder(object_folder=path_model_dir, urdf_prototype="proto.urdf")
b.build_library()