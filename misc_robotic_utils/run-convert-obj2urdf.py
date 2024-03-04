from core.simulator.sim_obj2urdf import ObjectUrdfBuilder

path_model_dir = "./resources/mesh/shelf"
path_proto = "./resources/proto.urdf"
b = ObjectUrdfBuilder(object_folder=path_model_dir, urdf_prototype=path_proto)
b.build_library()