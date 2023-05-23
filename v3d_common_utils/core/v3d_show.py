import numpy as np
import open3d as o3d

def v3d_show_depth_map(image: np.ndarray, depth: np.ndarray, fx = None, fy = None, cx = None, cy = None):

    pcd = []
    height, width = depth.shape
    for i in range(height):
        for j in range(width):
            z = depth[i][j]
            # ---- Conversion
            if fx is not None and fy is not None and cx is not None and cy is not None:
                x = (j - cx) * z / fx
                y = (i - cy) * z / fy
            else:
                x = i
                y = j

            pcd.append([x, y, z])

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    o3d.visualization.draw_geometries([pcd_o3d])
