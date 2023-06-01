import cv2
import numpy as np
import open3d as o3d


def v3d_reconstruct_depth_map(image: np.ndarray, depth: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Generate a 3D representation of the depth map as the point cloud
    :param image: numpy array [H,W,3] representing the image
    :param depth: numpy array of [H,W] each pixel representing depth
    :return: point cloud
    """
    pcd = []
    height, width = depth.shape
    for i in range(height):
        for j in range(width):
            z = depth[i][j]
            x = i
            y = j

            pcd.append([x, y, z])

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    return pcd


def v3d_reconstruct_pcd(image: np.ndarray, depth: np.ndarray, fx: float = None, fy: float = None, cx: float = None, cy: float = None, depth_threshold: float = None) -> o3d.geometry.PointCloud:
    """
    Reconstruct PCD from the given image, depth, intrinsic camera parameters, optionally also threshold by depth
    :param image: numpy array [H,W,3] (BGR) image
    :param depth: numpy array [H,W] depth map
    :param fx: intrinsic camera parameter fx
    :param fy: intrinsic camera parameter fy
    :param cx: intrinsic camera parameter cx
    :param cy: intrinsic camera parameter cy
    :param depth_threshold: threshold in meters
    :return: point cloud
    """
    h, w, c = image.shape

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    o3_image = o3d.geometry.Image(image_rgb)
    o3_depth = o3d.geometry.Image(depth)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3_image, o3_depth, depth_trunc=depth_threshold, convert_rgb_to_intensity=False)

    if fx is None and fy is None and cx is None and cy is None:
        pinhole_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    else:
        pinhole_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy)

    pcd = o3d.geometry.create_point_cloud_from_rgbd_image(rgbd, pinhole_intrinsic)
    return pcd


def v3d_reconstruct_pcd_scene(rgbd_data: list, camera_extrinsic_data: list, camera_intrinsic_data: list = None) -> o3d.geometry.PointCloud:
    """
    Reconstruct a PCD scene from multiple PCDs with given pose parameters
    :param rgbd_data: a list of image and depth pairs [[image, depth]] where image and depth are numpy arrays
    :param camera_extrinsic_data: for each pair [image,depth] camera pose matrix (in meters)
    :param camera_intrinsic_data: for each pair [image,depth] camera intrinsic as [fx, fy, cx, cy]
    :return: point cloud
    """
    assert len(rgbd_data) != 0
    assert len(camera_extrinsic_data) != 0
    assert len(rgbd_data) == len(camera_extrinsic_data)

    pcd_scene = None

    for i in range(rgbd_data):
        image = rgbd_data[i][0]
        depth = rgbd_data[i][1]

        # ---- Direct pose matrix
        camera_extrinsic = camera_extrinsic_data[i]

        if camera_intrinsic_data is None:
            fx = None
            fy = None
            cx = None
            cy = None
        else:
            camera_intrinsic = camera_intrinsic_data[i]
            fx = camera_intrinsic[0]
            fy = camera_intrinsic[1]
            cx = camera_intrinsic[2]
            cy = camera_intrinsic[3]

        pcd = v3d_reconstruct_pcd(image, depth, fx=fx, fy=fy, cx=cx, cy=cy)
        pcd.transform(camera_extrinsic)

        pcd_scene = pcd if pcd_scene is None else pcd_scene + pcd

    return pcd_scene


def v3d_reconstruct_volume_scene(rgbd_data: list, camera_extrinsic_data: list, camera_intrinsic_data: list = 0,
                                 voxel_length: float = 1/512, sdf_trunc: float = 1, depth_threshold: float = 1000) -> o3d.pipelines.integration.ScalableTSDFVolume:
    """
    Reconstruct volume (ScalableTSDFVolume) by automatically integrating different point clouds
    :param rgbd_data: a list of image and depth pairs [[image, depth]] where image and depth are numpy arrays
    :param camera_extrinsic_data: for each pair [image,depth] camera pose matrix (in meters)
    :param camera_intrinsic_data: for each pair [image,depth] camera intrinsic as [fx, fy, cx, cy]
    :param voxel_length: length of voxels the smaller the rougher
    :param sdf_trunc:
    :param depth_threshold: threshold in meters
    :return: volume
    """
    assert len(rgbd_data) != 0
    assert len(camera_extrinsic_data) != 0
    assert len(rgbd_data) == len(camera_extrinsic_data)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel_length, sdf_trunc=sdf_trunc,
                                                          color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    for i in range(rgbd_data):
        image = rgbd_data[i][0]
        depth = rgbd_data[i][1]

        h, w, c = image.shape

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        o3_image = o3d.geometry.Image(image_rgb)
        o3_depth = o3d.geometry.Image(depth)

        # ---- Direct pose matrix
        camera_extrinsic = camera_extrinsic_data[i]

        if camera_intrinsic_data is None:
            pinhole_intrinsic = o3d.camera.PinholeCameraIntrinsic()
        else:
            camera_intrinsic = camera_intrinsic_data[i]

            fx = camera_intrinsic[0]
            fy = camera_intrinsic[1]
            cx = camera_intrinsic[2]
            cy = camera_intrinsic[3]

            pinhole_intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)


        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3_image, o3_depth, depth_trunc=depth_threshold, convert_rgb_to_intensity=False)
        volume.integrate(rgbd, pinhole_intrinsic, np.linalg.inv(camera_extrinsic))

    return volume