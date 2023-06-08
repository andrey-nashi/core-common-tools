import cv2
import cv2.aruco as aruco

import numpy as np

TABLE_ARUCO_MARKERS = {
    "4X4_50": cv2.aruco.DICT_4X4_50,
    "4X4_100": cv2.aruco.DICT_4X4_100,
    "4X4_250": cv2.aruco.DICT_4X4_250,
    "4X4_1000": cv2.aruco.DICT_4X4_1000,
    "5X5_50": cv2.aruco.DICT_5X5_50,
    "5X5_100": cv2.aruco.DICT_5X5_100,
    "5X5_250": cv2.aruco.DICT_5X5_250,
    "5X5_1000": cv2.aruco.DICT_5X5_1000,
    "6X6_50": cv2.aruco.DICT_6X6_50,
    "6X6_100": cv2.aruco.DICT_6X6_100,
    "6X6_250": cv2.aruco.DICT_6X6_250,
    "6X6_1000": cv2.aruco.DICT_6X6_1000,
    "7X7_50": cv2.aruco.DICT_7X7_50,
    "7X7_100": cv2.aruco.DICT_7X7_100,
    "7X7_250": cv2.aruco.DICT_7X7_250,
    "7X7_1000": cv2.aruco.DICT_7X7_1000,
    "ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

def cv2d_detect_aruco_markers(image: np.ndarray, marker_type=None):
    """
    Detect ARUCO marker, return image with painted markers and XY coordinates
    :param image: numpy array the RGB image
    :param marker_type: name of the marker type (see the table), if None is given
    then attempt to parse through all dictionaries
    :return: image, list of dictionaries - found markers
    [<image>, [{"dict": <dict_id>, "id": <marker_id>, "xy": [[x, y], ... ]}]]
    """
    if marker_type is not None and marker_type in TABLE_ARUCO_MARKERS:
        aruco_dict = TABLE_ARUCO_MARKERS[marker_type]
        aruco_detection_list = [aruco_dict]
    else:
        aruco_detection_list = [TABLE_ARUCO_MARKERS[x] for x in TABLE_ARUCO_MARKERS]

    output_image = image.copy()
    output_data = []

    for aruco_dict in aruco_detection_list:
        aruco_marker_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        aruco_marker_parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_marker_dict, aruco_marker_parameters)
        (corners, ids, rejected) = detector.detectMarkers(image)

        if len(corners) != 0:
            output_image = aruco.drawDetectedMarkers(output_image, corners, ids)
            for i in range(0, len(corners)):
                marker_index = ids[i][0].tolist()
                marker_xy = corners[i][0].astype(np.uint32).tolist()
                output_data.append({"dict": aruco_dict, "id": marker_index, "xy": marker_xy})

    return output_image, output_data

def cv2d_detect_aruco_markers_pose(image: np.ndarray, marker_type: str, marker_size: float, cam_matrix: list = None, cam_distortion: list = None):
    """
    Detect markers on an image, estimate their pose
    :param image: numpy array the RGB image
    :param marker_type: name of the marker type (see the table), if None is given
    :param marker_size: size of the marker in cm
    :param cam_matrix: camera matrix list of [[fx, 0, cx],[0, fy, cy], [0, 0, 1]]
    :param cam_distortion: camera distortion vector of size (4,1)
    :return: image with painted markers, pose and marker info
    """

    # ---- Generate camera matrix and distortion vector if not given
    if cam_matrix is None:
        size = image.shape
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        cam_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
    else:
        cam_matrix = np.array(cam_matrix)

    if cam_distortion is None:
        cam_distortion = np.zeros((4, 1))
    else:
        cam_distortion = np.array(cam_distortion)

    # ---- Marker in 3D
    marker_3d = [
        [-marker_size / 2.0, marker_size / 2.0, 0],
        [marker_size / 2.0, marker_size / 2.0, 0],
        [marker_size / 2.0, -marker_size / 2.0, 0],
        [-marker_size / 2.0, -marker_size / 2.0, 0],
    ]
    marker_3d = np.array(marker_3d)

    # ---- Detect
    aruco_dict = TABLE_ARUCO_MARKERS[marker_type]
    aruco_marker_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
    aruco_marker_parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_marker_dict, aruco_marker_parameters)
    (corners, ids, rejected) = detector.detectMarkers(image)

    output_image = image.copy()
    output_data = []

    # ---- Compute pose & draw
    output_image = aruco.drawDetectedMarkers(output_image, corners, ids)
    for i in range(0, len(corners)):

        success, vector_rotation, vector_translation = cv2.solvePnP(marker_3d, corners[i][0], cam_matrix, cam_distortion, flags=0)
        marker_index = ids[i][0].tolist()
        marker_xy = corners[i][0].astype(np.uint32).tolist()
        marker_rotation = vector_rotation.tolist()
        marker_translation = vector_translation.tolist()
        output_data.append({"dict": aruco_dict, "id": marker_index, "xy": marker_xy, "rot": marker_rotation, "tr": marker_translation})
        cv2.drawFrameAxes(output_image, cam_matrix, cam_distortion, vector_rotation, vector_translation, 0.05);

    return output_image, output_data


path = "/home/andrey/test-ix.png"
image = cv2.imread(path)
cam_matrix =  [[533.19, 0.0, 640.77], [0.0, 533.42, 364.838], [0.0, 0.0, 1.0]]
cam_d = [-0.0564249, 0.0298475, -0.0114727, -7.10619e-05, -0.000308677]
image, x = cv2d_detect_aruco_markers_pose(image, "5X5_50", 0.03, cam_matrix, cam_d)
cv2.drawFrameAxes(image, np.array(cam_matrix), np.array(cam_d), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 0.05)
print(x)
cv2.imwrite("/home/andrey/test-out.png", image)