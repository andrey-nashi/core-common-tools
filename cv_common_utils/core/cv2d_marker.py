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

    if marker_type is not None and marker_type in TABLE_ARUCO_MARKERS:
        aruco_dict = TABLE_ARUCO_MARKERS[marker_type]
        aruco_detection_list = [aruco_dict]
    else:
        aruco_detection_list = [TABLE_ARUCO_MARKERS[x] for x in TABLE_ARUCO_MARKERS]

    output_image = image.copy()
    output_polygons = []

    for aruco_dict in aruco_detection_list:
        aruco_marker_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        aruco_marker_parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_marker_dict, aruco_marker_parameters)
        (corners, ids, rejected) = detector.detectMarkers(image)

        if len(corners):
            print(corners.shape)
            output_image = aruco.drawDetectedMarkers(output_image, corners, ids)
            for polygon in corners:
                print(polygon[0].shape, polygon[1])

    return output_image, output_polygons

def cv2d_detect_aruco_markers_pose(image: np.ndarray, marker_type, marker_size, cam_matrix, cam_distortion):
    aruco_dict = TABLE_ARUCO_MARKERS[marker_type]
    aruco_marker_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
    aruco_marker_parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_marker_dict, aruco_marker_parameters)
    (corners, ids, rejected) = detector.detectMarkers(image)

    for i in range(0, len(corners)):
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[i], marker_size, cam_matrix, cam_distortion)