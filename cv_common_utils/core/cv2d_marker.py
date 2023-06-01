import inspect
import cv2
import cv2.aruco as aruco

def detect_aruco_markers(image_path):
    # Load the input image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    arucoParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

    (corners, ids, rejected) = detector.detectMarkers(image)

    # Draw detected markers on the image
    aruco.drawDetectedMarkers(image, corners, ids)

    print(corners)
    # Display the image with markers
    cv2.imwrite("/home/andrey/test-out.png", image)