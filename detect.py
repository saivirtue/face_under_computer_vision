# import the necessary packages
import argparse
import math
import time
import winsound

import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils
from imutils.video import VideoStream
from scipy.spatial import distance as dist

# 3D model points
model_points = np.array([
    (0.0, 0.0, 0.0),  # nose tip
    (0.0, -330.0, -65.0),  # chin
    (-225.0, 170.0, -135.0),  # left eye left corner
    (225.0, 170.0, -135.0),  # right eye right corner
    (-150.0, -150.0, -125.0),  # left mouth corner
    (150.0, -150.0, -125.0)  # right mouth corner
])


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


def get_pupil(eyeHull):
    (x, y, w, h) = cv2.boundingRect(eyeHull)
    roi = frame[y:y + h, x:x + w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) > 0:
        cnts = max(cnts, key=cv2.contourArea)
        (_, radius) = cv2.minEnclosingCircle(cnts)
        # only proceed if the radius meets a minimum size
        if radius > 2:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            M = cv2.moments(cnts)
            if M["m00"] > 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                # cv2.circle(frame, (int(x + center[0]), int(y + center[1])), 1, (0, 0, 255), -1)
                return int(x + center[0]), int(y + center[1])


def horizontal_ratio(eyePupil, eye):
    return (eyePupil[0] - eye[0][0]) / (eye[3][0] - eye[0][0])


def vertical_ratio(eyePupil, eye):
    top = (eye[1][1] + eye[2][1]) / 2
    bottom = (eye[4][1] + eye[5][1]) / 2
    # do not calculate the ratio if pupil detect out of bound
    if eyePupil[1] < top or eyePupil[1] > bottom:
        return 0.5
    return (eyePupil[1] - top) / (bottom - top)


def get_2d_points(rotation_vector, translation_vector, camera_matrix, val):
    point_3d = []
    dist_coeffs = np.zeros((4, 1))
    rear_size = val[0]
    rear_depth = val[1]
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = val[2]
    front_depth = val[3]
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    # Map to 2d img points
    (point_2d, _) = cv2.projectPoints(point_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    return point_2d


def head_pose_points(rotation_vector, translation_vector, camera_matrix):
    rear_size = 1
    rear_depth = 0
    front_size = frame.shape[1]
    front_depth = front_size*2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(rotation_vector, translation_vector, camera_matrix, val)
    y = (point_2d[5] + point_2d[8])//2
    x = point_2d[2]
    return x, y


def head_pose_estimate(shape, size):
    image_points = np.array([
        shape[33],  # Nose tip
        shape[8],  # chin
        shape[36],  # left eye left corner
        shape[45],  # right eye right corner
        shape[48],  # left mouth corner
        shape[54]  # right mouth corner
    ], dtype="double")
    # camera internals
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array([[
        focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # project a 3D point (0, 0, 10000) onto the image plane.
    # we use this to draw a line sticking out of the nose
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([0.0, 0.0, 1000.0]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    # draw the "corners" of the face
    # for p in image_points:
    #     cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    (x1, x2) = head_pose_points(rotation_vector, translation_vector, camera_matrix)
    # cv2.line(frame, p1, p2, (255, 0, 0), 2)
    return p1, p2, x1, x2


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", default="D:\\Learn_Python\\easy-eye-app\\shape_predictor_68_face_landmarks.dat", help="path to facial landmark predictor")
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
args = vars(ap.parse_args())

# global variables
IS_FACE_CLOSED = False

# exercise 1 variables
EXERCISE_1_COMPLETED = False
RE_CHECK = False
EXERCISE_1_TIMES = 0
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 25
COUNTER = 0
MAX_TIMES = 2

# exercise 2 variables
EXERCISE_2_COMPLETED = False
FACE_ALIGNED = False
FACE_ALIGNED_TIMES = 0
MAX_FACE_ALIGNED_TIMES = 20
LOOK_DIRECTION = None
LOOK_COUNT = 0
MAX_COUNT = 25

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)


# loop over frames from the video stream
while True:
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    size = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        if w < 170 and not IS_FACE_CLOSED:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Please get closely to the camera :)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            break
        else:
            IS_FACE_CLOSED = True
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        # cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        # cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Exercise 1
        if not EXERCISE_1_COMPLETED:
            if ear < EYE_AR_THRESH and not RE_CHECK:
                COUNTER += 1

                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    EXERCISE_1_TIMES += 1
                    COUNTER = 0
                    winsound.Beep(500, 500)
                    RE_CHECK = True
            # if opening eyes, and do re-check
            elif ear > EYE_AR_THRESH and RE_CHECK:
                RE_CHECK = False
                COUNTER = 0
        # is exercise 1 completed?
        if EXERCISE_1_TIMES == MAX_TIMES:
            EXERCISE_1_COMPLETED = True

        # Exercise 2
        if EXERCISE_1_COMPLETED and not EXERCISE_2_COMPLETED:
            # find the pupil of left eye and right eye
            leftPupil = get_pupil(leftEyeHull)
            rightPupil = get_pupil(rightEyeHull)

            # estimate the face pose and aligned it
            if not FACE_ALIGNED and FACE_ALIGNED_TIMES < MAX_FACE_ALIGNED_TIMES:
                (p1, p2, x1, x2) = head_pose_estimate(shape, size)
                try:
                    m = (p2[1] - p1[1])/(p2[0] - p1[0])
                    ang1 = int(math.degrees(math.atan(m)))
                except:
                    ang1 = 90
                try:
                    m = (x2[1] - x1[1])/(x2[0] - x1[0])
                    ang2 = int(math.degrees(math.atan(-1/m)))
                except:
                    ang2 = 90
                if -80 <= ang1 <= -20 and -50 <= ang2 <= 50:
                    FACE_ALIGNED_TIMES += 1
                    cv2.putText(frame, f"face aligned!", (550, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 2)
                else:
                    FACE_ALIGNED_TIMES = 0
            else:
                FACE_ALIGNED = True

            if leftPupil is not None and rightPupil is not None and FACE_ALIGNED:
                left_hor = horizontal_ratio(leftPupil, leftEye)
                right_hor = horizontal_ratio(rightPupil, rightEye)
                left_ver = vertical_ratio(leftPupil, leftEye)
                right_ver = vertical_ratio(rightPupil, rightEye)
                ratio_hor = (left_hor + right_hor) / 2
                ratio_ver = (left_ver + right_ver) / 2
                if LOOK_DIRECTION is None:
                    cv2.putText(frame, f"Move your eyeball from ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    cv2.putText(frame, f"Top ", (204, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(frame, f"=> Left => Bottom => Right", (239, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                elif LOOK_DIRECTION == "Top":
                    cv2.putText(frame, f"Move your eyeball from Top ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    cv2.putText(frame, f"=> Left ", (239, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(frame, f"=> Bottom => Right", (310, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                elif LOOK_DIRECTION == "Left":
                    cv2.putText(frame, f"Move your eyeball from Top => Left ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    cv2.putText(frame, f"=> Bottom ", (310, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(frame, f"=> Right", (407, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                elif LOOK_DIRECTION == "Bottom":
                    cv2.putText(frame, f"Move your eyeball from Top => Left => Bottom ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    cv2.putText(frame, f"=> Right", (407, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                elif LOOK_DIRECTION == "Right":
                    cv2.putText(frame, f"Completed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                if ratio_ver < 0.1 and (0.4 < ratio_hor < 0.6) and LOOK_DIRECTION is None:
                    if LOOK_COUNT < MAX_COUNT:
                        LOOK_COUNT += 1
                    else:
                        LOOK_COUNT = 0
                        LOOK_DIRECTION = "Top"
                        winsound.Beep(500, 500)
                elif ratio_hor > 0.7 and LOOK_DIRECTION == "Top":
                    if LOOK_COUNT < MAX_COUNT:
                        LOOK_COUNT += 1
                    else:
                        LOOK_COUNT = 0
                        LOOK_DIRECTION = "Left"
                        winsound.Beep(500, 500)
                elif ratio_ver > 0.15 and (0.4 < ratio_hor < 0.6) and LOOK_DIRECTION == "Left":
                    if LOOK_COUNT < MAX_COUNT:
                        LOOK_COUNT += 1
                    else:
                        LOOK_COUNT = 0
                        LOOK_DIRECTION = "Bottom"
                        winsound.Beep(500, 500)
                elif ratio_hor < 0.2 and LOOK_DIRECTION == "Bottom":
                    if LOOK_COUNT < MAX_COUNT:
                        LOOK_COUNT += 1
                    else:
                        LOOK_COUNT = 0
                        LOOK_DIRECTION = "Done"
                        winsound.Beep(500, 500)
                        EXERCISE_2_COMPLETED = True

        if not EXERCISE_1_COMPLETED:
            cv2.putText(frame, f"Blink for 5 secs: ({EXERCISE_1_TIMES}/{MAX_TIMES})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif EXERCISE_1_COMPLETED and not EXERCISE_2_COMPLETED and not FACE_ALIGNED:
            cv2.putText(frame, f"Exercise 1 Completed! Please align your face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif EXERCISE_1_COMPLETED and EXERCISE_2_COMPLETED:
            cv2.putText(frame, f"Exercise 2 Completed!, 'r' for re-run ; 'q' for quit...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                IS_FACE_CLOSED = False
                # exercise 1 variables
                EXERCISE_1_COMPLETED = False
                RE_CHECK = False
                EXERCISE_1_TIMES = 0
                COUNTER = 0
                # exercise 2 variables
                EXERCISE_2_COMPLETED = False
                FACE_ALIGNED = False
                FACE_ALIGNED_TIMES = 0
                LOOK_DIRECTION = None
                LOOK_COUNT = 0

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
