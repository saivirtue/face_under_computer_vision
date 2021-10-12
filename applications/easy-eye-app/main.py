import math
import time

import cv2
import imutils
from imutils import face_utils
from imutils.video import WebcamVideoStream
from scipy.spatial import distance as dist

from utils.face_detector import FaceDetector
from utils.head_pose_estimator import HeadPoseEstimator
from utils.landmark_detector import LandmarkDetector

PROPERTIES = {
    # 眨眼運動
    "exercise_1": {
        # 是否已完成
        "is_completed": False,
        # 每次完成眨眼，重新判斷
        "re_check": False,
        # 完成次數
        "completed_times": 0,
        # 總次數
        "total_times": 2,
        # 使用眼睛長寬比來判斷是否為閉上眼睛
        "eye_ar_thresh": 0.3,
        # 經過多少個frames才判斷不是"正常眨眼動作"
        "eye_ar_consec_frames": 25,
        # 計算閉眼經過frames
        "eye_ar_consec_counter": 0
    },
    # 眼球運動
    "exercise_2": {
        # 是否已完成
        "is_completed": False,
        # 目前眼球看的方向
        "look_direction": None,
        # 眼球看的方向計算frame
        "look_count": 0,
        # 眼球看的方向需要的frames
        "look_max_count": 25
    }
}


def eye_aspect_ratio(eye):
    # 計算兩個垂直方向的距離
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # 計算水平方向的距離
    C = dist.euclidean(eye[0], eye[3])

    # 計算眼睛長寬比 (EAR)
    ear = (A + B) / (2.0 * C)

    return ear


def get_pupil(frame, eyeHull):
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
        if radius > 2:
            M = cv2.moments(cnts)
            if M["m00"] > 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                return int(x + center[0]), int(y + center[1])


# 計算眼球水平方向的位置比例
def horizontal_ratio(eyePupil, eye):
    return (eyePupil[0] - eye[0][0]) / (eye[3][0] - eye[0][0])


# 計算眼球垂直方向的位置比例
def vertical_ratio(eyePupil, eye):
    top = (eye[1][1] + eye[2][1]) / 2
    bottom = (eye[4][1] + eye[5][1]) / 2
    # do not calculate the ratio if pupil detect out of bound
    if eyePupil[1] < top or eyePupil[1] > bottom:
        return 0.5
    return (eyePupil[1] - top) / (bottom - top)


def main():
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    face_aligned = False
    face_aligned_times = 0
    face_aligned_max_times = 20

    # 啟動WebCam
    print("[INFO] starting webcam...")
    vs = WebcamVideoStream().start()
    time.sleep(2.0)
    first_frame = vs.read()

    face_detector = FaceDetector()
    landmark_detector = LandmarkDetector(68)
    head_pose_estimator = HeadPoseEstimator(first_frame.shape[1], first_frame.shape[0])

    while True:
        frame = vs.read()
        rects = face_detector.detect(frame)
        shapes = landmark_detector.detect(frame, rects)

        for shape in shapes:
            (x, y, w, h) = face_utils.rect_to_bb(rects[0])
            if w < 170:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Please get closely to the camera :)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 0, 0), 2)
                break

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # 還未完成第一個訓練
            if not PROPERTIES["exercise_1"]["is_completed"]:
                # 如果是閉眼
                if ear < PROPERTIES["exercise_1"]["eye_ar_thresh"] and not PROPERTIES["exercise_1"]["re_check"]:
                    PROPERTIES["exercise_1"]["eye_ar_consec_counter"] += 1

                    # 計算閉眼是否達到一定時間
                    if PROPERTIES["exercise_1"]["eye_ar_consec_counter"] >= PROPERTIES["exercise_1"]["eye_ar_consec_frames"]:
                        PROPERTIES["exercise_1"]["completed_times"] += 1
                        PROPERTIES["exercise_1"]["eye_ar_consec_counter"] = 0
                        PROPERTIES["exercise_1"]["re_check"] = True
                # 如果是睜開眼睛
                elif ear > PROPERTIES["exercise_1"]["eye_ar_thresh"] and PROPERTIES["exercise_1"]["re_check"]:
                    PROPERTIES["exercise_1"]["re_check"] = False
                    PROPERTIES["exercise_1"]["eye_ar_consec_counter"] = 0
            # 判斷是否完成第一個訓練
            if PROPERTIES["exercise_1"]["completed_times"] == PROPERTIES["exercise_1"]["total_times"]:
                PROPERTIES["exercise_1"]["is_completed"] = True

            # 還未完成第二個訓練
            if PROPERTIES["exercise_1"]["is_completed"] and not PROPERTIES["exercise_2"]["is_completed"]:
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                leftPupil = get_pupil(frame, leftEyeHull)
                rightPupil = get_pupil(frame, rightEyeHull)

                # 先做臉部對齊
                if not face_aligned and face_aligned_times < face_aligned_max_times:
                    (v_p1, v_p2, h_p1, h_p2) = head_pose_estimator.head_pose_estimate(shape)
                    try:
                        # 垂直臉部角度判斷
                        m = (v_p2[1] - v_p1[1]) / (v_p2[0] - v_p1[0])
                        ang1 = int(math.degrees(math.atan(m)))
                    except:
                        ang1 = 90
                    try:
                        # 水平臉部角度判斷
                        m = (h_p2[1] - h_p1[1]) / (h_p2[0] - h_p1[0])
                        ang2 = int(math.degrees(math.atan(-1 / m)))
                    except:
                        ang2 = 90
                    if -80 <= ang1 <= -20 and -50 <= ang2 <= 50:
                        face_aligned_times += 1
                        cv2.putText(frame, f"face aligned!", (550, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 2)
                    else:
                        face_aligned_times = 0
                else:
                    face_aligned = True

                # 找到眼球
                if leftPupil is not None and rightPupil is not None and face_aligned:
                    left_hor = horizontal_ratio(leftPupil, leftEye)
                    right_hor = horizontal_ratio(rightPupil, rightEye)
                    left_ver = vertical_ratio(leftPupil, leftEye)
                    right_ver = vertical_ratio(rightPupil, rightEye)
                    ratio_hor = (left_hor + right_hor) / 2
                    ratio_ver = (left_ver + right_ver) / 2
                    if PROPERTIES["exercise_2"]["look_direction"] is None:
                        cv2.putText(frame, f"Move your eyeball from ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 255), 2)
                        cv2.putText(frame, f"Top ", (204, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.putText(frame, f"=> Left => Bottom => Right", (239, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 255), 2)
                    elif PROPERTIES["exercise_2"]["look_direction"] == "Top":
                        cv2.putText(frame, f"Move your eyeball from Top ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 255), 2)
                        cv2.putText(frame, f"=> Left ", (239, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.putText(frame, f"=> Bottom => Right", (310, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 255), 2)
                    elif PROPERTIES["exercise_2"]["look_direction"] == "Left":
                        cv2.putText(frame, f"Move your eyeball from Top => Left ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 255, 255), 2)
                        cv2.putText(frame, f"=> Bottom ", (310, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.putText(frame, f"=> Right", (407, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    elif PROPERTIES["exercise_2"]["look_direction"] == "Bottom":
                        cv2.putText(frame, f"Move your eyeball from Top => Left => Bottom ", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        cv2.putText(frame, f"=> Right", (407, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    elif PROPERTIES["exercise_2"]["look_direction"] == "Right":
                        cv2.putText(frame, f"Completed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    if ratio_ver < 0.1 and (0.4 < ratio_hor < 0.6) and PROPERTIES["exercise_2"]["look_direction"] is None:
                        if PROPERTIES["exercise_2"]["look_count"] < PROPERTIES["exercise_2"]["look_max_count"]:
                            PROPERTIES["exercise_2"]["look_count"] += 1
                        else:
                            PROPERTIES["exercise_2"]["look_count"] = 0
                            PROPERTIES["exercise_2"]["look_direction"] = "Top"
                    elif ratio_hor > 0.7 and PROPERTIES["exercise_2"]["look_direction"] == "Top":
                        if PROPERTIES["exercise_2"]["look_count"] < PROPERTIES["exercise_2"]["look_max_count"]:
                            PROPERTIES["exercise_2"]["look_count"] += 1
                        else:
                            PROPERTIES["exercise_2"]["look_count"] = 0
                            PROPERTIES["exercise_2"]["look_direction"] = "Left"
                    elif ratio_ver > 0.15 and (0.4 < ratio_hor < 0.6) and PROPERTIES["exercise_2"]["look_direction"] == "Left":
                        if PROPERTIES["exercise_2"]["look_count"] < PROPERTIES["exercise_2"]["look_max_count"]:
                            PROPERTIES["exercise_2"]["look_count"] += 1
                        else:
                            PROPERTIES["exercise_2"]["look_count"] = 0
                            PROPERTIES["exercise_2"]["look_direction"] = "Bottom"
                    elif ratio_hor < 0.2 and PROPERTIES["exercise_2"]["look_direction"] == "Bottom":
                        if PROPERTIES["exercise_2"]["look_count"] < PROPERTIES["exercise_2"]["look_max_count"]:
                            PROPERTIES["exercise_2"]["look_count"] += 1
                        else:
                            PROPERTIES["exercise_2"]["look_count"] = 0
                            PROPERTIES["exercise_2"]["look_direction"] = "Done"
                            PROPERTIES["exercise_2"]["is_completed"] = True

            if not PROPERTIES["exercise_1"]["is_completed"]:
                cv2.putText(frame, f'Blink for 5 secs: ({PROPERTIES["exercise_1"]["completed_times"]}/{PROPERTIES["exercise_1"]["total_times"]})', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif PROPERTIES["exercise_1"]["is_completed"] and not PROPERTIES["exercise_2"]["is_completed"] and not face_aligned:
                cv2.putText(frame, f"Exercise 1 Completed! Please align your face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)
            elif PROPERTIES["exercise_1"]["is_completed"] and PROPERTIES["exercise_2"]["is_completed"]:
                cv2.putText(frame, f"Exercise 2 Completed!, 'r' for re-run ; 'q' for quit...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            face_aligned = False
            face_aligned_times = 0
            # exercise 1 variables
            PROPERTIES["exercise_1"]["is_completed"] = False
            PROPERTIES["exercise_1"]["re_check"] = False
            PROPERTIES["exercise_1"]["completed_times"] = 0
            PROPERTIES["exercise_1"]["eye_ar_consec_counter"] = 0
            # exercise 2 variables
            PROPERTIES["exercise_2"]["is_completed"] = False
            PROPERTIES["exercise_2"]["look_direction"] = None
            PROPERTIES["exercise_2"]["look_count"] = 0
        elif key == ord("q"):
            break
        cv2.imshow("Frame", frame)

    cv2.destroyAllWindows()
    vs.stop()


if __name__ == '__main__':
    main()
