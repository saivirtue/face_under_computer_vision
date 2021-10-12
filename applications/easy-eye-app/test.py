import unittest

import cv2

from utils.face_detector import FaceDetector
from utils.head_pose_estimator import HeadPoseEstimator
from utils.landmark_detector import LandmarkDetector


class TestClass(unittest.TestCase):
    def setUp(self):
        self.img = cv2.imread("sample_1.jpg")

        self.face_detector = FaceDetector()
        self.landmark_detector = LandmarkDetector(68)
        self.head_pose_estimator = HeadPoseEstimator(self.img.shape[1], self.img.shape[0])

    def test_face_detector(self):
        rects = self.face_detector.detect(self.img)

        # 應該只有一張人臉被辨識
        assert len(rects) == 1
        # 每個人臉的bounding box有四個點 (top, right, bottom, left)
        assert hasattr(rects[0], "top")
        assert hasattr(rects[0], "right")
        assert hasattr(rects[0], "bottom")
        assert hasattr(rects[0], "left")

    def test_landmark_detector(self):
        rects = self.face_detector.detect(self.img)
        shapes = self.landmark_detector.detect(self.img, rects)

        # 應該只有一張人臉被辨識
        assert len(shapes) == 1

        # 人臉關鍵點應該有68個點
        assert len(shapes[0]) == 68

    def test_head_pose_estimator(self):
        rects = self.face_detector.detect(self.img)
        shapes = self.landmark_detector.detect(self.img, rects)
        for shape in shapes:

            for s in shape:
                cv2.circle(self.img, (int(s[0]), int(s[1])), 3, (255, 0, 0), -1)

            pts = self.head_pose_estimator.head_pose_estimate(shape)

            # 共有四個點 (兩兩相對，分別對應垂直與水平方向角度判斷用)
            assert len(pts) == 4
            # 每個點都是tuple: (x, y)
            assert [type(pt) == tuple and len(pt) == 2 for pt in pts]
