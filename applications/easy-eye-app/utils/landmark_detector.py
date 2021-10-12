import ntpath
import os
from bz2 import decompress
from urllib.request import urlretrieve

import cv2
import dlib
from imutils import face_utils


class LandmarkDetector:
    def __init__(self, predictor_type):
        if predictor_type == 5:
            model_url = f"http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2"
            model_name = "shape_predictor_5_face_landmarks.dat"
        elif predictor_type == 68:
            model_url = f"https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks_GTX.dat.bz2"
            model_name = "shape_predictor_68_face_landmarks_GTX.dat"
        else:
            raise ValueError(f"un-support predictor type: {predictor_type}, must be 5 or 68!")

        model_path = os.sep.join([ntpath.dirname(ntpath.abspath(__file__)), model_name])
        if not os.path.exists(model_path):
            urlretrieve(model_url, model_name + ".bz2")
            with open(model_name, "wb") as new_file, open(model_name + ".bz2", "rb") as file:
                data = decompress(file.read())
                new_file.write(data)
            os.remove(model_name + ".bz2")

        # 初始化關鍵點偵測模型
        self._predictor = dlib.shape_predictor(model_path)

    def detect(self, img, rects):
        shapes = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for rect in rects:
            shape = self._predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            shapes.append(shape)
        return shapes
