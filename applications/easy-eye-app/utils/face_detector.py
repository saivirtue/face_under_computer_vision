# 匯入必要套件
import ntpath
import os
from bz2 import decompress
from urllib.request import urlretrieve

import cv2
import dlib


class FaceDetector:
    def __init__(self):
        # 下載模型檔案(.bz2)與解壓縮
        model_name = "mmod_human_face_detector.dat"
        model_path = os.sep.join([ntpath.dirname(ntpath.abspath(__file__)), model_name])
        if not os.path.exists(model_path):
            urlretrieve(f"https://github.com/davisking/dlib-models/raw/master/mmod_human_face_detector.dat.bz2",
                        model_name + ".bz2")
            with open(model_name, "wb") as new_file, open(model_name + ".bz2", "rb") as file:
                data = decompress(file.read())
                new_file.write(data)
            os.remove(model_name + ".bz2")

        # 初始化模型
        self._detector = dlib.cnn_face_detection_model_v1(model_path)

    def detect(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self._detector(rgb, 1)
        rects = [r.rect for r in results]
        return rects
