import ntpath
import sys

# resolve module import error in PyCharm
sys.path.append(ntpath.dirname(ntpath.dirname(ntpath.abspath(__file__))))

import argparse
import os
from bz2 import decompress
from urllib.request import urlretrieve

import cv2
import dlib
import numpy as np
from imutils.face_utils import rect_to_bb
from imutils.face_utils.facealigner import FaceAligner

from face_detection.dlib_mmod import detect


def main():
    # 初始化arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    args = vars(ap.parse_args())

    # 初始化68-points facial landmark模型
    model_url = f"https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks_GTX.dat.bz2"
    model_name = "shape_predictor_68_face_landmarks_GTX.dat"
    if not os.path.exists(model_name):
        urlretrieve(model_url, model_name + ".bz2")
        with open(model_name, "wb") as new_file, open(model_name + ".bz2", "rb") as file:
            data = decompress(file.read())
            new_file.write(data)
        os.remove(model_name + ".bz2")
    predictor = dlib.shape_predictor(model_name)

    # 初始化Face Aligner
    aligner = FaceAligner(predictor, desiredFaceWidth=256)

    # 載入圖片並偵測人臉
    image = cv2.imread(args["image"])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detect(image, return_ori_result=True)

    for rect in rects:
        (x, y, w, h) = rect_to_bb(rect)
        orign_face = cv2.resize(image[y:y + h, x:x + w], (256, 256))
        # 注意：如果此方法丟出getRotationMatrix2D()的型別錯誤，請將opencv-contrib-python的版本降版到4.1
        align_face = aligner.align(image, gray, rect)

        # 顯示結果
        result = np.hstack([orign_face, align_face])
        cv2.imshow("Result", result)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
