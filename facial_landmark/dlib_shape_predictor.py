import ntpath
import sys

# resolve module import error in PyCharm
sys.path.append(ntpath.dirname(ntpath.dirname(ntpath.abspath(__file__))))

import argparse
import os
import time
from bz2 import decompress
from urllib.request import urlretrieve

import cv2
import dlib
from imutils import face_utils
from imutils.video import WebcamVideoStream

# 注意這裡我們用的是dlib: MMOD的方法來偵測人臉；你可以試著換dlib: HOG + SVM方法看看結果
from face_detection.dlib_mmod import detect


def download(type):
    if type == 5:
        model_url = f"http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2"
        model_name = "shape_predictor_5_face_landmarks.dat"
    else:
        model_url = f"https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks_GTX.dat.bz2"
        model_name = "shape_predictor_68_face_landmarks_GTX.dat"

    if not os.path.exists(model_name):
        urlretrieve(model_url, model_name + ".bz2")
        with open(model_name, "wb") as new_file, open(model_name + ".bz2", "rb") as file:
            data = decompress(file.read())
            new_file.write(data)
        os.remove(model_name + ".bz2")
    return model_name


def main():
    # 初始化arguments
    ap = argparse.ArgumentParser()
    # 預設使用5-points的關鍵點方法，可以改參數使用68-points
    ap.add_argument("-t", "--type", type=int, default=5, choices=[5, 68],
                    help="the shape predictor type for prediction")
    args = vars(ap.parse_args())

    # 下載模型相關檔案
    model_name = download(args["type"])

    # 初始化關鍵點偵測模型
    predictor = dlib.shape_predictor(model_name)

    # 啟動WebCam
    vs = WebcamVideoStream().start()
    time.sleep(2.0)

    while True:
        frame = vs.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detect(frame, return_ori_result=True)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            for (s0, s1) in shape:
                cv2.circle(frame, (s0, s1), 1, (0, 0, 255), -1)

            # 另一種visualize人臉關鍵點結果的方法
            # frame = face_utils.visualize_facial_landmarks(frame, shape)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # 清除用不到的物件
    cv2.destroyAllWindows()
    vs.stop()


if __name__ == '__main__':
    main()
