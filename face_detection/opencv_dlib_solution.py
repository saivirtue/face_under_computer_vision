import ntpath
import sys

# resolve module import error in PyCharm
sys.path.append(ntpath.dirname(ntpath.dirname(ntpath.abspath(__file__))))

# 匯入必要套件
import argparse

import cv2
import imutils
import numpy as np

from face_detection.dlib_hog_svm import detect as detect_hog
from face_detection.dlib_mmod import detect as detect_mmod
from face_detection.opencv_dnns import detect as detect_dnns
from face_detection.opencv_haar_cascade import detect as detect_haar


def main():
    # 初始化arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="the image to be detected")
    args = vars(ap.parse_args())

    # 載入圖片與轉換
    img = cv2.imread(args["image"])
    img = imutils.resize(img, width=800)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w = img.shape[:2]

    # 偵測人臉
    # opencv_haar_cascade
    rects = detect_haar(img, scale_factor=1.1, min_neighbors=8, min_size=(50, 50))
    for rect in rects:
        (x, y, w, h) = rect.astype("int")
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 102, 255), 2)
        cv2.putText(img, "opencv_haar_cascade", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 102, 255), 2)

    # opencv_dnns
    rects = detect_dnns(img)
    rects = [rect["box"] for rect in rects]
    for rect in rects:
        (x, y, w, h) = np.array(rect).astype("int")
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "opencv_dnns", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # dlib_hog_svm
    rects = detect_hog(img)
    for rect in rects:
        (x, y, w, h) = np.array(rect).astype("int")
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(img, "dlib_hog", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # dlib_mmod
    rects = detect_mmod(img)
    for rect in rects:
        (x, y, w, h) = np.array(rect).astype("int")
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, "dlib_mmod", (x + w, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Result", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
