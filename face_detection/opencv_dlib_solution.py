# 匯入必要套件
import argparse
import ntpath
import os

import cv2
import dlib
import imutils
import numpy as np
from imutils.face_utils import rect_to_bb


def main():
    # 初始化arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="the image to be detected")
    args = vars(ap.parse_args())

    # 載入圖片與轉換
    img = cv2.imread(args["image"])
    img = imutils.resize(img, width=800)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = img.shape[:2]

    # 初始化各detector
    dect_haar = cv2.CascadeClassifier(os.path.sep.join([ntpath.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml']))
    dect_dnns = cv2.dnn.readNetFromCaffe(prototxt="deploy.prototxt", caffeModel="res10_300x300_ssd_iter_140000.caffemodel")
    dect_hog = dlib.get_frontal_face_detector()
    dect_mmod = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

    # 偵測人臉
    rects = dect_haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(50, 50))
    for rect in rects:
        (x, y, w, h) = rect.astype("int")
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 102, 255), 2)
        cv2.putText(img, "opencv_haar_cascade", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 102, 255), 2)

    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    dect_dnns.setInput(blob)
    results = dect_dnns.forward()
    for i in range(0, results.shape[2]):
        confidence = results[0, 0, i, 2]
        if confidence < 0.5:
            continue
        box = results[0, 0, i, 3:7] * np.array([img_w, img_h, img_w, img_h])
        (x0, y0, x1, y1) = box.astype("int")
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(img, "opencv_dnns", (x1 + 10, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    rects = dect_hog(rgb, 0)
    boxes = [rect_to_bb(rect) for rect in rects]
    for box in boxes:
        (x, y, w, h) = np.array(box).astype("int")
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(img, "dlib_hog", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    rects = dect_mmod(rgb, 0)
    boxes = [rect_to_bb(r.rect) for r in rects]
    for box in boxes:
        (x, y, w, h) = np.array(box).astype("int")
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, "dlib_mmod", (x + w, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Result", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
