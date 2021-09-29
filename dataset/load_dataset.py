# 匯入必要套件
import ntpath
import os
import pickle
from itertools import groupby

import cv2
import numpy as np
from imutils import paths

# 匯入人臉偵測方法 (你可以依據喜好更換不同方法)
from face_detection.opencv_dnns import detect


def load_images(input_path, min_size=15):
    # 載入所有圖片
    image_paths = list(paths.list_images(input_path))
    # 將圖片屬於"哪一個人"的名稱取出 (如：man_1, man_2,...)，並以此名稱將圖片分群
    groups = groupby(image_paths, key=lambda path: ntpath.normpath(path).split(os.path.sep)[-2])

    # 初始化結果 (images, names)
    images = []
    names = []

    # loop我們分群好的圖片
    for name, group_image_paths in groups:
        group_image_paths = list(group_image_paths)

        # 如果樣本圖片數小於15張，則不考慮使用該人的圖片 (因為會造成辨識結果誤差)；可以嘗試將下面兩行註解看準確度的差異
        if (len(group_image_paths)) < min_size:
            continue

        for imagePath in group_image_paths:
            # 將圖片依序載入，取得人臉矩形框
            img = cv2.imread(imagePath)
            # 更新結果
            images.append(img)
            names.append(name)

    # 將結果轉成numpy array，方便後續進行訓練
    images = np.array(images)
    names = np.array(names)
    return (images, names)


def images_to_faces(input_path):
    """
    將資料集內的照片依序擷取人臉後，轉成灰階圖片，回傳後續可以用作訓練的資料
    :return: (faces, labels)
    """
    # 判斷是否需要重新載入資料
    data_file = ntpath.sep.join([ntpath.dirname(ntpath.abspath(__file__)), "faces.pickle"])
    if os.path.exists(data_file):
        with open(data_file, "rb") as f:
            (faces, labels) = pickle.load(f)
            return (faces, labels)

    (images, names) = load_images(input_path)

    # 初始化結果 (faces, labels)
    faces = []
    labels = []

    # loop我們分群好的圖片
    for (img, name) in zip(images, names):
        rects = detect(img)
        # loop各矩形框
        for rect in rects:
            (x, y, w, h) = rect["box"]
            # 取得人臉ROI (注意在用陣列操作時，順序是 (rows, columns) => 也就是(y, x) )
            roi = img[y:y + h, x:x + w]
            # 將人臉的大小都轉成50 x 50的圖片
            roi = cv2.resize(roi, (50, 50))
            # 轉成灰階
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # 更新結果
            faces.append(roi)
            labels.append(name)

    # 將結果轉成numpy array，方便後續進行訓練
    faces = np.array(faces)

    with open(data_file, "wb") as f:
        pickle.dump((faces, labels), f)

    return (faces, labels)
