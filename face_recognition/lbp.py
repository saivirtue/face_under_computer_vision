import ntpath
import sys

# resolve module import error in PyCharm
sys.path.append(ntpath.dirname(ntpath.dirname(ntpath.abspath(__file__))))

# 匯入必要套件
import argparse
import random
import time

import cv2
import numpy as np
import imutils
from imutils import paths
from skimage import feature
from skimage.exposure import rescale_intensity
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from dataset.load_dataset import images_to_faces
# 匯入人臉偵測方法 (你可以依據喜好更換不同方法)
from face_detection.opencv_dnns import detect


def main():
    # 初始化arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str, required=True, help="the input dataset path")
    args = vars(ap.parse_args())

    print("[INFO] loading dataset....")
    (faces, labels) = images_to_faces(args["input"])
    print(f"[INFO] {len(faces)} images in dataset")

    # 將名稱從字串轉成整數 (在做訓練時時常會用到這個方法：label encoding)
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    # 將資料拆分訓練用與測試用；測試資料佔總資料1/4 (方便後續我們判斷這個方法的準確率)
    split = train_test_split(faces, labels, test_size=0.25, stratify=labels, random_state=9527)
    (trainX, testX, trainY, testY) = split

    print("[INFO] training...")
    start = time.time()
    recognizer = cv2.face_LBPHFaceRecognizer().create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(trainX, trainY)
    end = time.time()
    print(f"[INFO] training took: {round(end - start, 2)} seconds")

    # 辨識測試資料
    print("[INFO] predicting...")
    start = time.time()
    predictions = []
    confidence = []
    # loop over the test data
    for i in range(0, len(testX)):
        (prediction, conf) = recognizer.predict(testX[i])
        predictions.append(prediction)
        confidence.append(conf)
    end = time.time()
    print(f"[INFO] predicting took: {round(end - start, 2)} seconds")
    print(classification_report(testY, predictions, target_names=le.classes_))

    # 隨機挑選測試資料來看結果
    idxs = np.random.choice(range(0, len(testY)), size=10, replace=False)
    for i in idxs:
        predName = le.inverse_transform([predictions[i]])[0]
        actualName = le.classes_[testY[i]]

        face = np.dstack([testX[i]] * 3)
        face = imutils.resize(face, width=250)

        cv2.putText(face, f"pred:{predName}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(face, f"actual:{actualName}", (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        print(f"[INFO] prediction: {predName}, actual: {actualName}")
        cv2.imshow("Face", face)
        cv2.waitKey(0)

    # 隨機選取一張照片來看LBP的結果
    image_path = random.choice(list(paths.list_images(args["input"])))
    image = cv2.imread(image_path)
    rects = detect(image)

    (x, y, w, h) = rects[0]["box"]
    roi = image[y:y + h, x:x + w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(gray, 8, 1, method="default")
    lbp = rescale_intensity(lbp, out_range=(0, 255))
    lbp = lbp.astype("uint8")

    img = np.hstack([roi, np.dstack([lbp] * 3)])
    cv2.imshow("img", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
