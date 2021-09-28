import ntpath
import sys
# resolve module import error in PyCharm
sys.path.append(ntpath.dirname(ntpath.dirname(ntpath.abspath(__file__))))

# 匯入必要套件
import argparse
import time

import cv2
import imutils
import numpy as np
from imutils import build_montages
from matplotlib import pyplot as plt
from skimage.exposure import rescale_intensity
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from dataset.load_dataset import images_to_faces


def main():
    # 初始化arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str, required=True, help="the input dataset path")
    ap.add_argument("-n", "--components", type=int, default=25, help="number of components")
    args = vars(ap.parse_args())

    print("[INFO] loading dataset....")
    (faces, labels) = images_to_faces(args["input"])
    print(f"[INFO] {len(faces)} images in dataset")

    # 進行主成分分析時需要將資料轉成一維陣列
    pca_faces = np.array([face.flatten() for face in faces])

    # 將名稱從字串轉成整數 (在做訓練時時常會用到這個方法：label encoding)
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    # 將資料拆分訓練用與測試用；測試資料佔總資料1/4 (方便後續我們判斷這個方法的準確率)
    split = train_test_split(faces, pca_faces, labels, test_size=0.25, stratify=labels, random_state=9527)
    (oriTrain, oriTest, trainX, testX, trainY, testY) = split

    print("[INFO] creating eigenfaces...")
    pca = PCA(svd_solver="randomized", n_components=args["components"], whiten=True)
    start = time.time()
    pca_trainX = pca.fit_transform(trainX)
    end = time.time()
    print(f"[INFO] computing eigenfaces for {round(end - start, 2)} seconds")

    # 確認使用的主成分可以解釋多少資料的變異
    cum_ratio = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(cum_ratio)
    plt.xlabel("number of components")
    plt.ylabel("cumulative explained variance")
    plt.draw()
    plt.waitforbuttonpress(0)

    # 來用"圖像"看一下PCA的結果
    vis_images = []
    for (i, component) in enumerate(pca.components_):
        component = component.reshape((50, 50))
        component = rescale_intensity(component, out_range=(0, 255))
        component = np.dstack([component.astype("uint8")] * 3)
        vis_images.append(component)
    montage = build_montages(vis_images, (50, 50), (5, 5))[0]
    montage = imutils.resize(montage, width=250)

    mean = pca.mean_.reshape((50, 50))
    mean = rescale_intensity(mean, out_range=(0, 255)).astype("uint8")
    mean = imutils.resize(mean, width=250)

    cv2.imshow("Mean", mean)
    cv2.imshow("Components", montage)
    cv2.waitKey(0)

    # 建立SVM模型來訓練
    model = SVC(kernel="rbf", C=10.0, gamma=0.001, random_state=9527)
    model.fit(pca_trainX, trainY)

    # 驗證模型的準確度 (記得將測試資料轉成PCA的格式)
    pca_testX = pca.transform(testX)
    predictions = model.predict(pca_testX)
    print(classification_report(testY, predictions, target_names=le.classes_))

    # Optional: 你也可以直接用OpenCV內建的模型來訓練；下面的程式碼可以取代前面的SVM模型訓練
    # recognizer = cv2.face_EigenFaceRecognizer().create(num_components=args["components"])
    # recognizer.train(trainX, trainY)
    # predictions = []
    # for i in range(0, len(testX)):
    #     (prediction, _) = recognizer.predict(testX[i])
    #     predictions.append(prediction)
    # print(classification_report(testY, predictions, target_names=le.classes_))

    # 隨機挑選測試資料來看結果
    idxs = np.random.choice(range(0, len(testY)), size=10, replace=False)
    for i in idxs:
        predName = le.inverse_transform([predictions[i]])[0]
        actualName = le.classes_[testY[i]]

        face = np.dstack([oriTest[i]] * 3)
        face = imutils.resize(face, width=250)

        cv2.putText(face, f"pred:{predName}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(face, f"actual:{actualName}", (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        print(f"[INFO] prediction: {predName}, actual: {actualName}")
        cv2.imshow("Face", face)
        cv2.waitKey(0)
    plt.close()


if __name__ == '__main__':
    main()
