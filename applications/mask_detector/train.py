import argparse
import ntpath
import os

import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical


def main():
    # 初始化Arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
    ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")
    ap.add_argument("-m", "--model", type=str, default="mask_detector.model",
                    help="path to output face mask detector model")
    args = vars(ap.parse_args())

    # 初始化訓練用參數與Batch Size
    INIT_LR = 1e-4
    EPOCHS = 20
    BS = 32

    # 載入圖片
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(args["dataset"]))
    data = []
    labels = []

    # 將訓練圖片進行前處理與建立訓練data
    for imagePath in imagePaths:
        label = ntpath.normpath(imagePath).split(os.path.sep)[-2]

        # 注意這裡將圖片轉成224 x 224，與MobileNetV2模型需要的Input一樣大小
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(label)

    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    # 將類別encoding成數值方便訓練
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)

    # 切分訓練資料與測試資料
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=9527)

    # 做Data Argumentation，強化模型的辨識能力
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")

    # 載入模型，去除模型最後一層 (等等要改為我們要辨識的"兩種類別")
    baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

    # 組合自定義的最後層
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)

    # 建立模型
    model = Model(inputs=baseModel.input, outputs=headModel)

    # 確認模型只有我們新增的最後層可以訓練 (transfer learning)
    for layer in baseModel.layers:
        layer.trainable = False

    # 編譯模型
    print("[INFO] compiling model...")
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    # 開始訓練
    print("[INFO] training head...")
    H = model.fit(
        aug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=(testX, testY),
        validation_steps=len(testX) // BS,
        epochs=EPOCHS)

    # 使用測試資料驗證模型準確率
    print("[INFO] evaluating network...")
    predIdxs = model.predict(testX, batch_size=BS)
    predIdxs = np.argmax(predIdxs, axis=1)

    # 印出測試結果
    print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

    # 儲存模型
    print("[INFO] saving mask detector model...")
    model.save(args["model"], save_format="h5")

    # 劃出訓練結果
    N = EPOCHS
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])


if __name__ == '__main__':
    main()
