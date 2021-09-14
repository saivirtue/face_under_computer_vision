# 匯入必要套件
import argparse
import ntpath
import os
import time

import cv2
import imutils
from imutils.video import WebcamVideoStream


def main():
    # 初始化arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--part", type=str, choices=["eye", "face"], default="face", help="detect which part of face")
    args = vars(ap.parse_args())

    # 初始化Haar cascades函數
    detectors = {
        "eye": os.path.sep.join([ntpath.dirname(cv2.__file__), 'data', 'haarcascade_eye.xml']),
        "face": os.path.sep.join([ntpath.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml'])
    }
    detector = cv2.CascadeClassifier(detectors[args["part"]])

    # 啟動WebCam
    vs = WebcamVideoStream().start()
    time.sleep(2.0)
    start = time.time()
    fps = vs.stream.get(cv2.CAP_PROP_FPS)
    print("Frames per second using cv2.CAP_PROP_FPS : {0}".format(fps))

    while True:
        # 取得當前的frame，變更比例為寬300，並且轉成灰階圖片
        frame = vs.read()
        img = frame.copy()
        img = imutils.resize(img, width=300)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 根據選擇的模型偵測
        rects = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(15, 15),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

        # 繪出偵測結果 (記得將偵測的座標轉回原本的frame大小)
        ratio = frame.shape[1] / img.shape[1]
        for rect in rects:
            rect = rect * ratio
            (x, y, w, h) = rect.astype("int")
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 標示FPS
        end = time.time()
        text = f"FPS: {str(int(1 / (end - start)))}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        start = end

        # 顯示影像
        cv2.imshow("Frame", frame)

        # 判斷是否案下"q"；跳離迴圈
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break

    #  清除畫面與結束WebCam
    cv2.destroyAllWindows()
    vs.stop()


if __name__ == '__main__':
    main()
