# 匯入必要套件
import time

import cv2
import dlib
import imutils
import numpy as np
from imutils.face_utils import rect_to_bb
from imutils.video import WebcamVideoStream

# 初始化模型
detector = dlib.get_frontal_face_detector()


# 定義人臉偵測函數方便重複使用
def detect(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 偵測人臉，將辨識結果轉為(x, y, w, h)的bounding box
    results = detector(rgb, 0)
    rects = [rect_to_bb(rect) for rect in results]
    return rects


def main():
    # 啟動WebCam
    vs = WebcamVideoStream().start()
    time.sleep(2.0)
    start = time.time()
    fps = vs.stream.get(cv2.CAP_PROP_FPS)
    print("Frames per second using cv2.CAP_PROP_FPS : {0}".format(fps))

    while True:
        # 取得當前的frame，變更比例為寬300，並且轉成RGB圖片
        frame = vs.read()
        img = frame.copy()
        img = imutils.resize(img, width=300)

        # 取得frame的大小(高，寬)
        ratio = frame.shape[1] / img.shape[1]

        # 呼叫偵測函數，取得結果
        rects = detect(img)

        # loop所有預測結果
        for rect in rects:
            # 計算bounding box(邊界框)與準確率 - 取得(左上X，左上Y，右下X，右下Y)的值 (記得轉換回原始frame的大小)
            box = np.array(rect) * ratio
            (x, y, w, h) = box.astype("int")

            # 畫出邊界框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 標示FPS
        end = time.time()
        cv2.putText(frame, f"FPS: {str(int(1 / (end - start)))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
        start = end

        # 顯示影像
        cv2.imshow("Frame", frame)

        # 判斷是否案下"q"；跳離迴圈
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break


if __name__ == '__main__':
    main()
