import ntpath
import sys

# resolve module import error in PyCharm
sys.path.append(ntpath.dirname(ntpath.dirname(ntpath.dirname(ntpath.abspath(__file__)))))

import argparse
import time

import cv2
from imutils.video import WebcamVideoStream, FPS

from face_detection.dlib_mmod import detect

# 初始化OpenCV提供的追蹤方法列表
OPENCV_OBJECT_TRACKERS = {
    # 更準確但比kcf慢
    "csrt": cv2.TrackerCSRT_create,
    # 速度與準確兼備
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    # 更快但比kcf不準
    "mosse": cv2.TrackerMOSSE_create
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--tracker", type=str,
                    default="kcf",
                    choices=["csrt", "kcf", "boosting", "mil", "tld", "medianflow", "mosse"],
                    help="the object tracker type")
    args = vars(ap.parse_args())

    # 初始化object tracker
    trackers = cv2.MultiTracker_create()

    # 啟動WebCam
    vs = WebcamVideoStream().start()
    time.sleep(2.0)
    fps = FPS().start()

    initFace = False

    while True:
        # 取得當前的frame
        frame = vs.read()

        # 使用dlib: MMOD來偵測人臉並追蹤
        if not initFace:
            rects = detect(frame)
            for rect in rects:
                print(rect)
                tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
                trackers.add(tracker, frame, rect)
            initFace = True

        # 物件追蹤更新
        (_, boxes) = trackers.update(frame)

        # 偵測物件位置與畫圖
        if len(boxes) > 0:
            for box in boxes:
                (x, y, w, h) = [int(b) for b in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            fps.update()
            fps.stop()
            cv2.putText(frame, "FPS: {:.2f}".format(fps.fps()), (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("frame", frame)

        key = cv2.waitKey(1) & 0xff

        # 你也可以用判斷是否需要圈選物件 (按下c按鍵)
        if key == ord('c'):
            # 圈選矩形框 (需按下空白鍵或Enter確認)
            box = cv2.selectROI("frame", frame, fromCenter=False, showCrosshair=True)
            print(box)
            tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
            trackers.add(tracker, frame, box)
            fps = FPS().start()
        # 按下q鍵就離開程式
        elif key == ord('q'):
            break

    vs.stop()
    fps.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
