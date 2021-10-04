import time

import cv2
import mtcnn
from imutils.video import WebcamVideoStream


def main():
    # 初始化模型
    detector = mtcnn.MTCNN()

    # 啟動WebCam
    vs = WebcamVideoStream().start()
    time.sleep(2.0)

    while True:
        frame = vs.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)
        for face in faces:
            (x, y, w, h) = face['box']
            keypoints = face['keypoints']
            conf = face['confidence']

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"confidence: {str(round(conf, 3))}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            for (s0, s1) in keypoints.values():
                cv2.circle(frame, (s0, s1), 2, (0, 0, 255), -1)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # 清除用不到的物件
    cv2.destroyAllWindows()
    vs.stop()


if __name__ == '__main__':
    main()
