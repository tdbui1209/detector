import cv2
import time
from face import Face
from hand import Hand
from pose import Pose


def main():
    capture = cv2.VideoCapture(0)

    pose_detector = Pose()
    hand_detector = Hand()
    face_detector = Face()

    t0 = 0
    while True:
        success, frame = capture.read()
        frame = hand_detector.detect(frame)
        t1 = time.time()
        fps = 1 / (t1 - t0)
        t0 = t1
        frame = cv2.putText(frame, str(int(fps)), (20, 80), cv2.FONT_ITALIC,
                            1, (0, 255, 0), 2)
        cv2.imshow('Detector', frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()