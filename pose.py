import cv2
import mediapipe as mp


class Pose:
    """
    Phát hiện 33 điểm pose landmarks.
    """
    def __init__(self, static_image_mode=False,
                 model_complexity=1, smooth_landmarks=False,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        Args:
            static_image_mode: Nếu được set là False,
             detector sẽ coi input image là video frame. Nếu được set là True,
              detector sẽ detect mọi input image. Mặc định là False.

            model_complexity: Độ phức tạp pose landmarks model (0, 1, hoặc 2).
             Độ chính xác tỉ lệ thuận với độ trễ và độ phức tạp của model.
              Mặc định là 1.

            smooth_landmarks: Nếu được set là True, input images sẽ được filted
             để giảm jitter, tham số này sẽ bị bỏ qua nếu static_image_mode = True.
              Mặc định là True.

            min_tracking_confidence: Ngưỡng confidence nhỏ nhất ([0.0, 1.0])
             để được coi là detect thành công. Mặc định là 0.5.

            min_detection_confidence: Ngưỡng confidence nhỏ nhất ([0.0, 0.1])
             để được coi là track thành công. Nếu bất kỳ điểm nào dưới ngưỡng này,
              pose detect sẽ tự động được gọi và detect lại ở frame tiếp theo.
               Đặt với giá trị cao sẽ tăng chất lượng, nhưng cũng làm tăng độ trễ.
                Tham số này sẽ bị bỏ qua nếu static_image_mode = True.
                 Mặc định là 0.5

        """
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.min_dectection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.pose = mp.solutions.pose
        self.pose_detection = self.pose.Pose(self.static_image_mode,
                                             self.model_complexity,
                                             self.smooth_landmarks,
                                             self.min_dectection_confidence,
                                             self.min_tracking_confidence)

        # Module draw của MediaPipe
        self.mp_draw = mp.solutions.drawing_utils

    def detect(self, frame, draw=True):
        """
        Args:
            frame: imput frame.
            draw: Nếu draw là True, sẽ vẽ landmarks và connections của 16 điểm
             pose landmakrs, không bao gồm tay và mặt.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose_detection.process(rgb_frame)
        if self.results.pose_landmarks:
            if draw:
                self.pos = self.find_pos(frame)
                frame = self.draw_landmark(frame)
                frame = self.draw_connection(frame)
        return frame

    def find_pos(self, frame):
        """
        Lấy tọa độ các landmarks.
        """
        pos = []
        h, w, _ = frame.shape
        for idx, lmk in enumerate(self.results.pose_landmarks.landmark):
            x = int(lmk.x * w)
            y = int(lmk.y * h)
            pos.append((idx, x, y))
        return pos

    def draw_landmark(self, frame):
        """
        Vẽ các điểm landmarks.
        """
        # Drop face, hand landmarks
        drop_landmarks = [i for i in range(11)] + [i for i in range(17, 23)]
        for point in self.pos:
            if point[0] not in drop_landmarks:
                frame = cv2.circle(frame, (point[1], point[2]), 5,
                                   (0, 0, 255), cv2.FILLED)
        return frame

    def draw_connection(self, frame):
        """
        Vẽ các connections nối các landmarks.
        """
        connection = [(11, 12),  # left_shoulder -> right_shoulder
                      (11, 13),  # left_shoulder -> left_elbow
                      (13, 15),  # left_elbow -> left_wrist
                      (12, 14),  # right_shoulder -> right_elbow
                      (14, 16),  # right_elbow -> right_wrist
                      (11, 23),  # left_shoulder -> left_hip
                      (23, 25),  # left_hip -> left_knee
                      (25, 27),  # left_knee -> left_ankle
                      (27, 29),  # left_ankle -> left_heel
                      (29, 31),  # lef_heel -> left_foot_index
                      (31, 27),  # left_foot_index -> left_ankle
                      (12, 24),  # right_shoulder -> right_hip
                      (24, 26),  # right_hip -> right_knee
                      (26, 28),  # right_knee -> right_ankle
                      (28, 30),  # right_ankle -> right_heel
                      (30, 32),  # right_heel -> right_foot_index
                      (32, 28),  # right_foot_index -> right_ankle
                      (23, 24)]  # left_hip -> right_hip

        for con in connection:
            p1 = self.pos[con[0]]
            p2 = self.pos[con[1]]
            frame = cv2.line(frame, (p1[1], p1[2]), (p2[1], p2[2]),
                             (0, 255, 0), 2)
        return frame


def main():
    capture = cv2.VideoCapture(0)
    pose_detector = Pose()
    while True:
        _, frame = capture.read()
        frame = pose_detector.detect(frame)
        cv2.imshow('Detector', frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
