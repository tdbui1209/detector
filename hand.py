import cv2
import mediapipe as mp


class Hand:
    """
    Phát hiện 21 điểm hand landmarks.
    """
    def __init__(self, static_image_model=False, max_num_hands=2,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Args:
            static_image_model: Nếu được set là False,
             detector sẽ coi input image là video frame. Nếu được set là True,
              detector sẽ detect mọi input image. Mặc định là False.

            max_num_hands: Số lượng bàn tay tối đa sẽ detect.
             Mặc định là 2.

            min_tracking_confidence: Ngưỡng confidence nhỏ nhất ([0.0, 1.0])
             để được coi là detect thành công. Mặc định là 0.5.

            min_detection_confidence: Ngưỡng confidence nhỏ nhất ([0.0, 0.1])
             để được coi là track thành công. Nếu bất kỳ điểm nào dưới ngưỡng này,
              pose detect sẽ tự động được gọi và detect lại ở frame tiếp theo.
               Đặt với giá trị cao sẽ tăng chất lượng, nhưng cũng làm tăng độ trễ.
                Tham số này sẽ bị bỏ qua nếu static_image_mode = True.
                 Mặc định là 0.5
        """
        self.static_image_model = static_image_model
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.hand = mp.solutions.hands
        self.hand_detection = self.hand.Hands(self.static_image_model,
                                              self.max_num_hands,
                                              self.min_detection_confidence,
                                              self.min_tracking_confidence)

        # Module draw của MediaPipe
        self.mp_draw = mp.solutions.drawing_utils

    def detect(self, frame, draw=True):
        """
        Args:
            frame: imput frame.
            draw: Nếu draw là True, sẽ vẽ landmarks và connections của
             21 điểm hand landmakrs.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hand_detection.process(rgb_frame)
        # if self.results.multi_handedness:
        #     for self.hand in self.results.multi_handedness:
        #         if self.hand.classification[0].index == 0:
        #             cv2.putText(frame, self.hand.classification[0].label,
        #                         (400, 50), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)
        #         else:
        #             cv2.putText(frame, self.hand.classification[0].label,
        #                         (100, 50), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)
        if self.results.multi_hand_landmarks:
            self.landmark_positions = []
            self.data = []
            for self.hand in self.results.multi_hand_landmarks:
                self.landmark_positions.append(self.find_landmark(frame))
                self.data.append(self.take_data(frame))
                if draw:
                    frame = self.draw_connection(frame)
                    frame = self.draw_landmark(frame)
        return frame

    def find_landmark(self, frame):
        """
        Lấy tọa độ các landmarks.
        """
        landmark_positions = []
        h, w, _ = frame.shape
        for idx, lnk in enumerate(self.hand.landmark):
            x = int(lnk.x * w)
            y = int(lnk.y * h)
            landmark_positions.append((idx, x, y))
        return landmark_positions

    def draw_landmark(self, frame):
        """
        Vẽ các điểm landmarks.
        """
        for hand in self.landmark_positions:
            for point in hand:
                frame = cv2.circle(frame, (point[1], point[2]), 3,
                                   (0, 0, 255), cv2.FILLED)
        return frame

    def draw_connection(self, frame):
        """
        Vẽ các connections nối các landmarks.
        """
        connection = [(0, 1),  # wrist -> thumb_cmc
                      (1, 2),  # thumb_cmc -> thumb_mcp
                      (2, 3),  # thumb_mcp -> thumb_ip
                      (3, 4),  # thumb_ip -> thumb_tip
                      (0, 5),  # wrist -> index_finger_mcp
                      (5, 6),  # index_finger_mcp -> index_finger_pip
                      (6, 7),  # index_finger_pip -> index_finger_dip
                      (7, 8),  # index_finger_dip -> index_finger_tip
                      (9, 10),  # middle_finger_mcp -> middle_finger_pip
                      (10, 11),  # middle_finger_pip -> middle_finger_dip
                      (11, 12),  # middle_finger_dip -> middle_finger_tip
                      (13, 14),  # ring_finger_mcp -> ring_finger_pip
                      (14, 15),  # ring_finger_pip -> ring_finger_dip
                      (15, 16),  # ring_finger_dip -> ring_finger_tip
                      (17, 18),  # pinky_finger_mcp -> pinky_finger_pip
                      (18, 19),  # pinky_finger_pip -> pinky_finger_dip
                      (19, 20),  # pinky_finger_dip -> pinky_finger_tip
                      (0, 17),  # wrist - > pinky_mcp
                      (5, 9),  # index_finger_mcp -> middle_finger_mcp
                      (9, 13),  # middle_finger_mcp -> ring_finger_mcp
                      (13, 17)]  # ring_finger_mcp -> pinky_mcp
        for hand in self.landmark_positions:
            for con in connection:
                p1 = hand[con[0]]
                p2 = hand[con[1]]
                frame = cv2.line(frame, (p1[1], p1[2]), (p2[1], p2[2]),
                                 (0, 255, 0), 2)
        return frame

    def take_data(self, frame):
        data = []
        h, w, _ = frame.shape
        for idx, lnk in enumerate(self.hand.landmark):
            x = int(lnk.x * w)
            y = int(lnk.y * h)
            data.append(x)
            data.append(y)
        return data

import pickle
import numpy as np
import time
import pandas as pd
def main():
    filename = 'finalized_model.sav'
    capture = cv2.VideoCapture(0)
    hand_detector = Hand(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
    loaded_model = pickle.load(open(filename, 'rb'))
    t0 = 0
    while True:
        _, frame = capture.read()
        frame = hand_detector.detect(frame)
        try:
            X = np.array(hand_detector.data).reshape(1, -1)
            features = ['x0', 'y0', 'x4', 'y4', 'x5', 'y5', 'x8', 'y8', 'x9',
                        'y9', 'x12', 'y12', 'x13', 'y13', 'x17', 'y17', 'x20',
                        'y20']
            X = np.array(X).reshape(1, -1)
            actual = pd.DataFrame(
                X,
                columns=['x0', 'y0', 'x1', 'y1', 'x2', 'y2',
                         'x3', 'y3', 'x4', 'y4', 'x5', 'y5',
                         'x6', 'y6', 'x7', 'y7', 'x8', 'y8',
                         'x9', 'y9', 'x10', 'y10', 'x11', 'y11',
                         'x12', 'y12', 'x13', 'y13', 'x14', 'y14',
                         'x15', 'y15', 'x16', 'y16', 'x17', 'y17',
                         'x18', 'y18', 'x19', 'y19', 'x20','y20']
            )

            result = loaded_model.predict(actual[features])
            print(result)
            cv2.putText(frame, result[0], (30, 400), cv2.FONT_ITALIC, 2,
                        (0, 255, 0), 2)
        except:
            pass
        t1 = time.time()
        fps = 1 / (t1 - t0)
        t0 = t1
        frame = cv2.putText(frame, str(int(fps)), (20, 80), cv2.FONT_ITALIC,
                            1, (0, 255, 0), 2)
        cv2.imshow('Detector', frame)
        cv2.waitKey(1)



def data():
    capture = cv2.VideoCapture(0)
    hand_detector = Hand(max_num_hands=1)
    count = 0
    while True:
        _, frame = capture.read()
        frame = hand_detector.detect(frame)
        cv2.imshow('Detector', frame)
        cv2.waitKey(1)
        try:
            with open('test.txt', encoding='utf-8', mode='a') as file:
                for i in hand_detector.data:
                    if i == hand_detector.data[-1]:
                        file.write(str(i))
                    else:
                        file.write(str(i) + ',')
                count += 1
                file.write('\n')
        except:
            continue
        print(count)


def euclid():
    filename = 'euclid_model.sav'
    capture = cv2.VideoCapture(0)
    hand_detector = Hand(max_num_hands=2)
    loaded_model = pickle.load(open(filename, 'rb'))
    t0 = 0
    while True:
        _, frame = capture.read()
        frame = hand_detector.detect(frame)
        try:
            if len(hand_detector.data) > 1:
                X = np.array(hand_detector.data[0])
                d04 = ((X[8] - X[0])**2 + (X[9] - X[1])**2)**(1/2)
                d08 = ((X[16] - X[0])**2 + (X[17] - X[1])**2)**(1/2)
                d012 = ((X[24] - X[0])**2 + (X[25] - X[1])**2)**(1/2)
                d016 = ((X[32] - X[0])**2 + (X[33] - X[1])**2)**(1/2)
                d020 = ((X[40] - X[0]) ** 2 + (X[41] - X[1]) ** 2)**(1 / 2)
                total = d04 + d08 + d012 + d016 + d020
                r04 = d04 / total
                r08 = d08 / total
                r012 = d012 / total
                r016 = d016 / total
                r020 = d020 / total
                actual = np.array([r04, r08, r012, r016, r020]).reshape(1, -1)
                result = loaded_model.predict(actual)
                cv2.putText(frame, result[0], (30, 400), cv2.FONT_ITALIC, 2,
                            (0, 255, 0), 2)

                X_right = np.array(hand_detector.data[1])
                d04_right = ((X_right[8] - X_right[0]) ** 2 + (X_right[9] - X_right[1]) ** 2) ** (1 / 2)
                d08_right = ((X_right[16] - X_right[0]) ** 2 + (X_right[17] - X_right[1]) ** 2) ** (1 / 2)
                d012_right = ((X_right[24] - X_right[0]) ** 2 + (X_right[25] - X_right[1]) ** 2) ** (1 / 2)
                d016_right = ((X_right[32] - X_right[0]) ** 2 + (X_right[33] - X_right[1]) ** 2) ** (1 / 2)
                d020_right = ((X_right[40] - X_right[0]) ** 2 + (X_right[41] - X_right[1]) ** 2) ** (1 / 2)
                total_right = d04_right + d08_right + d012_right + d016_right + d020_right
                r04_right = d04_right / total_right
                r08_right = d08_right / total_right
                r012_right = d012_right / total_right
                r016_right = d016_right / total_right
                r020_right = d020_right / total_right

                actual_right = np.array([r04_right, r08_right, r012_right, r016_right, r020_right]).reshape(1, -1)
                result_right = loaded_model.predict(actual_right)
                cv2.putText(frame, result_right[0], (300, 400), cv2.FONT_ITALIC, 2,
                            (0, 255, 0), 2)
        except:
            pass

        t1 = time.time()
        fps = 1 / (t1 - t0)
        t0 = t1
        frame = cv2.putText(frame, str(int(fps)), (20, 80), cv2.FONT_ITALIC,
                            1, (0, 255, 0), 2)
        cv2.imshow('Detector', frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    euclid()
