import cv2
import mediapipe as mp
import pickle
import numpy as np
import time
import pandas as pd


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

        Return:
            frame đã phát hiện bàn tay.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hand_detection.process(rgb_frame)
        if self.results.multi_hand_landmarks:
            self.hands = []
            self.data = []
            for self.hand in self.results.multi_hand_landmarks:
                self.hands.append(self.find_landmark(frame))
                self.data.append(self.take_data(frame))
                if draw:
                    frame = self.draw_connection(frame)
                    frame = self.draw_landmark(frame)
        return frame

    def find_landmark(self, frame):
        """
        Lấy tọa độ 21 landmarks.
        Args:
            frame: input frame.

        Return:
            list 21 landmarks, mỗi landmark là một set gồm
             index, tọa độ x, tọa độ y.
        """
        landmark_positions = []
        frame_height, frame_width, _ = frame.shape
        for index_landmark, landmark in enumerate(self.hand.landmark):
            x_landmark = int(landmark.x * frame_width)
            y_landmark = int(landmark.y * frame_height)
            landmark_positions.append((index_landmark, x_landmark, y_landmark))
        return landmark_positions

    def draw_landmark(self, frame):
        """
        Vẽ các điểm landmarks.
        Args:
            frame: input frame.

        Return:
            frame có các điểm landmars được vẽ hình tròn.
        """
        CIRCLE_SIZE = 3
        CIRCLE_COLOR = (0, 0, 255) # Red
        for hand in self.hands:
            for landmark in hand:
                x_landmark = landmark[1]
                y_landmark = landmark[2]
                frame = cv2.circle(frame, (x_landmark, y_landmark), CIRCLE_SIZE,
                                   CIRCLE_COLOR, cv2.FILLED)
        return frame

    def draw_connection(self, frame):
        """
        Vẽ các connections nối các landmarks.
        """
        LINE_COLOR = (0, 255, 0) # Green
        LINE_SIZE = 2
        CONNECTIONS = [(0, 1),  # wrist -> thumb_cmc
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
        for hand in self.hands:
            for connect in CONNECTIONS:
                first_point = hand[connect[0]]
                second_point = hand[connect[1]]

                x_first_point = first_point[1]
                y_first_point = first_point[2]
                x_second_point = second_point[1]
                y_second_point = second_point[2]

                frame = cv2.line(frame, (x_first_point, y_first_point),
                    (x_second_point, y_second_point), LINE_COLOR, LINE_SIZE)
        return frame

    def take_data(self, frame):
        """
        Lấy dữ liệu để dự đoán cử chỉ.

        Return: list tọa độ x, y của từng landmark liên tiếp nhau.
         [x1, y1, x2, y2, ..., xn, yn]
        """
        data = []
        frame_height, frame_width, _ = frame.shape
        for _, landmark in enumerate(self.hand.landmark):
            x_landmark = int(landmark.x * frame_width)
            y_landmark = int(landmark.y * frame_height)
            data.append(x_landmark)
            data.append(y_landmark)
        return data

    def predict_gesture(self, loaded_model, frame):
        if len(self.data) > 1:
            first_hand = self.data[0]
            second_hand = self.data[1]

            X_rescaled = []
            for hand in [first_hand, second_hand]:
                X_rescaled.append(self.preprocess_input(hand))

            X_first_hand = np.array(X_rescaled[0]).reshape(1, -1)
            result_first_hand = loaded_model.predict(X_first_hand)
            cv2.putText(frame, result_first_hand[0], (30, 400), cv2.FONT_ITALIC, 2,
                        (0, 255, 0), 2)

            X_second_hand = np.array(X_rescaled[1]).reshape(1, -1)
            result_second_hand = loaded_model.predict(X_second_hand)
            cv2.putText(frame, result_second_hand[0], (300, 400), cv2.FONT_ITALIC, 2,
                        (0, 255, 0), 2)


    def calculate_euclid_distance(self, x_first_point, y_first_point,
                                  x_second_point, y_second_point):

        distance = ((x_first_point - x_second_point)**2
            + (y_first_point - y_second_point)**2)**(1/2)

        return distance

    def preprocess_input(self, hand):
        X = np.array(hand)
        total = 0
        distancies = []
        for x, y in zip([8, 16, 24, 32, 40], [9, 17, 25, 33, 41]):
            distance = self.calculate_euclid_distance(X[x], X[y], X[0], X[1])
            distancies.append(distance)
            total += distance

        rescaled = []
        for distance in distancies:
            rescaled.append(distance / total)
        return rescaled


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


def main():
    filename = 'euclid_model.sav'
    capture = cv2.VideoCapture(0)
    hand_detector = Hand(max_num_hands=2)
    loaded_model = pickle.load(open(filename, 'rb'))
    t0 = 0
    while True:
        _, frame = capture.read()
        frame = hand_detector.detect(frame)
        hand_detector.predict_gesture(loaded_model, frame)
        t1 = time.time()
        fps = 1 / (t1 - t0)
        t0 = t1
        frame = cv2.putText(frame, str(int(fps)), (20, 80), cv2.FONT_ITALIC,
                            1, (0, 255, 0), 2)
        cv2.imshow('Detector', frame)
        cv2.waitKey(1)



if __name__ == '__main__':
    main()
