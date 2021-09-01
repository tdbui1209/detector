import cv2
import mediapipe as mp


class Face:
    """
    Phát hiện khuôn mặt với 6 landmarks và tọa độ của bounding box.
    """
    def __init__(self, model_selection=0, min_detection_confidence=0.5):
        """
        Args:
            model_selection: Nhận hai giá trị 0 hoặc 1. 0 để sử dụng với ảnh
             trong khoảng cách 2 mét. 1 sử dụng với ảnh trong khoảng cách 5 mét.
              Mặc định là 0.
            min_detection_confidence: Ngưỡng tối thiểu sẽ phát hiện khuôn mặt
             ([0.0, 1.0]). Mặc định là 0.5.
        """
        self.model_selection = model_selection
        self.min_detection_confidence = min_detection_confidence
        self.face = mp.solutions.face_detection
        self.face_detection = self.face.FaceDetection(
            self.model_selection,
            self.min_detection_confidence)

        # Module draw của MediaPipe
        self.mp_draw = mp.solutions.drawing_utils

    def detect(self, frame, draw_bbox_only=True):
        """
        Args:
            frame: imput frame.
            draw_bbox_only: Nếu draw là True, sẽ chỉ vẽ bounding box. Ngược lại
             sẽ vẽ bounding box và 6 landmarks.
              Mặc định là True.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.face_detection.process(rgb_frame)
        if self.results.detections:
            if draw_bbox_only == False:
                self.pos = self.find_landmark(frame)
                self.draw_landmark(frame)
            self.bbox = self.find_bbox(frame)
            self.draw_bbox(frame)
        return frame

    def find_landmark(self, frame):
        """
        Lấy tọa độ các landmarks.
        """
        pos = []
        h, w, _ = frame.shape
        for face in self.results.detections:
            if face.score[0] > 0.5:
                lmk_class = face.location_data.relative_keypoints
                for idx, i in enumerate(lmk_class):
                    if i.x and i.y:
                        x = int(i.x * w)
                        y = int(i.y * h)
                        pos.append((idx, x, y))
        return pos

    def find_bbox(self, frame):
        """
        Lấy tọa độ xmin, ymin (góc trên cùng bên trái) và width, height của
         bounding box.
        """
        bbox = []
        h, w, _ = frame.shape
        for face in self.results.detections:
            if face.score[0] > 0.5:
                bbox_class = face.location_data.relative_bounding_box
                xmin = int(bbox_class.xmin * w)
                ymin = int(bbox_class.ymin * h)
                width = int(bbox_class.width * w)
                height = int(bbox_class.height * h)
                bbox.append((xmin, ymin, width, height))
        return bbox

    def draw_landmark(self, frame):
        """
        Vẽ các điểm landmarks.
        """
        for point in self.pos:
            frame = cv2.circle(frame, (point[1], point[2]), 3,
                               (0, 0, 255), cv2.FILLED)
        return frame

    def draw_bbox(self, frame):
        """
        Vẽ bounding box.
         """
        if self.bbox:
            frame = cv2.rectangle(
                frame, (self.bbox[0][0], self.bbox[0][1]),
                (self.bbox[0][0] + self.bbox[0][2], self.bbox[0][1] + self.bbox[0][3]),
                (0, 255, 0), 2
            )
        return frame

def main():
    capture = cv2.VideoCapture(0)
    face_detector = Face()
    while True:
        _, frame = capture.read()
        frame = face_detector.detect(frame, draw_bbox_only=False)
        cv2.imshow('Detector', frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
