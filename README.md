# TDDetector
This detector based on three solutions of MediaPipe: [Face Detection][1], [Hands][2], [Pose][3]<br>
I build this package with three modules for each solutions. So you can detect face, hands or pose, even all togethers with few of codes.
Not just detect, i will develop some features base on MediaPipe in the future.
<hr>

### Install requirements.txt first

pip install -r requirements.txt

### Example:

<pre>
import cv2 # open-cv
import mediapipe as mp
from hand import Hand

capture = cv2.VideoCapture(0)
hand_detector = Hand(max_num_hands=2)  # detector will detect maximum two hands in one frame
while True:
    _, frame = capture.read()
    frame = hand_detector.detect(frame)
    cv2.imshow('Frame', frame)
    cv2.waitKey(1)
    
# Ctrl + C to quit
</pre>
![alt][4]

### PREDICT GESTURE
<pre>
import cv2
import mediapipe as mp
from hand import Hand
import pickle

MODEL_PATH = 'euclid_model.sav'
LOADED_MODEL = pickle.load(open(MODEL_PATH, 'rb'))

capture = cv2.VideoCapture(0)
hand_detector = Hand(max_num_hands=2)

while True:
    _, frame = capture.read()
    frame = hand_detector.detect(frame)
    hand_detector.predict_gesture(LOADED_MODEL, frame)
    
    cv2.imshow('Frame', frame)
    cv2.waitKey(1)
    
# Ctrl + C to quit

</pre>




[1]: https://google.github.io/mediapipe/solutions/face_detection.html
[2]: https://google.github.io/mediapipe/solutions/hands.html
[3]: https://google.github.io/mediapipe/solutions/pose.html

[4]: https://i.imgur.com/rLbi0Zf.png
