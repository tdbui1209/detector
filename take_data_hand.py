def data(output_file):
    capture = cv2.VideoCapture(0)
    hand_detector = Hand(max_num_hands=1)
    count = 0
    while True:
        _, frame = capture.read()
        frame = hand_detector.detect(frame)
        cv2.imshow('Detector', frame)
        cv2.waitKey(1)
        try:
            with open(output_file, encoding='utf-8', mode='a') as file:
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

if __name__ == '__main__':
    data('test.txt')
