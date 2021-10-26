import cv2 as cv
import numpy as np
import mediapipe as mp
import time


class HandDetector():
    def __init__(self, mode=False, max_hands=2, detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(False, 2)
        self.mp_draw = mp.solutions.drawing_utils
        self.PURPLE = 255, 0, 255

    def find_hands(self, frame):
        frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, handLms, self.mp_hands.HAND_CONNECTIONS)

    def find_position(self, frame, hand_number=0):
        arr_of_positions = []

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    arr_of_positions.append([id, cx, cy])
                    cv.circle(frame, (cx, cy), 10, self.PURPLE, cv.FILLED)

        return arr_of_positions

    def writeText(self, frame, fps, x, y):
        cv.putText(frame, str(int(fps)), (x, y),
                   cv.FONT_HERSHEY_PLAIN, 3, self.PURPLE, 3)


def main():
    handDetector = HandDetector()

    cap = cv.VideoCapture(0)

    p_time = 0
    c_time = 0

    while True:
        ret, frame = cap.read()
        frame = cv.flip(frame, 1)

        if not ret:
            print("Can't receive frame. Exiting..")
            break

        handDetector.find_hands(frame)

        c_time = time.time()
        fps = 1/(c_time-p_time)
        p_time = c_time

        handDetector.writeText(frame, fps, 10, 70)

        cv.imshow("image", frame)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
