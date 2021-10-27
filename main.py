import cv2 as cv
import numpy as np
import mediapipe as mp
import time

PEN = cv.imread('./images/pen.png')
RUBBER = cv.imread('./images/rubber.png')
IMAGE_ELEMENTS = cv.imread('./images/elements.png')

# cv.namedWindow("image", cv.WINDOW_NORMAL)
# cv.resizeWindow("image", 400, 400)
imageCanvas = np.zeros((400, 400, 3), np.uint8)


class HandDetector():
    def __init__(self, mode=False, max_hands=2, detection_con=0.85, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(False, 2)
        self.mp_draw = mp.solutions.drawing_utils
        self.PURPLE = 255, 0, 255

        self.tipIds = [4, 8, 12, 16, 20]

    def find_hands(self, frame):
        frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, handLms, self.mp_hands.HAND_CONNECTIONS)

    def is_finger_up(self):
        fingers = []

        if self.arr_of_positions[self.tipIds[0]][1] < self.arr_of_positions[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if self.arr_of_positions[self.tipIds[id]][2] < self.arr_of_positions[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def find_positions(self, frame, hand_number=0):
        self.arr_of_positions = []

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.arr_of_positions.append([id, cx, cy])
                    cv.circle(frame, (cx, cy), 3, self.PURPLE, cv.FILLED)

        return self.arr_of_positions

    def writeText(self, frame, fps, x, y):
        cv.putText(frame, str(int(fps)), (x, y),
                   cv.FONT_HERSHEY_PLAIN, 3, self.PURPLE, 3)


class Draw():
    def __init__(self, can_draw=True):
        self.x = 0
        self.y = 0
        self.prev_x = 0
        self.prev_y = 0
        self.can_draw = can_draw

    def draw(self, frame):
        cv.line(frame, (self.prev_x, self.prev_y),
                (self.x, self.y), (255, 0, 255), 15)
        cv.line(imageCanvas, (self.prev_x, self.prev_y),
                (self.x, self.y), (255, 0, 255), 15)

    def is_hand_on_pen(self, frame):
        self.can_draw = True

    def is_hand_on_rubber(self, frame):
        pass

    def coordinates(self, frame, coordinates_arr, drawing_mode):
        print(drawing_mode)

        self.x, self.y = coordinates_arr[8][1:]

        if self.prev_x == 0 and self.prev_y == 0:
            self.prev_x = self.x
            self.prev_y = self.y

        if drawing_mode:
            self.draw(frame)
        elif drawing_mode == False:
            pass

        self.prev_x, self.prev_y = self.x, self.y

        # if is in select mode
        # check if coord is close to pen or rubber image
        # if clicked either do what the function does

        # if is in drawing mode

        # self.draw(frame, imageCanvas)


def main():
    handDetector = HandDetector()
    draw = Draw(True)

    coord_arr = []

    cap = cv.VideoCapture(0)

    p_time = 0
    c_time = 0

    while True:
        ret, frame = cap.read()
        frame = cv.flip(frame, 1)
        frame = cv.resize(frame, (600, 600))

        if not ret:
            print("Can't receive frame. Exiting..")
            break

        handDetector.find_hands(frame)

        c_time = time.time()
        fps = 1/(c_time-p_time)
        p_time = c_time

        coord_arr = list(handDetector.find_positions(frame, 2))
        if(len(coord_arr) > 0):
            detect_fingers = handDetector.is_finger_up()
            print(detect_fingers)
            if detect_fingers[1] and detect_fingers[2]:
                draw.coordinates(frame, coord_arr, drawing_mode=False)
            elif detect_fingers[1] and detect_fingers[2] == False:
                draw.coordinates(frame, coord_arr, drawing_mode=True)

        handDetector.writeText(frame, fps, 10, 70)

        cv.imshow("image", frame)
        cv.imshow("Canvas", imageCanvas)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
