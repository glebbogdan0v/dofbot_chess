import cv2
import numpy as np
from typing import List
from square_data import SquareData

class ChessBoardDetector:
    def __init__(self):
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.square_info = []

    def drawSquare(self, img, corners):
        corners = corners.reshape(-1, 2)
        top_left = tuple(corners[0].ravel())
        top_right = tuple(corners[8].ravel())
        bottom_right = tuple(corners[-1].ravel())
        bottom_left = tuple(corners[-9].ravel())
        center = tuple([abs(int((top_left[0] + top_right[0]) / 2)), abs(int((top_right[1] + bottom_right[1]) / 2))])
        top_side_len = abs(int(top_left[0] - top_right[0]))

        pts = np.array([top_left, top_right, bottom_right, bottom_left], np.int32)
        pts = pts.reshape((-1, 1, 2))

        cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.circle(img, center, radius=5, color=(0, 0, 255), thickness=2)

        return tuple([img, center, top_side_len])

    def run(self):
        cap = cv2.VideoCapture(0)

        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('img', 1280, 760)

        count = 0
        while True:
            ret, frame = cap.read()

            frame = cv2.flip(frame, 1)
            frame = draw_circle(frame)
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                frame, center, top_side_len = self.drawSquare(frame, corners2)

            cv2.imshow('img', frame)
            key = cv2.waitKey(1) & 0xFF

            if count == 2:
                get_shift_info(self.square_info)
                break

            if key == ord('p') and count < 2 and ret:
                is_saved = cv2.imwrite(f"chessboard/{count}_photo.png", frame)
                if is_saved:
                    self.square_info.append(SquareData(center, top_side_len))
                    count += 1
                    print('Successfully saved')


            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
		
def get_shift_info(square_info: List):
    start_dir = 50
    f = 63

    k = square_info[0].top_side_len / square_info[1].top_side_len
    x = square_info[0].center[0] - square_info[1].center[0]
    y = square_info[0].center[1] - square_info[1].center[1]

    z_dist = k * start_dir - start_dir
    x_dist = int(x * (start_dir + z_dist) / f ) // 10
    y_dist = int(y * (start_dir + z_dist) / f) // 10

    print(f"По координате X: {x_dist}см")
    print(f"По координате y: {y_dist}см")
    print(f"Глубина: {z_dist}см")
	
def draw_circle(frame):
    cv2.circle(frame, (320, 240), radius=5, color=(0, 255, 0), thickness=2)
    return frame
	
if __name__ == "__main__":
    chess_board_detector = ChessBoardDetector()
    chess_board_detector.run()