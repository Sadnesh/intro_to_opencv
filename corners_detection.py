"""Basically vertex detection"""

import cv2 as cv
import numpy as np


def main():
    chess_board = "images/chess_board.png"
    chess_img = cv.imread(chess_board)
    gray = cv.cvtColor(chess_img, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    res = cv.cornerHarris(gray, 2, 3, 0.04)  # type:ignore
    res = cv.dilate(res, None)  # type:ignore

    chess_img[res > 0.01 * res.max()] = [0, 0, 255]

    cv.imshow("result", chess_img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
