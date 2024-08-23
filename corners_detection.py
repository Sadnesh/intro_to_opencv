"""Basically vertex detection"""

import cv2 as cv
import numpy as np


def detect_corner(gray_image):
    gray = np.float32(gray_image)
    res = cv.cornerHarris(gray, 2, 3, 0.04)  # type:ignore
    res = cv.dilate(res, None)  # type:ignore
    return res


def display_image(title, image):
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def main():
    chess_board = "images/chess_board.png"
    twisted_chess_board = "images/chess_board_warped.png"
    chess_img = cv.imread(chess_board)
    twisted_chess_img = cv.imread(twisted_chess_board)

    res1 = detect_corner(cv.cvtColor(chess_img, cv.COLOR_BGR2GRAY))
    res2 = detect_corner(cv.cvtColor(twisted_chess_img, cv.COLOR_BGR2GRAY))

    chess_img[res1 > 0.01 * res1.max()] = [0, 0, 255]
    twisted_chess_img[res2 > 0.01 * res2.max()] = [0, 0, 255]

    display_image("Normal board", chess_img)
    display_image("Twisted board", twisted_chess_img)


if __name__ == "__main__":
    main()
