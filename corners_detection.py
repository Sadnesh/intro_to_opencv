"""Basically vertex detection"""

import cv2 as cv
import numpy as np


def detect_corner(gray_image):
    gray = np.float32(gray_image)
    res = cv.cornerHarris(gray, 2, 3, 0.04)  # type:ignore
    res = cv.dilate(res, None)  # type:ignore
    return res


def detect_corner_W_subpixel_accuracy(gray_image):
    gray = np.float32(gray_image)
    res = cv.cornerHarris(gray, 2, 3, 0.04)  # type:ignore
    res = cv.dilate(res, None)  # type:ignore
    _, dst = cv.threshold(res, 0.01 * res.max(), 255, 0)
    dst = np.uint8(dst)

    _, labels, stats, centriods = cv.connectedComponentsWithStats(dst)  # type:ignore

    criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_MAX_ITER, 100, 0.001)
    corners = cv.cornerSubPix(
        gray, np.float32(centriods), (5, 5), (-1, -1), criteria  # type:ignore
    )
    ret = np.hstack((centriods, corners))
    return np.int_(ret)


def display_image(title, image):
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def main():
    chess_board = "images/chess_board.png"
    twisted_chess_board = "images/chess_board_warped.png"
    pixeled_path = "images/pixeled_image.png"

    chess_img = cv.imread(chess_board)
    twisted_chess_img = cv.imread(twisted_chess_board)
    pix_img = cv.imread(pixeled_path)

    res1 = detect_corner(cv.cvtColor(chess_img, cv.COLOR_BGR2GRAY))
    res2 = detect_corner(cv.cvtColor(twisted_chess_img, cv.COLOR_BGR2GRAY))

    chess_img[res1 > 0.01 * res1.max()] = [0, 0, 255]
    twisted_chess_img[res2 > 0.1 * res2.max()] = [0, 0, 255]

    display_image("Normal board", chess_img)
    display_image("Twisted board", twisted_chess_img)

    res3 = detect_corner_W_subpixel_accuracy(cv.cvtColor(pix_img, cv.COLOR_BGR2GRAY))
    pix_img[res3[:, 1], res3[:, 0]] = [0, 0, 255]  # type:ignore
    pix_img[res3[:, 3], res3[:, 2]] = [0, 255, 0]  # type:ignore

    display_image("Pixel Corner detection", pix_img)


if __name__ == "__main__":
    main()
