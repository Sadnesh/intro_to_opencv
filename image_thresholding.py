import cv2 as cv
import numpy as np


def display_img(title: str, img) -> None:
    cv.imshow(title, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def simple_thresholding(img) -> None:
    """USEAGE OF SIMPLE THRESHOLDING"""
    _, thres1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    _, thres2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
    _, thres3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
    _, thres4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
    _, thres5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)

    titles = [
        "Original image",
        "Binary",
        "Binary inverse",
        "Trunc",
        "Tozero",
        "Tozero inverse",
    ]
    images = [
        img,
        thres1,
        thres2,
        thres3,
        thres4,
        thres5,
    ]

    for i in range(6):
        display_img(titles[i], images[i])


def adaptive_thresholding(img) -> None:
    """USEAGE OF ADAPTIVE THRESHOLDING"""
    blur_img = cv.medianBlur(img, 5)

    _, thres1 = cv.threshold(blur_img, 127, 255, cv.THRESH_BINARY)
    thres2 = cv.adaptiveThreshold(
        blur_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2
    )
    thres3 = cv.adaptiveThreshold(
        blur_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2
    )
    titles = [
        "Original Image",
        "Global thresholding W v=127",
        "Adaptive thresholding",
        "Adaptive gaussian thresholding",
    ]
    images = [img, thres1, thres2, thres3]

    for i in range(4):
        display_img(titles[i], images[i])


def main() -> None:
    img = cv.imread("images/gradient.png", cv.IMREAD_GRAYSCALE)
    img = cv.resize(
        img, None, fx=0.5, fy=0.5
    )  # image was way too big, so scaling down to half
    assert img is not None, "file could not be read, check with os.path.exists()"
    simple_thresholding(img)
    img = cv.imread("images/sudoku.png", cv.IMREAD_GRAYSCALE)
    adaptive_thresholding(img)


if __name__ == "__main__":
    main()
