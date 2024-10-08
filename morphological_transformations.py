"""
-To learn different morphological operations like Erosion, Dilation, Opening, Closing etc.
-To see different functions like : cv.erode(), cv.dilate(), cv.morphologyEx() etc.
"""

import cv2 as cv
import numpy as np


def display_image(title, img) -> None:
    cv.imshow(title, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def get_image(path):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    assert img is not None, "File couldn't be read, check with os.path.exist()"
    return img


def do_colored_erosion() -> None:
    img = cv.imread("images/colored_noisy.png")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kernel = np.ones((10, 10), np.uint8)

    eroded_img = cv.erode(gray, kernel, iterations=1)
    # Opening will return the same image will noise removed
    # eroded_img = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel, iterations=1)
    _, mask = cv.threshold(eroded_img, 1, 255, cv.THRESH_BINARY)
    img_color_part = cv.bitwise_and(img, img, mask=mask)

    display_image("erode", eroded_img)
    display_image("Mask", mask)
    display_image("Colored Part", img_color_part)


def main() -> None:
    img = get_image("images/letter_j.png")
    kernel = np.ones((5, 5), np.uint8)

    # EROSION
    erosion = cv.erode(img, kernel, iterations=1)
    display_image("Erosion", erosion)

    # DILATION
    dilation = cv.dilate(img, kernel, iterations=1)
    display_image("Dilation", dilation)

    # OPENING
    noisy_img1 = get_image("images/outer_noisy_j.png")
    # a bit larger images made me increase the kernel size to show the effect of noise reduction
    noisy_kernel = np.ones((15, 15), np.uint8)

    opening = cv.morphologyEx(noisy_img1, cv.MORPH_OPEN, noisy_kernel)
    display_image("Noisy image", noisy_img1)
    display_image("Opening", opening)

    # CLOSING
    noisy_img2 = get_image("images/inner_noisy_j.png")
    closing = cv.morphologyEx(noisy_img2, cv.MORPH_CLOSE, noisy_kernel)
    display_image("Noisy image", noisy_img2)
    display_image("Closing", closing)

    # MORPHOLOGICAL GRADIENT
    # difference between dilation and erosion of an image
    gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
    display_image("Gradient", gradient)

    # TOP HAT
    # difference between input image and opening of the image
    kernel = np.ones((9, 9), np.uint8)  # to demonstrate use-case
    tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
    display_image("Top Hat", tophat)

    # BLACK HAT
    # difference between input image and closing of the image
    blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)
    display_image("Top Hat", blackhat)

    do_colored_erosion()


if __name__ == "__main__":
    main()
