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

