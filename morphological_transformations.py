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


img = cv.imread("images/letter_j.png", cv.IMREAD_GRAYSCALE)
assert img is not None, "File couldn't be read, check with os.path.exist()"
kernel = np.ones((5, 5), np.uint8)

erosion = cv.erode(img, kernel, iterations=1)
display_image("Erosion", erosion)

dilation = cv.dilate(img, kernel, iterations=1)
display_image("Dilation", dilation)

