"""
TO LEARN:
    - Access pixel values and modify them
    - Access image properties
    - Set a Region of Interest (ROI)
    - Split and merge images
"""

import cv2 as cv
import numpy as np


def display_image(title: str, img):
    cv.imshow(title, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


img = cv.imread("images/messi.png")
assert img is not None, "File cannot be read, check it with os.path.exists()"

# changing specific pixel value
# Modify the pixel value at (10, 10) in the blue channel (index 2)
img[10, 10, 2] = 100

# changing pixel value in the given range
lower_blue = np.array([100, 0, 0])
upper_blue = np.array([255, 70, 70])

mask = cv.inRange(img, lower_blue, upper_blue)

# select all the white areas of the mask and change to green
img[mask != 0] = [0, 255, 0]

display_image("sth", img)
