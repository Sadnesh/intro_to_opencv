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

display_image("Color changed Image", img)

# if it is grayscale image then it will return only rows and cols
(rows, cols, channels) = img.shape

# for image size just do
size = img.size

# for data type of the image
datatype = img.dtype

print(rows, cols, channels, size, datatype)

ball = img[230:283, 270:320]
# testing
# display_image("test", ball)
img[225:278, 130:180] = ball
display_image("cropped and pasted", img)

# we can split this way as well
# b, g, r = cv.split(img)

img[:, :, 0] = 0  # to make blue pixels to zero (image will appear a bit yellowish)
img[:, :, 1] = 0  # to make green pixels to zero (image will appear a bit magenta-ish)
img[:, :, 2] = 0  # to make red pixels to zero (image will appear a bit cyan-ish)

# and this way we can merge
# img = cv.merge((b, g, r))
display_image("merged", img)

