import cv2 as cv
import numpy as np

"""USEAGE OF SIMPLE THRESHOLDING"""
img = cv.imread("gradient.png", cv.IMREAD_GRAYSCALE)
img = cv.resize(
    img, None, fx=0.5, fy=0.5
)  # image was way too big, so scaling it to half
assert img is not None, "file could not be read, check with os.path.exists()"
ret, thres1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
ret, thres2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
ret, thres3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
ret, thres4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
ret, thres5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)

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
    cv.imshow(titles[i], images[i])
    cv.waitKey(0)
    cv.destroyAllWindows()

