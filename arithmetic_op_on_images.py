import cv2 as cv
import numpy as np
from PIL import Image

# image addition
x = np.uint8([250])  # type:ignore
y = np.uint8([10])  # type:ignore

print(cv.add(x, y))  # type:ignore
# output: 260 as 250 + 10 --> 260

# but
print(x + y)  # output: 4 as 250 + 10 = 260 % 256 --> 4


def rotate(image, angle, scale):
    # for now center of the image itself
    center = image.shape[0] // 2, image.shape[1] // 2

    rows, cols = image.shape[:2]
    matrix = cv.getRotationMatrix2D(center, angle, scale)
    rotated = cv.warpAffine(image, matrix, (cols, rows))
    return rotated


# image blending example using same image
img1 = cv.imread("images/letter_j.png")
img2 = rotate(img1, 180, 1)

dst = cv.addWeighted(img1, 0.7, img2, 0.3, 0)
cv.imshow("dst", dst)
cv.waitKey(0)
cv.destroyAllWindows()
