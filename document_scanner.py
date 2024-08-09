# To use simply run
"""python document_scanner.py -i images/hello.png"""

import imutils
import argparse
import cv2 as cv
import numpy as np
from transform import four_point_transform
from skimage.filters import threshold_local

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image", required=True, help="Path to the image to be scanned")
args = vars(ap.parse_args())


image = cv.imread(args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height=500)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray, (5, 5), 0)
edged = cv.Canny(gray, 75, 200)

print("Step 1: Edge Detection")
cv.imshow("Image", image)
cv.imshow("Edged", edged)
cv.waitKey(0)
cv.destroyAllWindows()

cnts = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:5]

screenCnt = np.array([])  # defining empty matlike first

for con in cnts:
    peri = cv.arcLength(con, True)
    approx = cv.approxPolyDP(con, 0.02 * peri, True)

    if len(approx) == 4:
        screenCnt = approx
        break

print("Step 2: Find contours of paper")
cv.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv.imshow("Outline", image)
cv.waitKey(0)
cv.destroyAllWindows()


warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset=10, method="gaussian")
warped = (warped > T).astype("uint8") * 255


print("Step 3: Apply perspective transform")
cv.imshow("Original", imutils.resize(orig, height=650))
cv.imshow("Warped", imutils.resize(warped, height=650))
cv.waitKey(0)
cv.destroyAllWindows()
