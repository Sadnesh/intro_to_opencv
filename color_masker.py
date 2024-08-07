"""I learn how to convert images from one color-space to another, like BGR to Gray, BGR to HSV, etc.
In addition to that, I will create an application to extract a colored object in a video
I also learned the following functions: cv.cvtColor(), cv.inRange(), etc."""

import cv2 as cv
import numpy as np

cam = cv.VideoCapture(0)

while True:
    _, frame = cam.read()

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # finds red in video
    lower_limit = np.array([100, 100, 100])
    upper_limit = np.array([255, 255, 255])

    mask = cv.inRange(hsv, lower_limit, upper_limit)

    res = cv.bitwise_and(frame, frame, mask=mask)

    cv.imshow("frame", frame)
    cv.imshow("mask", mask)
    cv.imshow("res", res)

    # if cv.waitKey(5) & 0xFF == 27: close when 'esc' pressed
    if cv.waitKey(5) & 0xFF == ord("q"):
        break
cv.destroyAllWindows()

# find hsv value by giving bgr value

# red = np.uint8([[[0, 0, 255]]])
# hsv = cv.cvtColor(red, cv.COLOR_BGR2HSV)
# print(hsv)
