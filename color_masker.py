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
    lower_red = np.array([100, 100, 100])
    upper_red = np.array([255, 255, 255])
    red_mask = cv.inRange(hsv, lower_red, upper_red)

    # finds blue in video
    lower_blue = np.array([110, 100, 100])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv.inRange(hsv, lower_blue, upper_blue)

    combined_mask = cv.bitwise_or(red_mask, blue_mask)

    res = cv.bitwise_and(frame, frame, mask=combined_mask)

    cv.imshow("frame", frame)
    cv.imshow("blue mask", blue_mask)
    cv.imshow("red mask", red_mask)
    cv.imshow("res", res)

    # if cv.waitKey(5) & 0xFF == 27: close when 'esc' pressed
    if cv.waitKey(5) & 0xFF == ord("q"):
        break

cam.release()
cv.destroyAllWindows()

# find hsv value by giving bgr value

# red = np.uint8([[[0, 0, 255]]])
# hsv = cv.cvtColor(red, cv.COLOR_BGR2HSV)
# print(hsv)
