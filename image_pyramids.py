import cv2 as cv
import numpy as np

img = cv.imread("images/messi.png")
# examples of gaussian pyramid down (upto 4 levels)
lower_res = cv.pyrDown(img)
even_lower_res = cv.pyrDown(lower_res)
even_more_lower_res = cv.pyrDown(even_lower_res)

cv.imshow("original res", img)
cv.waitKey(0)
cv.imshow("lower res", lower_res)
cv.waitKey(0)
cv.imshow("even lower res", even_lower_res)
cv.waitKey(0)
cv.imshow("even more lower res", even_more_lower_res)
cv.waitKey(0)
cv.destroyAllWindows()

# examples of gaussian pyramind up (upto 4 levels )
higher_res = cv.pyrUp(
    img
)  # to see the consequences of scaling up and down these images use even_more_lower_res in this function
even_higher_res = cv.pyrUp(higher_res)
even_more_higher_res = cv.pyrUp(even_higher_res)
cv.imshow("original res", img)
cv.waitKey(0)
cv.imshow("higher res", higher_res)
cv.waitKey(0)
cv.imshow("even higher res", even_higher_res)
cv.waitKey(0)
cv.imshow("even more higher res", even_more_higher_res)
cv.waitKey(0)
cv.destroyAllWindows()
