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

# Image blending using pyramids

apple = cv.imread("images/apple.png")
orange = cv.imread("images/orange.png")

# generation of gaussian pyramid for apple
A_copy = apple.copy()
gpA = [A_copy]
for i in range(6):
    A_copy = cv.pyrDown(A_copy)
    gpA.append(A_copy)

# generation of gaussian pyramid for orange
O_copy = orange.copy()
gpO = [O_copy]
for i in range(6):
    O_copy = cv.pyrDown(O_copy)
    gpO.append(O_copy)

# generation of laplacian pyramid for apple
lpA = [gpA[5]]
for i in range(5, 0, -1):
    GE = cv.pyrUp(gpA[i])
    if GE.shape != gpA[i - 1].shape:
        GE = cv.resize(GE, (gpA[i - 1].shape[1], gpA[i - 1].shape[0]))
    # L = cv.subtract(gpA[i - 1], GE)
    L = cv.subtract(gpA[i - 1], GE)
    lpA.append(L)

# generation of laplacian pyramid for orange
lpO = [gpO[5]]
for i in range(5, 0, -1):
    GE = cv.pyrUp(gpO[i])
    if GE.shape != gpO[i - 1].shape:
        GE = cv.resize(GE, (gpO[i - 1].shape[1], gpO[i - 1].shape[0]))
    # L = cv.subtract(gpO[i - 1], GE)
    L = cv.subtract(gpO[i - 1], GE)
    lpO.append(L)

# adding left and right halves of images in each level
LS = []
for la, lo in zip(lpA, lpO):
    rows, cols, dpt = la.shape
    ls = np.hstack((la[:, 0 : cols // 2], lo[:, cols // 2 :]))
    LS.append(ls)

ls_ = LS[0]
for i in range(1, 6):
    ls_ = cv.pyrUp(ls_)
    if ls_.shape != LS[i].shape:
        ls_ = cv.resize(ls_, (LS[i].shape[1], LS[i].shape[0]))
    # ls_ = cv.add(ls_, LS[i])
    ls_ = cv.add(ls_, LS[i])

real = np.hstack((apple[:, : apple.shape[1] // 2], orange[:, orange.shape[1] // 2 :]))

cv.imshow("pyramid_blending", ls_)
cv.waitKey(0)
cv.imshow("direct blending", real)
cv.waitKey(0)
cv.destroyAllWindows()
