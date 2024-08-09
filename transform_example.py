# run this in command line as
""" python transform_example.py --image <path-to-image> --coords "[(top-left x and y), (top-right x and y), (bottom-right x and y), (bottom-left x and y)]"
"""

from transform import four_point_transform
import numpy as np
import cv2 as cv
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image file")
ap.add_argument("-c", "--coords", help="comma seperated list of source points")
args = vars(ap.parse_args())

image = cv.imread(args["image"])
pts = np.array(eval(args["coords"]), dtype="float32")

warped = four_point_transform(image, pts)

cv.imshow("Original", image)
cv.imshow("Warped", warped)
cv.waitKey(0)


# to test with this repo's hello.png the command will be,
""" python transform_example.py --image images/hello.png --coords "[(73, 239), (356, 117), (475, 265), (187, 443)]"
"""
