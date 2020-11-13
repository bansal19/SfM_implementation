import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def stereo_image_depth():
    imgL = cv.imread('examples/bahen_left.png', 0)
    imgR = cv.imread('examples/bahen_right.png', 0)

    stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(imgL, imgR)
    plt.imshow(disparity, 'gray')
    plt.show()

def epipolar_geometry():
    # Implement epipolar geometry here

if __name__ == "__main__":
    stereo_image_depth()
