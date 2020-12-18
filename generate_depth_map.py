import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt


def create_depth_map():
    imgL = cv2.imread('imgs/5.jpg',0)
    imgR = cv2.imread('imgs/6.jpg',0)


    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(imgL,imgR)
    plt.imshow(disparity,'gray')
    plt.show()

def use_depth_map():
    pass

if __name__ == '__main__':
    create_depth_map()