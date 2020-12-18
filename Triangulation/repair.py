import numpy as np
import cv2 as cv
import os
import sys

if len(sys.argv) < 2:
    print('Usage: repair.py <folder>')
    exit()

folder = sys.argv[1]
calibs = [os.path.join(folder,'calib',file) for file in os.listdir(os.path.join(folder,'calib'))]

for calib in calibs:
    repaired = False
    with open(calib, 'r') as file:
        header = file.readline()
        if header == 'CONTOUR':
            repaired = True
        else:
            lines = file.readlines()

    if repaired is False:
        with open(calib, 'w') as file:
            file.writelines(lines)

print("Repair done")