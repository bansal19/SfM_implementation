import cv2 as cv
import sys
import os

if len(sys.argv) < 3:
    print("Usage: resize.py <folder> <size>")
    print("\t folder - The folder containing the images")
    print("\t size - The new size in percentage (e.g. 50) to resize to")
    exit()

folder = sys.argv[1]
size = int(sys.argv[2]) / 100

files = [file for file in os.listdir(folder) if os.path.isfile(os.path.join(folder, file))]

for file in files:
    img = cv.imread(os.path.join(folder, file))
    aspect = img.shape[0]/img.shape[1]
    width = int(img.shape[1] * size)
    height = int(width * aspect)
    img = cv.resize(img, (width, height))
    cv.imwrite(os.path.join(folder, file), img)
