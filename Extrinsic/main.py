import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import funcs
import os

import scipy.optimize

from image import Image

# The fixed matrix was derived with:
# K, dist, rvecs, tvecs = funcs.calibrate('calibration.jpg', (6, 8), 1)
# The blocks are 15mm x 15mm
K = np.array([8.57388364e+02, 0.00000000e+00, 2.34326856e+03, \
    0.00000000e+00, 8.50006705e+02, 1.10511041e+03, \
    0.00000000e+00, 0.00000000e+00, 1.00000000e+00]).reshape((3,3))

# The folder
folder = 'house'

# Load the images
img_names = [file for file in os.listdir(folder)  if os.path.isfile(os.path.join(folder, file))]
images = [Image(fname, cv.cvtColor(cv.imread(os.path.join(folder, fname)), cv.COLOR_BGR2RGB), K) for fname in img_names[:5]]

# The matching filtering distance
dist = 0.95

# Get the features from the images
funcs.get_features(images)

# Get the matches
funcs.get_matches(images, dist)

# Get the camera matrices
funcs.get_cameras(images)

# Get the cloud
# Render the cloud
cloud = funcs.get_cloud(images)
funcs.plot(cloud)