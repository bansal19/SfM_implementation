import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import funcs
import os

import scipy.optimize

# The fixed matrix was derived with:
# K, dist, rvecs, tvecs = funcs.calibrate('calibration.jpg', (6, 8), 1)
# The blocks are 15mm x 15mm
K = np.array([8.57388364e+02, 0.00000000e+00, 2.34326856e+03, \
    0.00000000e+00, 8.50006705e+02, 1.10511041e+03, \
    0.00000000e+00, 0.00000000e+00, 1.00000000e+00]).reshape((3,3))

# folder = 'imgs'
# img_names = [os.path.join(folder, file) for file in os.listdir(folder)  if os.path.isfile(os.path.join(folder, file))]
# imgs = [cv.imread(fname) for fname in img_names]
# matches = [funcs.get_matches(imgs[i], imgs[i+1], 0.5) for i in range(len(imgs)-1)]
# essentials = [cv.findEssentialMat(a, b, K) for a, b in matches]
# decomp = [cv.recoverPose(E, a, b) for (E, _), (a, b) in zip(essentials, matches)]

# The matching filtering distance
dist = 0.70

# Load the images
img_1 = cv.imread('imgs/1.jpg')
img_2 = cv.imread('imgs/2.jpg')
img_3 = cv.imread('imgs/3.jpg')

# Find the features and descriptions of the images
kpts_1, desc_1 = funcs.get_features(img_1)
kpts_2, desc_2 = funcs.get_features(img_2)
kpts_3, desc_3 = funcs.get_features(img_3)

# For each pair of images find their matches
matches_12, mask_12, matches_21, mask_21 = funcs.find_matches(kpts_1, desc_1, kpts_2, desc_2, dist)
matches_13, mask_13, matches_31, mask_31 = funcs.find_matches(kpts_1, desc_1, kpts_3, desc_3, dist)
matches_23, mask_23, matches_32, mask_32 = funcs.find_matches(kpts_2, desc_2, kpts_3, desc_3, dist)

# Using the first set of image (1,2) find the essential matrix and calculate the relative position
E, mask = cv.findEssentialMat(matches_12, matches_21)

# Filter the matches further down
filtered_12 = matches_12[(mask == 1)[:,0]]
filtered_21 = matches_21[(mask == 1)[:,0]]

# Recover the pose from E
ret, R, t, _ = cv.recoverPose(E, filtered_12, filtered_21)

# The point cloud
cloud = []

# Build the camera matrix
# Assume 1 is at the chilling at the origin
P1 = funcs.camera_compose(K, np.diag([1,1,1]), np.zeros((3,1)))

# Create the minimization function
min_func = funcs.make_min_function(P1, filtered_12, filtered_21, K, R, t)

# Now it gets fun
# Miniminize the function to get an estimate of lambda (for the translation)
# Use 'Powell' (might need more testing)
res = scipy.optimize.minimize(min_func, 1, method='Powell')

# Create the second camera
P2 = funcs.camera_compose(K, R, t * res.x[0])

# Go through the matches
for point_a, point_b in zip(filtered_12, filtered_21):
    # Triangulate the points
    vec = funcs.triangulate(point_a, point_b, P1, P2)

    # Get the pixel
    pixel = img_1[int(point_a[1])][int(point_a[0])]

    # Add it to the cloud
    cloud.append([vec[0], vec[1], vec[2], pixel[2], pixel[1], pixel[0]])

# Render the point cloud
funcs.plot(np.array(cloud))