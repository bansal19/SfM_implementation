import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import funcs
import direct_linear_transform as dlt
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
print(len(kpts_1))
# For each pair of images find their matches
matches_12, mask_12, matches_21, mask_21 = funcs.find_matches(kpts_1, desc_1, kpts_2, desc_2, dist)

print(matches_12[0])
kpts_1 = np.array(kpts_1)
print(kpts_1[mask_12][0].pt)
# Using the first set of image (1,2) find the essential matrix and calculate the relative position
E, mask = cv.findEssentialMat(matches_12, matches_21)

print(len(matches_12))
# Filter the matches further down
filtered_12 = matches_12[(mask == 1)[:,0]]
filtered_21 = matches_21[(mask == 1)[:,0]]
print(len(filtered_12))
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

#Set up DLT for image 3
cloud = np.array(cloud)
old_points2D = [np.array(kpts_1)[mask_12][(mask == 1)[:,0]], desc_1[mask_12][(mask == 1)[:,0]]]
old_points3D = cloud[:,:3]
new_points = [kpts_3, desc_3]
calibration_matrix3 = dlt.perform_dlt(old_points2D, old_points3D, new_points)
cal_inverse = np.linalg.pinv(calibration_matrix3)
print(calibration_matrix3)
cloud_final = []
for point_a in kpts_3:
    point_a = point_a.pt
    pixel =  img_3[int(point_a[1])][int(point_a[0])]
    point_a = [point_a[0], point_a[1], 1]
    point_3d = np.dot(cal_inverse, point_a)
    point_3d = point_3d / point_3d[-1]
    cloud_final.append([point_3d[0], point_3d[1], point_3d[2], pixel[2], pixel[1], pixel[0]])

cloud_final = np.array(cloud_final)

# Render the point cloud
funcs.plot(cloud_final)