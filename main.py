import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import funcs

# The fixed matrix was derived with:
# K = funcs.calibrate('calibration.jpg', (6, 8), 0.015)
# The blocks are 15mm x 15mm
K = np.array([8.57388364e+02, 0.00000000e+00, 2.34326856e+03, \
    0.00000000e+00, 8.50006705e+02, 1.10511041e+03, \
    0.00000000e+00, 0.00000000e+00, 1.00000000e+00]).reshape((3,3))


img_names = ['img_left.jpg', 'img_right.jpg']
imgs = [cv.imread(fname) for fname in img_names]
pts_a, pts_b = funcs.get_matches(imgs[0], imgs[1], 0.95)

E, mask = cv.findEssentialMat(pts_a, pts_b, K)

R1, R2, t = cv.decomposeEssentialMat(E)

dmap = np.zeros((imgs[0].shape[0], imgs[0].shape[1]), np.float32)
x = [funcs.get_3d(pt, K, R1, R2, t) for pt in pts_b]
for vec, pt in zip(x, pts_b):
    v = vec[2]
    cv.circle(dmap, (int(pt[0]), int(pt[1])), 20, (v[2], v[2], v[2]), -1)

plt.imshow(dmap, cmap='gray')
plt.show()