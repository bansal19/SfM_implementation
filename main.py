import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import funcs
import os

# The fixed matrix was derived with:
# K, dist, rvecs, tvecs = funcs.calibrate('calibration.jpg', (6, 8), 1)
# The blocks are 15mm x 15mm
K = np.array([8.57388364e+02, 0.00000000e+00, 2.34326856e+03, \
    0.00000000e+00, 8.50006705e+02, 1.10511041e+03, \
    0.00000000e+00, 0.00000000e+00, 1.00000000e+00]).reshape((3,3))

folder = 'imgs'
img_names = [os.path.join(folder, file) for file in os.listdir(folder)  if os.path.isfile(os.path.join(folder, file))]
imgs = [cv.imread(fname) for fname in img_names]
matches = [funcs.get_matches(imgs[i], imgs[i+1], 0.95) for i in range(len(imgs)-1)]
essentials = [cv.findEssentialMat(a,b) for a, b in matches]
decomp = [cv.decomposeEssentialMat(E) for E, _ in essentials]

pc = []

for (_, points), (R1, R2, t), img in zip(matches, decomp, imgs[1:]):
    for point in points:
        vecs = funcs.get_3d(point, K, R1, R2, t)
        vec = vecs[0]
        pixel = img[int(point[1])][int(point[0])]
        pc.append([vec[0], vec[1], vec[2], pixel[0], pixel[1], pixel[2]])

np.savetxt('model.txt', np.array(pc))


# with open('model.txt', 'w') as file:
#     for (_, points), (R1, R2, t) in zip(matches, decomp):
#         for point in points:
#             vecs = funcs.get_3d(point, K, R1, R2, t)
#             vec = vecs[3]
#             file.write(f"{vec[0]},{vec[1]},{vec[2]}\n")


# pts_a, pts_b = funcs.get_matches(imgs[0], imgs[1], 0.95)

# E, mask = cv.findEssentialMat(pts_a, pts_b, K)

# Another way to calculate essential matrix (F = K'T * E * K)
# F, mask = cv.findFundamentalMat(pts_a, pts_b)
# Ki = np.linalg.inv(K)
# E2 = np.matmul(Ki.T, np.matmul(F, Ki))

# R1, R2, t = cv.decomposeEssentialMat(E2)

# dmap = np.zeros((imgs[0].shape[0], imgs[0].shape[1]), np.float32)
# x = [funcs.get_3d(pt, K, R1, R2, t) for pt in pts_b]
# for vec, pt in zip(x, pts_b):
    # v = vec[3]
    # cv.circle(dmap, (int(pt[0]), int(pt[1])), 20, (v[2], v[2], v[2]), -1)

# for r in range(imgs[0].shape[0]):
#     for c in range(imgs[0].shape[1]):
#         vec = funcs.get_3d((r,c), K, R1, R2, t)
#         dmap[r][c] = vec[3][2]

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for vec in x:
#     ax.scatter(vec[0], vec[1], vec[2], marker='o')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.imshow(dmap, cmap='gray')
# plt.show()