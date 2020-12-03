import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def calibrate(fname, num_corn, blk_size):
    crit = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    img = cv.imread(fname, cv.IMREAD_GRAYSCALE)

    ret, corners = cv.findChessboardCorners(img, num_corn, None)

    pts = np.zeros((num_corn[0] * num_corn[1], 3), np.float32)
    pts[:,:2] = np.mgrid[0:6,0:8].T.reshape(-1,2)
    pts *= blk_size

    if ret:
        ret, mat, dist, rvecs, tvecs = cv.calibrateCamera([pts], [corners], img.shape[::-1], None, None)
        return mat, dist, rvecs, tvecs

    return None

def get_matches(img_a, img_b, dist = 0.75, show = False):
    sift = cv.SIFT_create()
    kp_a, des_a = sift.detectAndCompute(img_a, None)
    kp_b, des_b = sift.detectAndCompute(img_b, None)
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des_a, des_b, k = 2)
    lst = []
    pts_a = []
    pts_b = []
    for a, b in matches:
        if a.distance < dist * b.distance:
            lst.append([a])
            pts_a.append(kp_a[a.queryIdx].pt)
            pts_b.append(kp_b[b.trainIdx].pt)
    if show:
        img = cv.drawMatchesKnn(img_a, kp_a, img_b, kp_b, lst, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img)
        plt.show()
    return np.array(pts_a, np.float32), np.array(pts_b, np.float32)

def cp_left_to_mat(vec):
    return np.array([0, -vec[2], vec[1], vec[2], 0, -vec[0], -vec[1], vec[0], 0]).reshape((3,3))

def get_sys_eqs(x, K, R, t):
    x_mat = cp_left_to_mat([x[0], x[1], 1])
    Rt = np.append(R, t.reshape((3,1)), axis=1)
    Mat = np.matmul(x_mat, np.matmul(K, Rt))
    A = Mat[:3,:3]
    b = Mat[:,-1] * -1
    return A, b

def get_3d(x, K, R1, R2, t):
    A1, b1 = get_sys_eqs(x, K, R1, t)
    A2, b2 = get_sys_eqs(x, K, R1, -t)
    A3, b3 = get_sys_eqs(x, K, R2, t)
    A4, b4 = get_sys_eqs(x, K, R2, -t)

    vecs = []

    vecs.append(np.matmul(np.linalg.pinv(A1), b1))
    vecs.append(np.matmul(np.linalg.pinv(A2), b2))
    vecs.append(np.matmul(np.linalg.pinv(A3), b3))
    vecs.append(np.matmul(np.linalg.pinv(A4), b4))

    # if np.linalg.matrix_rank(A1) == 3:
    #     vecs.append(np.linalg.solve(A1, b1))
    # if np.linalg.matrix_rank(A2) == 3:
    #     vecs.append(np.linalg.solve(A2, b2))
    # if np.linalg.matrix_rank(A3) == 3:
    #     vecs.append(np.linalg.solve(A3, b3))
    # if np.linalg.matrix_rank(A4) == 3:
    #     vecs.append(np.linalg.solve(A4, b4))



    return vecs