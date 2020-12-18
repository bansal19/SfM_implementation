import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.linalg
import plotly.graph_objects as go

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

def camera_decompose(P):
    # Get the instrinsic and rotation matrix
    K, R = scipy.linalg.rq(P[:3,:3])

    # Get the translation
    T = np.matmul(np.linalg.inv(K), P[:,3].reshape(3,1))

    # Return the result
    return K, R, np.matmul(R.T, T)

def camera_compose(K, R, T):
    # Append the translate to the rotation matrix
    Rt = np.append(R, np.matmul(R, T), axis = 1)
    
    # Return the camera matrix
    return np.matmul(K, Rt)

def get_features(img):
    sift = cv.SIFT_create()
    return sift.detectAndCompute(img, None)

def find_matches(kpts_a, desc_a, kpts_b, desc_b, dist):
    # Create the parameters for FLANN
    index_params = {'algorithm': 1, 'trees': 5}
    search_params = {'checks': 50}

    # Create the FLANN object
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc_a, desc_b, k=2)

    # The masks for the relation
    mask_a = np.zeros(desc_a.shape[0], dtype=np.bool)
    mask_b = np.zeros(desc_b.shape[0], dtype=np.bool)

    # The points array
    pts_a = []
    pts_b = []

    # Filter the matches
    for m, n in matches:
        if m.distance < dist * n.distance:
            # Add the points
            pts_a.append(kpts_a[m.queryIdx].pt)
            pts_b.append(kpts_b[m.trainIdx].pt)

            # Add the mask
            mask_a[m.queryIdx] = True
            mask_b[m.trainIdx] = True

    # Return the masks
    return np.array(pts_a, np.float32), mask_a, np.array(pts_b, np.float32), mask_b

def triangulate(pt1, pt2, P1, P2):
    A = np.zeros((4,3))
    b = np.zeros(4)

    A[0] = (pt1[0] * P1[2,:3]) - P1[0,:3]
    A[1] = (pt1[1] * P1[2,:3]) - P1[1,:3]
    A[2] = (pt2[0] * P2[2,:3]) - P2[0,:3]
    A[3] = (pt2[1] * P2[2,:3]) - P2[1,:3]
    
    b[0] = P1[0,3] - pt1[0] * P1[2,3]
    b[1] = P1[1,3] - pt1[1] * P1[2,3]
    b[2] = P2[0,3] - pt2[0] * P2[2,3]
    b[3] = P2[1,3] - pt2[1] * P2[2,3]

    return np.matmul(np.linalg.pinv(A), b)

def project(vec, P1, normalize = False):
    # Get the pixel
    pixel = np.matmul(P1, np.append(vec.reshape((3,1)), [1]))

    # Check if we normalize
    # Return the pixel
    return pixel if normalize is False else (pixel[:2] / pixel[-1])

def plot(pc):
    '''
    plots the Nx6 point cloud pc in 3D
    assumes (1,0,0), (0,1,0), (0,0,-1) as basis
    '''
    fig = go.Figure(data=[go.Scatter3d(
        x=pc[:, 0],
        y=pc[:, 1],
        z=-pc[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=pc[:, 3:],
            opacity=1
        )
    )])
    fig.show()

def make_min_function(P1, pts_1, pts_2, K, R, t):
    def _min(lam):
        # Compose the camera with the translation vector
        P2 = camera_compose(K, R, t * lam[0])

        # Triangulate the points
        vectors = [triangulate(pt1, pt2, P1, P2) for pt1, pt2 in zip(pts_1, pts_2)]

        # Project the points back
        points_1 = np.array([project(vec, P1, True) for vec in vectors])
        points_2 = np.array([project(vec, P2, True) for vec in vectors])

        # Create the difference
        diff_1 = points_1 - pts_1
        diff_2 = points_2 - pts_2

        # Square them
        diff_1 = diff_1 ** 2
        diff_2 = diff_2 ** 2

        # Return the square reprojection error
        return np.sum(diff_1) + np.sum(diff_2)

    # Return the minimization function
    return _min