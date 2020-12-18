import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.optimize
import plotly.graph_objects as go

from itertools import combinations
from relation import Relation
from random import randint
from math import atan2, sqrt, sin, cos, acos

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
    '''
    Breaks a camera projection matrix into intrinsic, rotation and translation
    P - The camera projection matrix
    '''
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

def rotation_decompose(R):
    # Find the rotation parameters
    theta_x = atan2(R[2,1], R[2,2])
    theta_y = atan2(-R[2,0], sqrt(R[2,1]**2 + R[2,2]**2))
    theta_z = atan2(R[1,0], R[0,0])

    # Return the thetas
    return theta_x, theta_y, theta_z

def rotation_compose(theta_x, theta_y, theta_z):
    # Create the rotation matrices
    rot_x = np.array([1, 0, 0, 0, cos(theta_x), -sin(theta_x), 0, sin(theta_x), cos(theta_x)]).reshape((3,3))
    rot_y = np.array([cos(theta_y), 0, sin(theta_y), 0, 1, 0, -sin(theta_y), 0, cos(theta_y)]).reshape((3,3))
    rot_z = np.array([cos(theta_z), -sin(theta_z), 0, sin(theta_z), cos(theta_z), 0, 0, 0, 1]).reshape((3,3))

    # Return the multiplication
    return np.matmul(rot_z, np.matmul(rot_y, rot_x))

def get_angle(a, b, deg = False):
    # Compute the dot product
    dot = np.dot(a.reshape((-1)), b.reshape((-1)))

    # Calculate the norms
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    # Compute the angle
    angle = acos( dot / (na * nb ))

    # Return the angle
    return angle if deg is False else np.rad2deg(angle)

def get_features(images):
    # Create the SIFT object
    sift = cv.SIFT_create()

    # Loop through the images
    for image in images:
        # Compute the keypoints
        kpts, desc = sift.detectAndCompute(image.data, None)

        # Get the points
        points = np.array([kpt.pt for kpt in kpts])
        
        # Save the information
        image.keypoints = kpts
        image.descriptions = desc
        image.points = points

def get_matches(images, dist):
    # Create the parameters for FLANN
    index_params = {'algorithm': 1, 'trees': 5}
    search_params = {'checks': 50}

    # Create the FLANN object
    flann = cv.FlannBasedMatcher(index_params, search_params)

    # Loop through the images
    for image_a, image_b in combinations(images, 2):
        # Get the descriptions
        desc_a = image_a.descriptions
        desc_b = image_b.descriptions

        # Get the matches
        matches = flann.knnMatch(desc_a, desc_b, k = 2)

        # The matches indices
        match_a = []
        match_b = []

        # Filter the matches
        for m, n in matches:
            if m.distance < dist * n.distance:
                # Add the point
                match_a.append(m.queryIdx)
                match_b.append(m.trainIdx)

        # Update the matches
        match_a = np.array(match_a).reshape((-1, 1))
        match_b = np.array(match_b).reshape((-1, 1))

        # Set the matches
        match_a, match_b = np.append(match_a, match_b, axis = 1), np.append(match_b, match_a, axis = 1)

        # Get the points
        points_a = image_a.points[match_a[:,0]]
        points_b = image_b.points[match_b[:,0]]

        # Calculate the essential matrix
        # F, mask = cv.findFundamentalMat(points_a, points_b)
        # E = np.matmul(image_b.intrinsic.T, np.matmul(F, image_a.intrinsic))
        E, mask = cv.findEssentialMat(points_a, points_b, image_a.intrinsic)
        # E, mask = cv.findEssentialMat(points_a, points_b)

        # Remove the points that don't match the mask
        match_a = match_a[(mask == 1)[:,0]]
        match_b = match_b[(mask == 1)[:,0]]

        # Create the relations
        rel_a = Relation(E, image_a, image_b, match_a, match_b)
        rel_b = Relation(E, image_b, image_a, match_b, match_a)

        # Add the relations
        image_a.add_relation(rel_a)
        image_b.add_relation(rel_b)

def get_cameras(images):
    # Find the best pair of images
    best_one, best_two, mask = _find_pair(images)

    # Recover the pose of the second image
    R, T = recover_pose(best_two, [best_one], [1])

    # Update the second image
    best_two.rotation = R
    best_two.translation = T

    # Add both images to the queue
    queue = [best_one, best_two]

    # The number of images left
    left = len(images) - 2

    # Loop as long as we have images left
    while left > 0:
        # Find an image
        idx, total, matches = _find_image(images, queue, mask)

        # Recover the pose
        R, T = recover_pose(images[idx], queue, matches)

        # Update the image rotation and translation
        images[idx].rotation = R
        images[idx].translation = T

        # Update the mask
        mask[idx] = 0

        # Add the image to the queue
        queue.append(images[idx])

        # Decrease the number of images left
        left -= 1        

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
    labels = [str(i) for i in range(len(pc))]
    fig = go.Figure(data=[go.Scatter3d(
        x=pc[:, 0],
        y=pc[:, 1],
        z=-pc[:, 2],
        mode='markers',
        text=labels,
        marker=dict(
            size=2,
            color=pc[:, 3:],
            opacity=1
        )
    )])
    fig.show()

def render_image(image):
    plt.imshow(image.data)
    plt.show()

def render_relation(image_a, image_b):
        # Get the relation
        relation = image_a.relations[image_b.name]

        # Get the keypoints
        keypoints_a = image_a.points[ relation.src_matches[:,0] ]
        keypoints_b = image_b.points[ relation.dst_matches[:,0] ]

        # Create the image
        image = np.zeros((max(image_a.data.shape[0], image_b.data.shape[0]), image_a.data.shape[1] + image_b.data.shape[1], 3))

        # Copy the images ontop
        image[0:image_a.data.shape[0],:image_a.data.shape[1],:] = image_a.data
        image[0:image_b.data.shape[0],image_a.data.shape[1]:,:] = image_b.data

        # Loop through the points
        for ptn_a, ptn_b in zip(keypoints_a, keypoints_b):
            start = (int(ptn_a[0]), int(ptn_a[1]))
            end = (int(ptn_b[0] + image_a.data.shape[1]), int(ptn_b[1]))
            image = cv.line(image, start, end, (randint(0, 255), randint(0, 255), randint(0, 255)), 2)

        # Show the image
        plt.imshow(image.astype(np.uint8))
        plt.show()

def reproj_error(pts_1, pts_2, P1, P2, norm = 2):
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
    return sqrt(np.sum(diff_1) + np.sum(diff_2))

def recover_pose(image, queue, matches):
    # Need to indices of the matches on the queue
    # Create the table and then sort the columns
    matches = np.append(matches, np.arange(len(queue)), axis = 0).reshape((2, -1))
    matches = matches[:,matches[1,:].argsort()]
    matches = matches[-1, :5]

    # This function construct the minimum function given points and other information
    def _const_min(image, queue, matches, R, T):
        def _min(lam):
            # The error
            error = 0

            # Construct the camera matrix
            P2 = camera_compose(image.intrinsic, R, T * lam[0])

            # Loop through the matches
            for i, idx in enumerate(matches):
                # Get the image
                q_image = queue[idx]

                # Get the relation
                relation = q_image.get_relation(image)

                # Get the camera
                P1 = q_image.get_camera()

                # Calculate the reprojection error
                error += reproj_error(relation.get_src_points(), relation.get_dst_points(), P1, P2)
            
            # Return error
            return error
        return _min

    # The projection error table
    table = np.zeros((len(matches), len(matches)))

    # The poses
    poses = []
    
    # Loop through the queue indices
    for i, idx in enumerate(matches):
        # Get the queue image
        q_image = queue[idx]

        # Get the relation between the two images
        relation = q_image.get_relation(image)

        # Get the rotation and translation between the cameras
        _, R, T, mask = cv.recoverPose(relation.E, relation.get_src_points(), relation.get_dst_points())

        # We need to rotation and translation based on the current image
        # R = R.T
        R = np.matmul(relation.src.get_rotation().T, R)
        T = relation.src.get_translation() + np.matmul(relation.src.get_rotation().T, T)

        # The minimum function
        _min = _const_min(image, queue, matches, R, T)

        # Minimize the error
        # res = scipy.optimize.minimize(_min, 1, method='Powell', bounds=[(-10, 10)])
        res = scipy.optimize.minimize(_min, 1, method='Powell')

        # Set the proper scale
        T *= res.x[0]

        # Add the poses
        poses.append([res.fun, R, T])

    # Sort the poses
    poses = sorted(poses, key = lambda i: i[0])
    
    # Return the best pose
    return poses[0][1], poses[0][2]

def combine_poses(image, queue, poses, matches):
    def _min(vec):
        # Get the scales
        data = np.array(vec)
        trans_scale = data[:len(matches)]
        trans_scale = trans_scale / np.sum(trans_scale)

        # The error
        error = 0

        # The rotation and translation
        R = poses[0][0]
        T = np.zeros(3).reshape((3,1))

        # Loop through the images
        for i, m in enumerate(matches):
            # Compute the rotation and translation matrices
            # TODO: update the rotation matrix
            # R = poses[i][0]
            T += poses[i][1] * trans_scale[i]

        # Create the matrix camera
        P2 = camera_compose(image.intrinsic, R, T)
        
        # Loop through the images
        for i, m in enumerate(matches):
            # Get the image
            img = queue[m]

            # Get the relation
            relation = img.get_relation(image)

            # Get the camera
            P1 = img.get_camera()            

            # Calculate the reprojection error
            error += reproj_error(relation.get_src_points(), relation.get_dst_points(), P1, P2)

        # Return the error
        return error
        
    # The initial vector
    init_vec = np.ones(len(matches))

    # Find the minimum of the function
    res = scipy.optimize.minimize(_min, init_vec, method='Powell')
    
    # Check the results
    if res.success:
        # Set the vector
        data = np.array(res.x)

    # Get the scales
    trans_scale = data[:len(matches)]
    trans_scale = trans_scale / np.sum(trans_scale)

    # The rotation and translation
    R = poses[0][0]
    T = np.zeros(3).reshape((3,1))

    # Loop through the images
    for i, m in enumerate(matches):
        # Compute the rotation and translation matrices
        # TODO: update the rotation matrix
        # R = poses[i][0]
        T += poses[i][1] * trans_scale[i]


    # Return the rotation and translation
    return R, T

def get_cloud(images):
    # The cloud
    cloud = []

    # Loop through the combinations of the images
    for image_a, image_b in combinations(images, 2):
        # Get the relation
        relation = image_a.get_relation(image_b)

        # Get the camera matrices
        P1 = image_a.get_camera()
        P2 = image_b.get_camera()

        # Loop through the points
        for pt1, pt2 in zip(relation.get_src_points(), relation.get_dst_points()):
            # Triangulate the point
            vec = triangulate(pt1, pt2, P1, P2)

            # Get the pixel
            pixel = image_a.data[int(pt1[1]), int(pt1[0])]

            # Add the vector the cloud
            cloud.append([vec[0], vec[1], vec[2], pixel[0], pixel[1], pixel[2]])

    # Return the cloud
    return np.array(cloud)

def _find_pair(images):
    # The number of images
    num = len(images)

    # The mask
    mask = np.ones(num)

    # The best data
    best_pair = (0, 0)
    best_matches = 0

    # Loop through the images
    for i in range(num):
        # Loop through the images
        for j in range(i + 1, num):
            # Get the relation between the two images
            relation = images[i].get_relation(images[j])

            # Check the number of matches
            if relation.get_num_matches() > best_matches:
                # Update the data
                best_pair = (i, j)
                best_matches = relation.get_num_matches()

    # Update the mast
    mask[best_pair[0]] = 0
    mask[best_pair[1]] = 0

    # Return the results
    return images[best_pair[0]], images[best_pair[1]], mask

def _find_image(images, queue, mask):
    # Create the info table
    table = np.zeros((int(np.sum(mask)), 2 + len(queue)), dtype=np.uint32)

    # The table index
    tbl_idx = 0
    
    # Loop through the images
    for i, image in enumerate(images):
        # Check the image is in the mask
        if mask[i] == 1:
            # The total
            total = 0

            # Loop through the images in the queue
            for j, img in enumerate(queue):
                # Get the relation
                relation = img.get_relation(image)

                # Add the total
                total += relation.get_num_matches()

                # Set the table information
                table[tbl_idx, j + 2] = relation.get_num_matches()

            # Update the table
            table[tbl_idx, 0] = i
            table[tbl_idx, 1] = total

            # Increase the table index
            tbl_idx += 1

    # Sort the table
    table = table[table[:,1].argsort()]
    
    # The bottom row contains the information we need
    idx = table[-1, 0]
    total = table[-1, 1]
    matches = table[-1, 2:]

    # Return the data
    return idx, total, matches
