import numpy as np
import funcs

def normalize_and_make_homogeneous(x_unnormalized):
    """Modify x_unnormalized to normalize the vector according to standard DLT methods and make homogeneous. Normalization is used to stabilize calculation of DLT
    
    x_unnormalized: 3 or 2 dimensional input data to be normalized
    """
    mean, std_dev = np.mean(x_unnormalized, 0), np.std(x_unnormalized)
    if x_unnormalized.shape[1] == 2:
        transform = np.array([[std_dev, 0, mean[0]], [0, std_dev, mean[1]], [0, 0, 1]])
    elif x_unnormalized.shape[1] == 3:
        transform = np.array([[std_dev, 0, 0, mean[0]], [0, std_dev, 0, mean[1]], [0, 0, std_dev, mean[2]], [0, 0, 0, 1]])
    else:
        print("Please use number of dimensions equal to 2 or 3.")
        assert False
    transform = np.linalg.inv(transform)
    x_unnormalized = np.dot(transform, np.concatenate((x_unnormalized.T, np.ones((1,x_unnormalized.shape[0])))))
    x_unnormalized = x_unnormalized[0:x_unnormalized.shape[1], :].T

    return transform, x_unnormalized

def calculate_calibration_matrix(points3D, points2D):
    """Use DLT to calculate the 11 params of the calibration matrix. Calibration matrix transforms from 3D to 2D."""
    transform3D, points3D_norm = normalize_and_make_homogeneous(points3D)
    transform2D, points2D_norm = normalize_and_make_homogeneous(points2D)
    
    matrix = []
    for i in range(points3D.shape[0]):
        X, Y, Z = points3D_norm[i,0], points3D_norm[i,1], points3D_norm[i,2]
        x, y = points2D_norm[i, 0], points2D_norm[i, 1]
        matrix.append([-X, -Y, -Z, -1, 0, 0, 0, 0, x*X, x*Y, x*Z])
        matrix.append([0,0,0,0,-X,-Y,-Z,-1,y*X,y*Y,y*Z])
    matrix = np.array(matrix)
    _,_,V = np.linalg.svd(matrix)
    calibration_matrix = np.reshape(V[-1,:], (3, 4))
    
    #Invert normalization with transform matrices
    calibration_matrix = np.dot(np.linalg.pinv(transform2D), np.dot(H, transform3D))
    return calibration_matrix
    
def get_calibration_points(old_points, old_points3D, new_image, dist):
    """Match feature points for the new image to the point cloud."""
    _, mask_a, points2D, _ = funcs.find_matches(old_points[:,0], old_points[:,1], new_image[:,0], new_image[:,1], dist)
    points_3D = old_points3D[mask_a]
    return points3D, points2D
    
def perform_dlt(old_points, old_points3D, new_image, dist=0.7):
    """Perform dlt on the new image to get camera matrix.
    
    old_points: 2D points in pointcloud from image 1 or 2. numpy array with old_points[:,0] being 2d points and old_points[:,1] being description vectors for the points
    old_points3D: same points as old_points but with depth. old_points3D are in the same order as old_points
    new_image: 2D points from image 3. numpy array with new_image[:,0] being 2d points and new_image[:,1] being description vectors for the points
    """
    points3D, points2D = get_calibration_points(old_points, old_points3D, new_image, dist)
    return calculate_calibration_matrix(points3D, points2D)
