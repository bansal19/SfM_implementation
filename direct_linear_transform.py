import numpy as np

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

    return transform, x

def calculate_calibration_matrix(points3D, points2D):
    """Use DLT to calculate the 11 params of the calibration matrix. Calibration matrix transforms from 3D to 2D.
    """
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
    
def get_calibration_points(old_points, old_points3D, new_image):
    """Match feature points for the new image to the point cloud."""
    
    
def perform_dlt(old_points, old_points3D, new_image):
    """Perform dlt on the new image to get camera matrix.
    
    old_points: points and """
    
    
    return calculate_calibration_matrix(points3D, points2D)

#Filter matches for image 3
if np.sum(mask_23) > np.sum(mask_13):
    E, mask = cv.findEssentialMat(matches_23, matches_32)

    # Filter the matches further down
    filtered_3_forward = matches_23[(mask == 1)[:,0]]
    filtered_3_backward = matches_32[(mask == 1)[:,0]]
    P = P2
    largest_sum =  np.sum(mask_23)
    current_img = img_2
else:
    E, mask = cv.findEssentialMat(matches_13, matches_31)

    # Filter the matches further down
    filtered_3_forward = matches_13[(mask == 1)[:,0]]
    filtered_3_backward = matches_31[(mask == 1)[:,0]]
    P = P1
    largest_sum =  np.sum(mask_13)
    current_image = img_1

print(largest_sum)
print(len(filtered_3_forward))
print(len(filtered_3_backward))
if len(filtered_3_forward) < 200:
    #Create new keyframe and triangulate
    ret, R, t, _ = cv.recoverPose(E, filtered_3_forward, filtered_3_backward)
    # Create the minimization function
    min_func = funcs.make_min_function(P, filtered_3_forward, filtered_3_backward, K, R, t)

    # Now it gets fun
    # Miniminize the function to get an estimate of lambda (for the translation)
    # Use 'Powell' (might need more testing)
    res = scipy.optimize.minimize(min_func, 1, method='Powell')

    # Create the second camera
    P3 = funcs.camera_compose(K, R, t * res.x[0])
    
    # Go through the matches
    for point_a, point_b in zip(filtered_3_forward, filtered_3_backward):
        # Triangulate the points
        vec = funcs.triangulate(point_a, point_b, P, P3)

        # Get the pixel
        pixel = current_img[int(point_a[1])][int(point_a[0])]

        # Add it to the cloud
        cloud.append([vec[0], vec[1], vec[2], pixel[2], pixel[1], pixel[0]])
else:
    #Apply DLT
    pass