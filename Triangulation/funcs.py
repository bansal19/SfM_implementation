import cv2 as cv
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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

def triangulate(pts, cameras):
    # The number of rows
    rows = len(pts) * 2

    # Create the matrix and vector
    A = np.zeros((rows, 3))
    b = np.zeros(rows)

    # Loop throught ever point and camera
    for i, (pt, camera) in enumerate(zip(pts, cameras)):
        # The index
        idx = i * 2

        # Append A
        A[idx] = (pt[0] * camera[2,:3]) - camera[0,:3]
        A[idx+1] = (pt[1] * camera[2,:3]) - camera[1,:3]

        # Build b
        b[idx] = camera[0,3] - (pt[0] * camera[2,3])
        b[idx+1] = camera[1,3] - (pt[1] * camera[2,3])

    # Return the closest point
    return np.matmul(np.linalg.pinv(A), b)

def plot_cloud(cloud):
    # Render the cloud
    fig = go.Figure(data=[go.Scatter3d(
        x=cloud[:,0], 
        y=cloud[:,1], 
        z=cloud[:,2], 
        mode='markers',
        marker=dict(
            size=3,
            color=cloud[:,3:][...,::-1],
            opacity=1
        )
    )])
    fig.show()


