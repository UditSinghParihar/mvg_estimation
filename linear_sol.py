"""
Linear Solution to finding 3D points
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.linalg as la


class CFG:
    """
    Config class for fixed parameters
    """
    path0 = 'problem1/image-cam-0-image.txt'
    path1 = 'problem1/image-cam-1-image.txt'
    path2 = 'problem1/image-cam-2-image.txt'
    path3 = 'problem1/image-cam-3-image.txt'
    path4 = 'problem1/image-cam-4-image.txt'

    plot_image = False

    checkerboard_rows = 10
    checkerboard_cols = 5

    fx = 200
    fy = 200

    use_opencv = True


def read_image(filename):
    """
    Read points: id,x,y
    """
    points = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                points.append([float(x) for x in line.split(',')])
    
    # 0-49 checkerboard points, 50-59 point cloud points
    checkerboard = np.array(points[:50])
    point_cloud = np.array(points[50:])

    # Plot checkerboard points, point cloud points
    if cfg.plot_image:
        plt.figure()
        plt.scatter(checkerboard[:, 1], checkerboard[:, 2], c='r')
        plt.scatter(point_cloud[:, 1], point_cloud[:, 2], c='b')
        plt.show()

    return checkerboard, point_cloud


def get_homography(checkerboard_px, checkerboard_3d):
    """
    Get homography matrix H
    """
    # 2D points 
    x = checkerboard_px[:, 0] # 50x1
    y = checkerboard_px[:, 1] 

    # 3D points
    X = checkerboard_3d[:, 0]
    Y = checkerboard_3d[:, 1]
    Z = np.ones_like(X)
    P = np.stack([X, Y, Z], axis=1) # 50x3

    # Preparing M matrix (2N x 9)
    M = np.zeros((2*len(x), 9)) # 100x9

    for i in range(len(x)):
        M[2*i, 0:3] = -P[i, :]
        M[2*i, 6:9] = x[i] * P[i, :]
        M[2*i+1, 3:6] = -P[i, :]
        M[2*i+1, 6:9] = y[i] * P[i, :]

    # SVD of M
    U, S, VT = la.svd(M, full_matrices=False)

    # Last column of V or last row of VT
    H = VT[-1, :].reshape(3, 3) # 3x3

    # Ensuring checkboard is in front of camera
    if H[2, 2] < 0:
        H = -H

    return H


def get_extrinsics(H, K):
    """
    Get extrinsics from homography
    """
    H_ = np.linalg.inv(K) @ H

    m1 = H_[:, 0] / np.linalg.norm(H_[:, 0])
    m2 = H_[:, 1] / np.linalg.norm(H_[:, 1])
    m3 = np.cross(m1, m2)

    M = np.stack([m1, m2, m3], axis=1) # 3x3

    U, S, VT = la.svd(M, full_matrices=False)

    # R = U @  diag(1, 1, det(U @ VT)) @ VT
    R = U @  np.diag([1, 1, np.linalg.det(U @ VT)]) @ VT

    # Scaling factor
    l = 2 / (np.linalg.norm(H_[:, 0]) + np.linalg.norm(H_[:, 1]))

    # Translation vector
    t = l * H_[:, 2]

    return R, t.reshape(3, 1)


if __name__ == '__main__':
    cfg = CFG()

    checkerboard_px0, point_cloud_px0 = read_image(cfg.path0) # 50x3, 10x3
    checkerboard_px1, point_cloud_px1 = read_image(cfg.path1)
    checkerboard_px2, point_cloud_px2 = read_image(cfg.path2)
    checkerboard_px3, point_cloud_px3 = read_image(cfg.path3)
    checkerboard_px4, point_cloud_px4 = read_image(cfg.path4)

    # Camera intrinsics
    K = np.array([[cfg.fx, 0, 0], [0, cfg.fy, 0], [0, 0, 1]])

    # 3D checkerboard points
    checkerboard_3d = []

    for i in range(cfg.checkerboard_rows):
        for j in range(cfg.checkerboard_cols):
            checkerboard_3d.append([i, j])
    
    checkerboard_3d = np.array(checkerboard_3d) # 50x2

    # Estimating homography and decomposing it into R and t
    if cfg.use_opencv:
        H0_cv2, _ = cv2.findHomography(checkerboard_px0[:, 1:], checkerboard_3d)   
        _, R0_cv2, t0_cv2, _ = cv2.decomposeHomographyMat(H0_cv2, K)
    

    H0 = get_homography(checkerboard_px0[:, 1:], checkerboard_3d)
    R0, t0 =  get_extrinsics(H0, K)

    # Reprojection error for checkerboard points using Homography
    checkerboard_3d_homo = np.concatenate([checkerboard_3d, np.ones((len(checkerboard_3d), 1))], axis=1) # 50x3
    projected_px = H0 @ checkerboard_3d_homo.T # 3x50
    projected_px = projected_px / projected_px[2, :] # 3x50

    error = np.sqrt(np.sum((projected_px[:2, :] - checkerboard_px0[:, 1:].T)**2, axis=0)) # 50x1
    print("Error", error.mean())

    # # Reprojection error for checkerboard using r and t
    checkerboard_3d_homo = np.concatenate([checkerboard_3d, np.zeros((len(checkerboard_3d), 1)), np.ones((len(checkerboard_3d), 1))], axis=1) # 50x4
    
    P = np.concatenate([R0, t0], axis=1) # 3x4
    projected_px = P @ checkerboard_3d_homo.T





    
    