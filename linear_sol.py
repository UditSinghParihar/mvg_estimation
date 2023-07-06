"""
Linear Solution to finding 3D points and camera poses
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.linalg as la
import os


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

    use_opencv = False
    plot_checkerboard = True
    plot_cameras = True
    plot_cameras_point_cloud = True

    save_solution = True
    save_dir = 'results/'


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
    # l = 2 / (np.linalg.norm(H_[:, 0]) + np.linalg.norm(H_[:, 1]))
    l = 1 / np.linalg.norm(H_[:, 0])

    # Translation vector
    t = l * H_[:, 2]

    return R, t.reshape(3, 1)


def reprojection_error(H, R, t, K, checkerboard_px, checkerboard_3d):
    # Using homography
    checkerboard_3d_homo = np.concatenate([checkerboard_3d, np.ones((len(checkerboard_3d), 1))], axis=1) # 50x3

    projected_px = H @ checkerboard_3d_homo.T # 3x50
    projected_px = projected_px / projected_px[2, :] # 3x50

    error_homo = np.sqrt(np.sum((projected_px[:2, :] - checkerboard_px[:, 1:].T)**2, axis=0)) # 50x1

    # Using R and t
    checkerboard_3d_homo = np.concatenate([checkerboard_3d, np.zeros((len(checkerboard_3d), 1)), np.ones((len(checkerboard_3d), 1))], axis=1) # 50x4

    P = K @ np.concatenate([R, t], axis=1) # 3x4
    projected_px = P @ checkerboard_3d_homo.T
    projected_px = projected_px / projected_px[2, :]

    error_rt = np.sqrt(np.sum((projected_px[:2, :] - checkerboard_px[:, 1:].T)**2, axis=0))

    return (error_homo.mean(), error_rt.mean())


def plot_checkerboard(checkerboard_3d):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D Checkerboard points with origin as world coordinate frame")

    ax.scatter(checkerboard_3d[:, 0], checkerboard_3d[:, 1], np.zeros(checkerboard_3d.shape[0]), color='k')

    ax.quiver(0, 0, 0, 1, 0, 0, color='r')
    ax.quiver(0, 0, 0, 0, 1, 0, color='g')
    ax.quiver(0, 0, 0, 0, 0, 1, color='b')

    ax.set_xlim(-1, 10)
    ax.set_ylim(-1, 10)
    ax.set_zlim(-1, 1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def plot_cameras(checkerboard_3d, R0, t0, R1, t1, R2, t2, R3, t3, R4, t4):
    """
    Plotting cameras and checkerboard in world coordinates
    """

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_title("Camera poses from Homogrpahy Decomposition of 2D-3D checkerboard correspondences")

    ax.scatter(checkerboard_3d[:, 0], checkerboard_3d[:, 1], 0, c='r', marker='o', label='Checkerboard 3D points')

    # Plot the world frame at the origin
    ax.quiver(0, 0, 0, 1, 0, 0, color='r', label='X')
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', label='Y')
    ax.quiver(0, 0, 0, 0, 0, 1, color='b', label='Z')
    

    # Ri, ti are world wrt camera, but for plotting we need camera wrt world
    R0_inv = np.linalg.inv(R0)
    t0_inv = -R0_inv @ t0
    R1_inv = np.linalg.inv(R1)
    t1_inv = -R1_inv @ t1
    R2_inv = np.linalg.inv(R2)
    t2_inv = -R2_inv @ t2
    R3_inv = np.linalg.inv(R3)
    t3_inv = -R3_inv @ t3
    R4_inv = np.linalg.inv(R4)
    t4_inv = -R4_inv @ t4

    # Plot camera 0, 1, 2, 3, 4 with different colors
    ax.scatter(t0_inv[0], t0_inv[1], t0_inv[2], c='g', marker='o', s=100, label='Camera 0')
    ax.scatter(t1_inv[0], t1_inv[1], t1_inv[2], c='b', marker='o', s=100, label='Camera 1')
    ax.scatter(t2_inv[0], t2_inv[1], t2_inv[2], c='y', marker='o', s=100, label='Camera 2')
    ax.scatter(t3_inv[0], t3_inv[1], t3_inv[2], c='m', marker='o', s=100, label='Camera 3')
    ax.scatter(t4_inv[0], t4_inv[1], t4_inv[2], c='c', marker='o', s=100, label='Camera 4')

    # Drawing camera frames
    ax.quiver(t0_inv[0], t0_inv[1], t0_inv[2], R0_inv[0, 0], R0_inv[1, 0], R0_inv[2, 0], color='r')
    ax.quiver(t0_inv[0], t0_inv[1], t0_inv[2], R0_inv[0, 1], R0_inv[1, 1], R0_inv[2, 1], color='g')
    ax.quiver(t0_inv[0], t0_inv[1], t0_inv[2], R0_inv[0, 2], R0_inv[1, 2], R0_inv[2, 2], color='b')

    ax.quiver(t1_inv[0], t1_inv[1], t1_inv[2], R1_inv[0, 0], R1_inv[1, 0], R1_inv[2, 0], color='r')
    ax.quiver(t1_inv[0], t1_inv[1], t1_inv[2], R1_inv[0, 1], R1_inv[1, 1], R1_inv[2, 1], color='g')
    ax.quiver(t1_inv[0], t1_inv[1], t1_inv[2], R1_inv[0, 2], R1_inv[1, 2], R1_inv[2, 2], color='b')

    ax.quiver(t2_inv[0], t2_inv[1], t2_inv[2], R2_inv[0, 0], R2_inv[1, 0], R2_inv[2, 0], color='r')
    ax.quiver(t2_inv[0], t2_inv[1], t2_inv[2], R2_inv[0, 1], R2_inv[1, 1], R2_inv[2, 1], color='g')
    ax.quiver(t2_inv[0], t2_inv[1], t2_inv[2], R2_inv[0, 2], R2_inv[1, 2], R2_inv[2, 2], color='b')

    ax.quiver(t3_inv[0], t3_inv[1], t3_inv[2], R3_inv[0, 0], R3_inv[1, 0], R3_inv[2, 0], color='r')
    ax.quiver(t3_inv[0], t3_inv[1], t3_inv[2], R3_inv[0, 1], R3_inv[1, 1], R3_inv[2, 1], color='g')
    ax.quiver(t3_inv[0], t3_inv[1], t3_inv[2], R3_inv[0, 2], R3_inv[1, 2], R3_inv[2, 2], color='b')

    ax.quiver(t4_inv[0], t4_inv[1], t4_inv[2], R4_inv[0, 0], R4_inv[1, 0], R4_inv[2, 0], color='r')
    ax.quiver(t4_inv[0], t4_inv[1], t4_inv[2], R4_inv[0, 1], R4_inv[1, 1], R4_inv[2, 1], color='g')
    ax.quiver(t4_inv[0], t4_inv[1], t4_inv[2], R4_inv[0, 2], R4_inv[1, 2], R4_inv[2, 2], color='b')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.legend()

    plt.show()


def triangulate_point(px0, px1, px2, px3, px4, R0, t0, R1, t1, R2, t2, R3, t3, R4, t4, K):
    """
    Triangulate a point in 3D space using 5 views
    """
    
    # Projection matrices
    P0 = np.dot(K, np.hstack((R0, t0)))
    P1 = np.dot(K, np.hstack((R1, t1)))
    P2 = np.dot(K, np.hstack((R2, t2)))
    P3 = np.dot(K, np.hstack((R3, t3))) 
    P4 = np.dot(K, np.hstack((R4, t4)))

    # Triangulate
    A = np.zeros((2 * 5, 4))

    A[0, :] = px0[0] * P0[2, :] - P0[0, :]
    A[1, :] = px0[1] * P0[2, :] - P0[1, :]

    A[2, :] = px1[0] * P1[2, :] - P1[0, :]
    A[3, :] = px1[1] * P1[2, :] - P1[1, :]

    A[4, :] = px2[0] * P2[2, :] - P2[0, :]
    A[5, :] = px2[1] * P2[2, :] - P2[1, :]
    
    A[6, :] = px3[0] * P3[2, :] - P3[0, :]
    A[7, :] = px3[1] * P3[2, :] - P3[1, :]

    A[8, :] = px4[0] * P4[2, :] - P4[0, :]
    A[9, :] = px4[1] * P4[2, :] - P4[1, :]

    # Solve
    U, S, VT = np.linalg.svd(A)

    # Get the last column of V or last row of V^T
    X = VT[-1, :]

    X = X / X[3]

    return X[:3]


def plot_point_cloud_cameras(checkerboard_3d, R0, t0, R1, t1, R2, t2, R3, t3, R4, t4, point_cloud_3d):
    """
    Plotting cameras, checkerboard and point cloud in world coordinates
    """

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_title("Linear Solution for estimation of 5 cameras and 10 points")

    ax.scatter(checkerboard_3d[:, 0], checkerboard_3d[:, 1], 0, c='r', marker='o', label='Checkerboard 3D points')

    ax.scatter(point_cloud_3d[:, 0], point_cloud_3d[:, 1], point_cloud_3d[:, 2], c='b', marker='o', label='Point cloud 3D points')

    # Plot the world frame at the origin
    ax.quiver(0, 0, 0, 1, 0, 0, color='r', label='X')
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', label='Y')
    ax.quiver(0, 0, 0, 0, 0, 1, color='b', label='Z')
    
    # Ri, ti are world wrt camera, but for plotting we need camera wrt world
    R0_inv = np.linalg.inv(R0)
    t0_inv = -R0_inv @ t0
    R1_inv = np.linalg.inv(R1)
    t1_inv = -R1_inv @ t1
    R2_inv = np.linalg.inv(R2)
    t2_inv = -R2_inv @ t2
    R3_inv = np.linalg.inv(R3)
    t3_inv = -R3_inv @ t3
    R4_inv = np.linalg.inv(R4)
    t4_inv = -R4_inv @ t4

    # Plot camera 0, 1, 2, 3, 4 with different colors
    ax.scatter(t0_inv[0], t0_inv[1], t0_inv[2], c='g', marker='o', s=100, label='Camera 0')
    ax.scatter(t1_inv[0], t1_inv[1], t1_inv[2], c='b', marker='o', s=100, label='Camera 1')
    ax.scatter(t2_inv[0], t2_inv[1], t2_inv[2], c='y', marker='o', s=100, label='Camera 2')
    ax.scatter(t3_inv[0], t3_inv[1], t3_inv[2], c='m', marker='o', s=100, label='Camera 3')
    ax.scatter(t4_inv[0], t4_inv[1], t4_inv[2], c='c', marker='o', s=100, label='Camera 4')

    # Drawing camera frames
    ax.quiver(t0_inv[0], t0_inv[1], t0_inv[2], R0_inv[0, 0], R0_inv[1, 0], R0_inv[2, 0], color='r')
    ax.quiver(t0_inv[0], t0_inv[1], t0_inv[2], R0_inv[0, 1], R0_inv[1, 1], R0_inv[2, 1], color='g')
    ax.quiver(t0_inv[0], t0_inv[1], t0_inv[2], R0_inv[0, 2], R0_inv[1, 2], R0_inv[2, 2], color='b')

    ax.quiver(t1_inv[0], t1_inv[1], t1_inv[2], R1_inv[0, 0], R1_inv[1, 0], R1_inv[2, 0], color='r')
    ax.quiver(t1_inv[0], t1_inv[1], t1_inv[2], R1_inv[0, 1], R1_inv[1, 1], R1_inv[2, 1], color='g')
    ax.quiver(t1_inv[0], t1_inv[1], t1_inv[2], R1_inv[0, 2], R1_inv[1, 2], R1_inv[2, 2], color='b')

    ax.quiver(t2_inv[0], t2_inv[1], t2_inv[2], R2_inv[0, 0], R2_inv[1, 0], R2_inv[2, 0], color='r')
    ax.quiver(t2_inv[0], t2_inv[1], t2_inv[2], R2_inv[0, 1], R2_inv[1, 1], R2_inv[2, 1], color='g')
    ax.quiver(t2_inv[0], t2_inv[1], t2_inv[2], R2_inv[0, 2], R2_inv[1, 2], R2_inv[2, 2], color='b')

    ax.quiver(t3_inv[0], t3_inv[1], t3_inv[2], R3_inv[0, 0], R3_inv[1, 0], R3_inv[2, 0], color='r')
    ax.quiver(t3_inv[0], t3_inv[1], t3_inv[2], R3_inv[0, 1], R3_inv[1, 1], R3_inv[2, 1], color='g')
    ax.quiver(t3_inv[0], t3_inv[1], t3_inv[2], R3_inv[0, 2], R3_inv[1, 2], R3_inv[2, 2], color='b')

    ax.quiver(t4_inv[0], t4_inv[1], t4_inv[2], R4_inv[0, 0], R4_inv[1, 0], R4_inv[2, 0], color='r')
    ax.quiver(t4_inv[0], t4_inv[1], t4_inv[2], R4_inv[0, 1], R4_inv[1, 1], R4_inv[2, 1], color='g')
    ax.quiver(t4_inv[0], t4_inv[1], t4_inv[2], R4_inv[0, 2], R4_inv[1, 2], R4_inv[2, 2], color='b')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.legend()

    plt.show()


def project_points(point_cloud, R, t, K):
    ones = np.ones((point_cloud.shape[0], 1))
    point_cloud = np.concatenate((point_cloud, ones), axis=1)

    P = np.matmul(K, np.append(R, t, axis=1))
    pixel_points = np.matmul(P, point_cloud.T).T

    pixel_points[:, 0] /= pixel_points[:, 2]
    pixel_points[:, 1] /= pixel_points[:, 2]

    return pixel_points[:, :2]


def reprojection_error_point_cloud(point_cloud, pixel_points, R, t, K):
    """
    Reprojection error of point cloud for single camera
    point_cloud: (N, 3)
    pixel_points: (N, 2)
    """

    # Project 3D points to 2D points
    pixel_points_pred = project_points(point_cloud, R, t, K)

    # Compute reprojection error
    error = np.sqrt(np.sum((pixel_points - pixel_points_pred) ** 2, axis=1))

    return error.mean()
    

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

    # Plot checkerboard points with coordinate frame at origin
    if cfg.plot_checkerboard:
        plot_checkerboard(checkerboard_3d)

    # Estimating homography and decomposing it into R and t
    if cfg.use_opencv:
        H0_cv2, _ = cv2.findHomography(checkerboard_px0[:, 1:], checkerboard_3d)   
        _, R0_cv2, t0_cv2, _ = cv2.decomposeHomographyMat(H0_cv2, K)
        error0_cv2 = reprojection_error(H0_cv2, R0_cv2[1], t0_cv2[1], K, checkerboard_px0, checkerboard_3d)

        print('OpenCV homography error: {} | R and t error: {}'.format(error0_cv2[0], error0_cv2[1]))

    

    H0 = get_homography(checkerboard_px0[:, 1:], checkerboard_3d)
    R0, t0 =  get_extrinsics(H0, K)
    error0 = reprojection_error(H0, R0, t0, K, checkerboard_px0, checkerboard_3d)

    H1 = get_homography(checkerboard_px1[:, 1:], checkerboard_3d)
    R1, t1 =  get_extrinsics(H1, K)
    error1 = reprojection_error(H1, R1, t1, K, checkerboard_px1, checkerboard_3d)

    H2 = get_homography(checkerboard_px2[:, 1:], checkerboard_3d)
    R2, t2 =  get_extrinsics(H2, K)
    error2 = reprojection_error(H2, R2, t2, K, checkerboard_px2, checkerboard_3d)

    H3 = get_homography(checkerboard_px3[:, 1:], checkerboard_3d)
    R3, t3 =  get_extrinsics(H3, K)
    error3 = reprojection_error(H3, R3, t3, K, checkerboard_px3, checkerboard_3d)

    H4 = get_homography(checkerboard_px4[:, 1:], checkerboard_3d)
    R4, t4 =  get_extrinsics(H4, K)
    error4 = reprojection_error(H4, R4, t4, K, checkerboard_px4, checkerboard_3d)

    print('Reprojection error using homography: ', error0[0], error1[0], error2[0], error3[0], error4[0])
    print('Reprojection error using R and t: ', error0[1], error1[1], error2[1], error3[1], error4[1])

    if cfg.plot_cameras:
        plot_cameras(checkerboard_3d, R0, t0, R1, t1, R2, t2, R3, t3, R4, t4)

    # Trianglate one 3D point from five images
    point_cloud_3d = []

    for i in range(len(point_cloud_px0)):
        point_3d = triangulate_point(point_cloud_px0[i][1:], point_cloud_px1[i][1:], point_cloud_px2[i][1:], point_cloud_px3[i][1:], point_cloud_px4[i][1:], R0, t0, R1, t1, R2, t2, R3, t3, R4, t4, K)
        point_cloud_3d.append(point_3d)

    point_cloud_3d = np.array(point_cloud_3d) # 10x3
    
    if cfg.plot_cameras_point_cloud:
        plot_point_cloud_cameras(checkerboard_3d, R0, t0, R1, t1, R2, t2, R3, t3, R4, t4, point_cloud_3d)

    # Reprojection error of point_cloud_3d for all cameras
    error_point_cloud_camera0 = reprojection_error_point_cloud(point_cloud_3d, point_cloud_px0[:, 1:], R0, t0, K)
    error_point_cloud_camera1 = reprojection_error_point_cloud(point_cloud_3d, point_cloud_px1[:, 1:], R1, t1, K)
    error_point_cloud_camera2 = reprojection_error_point_cloud(point_cloud_3d, point_cloud_px2[:, 1:], R2, t2, K)
    error_point_cloud_camera3 = reprojection_error_point_cloud(point_cloud_3d, point_cloud_px3[:, 1:], R3, t3, K)
    error_point_cloud_camera4 = reprojection_error_point_cloud(point_cloud_3d, point_cloud_px4[:, 1:], R4, t4, K)

    print('Reprojection error of point_cloud:', error_point_cloud_camera0, error_point_cloud_camera1, error_point_cloud_camera2, error_point_cloud_camera3, error_point_cloud_camera4)

    # Mean reprojection error of point_cloud_3d
    mean_error_point_cloud = np.mean([error_point_cloud_camera0, error_point_cloud_camera1, error_point_cloud_camera2, error_point_cloud_camera3, error_point_cloud_camera4])
    print('Mean reprojection error of point_cloud:', mean_error_point_cloud)
   
    # Saving point cloud and camera poses
    if cfg.save_solution:
        if cfg.save_dir is not None:
            os.makedirs(cfg.save_dir, exist_ok=True)

        np.savetxt(f'{cfg.save_dir}/point_cloud.txt', point_cloud_3d, delimiter=' ')

        # Converting rotation matrix to axis angle representation
        R0_aa = cv2.Rodrigues(R0)[0]
        R1_aa = cv2.Rodrigues(R1)[0]
        R2_aa = cv2.Rodrigues(R2)[0]
        R3_aa = cv2.Rodrigues(R3)[0]
        R4_aa = cv2.Rodrigues(R4)[0]

        # Saving camera poses in separate files
        np.savetxt(f'{cfg.save_dir}/camera0.txt', np.concatenate((R0_aa, t0), axis=1), delimiter=' ')
        np.savetxt(f'{cfg.save_dir}/camera1.txt', np.concatenate((R1_aa, t1), axis=1), delimiter=' ')
        np.savetxt(f'{cfg.save_dir}/camera2.txt', np.concatenate((R2_aa, t2), axis=1), delimiter=' ')
        np.savetxt(f'{cfg.save_dir}/camera3.txt', np.concatenate((R3_aa, t3), axis=1), delimiter=' ')
        np.savetxt(f'{cfg.save_dir}/camera4.txt', np.concatenate((R4_aa, t4), axis=1), delimiter=' ')

        # Saving checkerboard 3D points
        np.savetxt(f'{cfg.save_dir}/checkerboard.txt', checkerboard_3d, delimiter=' ')

        print('Saved solution to', cfg.save_dir)