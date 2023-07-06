"""
Bundle ajustment to refine the camera poses and 3D points from linear_sol.py
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.linalg as la
import os
from scipy.sparse import lil_matrix
import time
from scipy.optimize import least_squares


class CFG:
    """
    Config class for fixed parameters
    """

    input_dir = "results"

    path0 = 'problem1/image-cam-0-image.txt'
    path1 = 'problem1/image-cam-1-image.txt'
    path2 = 'problem1/image-cam-2-image.txt'
    path3 = 'problem1/image-cam-3-image.txt'
    path4 = 'problem1/image-cam-4-image.txt'

    focal_length = 200

    plot_initial_residuals = False
    plot_final_residuals = False
    plot_cameras_point_cloud = True

    print_optimized_point_cloud = False

def read_poses_point_cloud():
    """
    Read the camera rotation in axis-angle and translation from the file
    """
    R_t0 = np.loadtxt(os.path.join(cfg.input_dir, "camera0.txt"))
    R_t1 = np.loadtxt(os.path.join(cfg.input_dir, "camera1.txt"))
    R_t2 = np.loadtxt(os.path.join(cfg.input_dir, "camera2.txt"))
    R_t3 = np.loadtxt(os.path.join(cfg.input_dir, "camera3.txt"))
    R_t4 = np.loadtxt(os.path.join(cfg.input_dir, "camera4.txt"))

    R0_aa = R_t0[:, 0]
    t0 = R_t0[:, 1]
    R1_aa = R_t1[:, 0]
    t1 = R_t1[:, 1]
    R2_aa = R_t2[:, 0]
    t2 = R_t2[:, 1]
    R3_aa = R_t3[:, 0]
    t3 = R_t3[:, 1]
    R4_aa = R_t4[:, 0]
    t4 = R_t4[:, 1]

    return R0_aa, t0, R1_aa, t1, R2_aa, t2, R3_aa, t3, R4_aa, t4


def read_point_cloud():
    """
    Read the 3D points from the file
    """
    points = np.loadtxt(os.path.join(cfg.input_dir, "point_cloud.txt"))
    return points


def read_image_point_cloud(filename):
    """
    Read points: id,x,y
    x, y are image coordinates
    """
    points = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                points.append([float(x) for x in line.split(',')])
    
    # 0-49 checkerboard points, 50-59 point cloud points
    point_cloud = np.array(points[50:])
    point_cloud = point_cloud[:, 1:]

    return point_cloud


def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.
    
    Rodrigues' rotation formula is used.
    points : 50, 3
    rot_vecs : 50, 3
    """

    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    final_points = cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v # 50, 3

    return final_points


def project(points, r_t_vecs):
    """
    Convert 3-D points to 2-D by projecting onto images.
    points : 50, 3
    r_t_vecs : 5, 6
    points_proj : 50, 2
    """

    rot_vecs = r_t_vecs[:, :3]
    t = r_t_vecs[:, 3:]

    points_proj = rotate(points, rot_vecs) + t
    points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    
    f = cfg.focal_length
    points_proj = points_proj * f

    return points_proj


def residuals(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """
    Compute residuals.
    params contains camera parameters and 3-D coordinates.
    """

    r_t_vecs = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], r_t_vecs[camera_indices])
    return (points_proj - points_2d).ravel()


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size) # (50,)
    
    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

    return A

def rotation_matrix(r_vec):
    """
    Takes a rotation vector and returns the corresponding rotation matrix (Rodrigues' formula)
    rvec : (3,)
    R : (3, 3)
    """
    theta = np.linalg.norm(r_vec)
    w = r_vec / theta
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K

    return R


def plot_point_cloud_cameras(checkerboard_3d, R0, t0, R1, t1, R2, t2, R3, t3, R4, t4, point_cloud_3d):
    """
    Plotting cameras, checkerboard and point cloud in world coordinates
    """

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_title("Bundle Adjustment Solution for 5 cameras and 10 points")

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


if __name__ == "__main__":
    cfg = CFG()

    R0_aa, t0, R1_aa, t1, R2_aa, t2, R3_aa, t3, R4_aa, t4 = read_poses_point_cloud()
    points_3d = read_point_cloud()

    r_t_vecs = [np.concatenate((R0_aa, t0)), np.concatenate((R1_aa, t1)), np.concatenate((R2_aa, t2)), np.concatenate((R3_aa, t3)), np.concatenate((R4_aa, t4))]
    r_t_vecs = np.array(r_t_vecs)

    """
    camera_indices shape : (50, ) | [0, ..., 0, 1, ..., 1, 2, ..., 2, 3, ..., 3, 4, ..., 4] 
    point_indices shape: (50, ) | [0, ..., 9, 0, ..., 9, 0, ..., 9, 0, ..., 9, 0, ..., 9]
    """

    camera_indices = np.repeat(np.arange(5), 10)
    point_indices = np.tile(np.arange(10), 5)

    # Read image points
    points_2d = []
    points_2d.append(read_image_point_cloud(cfg.path0))
    points_2d.append(read_image_point_cloud(cfg.path1))
    points_2d.append(read_image_point_cloud(cfg.path2))
    points_2d.append(read_image_point_cloud(cfg.path3))
    points_2d.append(read_image_point_cloud(cfg.path4))
    points_2d = np.array(points_2d)
    points_2d = points_2d.reshape(-1, 2)
  
    """
    r_t_vecs: (5, 6)
    points_3d: (10, 3)
    points_2d: (50, 2)
    camera_indices: (50, )
    point_indices: (50, )
    """

    n_cameras = r_t_vecs.shape[0]
    n_points = points_3d.shape[0]

    # Initial parameters
    x0 = np.hstack((r_t_vecs.ravel(), points_3d.ravel()))

    f0 = residuals(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)

    print("Initial RMS residual:", np.sqrt(np.mean(f0 ** 2)))

    # Plot residuals
    if cfg.plot_initial_residuals:
        plt.figure()
        plt.plot(f0)
        plt.title('Initial residuals')
        plt.xlabel('Residual index')
        plt.ylabel('Residual value')
        # Y axis from -5 to 5
        plt.ylim([-5, 5])
        plt.show()

    # Sparse Jacobian matrix
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

    time0 = time.time()
    # result = least_squares(residuals, x0, verbose=2, x_scale='jac', ftol=1e-4, method='lm', loss='linear', args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
    result = least_squares(residuals, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf', args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
    time1 = time.time()

    # print("Optimization took {0:.0f} seconds".format(time1 - time0))

    if cfg.plot_final_residuals:
        plt.figure()
        plt.plot(result.fun)
        plt.title('Final residuals')
        plt.xlabel('Residual index')
        plt.ylabel('Residual value')
        # Y axis from -5 to 5
        plt.ylim([-5, 5])
        plt.show()

    # Extract optimized parameters
    r_t_vecs_optimized = result.x[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d_optimized = result.x[n_cameras * 6:].reshape((n_points, 3))

    # Final RMS residual
    f1 = residuals(result.x, n_cameras, n_points, camera_indices, point_indices, points_2d)
    print("Final RMS residual:", np.sqrt(np.mean(f1 ** 2)))

    # Rotation matrix and translation vector    
    R0_optimized = rotation_matrix(r_t_vecs_optimized[0, :3])
    t0_optimized = r_t_vecs_optimized[0, 3:].reshape(3, 1)
    R1_optimized = rotation_matrix(r_t_vecs_optimized[1, :3])
    t1_optimized = r_t_vecs_optimized[1, 3:].reshape(3, 1)
    R2_optimized = rotation_matrix(r_t_vecs_optimized[2, :3])
    t2_optimized = r_t_vecs_optimized[2, 3:].reshape(3, 1)
    R3_optimized = rotation_matrix(r_t_vecs_optimized[3, :3])
    t3_optimized = r_t_vecs_optimized[3, 3:].reshape(3, 1)
    R4_optimized = rotation_matrix(r_t_vecs_optimized[4, :3])
    t4_optimized = r_t_vecs_optimized[4, 3:].reshape(3, 1)

    # Reading checkerboard points
    checkerboard_3d = np.loadtxt(os.path.join(cfg.input_dir, 'checkerboard.txt')) # (50, 2)

    if cfg.plot_cameras_point_cloud:
        plot_point_cloud_cameras(checkerboard_3d, R0_optimized, t0_optimized, R1_optimized, t1_optimized, R2_optimized, t2_optimized, R3_optimized, t3_optimized, R4_optimized, t4_optimized, points_3d_optimized)
        
    if cfg.print_optimized_point_cloud:
        print("Final optimized point cloud with respect to world origin defined at checkerboard corner:\n", points_3d_optimized)
