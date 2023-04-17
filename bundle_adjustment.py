"""
Bundle ajustment to refine the camera poses and 3D points from linear_sol.py
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

    input_dir = "results"


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



if __name__ == "__main__":
    cfg = CFG()

    R0_aa, t0, R1_aa, t1, R2_aa, t2, R3_aa, t3, R4_aa, t4 = read_poses_point_cloud()
    point_cloud = read_point_cloud()

    