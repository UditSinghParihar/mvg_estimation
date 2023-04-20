# Problem Description

There is a camera, a checkerboard, and a point cloud in your environment.

The camera images the checkerboard and point cloud from 5 perspectives.  

Camera = focal length 200, principle-point = (0,0) and no distortion

Checkerboard = 1 m x 1 m in dimensions. Has 50 corners on it.

Point cloud = is positioned in the environment.

The zip file contains each image generated by the camera.

3D point index, X pixel coordinate, Y pixel coordinate = is the format of each image text file.

0-49 point indices correspond to the checkerboard.

All images have gaussian noise.  

You need to find the best possible 3D estimates of the points in the point cloud.


# Code
`python linear_sol.py`  
`python bundle_adjustment.py`