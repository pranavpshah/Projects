import numpy as np
import cv2
import pdb


EPS = 1e-8


def backproject_corners(K, width, height, depth, Rt):
    """
    Backproject 4 corner points in image plane to the imaginary depth plane

    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        width -- width of the image
        heigh -- height of the image
        depth -- depth value of the imaginary plane
    Output:
        points -- 2 x 2 x 3 array of 3D coordinates of backprojected points
    """

    points = np.array((
        (0, 0, 1),
        (width, 0, 1),
        (0, height, 1),
        (width, height, 1),
    ), dtype=np.float32).reshape(2, 2, 3)
    

    """ YOUR CODE HERE
    """
    R = Rt[0:3,0:3]
    t = Rt[:,3].reshape((3,1))
    K_inv = np.linalg.inv(K)

    for i in range(points.shape[0]):
        temp = points[i].T
        points[i] = (R.T @ (((K_inv @ temp)*depth) - t)).T 

    # pdb.set_trace()

    """ END YOUR CODE
    """
    return points

def project_points(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- points_height x points_width x 3 array of 3D points
    Output:
        projections -- points_height x points_width x 2 array of 2D projections
    """
    """ YOUR CODE HERE
    """
    height = points.shape[0]
    width = points.shape[1]
    R = Rt[0:3,0:3]
    t = Rt[:,3].reshape((3,1))
    projections = np.zeros((height, width, 2))
    # pdb.set_trace()
    for i in range(height):
        for j in range(width):
            temp1 = points[i,j].reshape((3,1))
            temp2 = K @ (R @ temp1 + t)
            projections[i,j] = temp2[:2,0]/temp2[2,0]

    """ END YOUR CODE
    """
    return projections

def warp_neighbor_to_ref(backproject_fn, project_fn, depth, neighbor_rgb, K_ref, Rt_ref, K_neighbor, Rt_neighbor):
    """ 
    Warp the neighbor view into the reference view 
    via the homography induced by the imaginary depth plane at the specified depth

    Make use of the functions you've implemented in the same file (which are passed in as arguments):
    - backproject_corners
    - project_points

    Also make use of the cv2 functions:
    - cv2.findHomography
    - cv2.warpPerspective

    Input:
        backproject_fn -- backproject_corners function
        project_fn -- project_points function
        depth -- scalar value of the depth at the imaginary depth plane
        neighbor_rgb -- height x width x 3 array of neighbor rgb image
        K_ref -- 3 x 3 camera intrinsics calibration matrix of reference view
        Rt_ref -- 3 x 4 camera extrinsics calibration matrix of reference view
        K_neighbor -- 3 x 3 camera intrinsics calibration matrix of neighbor view
        Rt_neighbor -- 3 x 4 camera extrinsics calibration matrix of neighbor view
    Output:
        warped_neighbor -- height x width x 3 array of the warped neighbor RGB image
    """

    height, width = neighbor_rgb.shape[:2]

    """ YOUR CODE HERE
    """
    p0 = np.array((
        (0, 0, 1),
        (width, 0, 1),
        (0, height, 1),
        (width, height, 1),
    ), dtype=np.float32).reshape(2, 2, 3)

    p0_temp = np.array((
        (0, 0),
        (width, 0),
        (0, height),
        (width, height),
    ), dtype=np.float32).reshape(2, 2, 2)
    

    p1 = backproject_fn(K_ref, neighbor_rgb.shape[1], neighbor_rgb.shape[0], depth, Rt_ref)
    p2 = project_fn(K_neighbor, Rt_neighbor, p1)
    matrix = cv2.findHomography(p0_temp.reshape((-1,2)), p2.reshape((-1,2)), cv2.RANSAC)[0]

    warped_neighbor = cv2.warpPerspective(neighbor_rgb, np.linalg.inv(matrix), (width, height))
    # pdb.set_trace()
    
    """ END YOUR CODE
    """
    return warped_neighbor


def zncc_kernel_2D(src, dst):
    """ 
    Compute the cost map between src and dst patchified images via the ZNCC metric
    
    IMPORTANT NOTES:
    - Treat each RGB channel separately but sum the 3 different zncc scores at each pixel

    - When normalizing by the standard deviation, add the provided small epsilon value, 
    EPS which is included in this file, to both sigma_src and sigma_dst to avoid divide-by-zero issues

    Input:
        src -- height x width x K**2 x 3
        dst -- height x width x K**2 x 3
    Output:
        zncc -- height x width array of zncc metric computed at each pixel
    """
    assert src.ndim == 4 and dst.ndim == 4
    assert src.shape[:] == dst.shape[:]

    """ YOUR CODE HERE
    """
    height = src.shape[0]
    width = dst.shape[1]
    zncc = np.zeros((height, width))

    # pdb.set_trace()

    src_mean = np.mean(src, axis = 2).reshape((height, width, 1, 3))
    dst_mean = np.mean(dst, axis = 2).reshape((height, width, 1, 3))
    src_std = np.std(src, axis = 2).reshape((height, width, 1, 3))
    dst_std = np.std(dst, axis = 2).reshape((height, width, 1, 3))

    zncc = np.sum(np.sum((src - src_mean)*(dst - dst_mean)/((src_std*dst_std) + EPS), axis = 2), axis = 2)

    """ END YOUR CODE
    """

    return zncc  # height x width


def backproject(dep_map, K):
    """ 
    Backproject image points to 3D coordinates wrt the camera frame according to the depth map

    Input:
        K -- camera intrinsics calibration matrix
        dep_map -- height x width array of depth values
    Output:
        points -- height x width x 3 array of 3D coordinates of backprojected points
    """
    _u, _v = np.meshgrid(np.arange(dep_map.shape[1]), np.arange(dep_map.shape[0]))

    """ YOUR CODE HERE
    """
    uv = np.stack((_u, _v, np.ones(_u.shape)), axis = 2)

    pts_c_frame = np.tensordot(np.linalg.inv(K), uv, axes = [1,2])
    pts_c_frame = np.transpose(pts_c_frame, (1,2,0))

    depth_map = dep_map.reshape((dep_map.shape[0], dep_map.shape[1], 1))
    xyz_cam = pts_c_frame*depth_map
    
    """ END YOUR CODE
    """
    return xyz_cam

