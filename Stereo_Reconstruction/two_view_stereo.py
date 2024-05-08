import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import imageio
from tqdm import tqdm
from transforms3d.euler import mat2euler, euler2mat
import pyrender
import trimesh
import cv2
import open3d as o3d
import pdb


from dataloader import load_middlebury_data
from utils import viz_camera_poses

EPS = 1e-8


def homo_corners(h, w, H):
    corners_bef = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    corners_aft = cv2.perspectiveTransform(corners_bef, H).squeeze(1)
    u_min, v_min = corners_aft.min(axis=0)
    u_max, v_max = corners_aft.max(axis=0)
    return u_min, u_max, v_min, v_max


def rectify_2view(rgb_i, rgb_j, R_irect, R_jrect, K_i, K_j, u_padding=20, v_padding=20):
    """Given the rectify rotation, compute the rectified view and corrected projection matrix

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    R_irect,R_jrect : [3,3]
        p_rect_left = R_irect @ p_i
        p_rect_right = R_jrect @ p_j
    K_i,K_j : [3,3]
        original camera matrix
    u_padding,v_padding : int, optional
        padding the border to remove the blank space, by default 20

    Returns
    -------
    [H,W,3],[H,W,3],[3,3],[3,3]
        the rectified images
        the corrected camera projection matrix. WE HELP YOU TO COMPUTE K, YOU DON'T NEED TO CHANGE THIS
    """
    # reference: https://stackoverflow.com/questions/18122444/opencv-warpperspective-how-to-know-destination-image-size
    assert rgb_i.shape == rgb_j.shape, "This hw assumes the input images are in same size"
    h, w = rgb_i.shape[:2]

    ui_min, ui_max, vi_min, vi_max = homo_corners(h, w, K_i @ R_irect @ np.linalg.inv(K_i))
    uj_min, uj_max, vj_min, vj_max = homo_corners(h, w, K_j @ R_jrect @ np.linalg.inv(K_j))

    # The distortion on u direction (the world vertical direction) is minor, ignore this
    w_max = int(np.floor(max(ui_max, uj_max))) - u_padding * 2
    h_max = int(np.floor(min(vi_max - vi_min, vj_max - vj_min))) - v_padding * 2

    assert K_i[0, 2] == K_j[0, 2], "This hw assumes original K has same cx"
    K_i_corr, K_j_corr = K_i.copy(), K_j.copy()
    K_i_corr[0, 2] -= u_padding
    K_i_corr[1, 2] -= vi_min + v_padding
    K_j_corr[0, 2] -= u_padding
    K_j_corr[1, 2] -= vj_min + v_padding

    """Student Code Starts"""

    H_i = K_i_corr @ R_irect @ np.linalg.inv(K_i)
    H_j = K_j_corr @ R_jrect @ np.linalg.inv(K_j)

    rgb_i_rect = cv2.warpPerspective(rgb_i, H_i, (w_max, h_max))
    rgb_j_rect = cv2.warpPerspective(rgb_j, H_j, (w_max, h_max))
    
    """Student Code Ends"""

    return rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr


def compute_right2left_transformation(R_wi, T_wi, R_wj, T_wj):
    """Compute the transformation that transform the coordinate from j coordinate to i

    Parameters
    ----------
    R_wi, R_wj : [3,3]
    T_wi, T_wj : [3,1]
        p_i = R_wi @ p_w + T_wi
        p_j = R_wj @ p_w + T_wj
    Returns
    -------
    [3,3], [3,1], float
        p_i = R_ji @ p_j + T_ji, B is the baseline
    """

    """Student Code Starts"""
    R_ji =  R_wi @ (R_wj.T)
    temp_T_w = ((R_wj.T) @ T_wj) - ((R_wi.T) @ T_wi)
    T_ji = - R_wi @ temp_T_w
    B = np.linalg.norm(T_ji)
    """Student Code Ends"""

    return R_ji, T_ji, B


def compute_rectification_R(T_ji):
    """Compute the rectification Rotation

    Parameters
    ----------
    T_ji : [3,1]

    Returns
    -------
    [3,3]
        p_rect = R_irect @ p_i
    """
    # check the direction of epipole, should point to the positive direction of y axis
    e_i = T_ji.squeeze(-1) / (T_ji.squeeze(-1)[1] + EPS)
    e_i = e_i.reshape((-1,1))
    """Student Code Starts"""
    r2 = T_ji / np.linalg.norm(T_ji)
    r1 = -(1/np.linalg.norm(T_ji[:2,0]))*np.array([[-T_ji[1,0]], [T_ji[0,0]], [0]])
    r3 = np.cross(np.squeeze(r1), np.squeeze(r2))
    r3 = r3.reshape((-1,1))
    R_irect = np.vstack((r1.T, r2.T, r3.T))
    # pdb.set_trace()
    # u1, s1, v1t = np.linalg.svd(R_temp)
    # R_irect = v1t.T @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, np.linalg.det(v1t.T @ u1.T)]]) @ u1.T 

    # pdb.set_trace()
    """Student Code Ends"""

    return R_irect


def ssd_kernel(src, dst):
    """Compute SSD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    ssd = np.zeros((src.shape[0], dst.shape[0]))
    """Student Code Starts"""
    for i in range(src.shape[0]):
        temp1 = src[i]
        temp2 = (dst - temp1)**2
        # pdb.set_trace()
        ssd[i] = np.sum(np.sum(temp2, axis = 2), axis = 1)

    # pdb.set_trace()
    
    """Student Code Ends"""

    return ssd  # M,N


def sad_kernel(src, dst):
    """Compute SSD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    """Student Code Starts"""
    sad = np.zeros((src.shape[0], dst.shape[0]))
    """Student Code Starts"""
    for i in range(src.shape[0]):
        temp1 = src[i]
        temp2 = np.abs(dst - temp1)
        sad[i] = np.sum(np.sum(temp2, axis = 2), axis = 1)
        # pdb.set_trace()

    
    """Student Code Ends"""

    return sad  # M,N


def zncc_kernel(src, dst):
    """Compute negative zncc similarity, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]
    src_copy = src.copy()
    dst_copy = dst.copy()
    src_mean = np.zeros((src.shape[0], src.shape[2]))
    dst_mean = np.zeros((dst.shape[0], dst.shape[2]))
    src_std = np.zeros((src.shape[0], src.shape[2]))
    dst_std = np.zeros((dst.shape[0], dst.shape[2]))
    zncc = np.zeros((src.shape[0], dst.shape[0]))
    """Student Code Starts"""
    for i in range(src.shape[0]):
        temp = src[i].copy()
        src_mean[i] = np.mean(temp, axis = 0)
        src_std[i] = np.std(temp, axis = 0)
        src_copy[i] = temp - src_mean[i].reshape((1,3))

    for i in range(dst.shape[0]):
        temp = dst[i].copy()
        dst_mean[i] = np.mean(temp, axis = 0)
        dst_std[i] = np.std(temp, axis = 0)
        dst_copy[i] = temp - dst_mean[i].reshape((1,3))

    for i in range(src_copy.shape[0]):
        std1 = src_std[i].reshape((1,3))
        temp1 = src_copy[i].copy()
        for j in range(dst_copy.shape[0]):
            std2 = dst_std[j].reshape((1,3))
            temp2 = dst_copy[j].copy()
            temp3 = (temp1*temp2)/((std1*std2) + EPS)
            zncc[i,j] = np.sum(temp3)

    
    # pdb.set_trace()
    
    """Student Code Ends"""

    return zncc * (-1.0)  # M,N


def image2patch(image, k_size):
    """get patch buffer for each pixel location from an input image; For boundary locations, use zero padding

    Parameters
    ----------
    image : [H,W,3]
    k_size : int, must be odd number; your function should work when k_size = 1

    Returns
    -------
    [H,W,k_size**2,3]
        The patch buffer for each pixel
    """

    """Student Code Starts"""
    patch_buffer = np.zeros((image.shape[0], image.shape[1], k_size**2, 3))

    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            x_indices = []
            y_indices = []
            const_x = -(int(k_size/2))
            const_y = -(int(k_size/2))
            patch_const_x = 0
            patch_const_y = 0
            patch_x_indices = []
            patch_y_indices = []
            patch_r = np.zeros((k_size, k_size))
            patch_g = np.zeros((k_size, k_size))
            patch_b = np.zeros((k_size, k_size))
            for i in range(k_size):
                if((x + const_x < 0) or (x + const_x >= image.shape[1])):
                    const_x += 1
                    patch_const_x += 1
                    #continue
                else:
                    # pdb.set_trace()
                    x_indices.append(x + const_x)
                    const_x += 1
                    patch_x_indices.append(patch_const_x)
                    patch_const_x += 1

                if((y + const_y < 0) or (y + const_y >= image.shape[0])):
                    const_y += 1
                    patch_const_y += 1
                    #continue
                else:
                    # pdb.set_trace()
                    y_indices.append(y + const_y)
                    const_y += 1
                    patch_y_indices.append(patch_const_y)
                    patch_const_y += 1

            nx, ny = np.meshgrid(y_indices, x_indices)
            patch_nx, patch_ny = np.meshgrid(patch_y_indices, patch_x_indices)
            # if((nx == 475).any()):
            #     pdb.set_trace()

            # pdb.set_trace()
            patch_r[patch_nx, patch_ny] = image[nx, ny, 0]
            patch_g[patch_nx, patch_ny] = image[nx, ny, 1]
            patch_b[patch_nx, patch_ny] = image[nx, ny, 2]
            # pdb.set_trace()

            patch_buffer[y,x,:,0] = patch_r.reshape((-1,))
            patch_buffer[y,x,:,1] = patch_g.reshape((-1,))
            patch_buffer[y,x,:,2] = patch_b.reshape((-1,))


    # pdb.set_trace()
    
    """Student Code Starts"""

    return patch_buffer  # H,W,K**2,3


def compute_disparity_map(rgb_i, rgb_j, d0, k_size=5, kernel_func=ssd_kernel):
    """Compute the disparity map from two rectified view

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    d0 : see the hand out, the bias term of the disparty caused by different K matrix
    k_size : int, optional
        The patch size, by default 3
    kernel_func : function, optional
        the kernel used to compute the patch similarity, by default ssd_kernel

    Returns
    -------
    disp_map: [H,W], dtype=np.float64
        The disparity map, the disparity is defined in the handout as d0 + vL - vR

    lr_consistency_mask: [H,W], dtype=np.float64
        For each pixel, 1.0 if LR consistent, otherwise 0.0
    """

    """Student Code Starts"""
    h, w = rgb_i.shape[:2]
    patches_i = image2patch(rgb_i.astype(float) / 255.0, k_size)  # [h,w,k*k,3]
    patches_j = image2patch(rgb_j.astype(float) / 255.0, k_size)  # [h,w,k*k,3]

    vi_idx, vj_idx = np.arange(h), np.arange(h)
    disp_candidates = vi_idx[:, None] - vj_idx[None, :] + d0
    # pdb.set_trace()
    valid_disp_mask = disp_candidates > 0.0
    lr_consistency_mask = np.zeros((h,w))
    disp_map = np.zeros((h,w))
    # pdb.set_trace()
    for u in range(patches_i.shape[1]):
        buf_i, buf_j = patches_i[:, u], patches_j[:, u]
        value = ssd_kernel(buf_i, buf_j)
        # if(kernel_func == ssd_kernel):
        #     value = ssd_kernel(buf_i, buf_j)
        # elif(kernel_func == sad_kernel):
        #     value = sad_kernel(buf_i, buf_j)
        # else:
        #     value = zncc_kernel(buf_i, buf_j)

        # pdb.set_trace()
        _upper = value.max() + 1.0
        value[~valid_disp_mask] = _upper

        best_matched_right_pixel = np.argmin(value, axis = 1)
        best_matched_left_pixel = np.argmin(value[:,best_matched_right_pixel], axis = 0)
        match_arr = best_matched_left_pixel == vi_idx
        # pdb.set_trace()
        lr_consistency_mask[:,u] = match_arr
        disp_map[vi_idx, u] = disp_candidates[vi_idx, best_matched_right_pixel]
        # pdb.set_trace()

    # disp_map = disp_candidates.copy()


    """Student Code Ends"""

    return disp_map, lr_consistency_mask


def compute_dep_and_pcl(disp_map, B, K):
    """Given disparity map d = d0 + vL - vR, the baseline and the camera matrix K
    compute the depth map and backprojected point cloud

    Parameters
    ----------
    disp_map : [H,W]
        disparity map
    B : float
        baseline
    K : [3,3]
        camera matrix

    Returns
    -------
    [H,W]
        dep_map
    [H,W,3]
        each pixel is the xyz coordinate of the back projected point cloud in camera frame
    """

    """Student Code Starts"""
    h = disp_map.shape[0]
    w = disp_map.shape[1]
    y = np.arange(h)
    x = np.arange(w)
    nx, ny = np.meshgrid(x, y)

    xp = np.stack([nx.flatten(),ny.flatten(),np.ones((h,w)).flatten()])
    # pdb.set_trace()

    fy = K[1,1]
    K_inv = np.linalg.inv(K)
    xp_cam = K_inv @ xp

    disp_map_flatten = disp_map.reshape((1,-1))
    depths = (fy*B)/disp_map_flatten
    dep_map = depths.reshape((h,w))

    xyz_cam = np.zeros((h,w,3))
    k = 0
    for i in range(h):
        for j in range(w):
            xyz_cam[i,j] = xp_cam[:,k].copy()
            # pdb.set_trace()
            # xyz_cam[i,j,2] = depths[0,k].copy()
            xyz_cam[i,j] = xyz_cam[i,j]*depths[0,k]
            k += 1
    pdb.set_trace()
    """Student Code Ends"""

    return dep_map, xyz_cam


def postprocess(
    dep_map,
    rgb,
    xyz_cam,
    R_wc,
    T_wc,
    consistency_mask=None,
    hsv_th=45,
    hsv_close_ksize=11,
    z_near=0.45,
    z_far=0.65,
):
    """
    Your goal in this function is: 
    given pcl_cam [N,3], R_wc [3,3] and T_wc [3,1]
    compute the pcl_world with shape[N,3] in the world coordinate
    """

    # extract mask from rgb to remove background
    mask_hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., -1]
    mask_hsv = (mask_hsv > hsv_th).astype(np.uint8) * 255
    # imageio.imsave("./debug_hsv_mask.png", mask_hsv)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hsv_close_ksize, hsv_close_ksize))
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, morph_kernel).astype(float)
    # imageio.imsave("./debug_hsv_mask_closed.png", mask_hsv)

    # constraint z-near, z-far
    mask_dep = ((dep_map > z_near) * (dep_map < z_far)).astype(float)
    # imageio.imsave("./debug_dep_mask.png", mask_dep)

    mask = np.minimum(mask_dep, mask_hsv)
    if consistency_mask is not None:
        mask = np.minimum(mask, consistency_mask)
    # imageio.imsave("./debug_before_xyz_mask.png", mask)

    # filter xyz point cloud
    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcl_cam.reshape(-1, 3).copy())
    cl, ind = o3d_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    _pcl_mask = np.zeros(pcl_cam.shape[0])
    _pcl_mask[ind] = 1.0
    pcl_mask = np.zeros(xyz_cam.shape[0] * xyz_cam.shape[1])
    pcl_mask[mask.reshape(-1) > 0] = _pcl_mask
    mask_pcl = pcl_mask.reshape(xyz_cam.shape[0], xyz_cam.shape[1])
    # imageio.imsave("./debug_pcl_mask.png", mask_pcl)
    mask = np.minimum(mask, mask_pcl)
    # imageio.imsave("./debug_final_mask.png", mask)

    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    pcl_color = rgb.reshape(-1, 3)[mask.reshape(-1) > 0]

    """Student Code Starts"""
    pcl_world = ((R_wc.T) @ (pcl_cam.T)) - (R_wc.T @ T_wc)
    pcl_world = pcl_world.T
    """Student Code Ends"""

    # np.savetxt("./debug_pcl_world.txt", np.concatenate([pcl_world, pcl_color], -1))
    # np.savetxt("./debug_pcl_rect.txt", np.concatenate([pcl_cam, pcl_color], -1))

    return mask, pcl_world, pcl_cam, pcl_color


def two_view(view_i, view_j, k_size=5, kernel_func=ssd_kernel):
    # Full pipeline

    # * 1. rectify the views
    R_wi, T_wi = view_i["R"], view_i["T"][:, None]  # p_i = R_wi @ p_w + T_wi
    R_wj, T_wj = view_j["R"], view_j["T"][:, None]  # p_j = R_wj @ p_w + T_wj

    R_ji, T_ji, B = compute_right2left_transformation(R_wi, T_wi, R_wj, T_wj)
    assert T_ji[1, 0] > 0, "here we assume view i should be on the left, not on the right"

    R_irect = compute_rectification_R(T_ji)

    rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr = rectify_2view(
        view_i["rgb"],
        view_j["rgb"],
        R_irect,
        R_irect @ R_ji,
        view_i["K"],
        view_j["K"],
        u_padding=20,
        v_padding=20,
    )

    # * 2. compute disparity
    assert K_i_corr[1, 1] == K_j_corr[1, 1], "This hw assumes the same focal Y length"
    assert (K_i_corr[0] == K_j_corr[0]).all(), "This hw assumes the same K on X dim"
    assert (
        rgb_i_rect.shape == rgb_j_rect.shape
    ), "This hw makes rectified two views to have the same shape"
    disp_map, consistency_mask = compute_disparity_map(
        rgb_i_rect,
        rgb_j_rect,
        d0=K_j_corr[1, 2] - K_i_corr[1, 2],
        k_size=k_size,
        kernel_func=kernel_func,
    )

    # * 3. compute depth map and filter them
    dep_map, xyz_cam = compute_dep_and_pcl(disp_map, B, K_i_corr)

    mask, pcl_world, pcl_cam, pcl_color = postprocess(
        dep_map,
        rgb_i_rect,
        xyz_cam,
        R_wc=R_irect @ R_wi,
        T_wc=R_irect @ T_wi,
        consistency_mask=consistency_mask,
        z_near=0.5,
        z_far=0.6,
    )

    return pcl_world, pcl_color, disp_map, dep_map


def main():
    DATA = load_middlebury_data("data/templeRing")
    # viz_camera_poses(DATA)
    two_view(DATA[0], DATA[3], 5, zncc_kernel)

    return


if __name__ == "__main__":
    main()
