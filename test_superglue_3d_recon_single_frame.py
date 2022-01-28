# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 12/31/21 4:43 PM
"""
import time

import cv2
import numpy as np
import open3d as o3
import torch
import transforms3d as tf3

import slam_lib.dataset
import slam_lib.feature
import slam_lib.mapping
from slam_lib.camera.cam import BiCamera
from slam_lib.camera.calibration import stereo_calibrate
import slam_lib.vis

from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)


def main():
    """3d reconstruction"""
    timer = AverageTimer(newline=True)

    # Load the SuperPoint and SuperGlue models.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_grad_enabled(False)
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
        },
        'superglue': {
            'weights': 'indoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }
    matching = Matching(config).eval().to(device)
    print('SuperGlue', config)
    print('Running inference on device \"{}\"'.format(device))

    # load dataset
    resize = [1024, 750]
    dataset_dir = '/home/cheng/Pictures/data/202201171510/'
    data = slam_lib.dataset.get_calibration_and_img(dataset_dir)

    timer.update('load_dataset')

    '''shorten calibration data'''
    binocular = BiCamera('./config/bicam_cal_para.json')
    print('calibration result')
    print(binocular.cam_left.camera_matrix)
    print(binocular.cam_right.camera_matrix)

    timer.update('load_camera_model')

    # 3d recon
    last_frame = {'pts_3d': None,
                  'pts_color': None,
                  'pts_2d_left': None,
                  'pts_2d_right': None,
                  'feats_left': None,
                  'feats_right': None,
                  'match': None}

    pc_general = o3.geometry.PointCloud()

    for i, (img_left_path, img_right_path) in enumerate(zip(data['left_general_img'], data['right_general_img'])):
        print('running stereo vision on ', i, img_left_path, img_right_path)
        # Load the image pair.
        img_org_left, img_org_right = cv2.imread(img_left_path), cv2.imread(img_right_path)
        # img_left, img_right = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        img_left, inp0, scales0 = read_image(img_left_path, device, resize, rotation=0, resize_float=True)
        img_right, inp1, scales1 = read_image(img_right_path, device, resize, rotation=0, resize_float=True)

        # extract feature
        # time_start = time.time()

        # general_pts_2d_left, general_pts_2d_right = feature.get_pts_pair_by_sift(img_left, img_right, flag_debug=True)

        # Perform the matching.
        pred = matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts_left, kpts_right = pred['keypoints0'], pred['keypoints1']
        feats_left, feats_right = pred['descriptors0'].T, pred['descriptors1'].T
        matches, conf = pred['matches0'], pred['matching_scores0']
        timer.update('matcher')

        # Keep the matching keypoints and scale points back
        valid = matches > -1
        mkpts_left = kpts_left[valid]
        mfeats_left = feats_left[valid]
        mkpts_right = kpts_right[matches[valid]]
        mfeats_right = feats_right[matches[valid]]
        mconf = conf[valid]
        # print('get features in', time.time() - time_start, 'seconds')

        # compute feature 3d coord
        pts_2d_left, pts_2d_right = slam_lib.mapping.scale_pts(scales0, mkpts_left), slam_lib.mapping.scale_pts(scales1, mkpts_right)
        general_pts_3d = binocular.transform_raw_pixel_to_world_coordiante(pts_2d_left, pts_2d_right)

        # get color for 3d feature points
        pts_color = (img_org_left[pts_2d_left.T[::-1].astype(int).tolist()] + img_org_right[pts_2d_right.T[::-1].astype(int).tolist()])[::-1] / 2 / 255.0   # take average of left and right bgr, then convert to normalized rgb

        # vis
        ### vis match
        # img3 = vis.draw_matches(img_left, mkpts_left, img_right, mkpts_right)
        # cv2.namedWindow('match', cv2.WINDOW_NORMAL)
        # cv2.imshow('match', img3)
        # cv2.waitKey(0)

        img3 = slam_lib.vis.draw_matches(img_org_left, pts_2d_left, img_org_right, pts_2d_right)
        cv2.namedWindow('match', cv2.WINDOW_NORMAL)
        cv2.imshow('match', img3)
        cv2.waitKey(0)

        # tracking
        if i > 0:
            pts1, des1, pts2, des2, index_match, good_matches = slam_lib.feature.match_filter_pts_pair(last_frame['pts_2d_left'],
                                                                                              last_frame['feats_left'],
                                                                                              pts_2d_left, mfeats_left)
            tf = slam_lib.mapping.umeyama_ransac(src=last_frame['pts_3d'][index_match[:, 0]],
                                        tgt=general_pts_3d[index_match[:, 1]])  # compute tf by common pts

            print('>>>>>>>>>>>>>>>>>>>>>>>> tracking ')
            print('fusion get ', len(good_matches), ' good matches')
            print(len(index_match), 'common pts between frame ')
            print('rotation', tf3.euler.mat2euler(tf[:3, :3]), '\ntranslation', tf[:3, -1])
            print('<<<<<<<<<<<<<<<<<<<<<<<< tracking ')

            img_left_last = cv2.imread(data['left_general_img'][i - 1])
            img3 = slam_lib.vis.draw_matches(img_left_last, pts1, img_org_left, pts2)
            cv2.namedWindow('match', cv2.WINDOW_NORMAL)
            cv2.imshow('match', img3)
            cv2.waitKey(10)

            # 3d points fusion
            pts_3d_last = slam_lib.mapping.transform_pt_3d(tf, np.copy(last_frame['pts_3d']))
            general_pts_3d = np.vstack([general_pts_3d, pts_3d_last])
            pts_color = np.vstack([pts_color, last_frame['pts_color']])

        # update keyframe
        # update last frame for tracking0
        last_frame['pts_3d'] = general_pts_3d
        last_frame['pts_color'] = pts_color
        last_frame['pts_2d_left'] = pts_2d_left
        last_frame['feats_left'] = mfeats_left
        last_frame['pts_2d_right'] = pts_2d_right
        last_frame['feats_right'] = mfeats_right

        ### vis point cloud
        mesh = o3.geometry.TriangleMesh()
        frame = mesh.create_coordinate_frame(size=25.0)

        pc_general.points = o3.utility.Vector3dVector(general_pts_3d)
        pc_general.colors = o3.utility.Vector3dVector(pts_color)

        o3.visualization.draw_geometries([pc_general, frame])
        o3.visualization.draw_geometries([pc_general])


if __name__ == '__main__':
    main()
