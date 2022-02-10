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
import transforms3d as tf3

import slam_lib.dataset
import slam_lib.feature
from slam_lib.camera.cam import BiCamera
from slam_lib.camera.calibration import stereo_calibrate
import slam_lib.mapping

import slam_lib.vis


def main():
    """3d reconstruction"""
    dataset_dir = '/home/cheng/Pictures/data/202201251506/'
    data = slam_lib.dataset.get_calibration_and_img(dataset_dir)
    # data['left_general_img'] = data['left_general_img'][4:]
    # data['right_general_img'] = data['right_general_img'][4:]

    for key in data.keys():
        print(key, 'has')
        for img_dir in data.get(key):
            print('     ', img_dir)

    binocular = BiCamera('./config/bicam_cal_para.json')
    print('calibration result')
    print(binocular.cam_left.camera_matrix)
    print(binocular.cam_right.camera_matrix)

    # 3d recon
    last_frame = {'pts_3d': None,
                  'pts_color': None,
                  'pts_2d_left': None,
                  'pts_2d_right': None,
                  'sift_left': None,
                  'sift_right': None,
                  'match': None}

    pc_general = o3.geometry.PointCloud()

    for i, (img_left_path, img_right_path) in enumerate(zip(data['left_general_img'], data['right_general_img'])):
        print('running stereo vision on ', i, img_left_path, img_right_path)

        # preprocess image
        img_left, img_right = cv2.imread(img_left_path), cv2.imread(img_right_path)
        gray_left, gray_right = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        # extract feature
        time_start = time.time()
        pts_2d_left, sift_left, pts_2d_right, sift_right = slam_lib.feature.get_epipolar_geometry_filtered_sift_matched_pts(gray_left, gray_right, flag_debug=True)

        print('get features in', time.time() - time_start, 'seconds')

        # compute feature 3d coord
        pts_3d = binocular.correspondence_to_3d_in_left_rectify(pts_2d_left, pts_2d_right)

        # get color for 3d feature points
        pts_color = (img_left[pts_2d_left.T[::-1].astype(int).tolist()] + img_right[pts_2d_right.T[::-1].astype(int).tolist()])[::-1] / 2 / 255.0   # take average of left and right bgr, then convert to normalized rgb

        # get triangle mesh for mesh body

        # tracking
        if i > 0:
            pts1, des1, pts2, des2, index_match = slam_lib.feature.match_sift_feats(last_frame['pts_2d_left'], last_frame['sift_left'], pts_2d_left, sift_left)
            pts1, des1, pts2, des2, mask = slam_lib.feature.epipolar_geometry_filter_matched_pts_pair(pts1, des1, pts2, des2)
            index_match = index_match[mask]
            tf = slam_lib.mapping.umeyama_ransac(src=last_frame['pts_3d'][index_match[:, 0]], tgt=pts_3d[index_match[:, 1]])     # compute tf by common pts

            print('>>>>>>>>>>>>>>>>>>>>>>>> tracking ')
            print('fusion get ', len(index_match), ' good matches')
            print(len(index_match), 'common pts between frame ')
            print('rotation', tf3.euler.mat2euler(tf[:3, :3]), '\ntranslation', tf[:3, -1])
            print('<<<<<<<<<<<<<<<<<<<<<<<< tracking ')

            img_left_last = cv2.imread(data['left_general_img'][i-1])
            img3 = slam_lib.vis.draw_matches(img_left_last, pts1, gray_left, pts2)
            cv2.namedWindow('match', cv2.WINDOW_NORMAL)
            cv2.imshow('match', img3)
            cv2.waitKey(0)

            # 3d points fusion
            pts_3d_last = slam_lib.mapping.transform_pt_3d(tf, np.copy(last_frame['pts_3d']))
            pts_3d = np.vstack([pts_3d, pts_3d_last])
            pts_color = np.vstack([pts_color, last_frame['pts_color']])

            print('new points added', len(pts_3d) - len(pts_3d_last))


        # update last frame for tracking
        last_frame['pts_3d'] = pts_3d
        last_frame['pts_color'] = pts_color
        last_frame['pts_2d_left'] = pts_2d_left
        last_frame['sift_left'] = sift_left
        last_frame['pts_2d_right'] = pts_2d_right
        last_frame['sift_right'] = sift_right

        # statistics
        # vis
        distance_max = 500
        mesh = o3.geometry.TriangleMesh()
        frame = mesh.create_coordinate_frame(size=25.0)

        # mask out some wrong point by distance away from camera
        # pts_3d = pts_3d[np.linalg.norm(pts_3d, axis=-1) < distance_max]
        pc_general.points = o3.utility.Vector3dVector(pts_3d)
        pc_general.colors = o3.utility.Vector3dVector(pts_color)

        o3.visualization.draw_geometries([pc_general, frame])
        o3.visualization.draw_geometries([pc_general])


if __name__ == '__main__':
    main()
