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

import dataset
import feature
import mapping
from camera.cam import BiCamera
from camera.calibration import stereo_calibrate
import vis


def main():
    """3d reconstruction"""
    dataset_dir = '/home/cheng/Pictures/data/202201131456/'
    data = dataset.get_calibration_and_img(dataset_dir)

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
                  'pts_2d_left': None,
                  'pts_2d_right': None,
                  'sift_left': None,
                  'sift_right': None,
                  'match': None}

    for i, (img_left_path, img_right_path) in enumerate(zip(data['left_general_img'], data['right_general_img'])):
        print('running stereo vision on ', i, img_left_path, img_right_path)

        # preprocess image
        img_left, img_right = cv2.imread(img_left_path), cv2.imread(img_right_path)
        img_left, img_right = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        # extract feature
        time_start = time.time()
        pts_2d_left, sift_left, pts_2d_right, sift_right = feature.get_sift_and_pts(img_left, img_right, flag_debug=True)

        print('get features in', time.time() - time_start, 'seconds')

        # compute feature 3d coord
        pts_3d = binocular.transform_raw_pixel_to_world_coordiante(pts_2d_left, pts_2d_right)

        # tracking
        if i > 0:
            pts1, des1, pts2, des2, index_match, good_matches = feature.match_filter_pts_pair(last_frame['pts_2d_left'], last_frame['sift_left'], pts_2d_left, sift_left)
            tf = mapping.umeyama_ransac(src=pts_3d[index_match[:, 1]], tgt=last_frame['pts_3d'][index_match[:, 0]])     # compute tf by common pts

            print('>>>>>>>>>>>>>>>>>>>>>>>> tracking ')
            print('fusion get ', len(good_matches), ' good matches')
            img_left_last = cv2.imread(data['left_general_img'][i-1])
            img3 = vis.draw_matches(img_left_last, pts1, img_left, pts2)
            cv2.namedWindow('match', cv2.WINDOW_NORMAL)
            cv2.imshow('match', img3)
            cv2.waitKey(0)

            # fusion
            pts_3d = mapping.transform_pt_3d(tf, pts_3d)
            pts_3d = np.vstack([pts_3d, last_frame['pts_3d']])

            print(len(index_match), 'common pts between frame ')
            print('rotation', tf3.euler.mat2euler(tf[:3, :3]), '\ntranslation', tf[:3, -1])
            print('<<<<<<<<<<<<<<<<<<<<<<<< tracking ')

        # update last frame for tracking
        last_frame['pts_3d'] = pts_3d
        last_frame['pts_2d_left'] = pts_2d_left
        last_frame['sift_left'] = sift_left
        last_frame['pts_2d_right'] = pts_2d_right
        last_frame['sift_right'] = sift_right

        # statistics

        # vis
        mesh = o3.geometry.TriangleMesh()
        frame = mesh.create_coordinate_frame(size=25.0)

        pc_general = o3.geometry.PointCloud()
        pc_general.points = o3.utility.Vector3dVector(pts_3d)

        o3.visualization.draw_geometries([pc_general, frame])
        o3.visualization.draw_geometries([pc_general])


if __name__ == '__main__':
    main()
