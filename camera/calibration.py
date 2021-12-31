# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 12/30/21 2:52 PM
"""
import os

import cv2
import numpy as np
import json
from camera.cam import PinHoleCamera, BiCamera
import feature
import dataset


def stereo_calibrate(square_size, checkboard_size, left_img_paths, right_img_paths, binocular=None, file_2_save=None):
    """

    :param square_size:
    :type square_size:
    :param checkboard_size: (board_width, board_height)
    :type checkboard_size:
    :param left_img_paths:
    :type left_img_paths:
    :param right_img_paths:
    :type right_img_paths:
    :param binocular:
    :type binocular:
    :param file_2_save:
    :type file_2_save:
    :return:
    :rtype:
    """

    img_size = None
    pts_2d_left, pts_2d_right = [], []

    # corner coord in checkboard frame
    chessboard_corners = feature.make_chessbaord_corners_coord(chessboard_size=checkboard_size, square_size=square_size).astype(
        np.float32)
    # chessboard_corners = np.expand_dims(chessboard_corners, axis=-2)
    chessboard_corners = [chessboard_corners] * len(left_img_paths)

    # corner coord in camera frame
    for i, (left_img_name, right_img_name) in enumerate(zip(left_img_paths, right_img_paths)):
        img_left, img_right = cv2.imread(left_img_name), cv2.imread(right_img_name)
        img_left, img_right = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        pts_2d_left.append(feature.get_checkboard_corners(img_left, checkboard_size))
        pts_2d_right.append(feature.get_checkboard_corners(img_right, checkboard_size))

        if i == 0:
            img_size = (img_left.shape[1], img_left.shape[0])
    # pts_2d_left, pts_2d_right = np.asarray(pts_2d_left), np.asarray(pts_2d_right)

    '''calibrate each camera'''
    ret_left, camera_matrix_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(chessboard_corners,
                                                                                          pts_2d_left,
                                                                                          imageSize=img_size,
                                                                                          cameraMatrix=np.eye(3),
                                                                                          distCoeffs=np.zeros(5))
    ret_right, camera_matrix_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(chessboard_corners,
                                                                                               pts_2d_right,
                                                                                               imageSize=img_size,
                                                                                               cameraMatrix=np.eye(3),
                                                                                               distCoeffs=np.zeros(5))

    '''calibrate binocular'''
    result = cv2.stereoCalibrate(chessboard_corners, pts_2d_left, pts_2d_right, camera_matrix_left, dist_left,
                                 camera_matrix_right, dist_right, imageSize=img_size)
    ret_stereo, camera_matrix_left, dist_left, camera_matrix_right, dist_right, rvecs_stereo, tvecs_stereo, essential, fundamental = result

    '''init camera object parameter'''
    if binocular is not None:
        binocular.cam_left.camera_matrix = camera_matrix_left
        binocular.cam_right.camera_matrix = camera_matrix_right
        binocular.cam_left.distortion_coefficient = dist_left
        binocular.cam_right.distortion_coefficient = dist_right

    '''record value'''
    if file_2_save is not None:
        if not file_2_save[-4:] == 'json':
            print(file_2_save, 'is not a valid json file path')
        else:
            f = open(file_2_save, 'w')
            stereo_calibration_result = {
                'camera_matrix_left': camera_matrix_left.tolist(),
                'distortion coefficients_left': dist_left.tolist(),
                'camera_matrix_right': camera_matrix_right.tolist(),
                'distortion coefficients_right': dist_right.tolist(),
                'rotation': rvecs_stereo.tolist(),
                'translation': tvecs_stereo.tolist(),
                'left_camera_reprojection_error': ret_left,
                'right_camera_reprojection_error': ret_right,
                'stereo_camera_reprojection_error': ret_stereo,
            }
            json.dump(stereo_calibration_result, f)
            f.close()
            print('Calibration result saved to', file_2_save)

    return True


def main():
    square_size = 0.02423
    checkboard_size = (6, 9)  # (board_width, board_height)

    img_dir = '../data/1/'
    img_left_paths, img_right_paths = dataset.get_left_right_img_path_in_one_folder(img_dir)
    binocular = BiCamera()
    stereo_calibrate(square_size, checkboard_size, img_left_paths, img_right_paths, binocular=binocular,
                     file_2_save='../config/bicam_cal_para.json')

    print('calibration result')
    print(binocular.cam_left.camera_matrix)
    print(binocular.cam_right.camera_matrix)


if __name__ == '__main__':
    main()
