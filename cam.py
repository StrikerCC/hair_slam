# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 12/1/21 6:54 PM
"""

import cv2
import numpy as np


class Camera:
    def __init__(self):
        self.type = 'basic'
        self.camera_matrix = np.eye(3)
        self.distortion_coefficient = np.zeros((3, 1))

    def get_img(self):
        return None


class StereoCamera:
    def __init__(self):
        self.type = 'stereo'
        self.left_camera = Camera()
        self.right_camera = Camera()
        self.R = np.eye(3)
        self.T = np.zeros(3)
        self.E = np.zeros((3, 3))
        self.F = np.zeros((3, 3))
        self.R1 = np.eye(3)
        self.P1 = np.eye(3)
        self.R2 = np.eye(3)
        self.P2 = np.eye(3)
        self.Q = np.eye(4)


    def get_left_img(self):
        return None

    def get_right_img(self):
        return None

    def set_camera_parameter(self, camera_parameter_file_path):
        f = cv2.FileStorage(camera_parameter_file_path, cv2.FILE_STORAGE_READ)
        self.left_camera.leftCameraMatrix, self.right_camera.camera_matrix = f.getNode('leftCameraMatrix').mat(), f.getNode(
            'rightCameraMatrix').mat()
        self.left_camera.distortion_coefficient, self.right_camera.distortion_coefficient = f.getNode('leftDistCoeffs').mat(), f.getNode(
            'rightDistCoeffs').mat()
        self.R, self.T, self.E, self.F = f.getNode('R').mat(), f.getNode('T').mat(), f.getNode('E').mat(), f.getNode(
            'F').mat()
        self.R1, self.P1, self.R2, self.P2, self.Q = f.getNode('R1').mat(), f.getNode('P1').mat(), f.getNode(
            'R2').mat(), f.getNode('P2').mat(), f.getNode('Q').mat()


class HairCamera(StereoCamera):
    def __init__(self):
        super().__init__()
        self.type = 'Hair'

    def get_left_hair_info(self):
        return None

    def get_right_hair_info(self):
        return None


def main():
    cam = HairCamera()
    cam.set_camera_parameter('/home/cheng/proj/3d/BiCameraSDKv2.0/test_data/saved_parameters.xml')
    print(cam.R)
    print(cam.T)
    print(cam.R1)
    print(cam.P1)
    print(cam.Q)


if __name__ == '__main__':
    main()
