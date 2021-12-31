# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 12/31/21 4:46 PM
"""
import os


def get_left_right_img_path_in_two_folder(left_img_dir, right_img_dir):
    left_img_paths, right_img_paths = [], []
    left_img_names = os.listdir(left_img_dir)
    right_img_names = os.listdir(right_img_dir)
    for left_img_name, right_img_name in zip(left_img_names, right_img_names):
        left_img_paths.append(left_img_dir + '/' + left_img_name)
        right_img_paths.append(right_img_dir + '/' + right_img_name)
    return left_img_paths, right_img_paths


def get_left_right_img_path_in_one_folder(img_dir):
    img_paths_left, img_paths_right = [], []
    if not os.path.isdir(img_dir):
        return [], []
    img_names = os.listdir(img_dir)
    for img_name in img_names:
        if img_name[:4] == 'left':
            img_paths_left.append(img_dir + '/' + img_name)
            img_paths_right.append(img_dir + '/' + 'righ' + img_name[len('lef'):])
    return img_paths_left, img_paths_right
