# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 12/31/21 4:46 PM
"""
import os


class Dataset:
    def __init__(self):
        # some data path naming convensions
        self.calibration_img_dir_name = '/calibration/'
        self.general_img_dir_name = '/img/'
        self.left_img_dir_name = '/left/'
        self.right_img_dir_name = '/right/'

        self.left_calibration_img = []
        self.right_calibration_img = []
        self.left_general_img = []
        self.right_general_img = []

    def load_from_dir(self, data_dir_path):
        if not os.path.isdir(data_dir_path):
            print('data dir not available')
            return

        if os.path.isdir(data_dir_path + self.calibration_img_dir_name):
            self.left_calibration_img, self.right_calibration_img = get_left_right_img_path_in_two_folder(
                data_dir_path + self.calibration_img_dir_name + self.left_img_dir_name,
                data_dir_path + self.calibration_img_dir_name + self.right_img_dir_name)
        else:
            print('no calibration data found')

        if os.path.isdir(data_dir_path + self.general_img_dir_name):
            self.left_general_img, self.right_general_img = get_left_right_img_path_in_two_folder(
                data_dir_path + self.general_img_dir_name + self.left_img_dir_name,
                data_dir_path + self.general_img_dir_name + self.right_img_dir_name)
        else:
            print('no general data found')


def get_left_right_img_path_in_two_folder(left_img_dir, right_img_dir):
    left_img_paths, right_img_paths = [], []
    left_img_names = sorted(os.listdir(left_img_dir))
    right_img_names = sorted(os.listdir(right_img_dir))
    for left_img_name, right_img_name in zip(left_img_names, right_img_names):
        left_img_paths.append(left_img_dir + '/' + left_img_name)
        right_img_paths.append(right_img_dir + '/' + right_img_name)
    return left_img_paths, right_img_paths


def get_left_right_img_path_in_one_folder(img_dir):
    img_paths_left, img_paths_right = [], []
    if not os.path.isdir(img_dir):
        return [], []
    img_names = sorted(os.listdir(img_dir))
    for img_name in img_names:
        if img_name[:4] == 'left':
            img_paths_left.append(img_dir + '/' + img_name)
            img_paths_right.append(img_dir + '/' + 'righ' + img_name[len('lef'):])
    return img_paths_left, img_paths_right


def get_calibration_and_img(data_dir):
    calibration_img_dir_name = '/calibration/'
    general_img_dir_name = '/img/'
    left_img_dir_name = '/left/'
    right_img_dir_name = '/right/'

    data = {'left_calibration_img': [],
            'right_calibration_img': [],
            'left_general_img': [],
            'right_general_img': []}

    if not os.path.isdir(data_dir):
        print('data dir not available')
        return data

    if os.path.isdir(data_dir + calibration_img_dir_name):
        data['left_calibration_img'], data['right_calibration_img'] = get_left_right_img_path_in_two_folder(
            data_dir + calibration_img_dir_name + left_img_dir_name,
            data_dir + calibration_img_dir_name + right_img_dir_name)
    else:
        print('no calibration data found')

    if os.path.isdir(data_dir + general_img_dir_name):
        data['left_general_img'], data['right_general_img'] = get_left_right_img_path_in_two_folder(
            data_dir + general_img_dir_name + left_img_dir_name,
            data_dir + general_img_dir_name + right_img_dir_name)
    else:
        print('no general data found')

    return data


def main():
    dataset = get_calibration_and_img('/home/cheng/Pictures/data/202201111639')
    for key in dataset.keys():
        print(key)
        for img_dir in dataset.get(key):
            print('     ', img_dir)


if __name__ == '__main__':
    main()
