import cv2

import slam_lib.format
import slam_lib.feature
import socket_msg
import tracking_roi
import threading
import multithread_func

file_path_dic = socket_msg.file_path_dic


def main():
    img_left_path, img_right_path = file_path_dic['Path_Picture_Left'], file_path_dic['Path_Picture_Right']
    pts_left_path, pts_right_path = file_path_dic['Path_Xml_Match_Left'], file_path_dic['Path_Xml_Match_Right']

    # img_left, img_right = cv2.imread(img_left_path), cv2.imread(img_right_path)
    # pts_left, _ = slam_lib.feature.sift_features(img_left)
    # pts_right, _ = slam_lib.feature.sift_features(img_right)

    # slam_lib.format.pts_2_xml(pts_left, pts_left_path)
    # slam_lib.format.pts_2_xml(pts_left, pts_right_path)

    t0 = threading.Thread(target=multithread_func.match_start_image_xml, name='match_start_image_xml')
    multithread_func.match_start_image_xml()
    t0.start()
    print('<<<<<<<<<<<<<<<<<<<<< main')
    print('<<<<<<<<<<<<<<<<<<<<< main')
    print('<<<<<<<<<<<<<<<<<<<<< main')
    print('<<<<<<<<<<<<<<<<<<<<< main')
    print('<<<<<<<<<<<<<<<<<<<<< main')
    print('<<<<<<<<<<<<<<<<<<<<< main')
    t0.join()
    print('<<<<<<<<<<<<<<<<<<<<< finish')

    return


if __name__ == '__main__':
    main()
