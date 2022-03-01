import cv2

import slam_lib.format
import hair_match_superglue
import tracking_roi
import socket_msg


'''functionalities'''
matcher = hair_match_superglue.Matcher()
# tracker_left = tracking_roi.Tracker()
# tracker_right = tracking_roi.Tracker()
# tracker_mask = tracking_roi.Tracker()

'''file paths'''
file_path_dic = socket_msg.file_path_dic


''''############################### match #############################'''


def match_start_image_xml():
    """"""
    '''reading data'''
    path_picture_left = file_path_dic['Path_Picture_Left']
    path_picture_right = file_path_dic['Path_Picture_Right']
    left_pts_xml_path = file_path_dic['Path_Xml_Match_Left']
    right_pts_xml_path = file_path_dic['Path_Xml_Match_Right']
    id_match_xml_path = file_path_dic['Path_Xml_Match_Id']

    img_left, img_right = cv2.imread(path_picture_left), cv2.imread(path_picture_right)
    pts_left, pts_right = slam_lib.format.xml_2_pts(left_pts_xml_path), slam_lib.format.xml_2_pts(right_pts_xml_path)

    '''exec match'''
    id_match = matcher.match(img_left, pts_left, img_right, pts_right)

    '''write result'''
    slam_lib.format.pts_2_xml(id_match, id_match_xml_path)
    return True


def match_end_xml():
    return


''''############################### track left #############################'''
def track_start_new_image_xml_left():
    path_picture = file_path_dic['Path_Picture_Left']

    return True

def track_start_image_left():
    path_picture = file_path_dic['Path_Picture_Left']

    return True


''''############################### track right #############################'''
def track_start_new_image_xml_right():
    path_picture = file_path_dic['Path_Picture_Right']

    return True


def track_start_image_right():
    path_picture = file_path_dic['Path_Picture_Right']

    return True

''''############################### track mask #############################'''
def mask_start_image_xml():
    path_picture = file_path_dic['Path_Picture_Right']

    return


def mask_respond_image():
    return


def mask_resquest_image():
    return
