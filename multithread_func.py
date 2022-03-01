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


def Match_End_Xml():
    return


def Track_Start_New_Image_Xml():
    path_picture_left = file_path_dic['Path_Picture_Left']
    path_picture_right = file_path_dic['Path_Picture_Right']

    return


''''############################### track left #############################'''


def Track_Start_Image_Left():
    path_picture = file_path_dic['Path_Picture_Left']

    return


def Track_Start_Image_Right():
    path_picture = file_path_dic['Path_Picture_Right']

    return


def Track_End_Xml():
    return


''''############################### track right #############################'''

''''############################### track mask #############################'''


def Mask_Start_Image_Xml():
    path_picture = file_path_dic['Path_Picture_Right']

    return


def Mask_End():
    return


def Mask_Respond_Image():
    return


def Mask_Resquest_Image():
    return
