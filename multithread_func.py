import cv2

import slam_lib.format
import hair_match_superglue
import tracking_roi
import socket_msg
from hair_match_server import send_msg


'''functionalities'''
matcher = hair_match_superglue.Matcher()
tracker_left = tracking_roi.Tracker()
tracker_right = tracking_roi.Tracker()
# tracker_mask = tracking_roi.Tracker()

'''file paths'''
file_path_dic = socket_msg.file_path_dic


''''############################### match #############################'''
def match_start_image_xml(sock, msg_out):
    """"""
    '''reading data'''
    path_picture_left = file_path_dic['Path_Picture_Match_Left']
    path_picture_right = file_path_dic['Path_Picture_Match_Right']
    left_pts_xml_path = file_path_dic['Path_Xml_Match_Left']
    right_pts_xml_path = file_path_dic['Path_Xml_Match_Right']
    id_match_xml_path = file_path_dic['Path_Xml_Match_Id']

    img_left, img_right = cv2.imread(path_picture_left), cv2.imread(path_picture_right)
    pts_left, pts_right = slam_lib.format.xml_2_pts(left_pts_xml_path), slam_lib.format.xml_2_pts(right_pts_xml_path)

    '''exec match'''
    id_match = matcher.match(img_left, pts_left, img_right, pts_right)
    if id_match is not None:
        print('match id\n       ', id_match[:max(len(id_match), 5)])
    matcher.save_pts(img_left, pts_left, img_right, pts_right)
    matcher.save_matches(img_left, pts_left, img_right, pts_right, id_match)

    '''write result'''
    print('writing start')
    slam_lib.format.pts_2_xml(id_match, id_match_xml_path)
    send_msg(sock, msg_out)
    print('writing done')

    print("out >>>>>>>>>>>>>>>>>>>>>>>>>", msg_out, '->', sock)
    return True


''''############################### track left #############################'''
def track_start_new_image_xml_left(sock, msg_out):
    """"""
    '''reading data'''

    path_picture = file_path_dic['Path_Picture_Track_Left']
    pts_input_xml_path = file_path_dic['Path_Xml_Track_Left']
    pts_updated_xml_path = file_path_dic['Path_Xml_Tracked_Left']

    img = cv2.imread(path_picture)
    rois = slam_lib.format.xml_2_rois(pts_input_xml_path)

    '''exec tracking'''
    tracker_left.track(img, rois)
    rois_updated = tracker_left.get_rois()

    '''send finish signal'''
    slam_lib.format.rois_2_xml(rois_updated, pts_updated_xml_path)
    send_msg(sock, msg_out)
    print("out >>>>>>>>>>>>>>>>>>>>>>>>>", msg_out, '->', sock)
    return True


def track_start_image_left(sock, msg_out):
    """"""
    '''reading data'''
    path_picture = file_path_dic['Path_Picture_Track_Left']
    pts_updated_xml_path = file_path_dic['Path_Xml_Tracked_Left']

    img = cv2.imread(path_picture)

    '''exec tracking'''
    tracker_left.track(img)
    pts_updated = tracker_left.get_rois()

    '''send finish signal'''
    slam_lib.format.rois_2_xml(pts_updated, pts_updated_xml_path)
    send_msg(sock, msg_out)
    print("out >>>>>>>>>>>>>>>>>>>>>>>>>", msg_out, '->', sock)
    return True


''''############################### track right #############################'''
def track_start_new_image_xml_right(sock, msg_out):
    """"""
    '''reading data'''
    path_picture = file_path_dic['Path_Picture_Track_Right']
    pts_input_xml_path = file_path_dic['Path_Xml_Track_Right']
    pts_updated_xml_path = file_path_dic['Path_Xml_Tracked_Right']

    img = cv2.imread(path_picture)
    pts = slam_lib.format.xml_2_rois(pts_input_xml_path)

    '''exec tracking'''
    tracker_right.track(img, pts)
    pts_updated = tracker_right.get_rois()

    '''send finish signal'''
    slam_lib.format.rois_2_xml(pts_updated, pts_updated_xml_path)
    send_msg(sock, msg_out)
    print("out >>>>>>>>>>>>>>>>>>>>>>>>>", msg_out, '->', sock)
    return True


def track_start_image_right(sock, msg_out):
    """"""
    '''reading data'''
    path_picture = file_path_dic['Path_Picture_Track_Right']
    pts_updated_xml_path = file_path_dic['Path_Xml_Tracked_Right']

    img = cv2.imread(path_picture)

    '''exec tracking'''
    tracker_right.track(img)
    pts_updated = tracker_right.get_rois()

    '''send finish signal'''
    slam_lib.format.rois_2_xml(pts_updated, pts_updated_xml_path)
    send_msg(sock, msg_out)
    print("out >>>>>>>>>>>>>>>>>>>>>>>>>", msg_out, '->', sock)
    return True


''''############################### track mask #############################'''
def mask_start_image_xml_left(sock, msg_out):
    """"""
    '''reading data'''
    path_picture = file_path_dic['Path_Picture_Track_Left']
    pts_input_xml_path = file_path_dic['Path_Xml_Mask_Track_Left']
    pts_updated_xml_path = file_path_dic['Path_Xml_Mask_Tracked_Left']

    img = cv2.imread(path_picture)
    pts = slam_lib.format.xml_2_rois(pts_input_xml_path)

    '''exec tracking'''
    tracker_right.track(img, pts)
    pts_updated = tracker_right.get_rois()

    '''send finish signal'''
    slam_lib.format.rois_2_xml(pts_updated, pts_updated_xml_path)
    send_msg(sock, msg_out)
    print("out >>>>>>>>>>>>>>>>>>>>>>>>>", msg_out, '->', sock)
    return True


def request_left_mask(sock, msg_out):
    """"""
    '''reading data'''
    path_picture = file_path_dic['Path_Picture_Track_Left']
    pts_updated_xml_path = file_path_dic['Path_Xml_Mask_Tracked_Left']
    img = cv2.imread(path_picture)

    '''exec tracking'''
    tracker_right.track(img)
    pts_updated = tracker_right.get_rois()

    '''send finish signal'''
    slam_lib.format.rois_2_xml(pts_updated, pts_updated_xml_path)
    send_msg(sock, msg_out)
    print("out >>>>>>>>>>>>>>>>>>>>>>>>>", msg_out, '->', sock)
    return True


def mask_start_image_xml_right(sock, msg_out):
    """"""
    '''reading data'''
    path_picture = file_path_dic['Path_Picture_Track_Right']
    pts_input_xml_path = file_path_dic['Path_Xml_Mask_Track_Right']
    pts_updated_xml_path = file_path_dic['Path_Xml_Mask_Tracked_Right']

    img = cv2.imread(path_picture)
    pts = slam_lib.format.xml_2_rois(pts_input_xml_path)

    '''exec tracking'''
    tracker_right.track(img, pts)
    pts_updated = tracker_right.get_rois()

    '''send finish signal'''
    slam_lib.format.rois_2_xml(pts_updated, pts_updated_xml_path)
    send_msg(sock, msg_out)
    print("out >>>>>>>>>>>>>>>>>>>>>>>>>", msg_out, '->', sock)
    return True


def request_right_mask(sock, msg_out):
    """"""
    '''reading data'''
    path_picture = file_path_dic['Path_Picture_Track_Right']
    pts_updated_xml_path = file_path_dic['Path_Xml_Mask_Tracked_Right']

    img = cv2.imread(path_picture)

    '''exec tracking'''
    tracker_right.track(img)
    pts_updated = tracker_right.get_rois()

    '''send finish signal'''
    slam_lib.format.rois_2_xml(pts_updated, pts_updated_xml_path)
    send_msg(sock, msg_out)
    print("out >>>>>>>>>>>>>>>>>>>>>>>>>", msg_out, '->', sock)
    return True
