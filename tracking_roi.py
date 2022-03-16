import os
import threading
import warnings

import cv2
import numpy as np

import slam_lib.feature
import slam_lib.geometry
import slam_lib.mapping


# class Tracker:
#     def __init__(self):
#         self.matcher_core = slam_lib.feature.Matcher()
#
#         self.track_successful = False
#
#         self.frame_last = None
#         self.pts_last = None
#         self.feats_last = None
#
#         self.rois_last = []
#
#         self.flag_debug = True
#
#     def track(self, img, rois_new=None):
#         """
#         """
#
#         if rois_new is None:
#             rois_new = []
#         for i in range(len(rois_new)):
#             rois_new[i] = np.asarray(rois_new[i])
#
#         tf = np.eye(3)
#
#         '''compute tf from last to current input'''
#         if self.frame_last is not None:
#             '''match'''
#             print('getting superglue match pts')
#             pts_last, _, pts, _ = self.matcher_core.match(self.frame_last, img)
#             # pts_last, pts, _ = slam_lib.geometry.epipolar_geometry_filter_matched_pts_pair(pts_last, pts)
#             print('got match for interesting pts, # of match: ', len(pts))
#
#             '''compute tf'''
#             if len(pts) > 4:
#                 self.track_successful = True
#                 tf, mask_ = cv2.findHomography(pts_last, pts)
#         else:
#             self.track_successful = True
#
#         '''update all roi and last frame'''
#         if self.track_successful:
#             '''update last data'''
#             self.frame_last = img
#
#             '''update rois'''
#             for i_roi, roi_last in enumerate(self.rois_last):
#                 self.rois_last[i_roi] = slam_lib.mapping.transform_pt_2d(tf, roi_last)
#         self.rois_last += rois_new
#         return
#
#     def get_rois(self):
#         if not self.track_successful:
#             return []
#         rois_output = []
#         for roi in self.rois_last:
#             rois_output.append(roi.astype(int).tolist())
#         return rois_output


class Tracker:
    def __init__(self):
        self.matcher_core = slam_lib.feature.Matcher()

        self.track_successful = False

        self.frame_first = None
        self.pts_first = None
        self.feats_first = None

        self.rois_in_first_frame = []
        self.masks_in_first_frame = []
        self.tf_first_2_new = None
        self.flag_debug = True

        self.lock = threading.Lock()

    def track(self, img, rois_new=None):
        """
        """
        '''no new img, no tracking'''
        if img is None or len(img) == 0:
            print('tracking receive empty img')
            return
        '''rois has nothing to track'''
        if len(self.rois_in_first_frame) == 0 and (rois_new is None or len(rois_new) == 0):
            print('tracking has no rois')
            return

        print(self.__class__, 'track get new img shape', img.shape, ', new rois', rois_new)
        if rois_new is None:
            rois_new = []
        for i in range(len(rois_new)):
            rois_new[i] = np.asarray(rois_new[i])

        tf = np.eye(3)

        '''compute tf from last to current input'''
        if self.frame_first is not None:
            '''match'''
            print('getting superglue match pts')
            pts_last, _, pts, _ = self.matcher_core.match(self.frame_first, img)
            # pts_last, pts, _ = slam_lib.geometry.epipolar_geometry_filter_matched_pts_pair(pts_last, pts)
            print('got match for interesting pts, # of match: ', len(pts))

            '''compute tf'''
            if len(pts) > 4:
                self.track_successful = True
                tf, mask_ = cv2.findHomography(pts_last, pts)
                self.tf_first_2_new = tf

        else:
            print(self.__class__, ' track initialized with img shape', img.shape, ', # new rois', len(rois_new))
            self.frame_first = img
            self.track_successful = True
            self.tf_first_2_new = tf

        '''update all roi and last frame'''
        if self.track_successful:
            print(self.__class__, ' track successful with img shape', img.shape, ', # new rois', len(rois_new))

            '''update last data'''

            '''update rois to first'''

            '''lock rois'''
            self.lock.acquire()
            for i_roi, roi_new in enumerate(rois_new):
                self.rois_in_first_frame.append(slam_lib.mapping.transform_pt_2d(np.linalg.inv(tf), roi_new))
            self.lock.release()
        return

    def get_rois(self):
        if not self.track_successful:
            return []
        rois_output = []

        self.lock.acquire()
        for roi in self.rois_in_first_frame:
            rois_output.append(slam_lib.mapping.transform_pt_2d(self.tf_first_2_new, roi).astype(int).tolist())
        self.lock.release()

        return rois_output

    def move_rois_2_masks(self, ids):
        if ids is None or len(ids) == 0:
            print(self.__class__, ' push_rois_2_masks, get empty ids to push rois to masks')
            return

        '''lock rois and masks'''
        self.lock.acquire()

        for index in ids:
            if index < 0 or index >= len(self.rois_in_first_frame):
                print('index', index, 'out of range for rois, rois has', len(self.rois_in_first_frame), 'ele')
            else:
                self.masks_in_first_frame.append(self.rois_in_first_frame[index])

        for index in ids:
            if index < 0 or index >= len(self.rois_in_first_frame):
                print('index', index, 'out of range for rois, rois has', len(self.rois_in_first_frame), 'ele')
            else:
                self.rois_in_first_frame.pop(index)

        self.lock.release()
        return

    def get_masks(self):
        if not self.track_successful:
            return []
        rois_output = []

        print('get_masks locking')
        self.lock.acquire()
        for roi in self.masks_in_first_frame:
            rois_output.append(slam_lib.mapping.transform_pt_2d(self.tf_first_2_new, roi).astype(int).tolist())
        self.lock.release()
        print('get_masks release')

        return rois_output


def main(name=''):
    print(name, 'at thread', threading.current_thread().name, 'by pid', os.getpid())
    t = Tracker()
    img_dir = '/home/cheng/proj/data/AutoPano/data/graf'

    marked = False
    for i, img_name in enumerate(os.listdir(img_dir)):
        if img_name[-3:] == 'ppm':
            img = cv2.imread(img_dir + '/' + img_name)
            img = cv2.resize(img, (50, 50))
            if not marked:
                roi = [np.asarray([[200, 200], [300, 200], [300, 500], [200, 500]])]
                t.track(img, roi)
                marked = True
            else:
                t.track(img)

            rois = t.get_rois()
            print(rois)
            for roi in rois:
                img = cv2.circle(img, center=roi[0], radius=50, thickness=3, color=(100, 0, 0))
                img = cv2.circle(img, center=roi[1], radius=50, thickness=3, color=(100, 0, 0))
                img = cv2.circle(img, center=roi[2], radius=50, thickness=3, color=(100, 0, 0))
                img = cv2.circle(img, center=roi[3], radius=50, thickness=3, color=(100, 0, 0))
            # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
    print(name)


if __name__ == '__main__':
    main()
