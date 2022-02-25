import os

import cv2
import numpy as np

import slam_lib.feature
import slam_lib.geometry
import slam_lib.mapping


class Tracker:
    def __init__(self):
        self.matcher_core = slam_lib.feature.Matcher()

        self.frame_last = None
        self.pts_last = None
        self.feats_last = None

        self.rois_last = []

        self.flag_debug = True

    def track(self, img, rois):
        """
        """

        tf = np.eye(3)
        update = False

        if self.frame_last is not None:
            '''match'''
            print('getting superglue match pts')
            pts_last, _, pts, _ = self.matcher_core.match(self.frame_last, img)
            # pts_last, pts, _ = slam_lib.geometry.epipolar_geometry_filter_matched_pts_pair(pts_last, pts)
            print('got match for interesting pts, # of match: ', len(pts))

            '''compute tf'''
            if len(pts) > 4:
                update = True
                tf, mask_ = cv2.findHomography(pts_last, pts)
        else:
            update = True

        if update:
            '''update last data'''
            self.frame_last = img

            '''update rois'''
            for i_roi, roi_last in enumerate(self.rois_last):
                self.rois_last[i_roi] = slam_lib.mapping.transform_pt_2d(tf, roi_last)
        self.rois_last += rois

        return

    def get_rois(self):
        rois_output = []
        for roi in self.rois_last:
            rois_output.append(roi.astype(int).tolist())
        return rois_output


def main():
    t = Tracker()
    img_dir = '/home/cheng/proj/data/AutoPano/data/graf'
    for i, img_name in enumerate(os.listdir(img_dir)):
        roi = [[]]
        if img_name[-3:] == 'ppm':
            roi = [np.asarray([[200, 200], [300, 200], [300, 500], [200, 500]])]
            img = cv2.imread(img_dir + '/' + img_name)
            t.track(img, roi)

            rois = t.get_rois()
            print(rois)

            for roi in rois:
                img = cv2.circle(img, center=roi[0], radius=50, thickness=3, color=(100, 0, 0))
                img = cv2.circle(img, center=roi[1], radius=50, thickness=3, color=(100, 0, 0))
                img = cv2.circle(img, center=roi[2], radius=50, thickness=3, color=(100, 0, 0))
                img = cv2.circle(img, center=roi[3], radius=50, thickness=3, color=(100, 0, 0))

            cv2.namedWindow('img', cv2.WINDOW_NORMAL)
            cv2.imshow('img', img)
            cv2.waitKey(0)


if __name__ == '__main__':
    main()
    device = 0

    gray_0, inp0, scales0 = read_image(img_1, self.device, self.resize, rotation=0, resize_float=True)
    gray_1, inp1, scales1 = read_image(img_2, self.device, self.resize, rotation=0, resize_float=True)

    Matching(config).eval().to(device)
    matcher = slam_lib
    pred = matching({'image0': inp0, 'image1': inp1})

    model = CNN()
    output = model(img)