import torch
import cv2

import slam_lib.dataset
import slam_lib.models
from slam_lib.models.matching import Matching
from slam_lib.models.utils import AverageTimer, read_image
import slam_lib.mapping
import slam_lib.vis


def match_2_imgs(matching, device, img_local_left_path, img_local_right_path, resize, timer):
    """"""

    img_left, img_right = cv2.imread(img_local_left_path), cv2.imread(img_local_right_path)

    gray_left, inp0, scales0 = read_image(img_local_left_path, device, resize, rotation=0, resize_float=True)
    gray_right, inp1, scales1 = read_image(img_local_right_path, device, resize, rotation=0, resize_float=True)

    # extract feature
    # time_start = time.time()

    # general_pts_2d_left, general_pts_2d_right = feature.get_pts_pair_by_sift(img_left, img_right, flag_debug=True)

    # Perform the matching.
    pred = matching({'image0': inp0, 'image1': inp1})
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    kpts_left, kpts_right = pred['keypoints0'], pred['keypoints1']
    feats_left, feats_right = pred['descriptors0'].T, pred['descriptors1'].T
    matches, conf = pred['matches0'], pred['matching_scores0']

    timer.update('matcher')

    # Keep the matching keypoints and scale points back
    matches = matches
    valid = matches > -1
    mkpts_left = kpts_left[valid]
    mfeats_left = feats_left[valid]
    mkpts_right = kpts_right[matches[valid]]
    mfeats_right = feats_right[matches[valid]]
    mconf = conf[valid]

    pts_2d_left, pts_2d_right = slam_lib.mapping.scale_pts(scales0, mkpts_left), slam_lib.mapping.scale_pts(scales1,
                                                                                                            mkpts_right)

    # get color for 3d feature points
    pts_color = (img_left[pts_2d_left.T[::-1].astype(int).tolist()] + img_right[
        pts_2d_right.T[::-1].astype(int).tolist()])[
                ::-1] / 2 / 255.0  # take average of left and right bgr, then convert to normalized rgb

    # vis
    ### vis match
    # img3 = vis.draw_matches(img_left, mkpts_left, img_right, mkpts_right)
    # cv2.namedWindow('match', cv2.WINDOW_NORMAL)
    # cv2.imshow('match', img3)
    # cv2.waitKey(0)

    img3 = slam_lib.vis.draw_matches(img_left, pts_2d_left, img_right, pts_2d_right)
    cv2.namedWindow('match', cv2.WINDOW_NORMAL)
    cv2.imshow('match', img3)
    cv2.waitKey(0)


def main():
    timer = AverageTimer(newline=True)

    # Load the SuperPoint and SuperGlue models.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_grad_enabled(False)
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
        },
        'superglue': {
            'weights': 'indoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }
    matching = Matching(config).eval().to(device)
    print('SuperGlue', config)
    print('Running inference on device \"{}\"'.format(device))

    # load dataset
    resize = [1024, 750]
    dataset_dir = '/home/cheng/Pictures/data/202201251506/'
    # data = slam_lib.dataset.get_calibration_and_img(dataset_dir)
    data_stereo = slam_lib.dataset.get_calibration_and_img(dataset_dir)
    data_binocular = slam_lib.dataset.get_calibration_and_img(dataset_dir, right_img_dir_name='global')
    timer.update('load_dataset')

    timer.update('load_camera_model')


    for i, (img_local_left_path, img_local_right_path, img_global_path) in enumerate(zip(data_stereo['left_general_img'], data_stereo['right_general_img'], data_binocular['right_general_img'])):
        print('running stereo vision on ', i, img_local_left_path, img_local_right_path, img_global_path)
        # Load the image pair.
        img_left, img_right, img_global = cv2.imread(img_local_left_path), cv2.imread(
            img_local_right_path), cv2.imread(img_global_path)
        match_2_imgs(matching, device, img_local_left_path, img_local_right_path, resize, timer)
        match_2_imgs(matching, device, img_local_left_path, img_global_path, resize, timer)

    return


if __name__ == '__main__':
    main()
