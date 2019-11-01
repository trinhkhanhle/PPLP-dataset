#!/usr/bin/python

'''
Created on Aug 17, 2018

@author: Trinh Le (trle@umich.edu)
'''

import os
import cam_utils
import numpy as np
import argparse

# HD_11: 3x3 [K]
# Kd_11: 1x4
# Tr_pan_to_cam_11: 3x4 [R|t]


def gen_cam_calib_info(dataset_path, calib_path, seq_name, idk, seq_uid, kcolor_index_list):
    cameras = cam_utils.get_cams_matrices(dataset_path, seq_name)
    cam = cameras[(50, idk)]  # 50 means Kinect
    K_cam = np.array(cam['K']).reshape(-1)
    dist_cam = np.array(cam['distCoef']).reshape(-1)
    R_cam = np.array(cam['R'])
    T_cam = np.array(cam['t'])
    R_T_cam = np.append(R_cam, T_cam, 1).reshape(-1)

    for k_idx in kcolor_index_list:
        try:
            with open(calib_path + '/{0:02d}{1:02d}{2:02d}{3:06d}.txt'.format(50, idk, seq_uid, k_idx), 'w') as cfile:
                cfile.write('HD_11: ' + ' '.join('{0:0.12f}'.format(i) for i in K_cam) + '\n')
                cfile.write('Kd_11: ' + ' '.join('{0:0.12f}'.format(i) for i in dist_cam) + '\n')
                cfile.write('Tr_pan_to_cam_11: ' + ' '.join('{0:0.12f}'.format(i) for i in R_T_cam))
        except IOError as e:
            print('Error writing file {0}\n'.format(calib_path) + e.strerror)
            continue


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Generate camera calibration info.")
    # Dataset path: /home/trinhle/panoptic-toolbox/, /mnt/fcav/datasets/panoptic/
    parser.add_argument("-i", "--input", help="Specify input path", type=str, required=True)
    # Sequence output path: /home/trinhle/panoptic-toolbox/, /mnt/fcav/projects/bodypose2dsim/
    parser.add_argument("-o", "--output", help="Specify output path", type=str, required=True)
    # Sequence name: 160422_haggling1,  160422_ultimatum1
    parser.add_argument("-s", "--seq", help="Specify sequence name", type=str, required=True)
    # Sequence length in terms of no of images
    parser.add_argument("-rb", "--rbegin", help="Specify Kinect image frame number to start", type=int, required=True)
    parser.add_argument("-re", "--rend", help="Specify Kinect image frame number to end", type=int, required=True)
    # Unique number (up to 2 digits) to distinguish samples from different sequences.
    # This will be part of sample filename.
    parser.add_argument("-n", "--num", help="Specify unique number (up to 2 digits) to distinguish sequences. This will be part of sample filename.",
                        type=int, choices=range(0, 100), required=True)

    args = parser.parse_args()

    kitti_calib_path = args.output + args.seq + '/training/calib/'
    if not os.path.exists(kitti_calib_path):
        os.mkdir(kitti_calib_path)
    idk = 1
    kcolor_index_list = range(args.rbegin, args.rend + 1)

    gen_cam_calib_info(args.input, kitti_calib_path, args.seq, idk, args.num, kcolor_index_list)






