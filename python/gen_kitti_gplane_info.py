#!/usr/bin/python

'''
Created on Aug 17, 2018

@author: Trinh Le (trle@umich.edu)
'''

import os
import cam_utils
import numpy as np
import argparse

# Plane
# Width 4
# Height 1
# -2.128048e-03 -8.009965e-01 -5.988e-01 1.866530e+00


def gen_gplane_info(dataset_path, planes_path, seq_name, idk, seq_uid, kcolor_index_list):
    cameras = cam_utils.get_cams_matrices(dataset_path, seq_name)
    cam = cameras[(50, idk)]  # 50 means Kinect
    K_cam = np.matrix(cam['K'])
    R_cam = np.matrix(cam['R'])
    T_cam = np.array(cam['t']).reshape((3, 1))
    dist_cam = np.array(cam['distCoef'])

    # convert normal vector to cam frame
    gplane_cam_start = np.asarray(R_cam * np.matrix([0, 0, 0]).transpose() + T_cam)
    gplane_cam_end = np.asarray(R_cam * np.matrix([0, -1, 0]).transpose() + T_cam)
    gplane_cam = gplane_cam_end - gplane_cam_start

    gplane_cam_norm = gplane_cam / (
        np.sqrt(gplane_cam[0] * gplane_cam[0] + gplane_cam[1] * gplane_cam[1] + gplane_cam[2] * gplane_cam[2]))

    a = gplane_cam_norm[0].item()
    b = gplane_cam_norm[1].item()
    c = gplane_cam_norm[2].item()

    gplane_cam_start = gplane_cam_start / 100.0  # convert from cm to meter
    d = -(a * gplane_cam_start[0] + b * gplane_cam_start[1] + c * gplane_cam_start[2]).item()

    print(a, b, c, d)
    for k_idx in kcolor_index_list:
        try:
            with open(planes_path + '/{0:02d}{1:02d}{2:02d}{3:06d}.txt'.format(50, idk, seq_uid, k_idx), 'w') as pfile:
                pfile.write('Width 4\n')
                pfile.write('Height 1\n')
                pfile.write(str(a) + ' ' + str(b) + ' ' + str(c) + ' ' + str(d))
        except IOError as e:
            print('Error writing file {0}\n'.format(planes_path) + e.strerror)
            continue


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Generate ground planes info.")
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
    parser.add_argument("-n", "--num",
                        help="Specify unique number (up to 2 digits) to distinguish sequences. This will be part of sample filename.",
                        type=int, choices=range(0, 100), required=True)

    args = parser.parse_args()

    kitti_planes_path = args.output + args.seq + '/training/planes/'
    if not os.path.exists(kitti_planes_path):
        os.mkdir(kitti_planes_path)
    kcolor_index_list = range(args.rbegin, args.rend + 1)
    idk = 1

    gen_gplane_info(args.input, kitti_planes_path, args.seq, idk, args.num, kcolor_index_list)


