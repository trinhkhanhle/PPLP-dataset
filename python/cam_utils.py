#!/usr/bin/python

'''
Created on Aug 14, 2018

@author: Trinh Le (trle@umich.edu)
'''

import sys
from os.path import dirname, realpath
import json
import numpy as np

'''
## Calculate Camera Intrinsic and Distortion Matrices in .txt
'''
def get_cams_matrices(data_path, seq_name):

    # Load camera calibration parameters
    try:
        with open(data_path + seq_name + '/calibration_{0}.json'.format(seq_name)) as cfile:
            calib = json.load(cfile)

    except IOError as e:
        print('Error reading calib json file.\n'+ e.strerror)
        return []

    # Cameras matrices are identified by a tuple of (panel#,node#)
    cameras = {(cam['panel'], cam['node']): cam for cam in calib['cameras']}

    return cameras


if __name__ == "__main__":
    pass
    # if len(sys.argv) < 2:
    #     print("Error: You must enter a sequence name.")
    #     sys.exit(1)
    #
    # data_path = dirname(dirname(realpath(__file__))) + "/"
    # seq_name = sys.argv[1]
    # camIdx = 11
    #
    # cameras = get_cams_matrices(data_path, seq_name)
    #
    # cam = cameras[(0, camIdx)]  # 0 means HD
    #
    # # Save Camera Intrinsic and Distortion Matrices in .txt
    # cam_K = ' '.join(' '.join('%.12f' % col for col in row) for row in cam['K'])
    # cam_dist = ' '.join('%.12f' % coeff for coeff in cam['distCoef'])
    #
    # with open(data_path + seq_name + '/calibration_{0}_{1}.txt'.format(seq_name, camIdx), 'w') as tfile:
    #     tfile.write('K_%02d: %s\n' % (camIdx, cam_K))
    #     tfile.write('Kd_%02d: %s' % (camIdx, cam_dist))
    #
    #
    # # cam['K'], cam['R'], cam['t'], cam['distCoef']
    # '''
    # ## Panoptic Origin w.r.t HD camera and Height of Camera w.r.t Ground Plane. Save in .txt
    # '''
    #
    # # Convert data into numpy arrays for convenience
    # for camIdx in range(11, 12):
    #     cam = cameras[(0, camIdx)]
    #     cam['K'] = np.matrix(cam['K'])
    #     cam['distCoef'] = np.array(cam['distCoef'])
    #     cam['R'] = np.matrix(cam['R'])
    #     cam['t'] = np.array(cam['t']).reshape((3, 1))
    #
    #     search_pos = np.array([0, 0, 0]).reshape((-1, 3)).transpose()
    #     x = np.asarray(cam['R'] * search_pos + cam['t'])
    #     print(str(x))






