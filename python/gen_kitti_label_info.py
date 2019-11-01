#!/usr/bin/python

"""
Created on Aug 17, 2018

@author: Trinh Le (trle@umich.edu)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import cam_utils
import panutils
import json
import os
from obj_utils import *
import argparse

# KITTI label format: https://github.com/NVIDIA/DIGITS/blob/v4.0.0-rc.3/digits/extensions/data/objectDetection/README.md
#Values    Name      Description
#----------------------------------------------------------------------------
#    1    type         Describes the type of object: 'Car', 'Van', 'Truck',
#                      'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
#                      'Misc' or 'DontCare'
#    1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
#                      truncated refers to the object leaving image boundaries
#    1    occluded     Integer (0,1,2,3) indicating occlusion state:
#                      0 = fully visible, 1 = partly occluded
#                      2 = largely occluded, 3 = unknown
#    1    alpha        Observation angle of object, ranging [-pi..pi]
#    4    bbox         2D bounding box of object in the image (0-based index):
#                      contains left, top, right, bottom pixel coordinates
#    3    dimensions   3D object dimensions: height, width, length (in meters)
#    3    location     3D object location x,y,z in camera coordinates (in meters)
#    1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
#    1    score        Only for results: Float, indicating confidence in
#                      detection, needed for p/r curves, higher is better.


def gen_label_info(dataset_path, label2_path, seq_name, idk, seq_uid, hd_skel_json_path):

    cameras = cam_utils.get_cams_matrices(dataset_path, seq_name)
    cam = cameras[(50, idk)]  # 50 means Kinect
    K_cam = np.matrix(cam['K'])
    R_cam = np.matrix(cam['R'])
    T_cam = np.array(cam['t']).reshape((3, 1))
    dist_cam = np.array(cam['distCoef'])

    with open(dataset_path + seq_name + '/synctables_{0}.json'.format(seq_name)) as sfile:
        psync = json.load(sfile)

    with open(dataset_path + seq_name + '/ksynctables_{0}.json'.format(seq_name)) as ksfile:
        ksync = json.load(ksfile)

    for (dirpath, dirnames, filenames) in os.walk(hd_skel_json_path):

        for filename in filenames:

            hd_index = int(filename[-13:-5])  # 8 numbers. 1-based

            # Find corresponding RGB frame
            # Compute universal time
            selUnivTime = psync['hd']['univ_time'][
                hd_index + 2 - 1]  # -2 is some weired offset in synctables. -1 to account for 1-based
            # print('hd_index: {0}, UnivTime: {1: 0.3f} \n'.format(hd_index, selUnivTime))

            k_color = ksync['kinect']['color']['KINECTNODE{0:d}'.format(idk)]
            time_diff_c = abs(selUnivTime - (np.array(k_color['univ_time']) - 6.25))
            time_distc = np.min(time_diff_c)
            cindex = np.argmin(time_diff_c)  # cindex: 0 based

            k_depth = ksync['kinect']['depth']['KINECTNODE{0:d}'.format(idk)]
            time_diff_d = abs(selUnivTime - np.array(k_depth['univ_time']))
            time_distd = np.min(time_diff_d)
            dindex = np.argmin(time_diff_d)  # dindex: 0 based

            # Time difference between HD and Depth camera
            print("hd_index:", hd_index, ", cindex=", cindex, ", dindex=", dindex,
                  'hd-depth diff={0: 0.4f}'.format(selUnivTime - k_depth['univ_time'][dindex]),
                  'hd-color diff= {0: 0.4f} \n'.format(selUnivTime - k_color['univ_time'][cindex] - 6.25))

            # Filtering if current kinect data is far from the selected time
            color_depth_diff = abs(k_depth['univ_time'][dindex] - k_color['univ_time'][cindex])
            if color_depth_diff > 6.5:
                print('Skipping {0}, depth-color diff {1:0.3f}\n'.format(hd_index, color_depth_diff))
                continue

            if time_distc > 30 or time_distd > 17:
                print('Skipping {0}\n'.format(hd_index))
                print(time_distc, time_distd)
                continue

            # print(hd_index, cindex, dindex)

            # Now there exists a corresponding cindex!!!
            # Get body data points from JSON files
            skel_json_fname = hd_skel_json_path + filename

            try:
                with open(skel_json_fname) as dfile:
                    bframe = json.load(dfile)
            except Exception as e:
                print('Error reading {0}\n'.format(skel_json_fname))
                continue

            label_str = ''
            for id_ in range(len(bframe['bodies'])):

                body = bframe['bodies'][id_]
                skel = np.array(body['joints19']).reshape((-1, 4)).transpose()

                # Filter point with low confidence values
                # Show only points detected with confidence
                valid = skel[3, :] > 0.1  # value is column-indexed base
                valid[3] = True  # Always keep shoulders
                valid[9] = True

                # skel = skel[:, valid] # Don't remove body points < 0.1
                skel = np.delete(skel, (3), axis=0)  # don't need confidence value anymore

                # (skel[:, 0], skel[:, 3], skel[:, 9]) are neck and 2 shoulders

                # Find middle point of 2 shoulders
                midpoint_along_shoulders = (skel[:, 3] + skel[:, 9]) / 2.0

                # Write unit vector (orientation) going through this midpoint and parallel to ground plane y = 0 and perpendicular to the line
                # => cross product between normal vector of ground plane and line vector
                n = np.array([0, -1, 0])
                m = skel[:, 3] - skel[:, 9]  # shoulder vector: from right to left shoulder

                # In opposite direction of Orientation vector
                z2 = np.cross(n, m)
                z2 = z2 / np.linalg.norm(z2)

                # Write unit vector perpendicular to ground plane
                y2 = np.array([0, -1, 0])

                x2 = np.cross(y2, z2)  # y2, z2 order
                x2 = x2 / np.linalg.norm(x2)

                origin2 = midpoint_along_shoulders
                origin1 = np.array([0, 0, 0])

                # T = np.matrix(origin1 - origin2).transpose()

                R = np.matrix([x2, y2, z2])

                # skel2 = np.zeros(shape=skel.shape)  # skel2 is matrix

                # Convert body points to body origin
                skel2 = R * skel
                xs = np.array(skel[0, :]).reshape(-1)
                ys = np.array(skel[1, :]).reshape(-1)
                zs = np.array(skel[2, :]).reshape(-1)

                xs2 = np.array(skel2[0, :]).reshape(-1)  # To remove the second dimension.
                ys2 = np.array(skel2[1, :]).reshape(-1)
                zs2 = np.array(skel2[2, :]).reshape(-1)

                # Find min, max
                min_x = np.inf
                max_x = -np.inf
                min_y = np.inf
                max_y = -np.inf
                min_z = np.inf
                max_z = -np.inf
                # Need mask here to avoid taking account points less than confidence threshold
                for i in range(0, 19):
                    if not valid[i]:
                        continue
                    if skel2[0, i] < min_x: min_x = skel2[0, i]
                    if skel2[0, i] > max_x: max_x = skel2[0, i]

                    if skel2[1, i] < min_y: min_y = skel2[1, i]
                    if skel2[1, i] > max_y: max_y = skel2[1, i]

                    if skel2[2, i] < min_z: min_z = skel2[2, i]
                    if skel2[2, i] > max_z: max_z = skel2[2, i]

                # Add a margin
                margin = 0  # 5 centimeters
                min_x -= margin
                min_y -= 10  # TODO: hacky
                min_z -= margin
                max_x += margin
                max_y += margin
                max_z += margin

                # Draw bounding box
                x_corners_local = [min_x, min_x, min_x, min_x, max_x, max_x, max_x, max_x]
                y_corners_local = [min_y, min_y, max_y, max_y, min_y, min_y, max_y, max_y]
                z_corners_local = [min_z, max_z, min_z, max_z, min_z, max_z, min_z, max_z]

                height = max_y - min_y  # 1 is +, other is -. Because origin is between shoulders
                width = max_x - min_x
                length = max_z - min_z

                # Find bottom center of bbox
                bottom_center_local = [(min_x + max_x) / 2.0, min_y, (min_z + max_z) / 2.0]
                # Translate back to panoptic
                bottom_center_panoptic = np.linalg.inv(R) * (np.matrix(bottom_center_local).transpose())

                # Translate to Kinect RGB cam
                bottom_center_cam = np.array(R_cam * bottom_center_panoptic + T_cam)
                bottom_center_pixel = panutils.projectPoints(bottom_center_panoptic, K_cam, R_cam, T_cam, dist_cam)

                # Translate all body points and bbox corners to Kinect RGB cam
                skel_pixel = panutils.projectPoints(skel, K_cam, R_cam, T_cam, dist_cam)
                bbox_corners_local = np.matrix([x_corners_local, y_corners_local, z_corners_local])
                bbox_corners_pan = np.linalg.inv(R) * (bbox_corners_local)
                bbox_corners_pixel = panutils.projectPoints(bbox_corners_pan, K_cam, R_cam, T_cam, dist_cam)

                # Calculate rotation_y
                orientation_pan_end = np.matrix(-z2).T  # panoptic
                orientation_pan_start = np.matrix([0, 0, 0]).T
                orientation_cam_end = np.array(R_cam * orientation_pan_end + T_cam)
                orientation_cam_start = np.array(R_cam * orientation_pan_start + T_cam)
                orientation_cam_vec = orientation_cam_end - orientation_cam_start
                ry = -np.arctan2(orientation_cam_vec[2], orientation_cam_vec[0])
                ry = ry.item()

                x = bottom_center_cam[0].item() / 100.0
                y = bottom_center_cam[1].item() / 100.0
                z = bottom_center_cam[2].item() / 100.0
                l = (length + 15) / 100.0
                w = (width + 15) / 100.0
                h = (height + 15) / 100.0
                box_3d = np.array([x, y, z, l, w, h, ry])
                # (3, 3) K_cam
                img_box = project_to_image_space(box_3d, K_cam, truncate=True, discard_before_truncation=False,
                                                 image_size=[1920, 1080])

                if img_box is not None:
                    label_str += 'Pedestrian 0.00 0 0 ' + str(img_box[0]) + ' ' + str(img_box[1]) + ' ' + str(
                        img_box[2]) + ' ' + str(img_box[3]) + ' ' + str(h) + ' ' + str(w) + ' ' + str(l) \
                                 + ' ' + str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(ry) + '\n'

                ### VISUALIZATION
                # origin = [0, 0, 0]
                # X, Y, Z = zip(origin, origin, origin)
                # # x2 = np.cross(y2, z2)
                # # U, V, W = zip(x2, y2, z2)
                # O, L, I = zip([0, 5, 0], [5, 0, 0], [0, 0, 5])
                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                #
                # # ax.quiver(X, Y, Z, U, V, W, arrow_length_ratio=0.01)
                # ax.quiver(X, Y, Z, O, L, I, arrow_length_ratio=0.01)
                # ax.plot3D([midpoint_along_shoulders[0]], [midpoint_along_shoulders[1]], [midpoint_along_shoulders[2]], '.')

                # fig = plt.figure()
                # ax = plt.axes(projection='3d')
                # colors = plt.cm.hsv(np.linspace(0, 1, 30)).tolist()
                # # Local
                # body_edges = np.array(
                #     [[1, 2], [1, 4], [4, 5], [5, 6], [1, 3], [3, 7], [7, 8], [8, 9], [3, 13], [13, 14], [14, 15],
                #      [1, 10], [10, 11], [11, 12]]) - 1
                # # Plot edges for each bone
                # # ax.plot3D([0], [0], [0], '.')
                # for edge in body_edges:
                #     ax.plot3D([skel2[0, edge[0]], skel2[0, edge[1]]], [skel2[1, edge[0]], skel2[1, edge[1]]], zs=[skel2[2, edge[0]], skel2[2, edge[1]]])
                #
                # ax.plot3D(x_corners_local, y_corners_local, z_corners_local, '.')  # Plot 3D bbox corners
                # ax.plot3D([bottom_center_local[0]], [bottom_center_local[1]], [bottom_center_local[2]], '.')  # Plot bottom center of bbox
                #
                # ax.set_xlabel('$Xlocal$', fontsize=20)
                # ax.set_ylabel('$Ylocal$')
                # ax.set_zlabel('$Zlocal$')
                # ax.set_aspect(1)

                # ax.plot3D([x2[0]], [x2[1]], [x2[2]], '.') # Plot 3 unit vectors local
                # ax.plot3D([y2[0]], [y2[1]], [y2[2]], '.')
                # ax.plot3D([z2[0]], [z2[1]], [z2[2]], '.')

                # Local
                # ax.plot3D([0], [0], [0], '.')
                # for edge in body_edges:
                #     ax.plot3D([skel[0, edge[0]], skel[0, edge[1]]], [skel[1, edge[0]], skel[1, edge[1]]], zs=[skel[2, edge[0]], skel[2, edge[1]]])
                #
                # ax.plot3D([midpoint_along_shoulders[0]], [midpoint_along_shoulders[1]], [midpoint_along_shoulders[2]], '.')
                #
                # ax.set_xlabel('$Xpan$', fontsize=20)
                # ax.set_ylabel('$Ypan$')
                # ax.set_zlabel('$Zpan$')

                # plt.show()
                #
                # # Load the corresponding Kinect image
                # colors = plt.cm.hsv(np.linspace(0, 1, 30)).tolist()
                # kinect_img_path = dataset_path + seq_name + '/kinectImgs/'
                # image_path = kinect_img_path + '{0:02d}_{1:02d}/{0:02d}_{1:02d}_{2:08d}.jpg'.format(cam['panel'],
                #                                                                                     cam['node'], cindex)
                # im = plt.imread(image_path)
                # plt.figure(figsize=(15, 15))
                # plt.title('3D Body Projection on Kinect RGB view ({0})'.format(cam['name']))
                # plt.imshow(im)
                # currentAxis = plt.gca()
                # currentAxis.set_autoscale_on(False)
                #
                #
                # # Plot bbox corners and bottom center in pixel frame
                # plt.plot(bbox_corners_pixel[0, :], bbox_corners_pixel[1, :], '.')
                # plt.plot(bottom_center_pixel[0], bottom_center_pixel[1], '.', color=colors[7])
                #
                # # Plot 2D image box
                # if img_box is not None:
                #     plt.plot(img_box[[0, 2]], img_box[[1, 3]], '.')
                #
                # # Plot bbox edges
                # plt.plot([bbox_corners_pixel[0, 0], bbox_corners_pixel[0, 1]], [bbox_corners_pixel[1, 0], bbox_corners_pixel[1, 1]], color='g', linestyle='-', linewidth=2)
                # plt.plot([bbox_corners_pixel[0, 0], bbox_corners_pixel[0, 2]], [bbox_corners_pixel[1, 0], bbox_corners_pixel[1, 2]], color='g', linestyle='-', linewidth=2)
                # plt.plot([bbox_corners_pixel[0, 2], bbox_corners_pixel[0, 3]], [bbox_corners_pixel[1, 2], bbox_corners_pixel[1, 3]], color='g', linestyle='-', linewidth=2)
                # plt.plot([bbox_corners_pixel[0, 3], bbox_corners_pixel[0, 1]],
                #          [bbox_corners_pixel[1, 3], bbox_corners_pixel[1, 1]], color='g', linestyle='-', linewidth=2)
                #
                # plt.plot([bbox_corners_pixel[0, 4], bbox_corners_pixel[0, 5]],
                #          [bbox_corners_pixel[1, 4], bbox_corners_pixel[1, 5]], color='g', linestyle='-', linewidth=2)
                # plt.plot([bbox_corners_pixel[0, 5], bbox_corners_pixel[0, 7]],
                #          [bbox_corners_pixel[1, 5], bbox_corners_pixel[1, 7]], color='g', linestyle='-', linewidth=2)
                # plt.plot([bbox_corners_pixel[0, 4], bbox_corners_pixel[0, 6]],
                #          [bbox_corners_pixel[1, 4], bbox_corners_pixel[1, 6]], color='g', linestyle='-', linewidth=2)
                # plt.plot([bbox_corners_pixel[0, 6], bbox_corners_pixel[0, 7]],
                #          [bbox_corners_pixel[1, 6], bbox_corners_pixel[1, 7]], color='g', linestyle='-', linewidth=2)
                #
                # plt.plot([bbox_corners_pixel[0, 4], bbox_corners_pixel[0, 0]],
                #          [bbox_corners_pixel[1, 4], bbox_corners_pixel[1, 0]], color='g', linestyle='-', linewidth=2)
                # plt.plot([bbox_corners_pixel[0, 1], bbox_corners_pixel[0, 5]],
                #          [bbox_corners_pixel[1, 1], bbox_corners_pixel[1, 5]], color='g', linestyle='-', linewidth=2)
                # plt.plot([bbox_corners_pixel[0, 2], bbox_corners_pixel[0, 6]],
                #          [bbox_corners_pixel[1, 2], bbox_corners_pixel[1, 6]], color='g', linestyle='-', linewidth=2)
                # plt.plot([bbox_corners_pixel[0, 3], bbox_corners_pixel[0, 7]],
                #          [bbox_corners_pixel[1, 3], bbox_corners_pixel[1, 7]], color='g', linestyle='-', linewidth=2)
                #
                # # Check if no points visible in the image, if no points, don't plot bbox in pixel frame, just save the skel_cam in the file only.
                #
                # if skel_pixel.size != 0:
                #     plt.plot(skel_pixel[0, :], skel_pixel[1, :], '.',color=colors[body['id']])  # Plot body points in pixel frame
                #
                #     num_points_out_range = 0
                #     for i in range(len(skel_pixel[0, :])): # There's 19 body points
                #         if skel_pixel[0, i] < 0 or skel_pixel[0, i] > 1920 or skel_pixel[1, i] < 0 or skel_pixel[1, i] > 1080:
                #             num_points_out_range += 1
                #
                # # Count percents
                # occlusion_percent = num_points_out_range / len(skel_pixel[0, :]) # 19
                # # Plot percent # on image
                # plt.text(bbox_corners_pixel[0, 7], bbox_corners_pixel[1, 7] - 5, '{0:0.3}'.format(occlusion_percent), color=colors[body['id']])
                #
                #
                # bbox_img_path = output_path + seq_name + '/training/bbox_imgs/'
                # if not os.path.exists(bbox_img_path):
                #     os.mkdir(bbox_img_path)
                # plt.savefig(bbox_img_path + '/{0:02d}{1:08d}.png'.format(idk, cindex))

                # plt.show()

            # If there're 2 HD frames corresponding to 1 Kinect frame, then the latter will be recorded.
            with open(label2_path + '/{0:02d}{1:02d}{2:02d}{3:06d}.txt'.format(50, idk, seq_uid, cindex), 'w') as lfile:
                lfile.write(label_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate ground truth labels based on no. of HD 3D pose data available.")
    # Dataset path: /home/trinhle/panoptic-toolbox/, /mnt/fcav/datasets/panoptic/
    parser.add_argument("-i", "--input", help="Specify input path", type=str, required=True)
    # Sequence output path: /home/trinhle/panoptic-toolbox/, /mnt/fcav/projects/bodypose2dsim/
    parser.add_argument("-o", "--output", help="Specify output path", type=str, required=True)
    # Sequence name: 160422_haggling1,  160422_ultimatum1
    parser.add_argument("-s", "--seq", help="Specify sequence name", type=str, required=True)
    # Unique number (up to 2 digits) to distinguish samples from different sequences.
    # This will be part of sample filename.
    parser.add_argument("-n", "--num",
                        help="Specify unique number (up to 2 digits) to distinguish sequences. This will be part of sample filename.",
                        type=int, choices=range(0, 100), required=True)

    args = parser.parse_args()

    kitti_label2_path = args.output + args.seq + '/training/label_2/'
    if not os.path.exists(kitti_label2_path):
        os.mkdir(kitti_label2_path)
    idk = 1
    hd_skel_json_path = args.input + args.seq + '/hdPose3d_stage1_coco19/'

    gen_label_info(args.input, kitti_label2_path, args.seq, idk, args.num, hd_skel_json_path)


