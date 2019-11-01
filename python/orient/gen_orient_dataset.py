#!/usr/bin/python

############## HOW TO EXECUTE ##############
# python3 gen_orient_dataset.py -p1 path1 -p2 path2 -p3 path3 -s sequence_name -rb frame_begin -re frame_end -n sequence_unique_number
# For example:
# python3 gen_orient_dataset.py -p1 /mnt/fcav/projects/bodypose2dsim/ -p2 /mnt/fcav/datasets/panoptic/ -p3 /mnt/fcav/projects/lpbev_mini_batches/iou_2d/panoptic/train/lidar/ -s 160422_ultimatum1 -rb 8798 -re 8799 -n 0

############## PARAMETERS ################
# path1: Specify path to where pointcloud, labels, imgs are stored
# path2: Specify path to sequence calibration file
# path3:  Specify output path to store orientation dataset (must be same directory to where mini batches were generated, i.e. /iou_2d/panoptic/train/lidar/)
# s: sequence name
# rb: frame to start
# re: frame to end
# n: unique number (from 0 - 99) to distinguish sequences. This will be part of sample filename. Has to be consistent among scripts.
# viz: visualize output
# list: instead of [rb, rb], specify a list of frames to generate.


import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import obj_panoptic_utils
from pyntcloud import PyntCloud
from voxel_grid_2d import VoxelGrid2D


area_extents = np.array([[-3.99, 3.99], [-5.0, 3.0], [0.0, 6.995]])
voxel_size = 0.01
ptcloud_threshold = 50


def read_lidar_xyz(velo_dir, sample_name):
    """Reads in PointCloud from Kitti Dataset.

        Keyword Arguments:
        ------------------
        velo_dir : Str
                    Directory of the velodyne files.

        sample_name : Int
                  Index of the image.

        Returns:
        --------
        x : Numpy Array
                   Contains the x coordinates of the pointcloud.
        y : Numpy Array
                   Contains the y coordinates of the pointcloud.
        z : Numpy Array
                   Contains the z coordinates of the pointcloud.
        i : Numpy Array
                   Contains the intensity values of the pointcloud.

        [] : if file is not found

        """
    velo_file = velo_dir + "/%s.ply" % sample_name

    try:
        ptclouds = PyntCloud.from_file(velo_file)  # ptclouds is Panda DataFrame
    except Exception:
        return []
    x = ptclouds.points["x"].values  # .values() converts Panda Series to NumPy array
    y = ptclouds.points["y"].values
    z = ptclouds.points["z"].values

    return np.array([x, y, z])


def read_mrcnn_from_file(mini_batch_dir, classes_name, sub_str, sample_name):
    """
    Reads the MRCNN info matrix from a file

    Args:
        classes_name (str): classes name, e.g. 'Car', 'Pedestrian',
            'Cyclist', 'People'
        sample_name (str): name of sample, e.g. '000123'

    Returns:
        mrcnn_results: {'scores': array(dtype=float32),
                        'features': array(dtype=float32),
                        'keypoints': array(),
                        'class_ids': array(dtype=int32),
                        'masks': array(dtype=float32),
                        'rois': array(dtype=int32),
                        'full_masks': array(dtype=uint8)
    """

    file_name = make_file_path(mini_batch_dir, classes_name,
                               sub_str,
                               sample_name)
    print('read_mrcnn_from_file :: file_name = ', file_name)
    # Load from npy file
    try:
        return np.load(file_name)
    except:
        print(sample_name + ': no npy file is found.')
        return None


def save_numpy_to_file(mini_batch_dir, classes_name, sub_str, sample_name, numpy_results=None):
    """
    Saves the MRCNN info matrix to a file

    Args:
        classes_name (str): classes name, e.g. 'Car', 'Pedestrian',
            'Cyclist', 'People'
        anchor_strides: anchor strides
        sample_name (str): name of sample, e.g. '000123'
        mrcnn_results: To Do
    """
    if numpy_results:
        # Save msakrcnn_result
        file_name = make_file_path(mini_batch_dir, classes_name,
                                   sub_str,
                                   sample_name)
        print('save_numpy_to_file :: file_name = ', file_name)
        np.save(file_name, numpy_results)
        # np.savetxt(file_name[:-4] + '.txt', numpy_results.get('boxes_3d'), fmt='%s')

    else:
        results = {}
        file_name = make_file_path(mini_batch_dir, classes_name,
                                   sub_str,
                                   sample_name)
        print('save_numpy_to_file : results empty : file_name = ', file_name)
        # Save to npy file
        np.save(file_name, results)
        # np.savetxt(file_name[:-4] + '.txt', [], fmt='%s')


def make_file_path(mini_batch_dir, classes_name, sub_str, sample_name, subsub_str=None):
    """Make a full file path to the mini batches

    Args:
        classes_name: name of classes ('Car', 'Pedestrian', 'Cyclist',
            'People')
        sub_str: a name for folder subname
        sample_name: sample name, e.g. '000123'

    Returns:
        The anchors info file path. Returns the folder if
            sample_name is None
    """

    if sample_name:
        if subsub_str:
            return mini_batch_dir + '/' + classes_name + \
                '[' + sub_str + ']/' + \
                subsub_str + '/' + \
                sample_name + ".npy"
        else:
            return mini_batch_dir + '/' + classes_name + \
                '[' + sub_str + ']/' + \
                sample_name + ".npy"
    else:
        if subsub_str:
            return mini_batch_dir + '/' + classes_name + \
                '[' + sub_str + ']/' + subsub_str
        else:
            return mini_batch_dir + '/' + classes_name + \
                '[' + sub_str + ']'


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


def plot_on_image(rgb_img_dir, sample_name, ptcloud_2d):
    image_filename = rgb_img_dir + sample_name + '.jpg'
    im = plt.imread(image_filename)
    plt.figure(figsize=(15, 15))
    plt.imshow(im)
    currentAxis = plt.gca()
    currentAxis.set_autoscale_on(False)
    plt.plot(ptcloud_2d[0, :], ptcloud_2d[1, :], '.')
    plt.show()


def project_points(dataset_path, seq_name, idk, point_cloud):
    cameras = get_cams_matrices(dataset_path, seq_name)
    cam = cameras[(50, idk)]  # 50 means Kinect
    K = np.matrix(cam['K'])
    dist = np.array(cam['distCoef'])

    pts_2d = point_cloud[0:2, :] / point_cloud[2, :]
    # pts_2d = [point_cloud[2, :],
    r = pts_2d[0, :] * pts_2d[0, :] + pts_2d[1, :] * pts_2d[1, :]
    pts_2d[0, :] = pts_2d[0, :] * (1 + dist[0] * r + dist[1] * r * r + dist[4] * r * r * r) \
                   + 2 * dist[2] * pts_2d[0, :] * pts_2d[ 1, :] + dist[3] * ( r + 2 * pts_2d[0, :] * pts_2d[0, :])

    pts_2d[1, :] = pts_2d[1, :] * (1 + dist[0] * r + dist[1] * r * r + dist[4] * r * r * r) \
                   + 2 * dist[3] * pts_2d[0, :] * pts_2d[ 1, :] + dist[2] * ( r + 2 * pts_2d[1, :] * pts_2d[1, :])

    pts_2d[0, :] = K[0, 0] * pts_2d[0, :] + K[0, 1] * pts_2d[1, :] + K[0, 2]
    pts_2d[1, :] = K[1, 0] * pts_2d[0, :] + K[1, 1] * pts_2d[1, :] + K[1, 2]
    return pts_2d


def project_ptcloud_onto_bev(ground_plane, ptcloud):

    all_points = np.transpose(ptcloud)

    # Create Voxel Grid 2D
    voxel_grid_2d = VoxelGrid2D()
    voxel_grid_2d.voxelize_2d(
        all_points, voxel_size,  # voxel_size: 0.01
        extents=area_extents,
        ground_plane=ground_plane,
        create_leaf_layout=False)

    # Remove y values (all 0)
    # voxel_indices all positive
    # voxel_indices: unique indexes - only 1 point is chosen to be occupied in one voxel
    voxel_indices = voxel_grid_2d.voxel_indices[:, [0, 2]]

    occupancy_map = np.zeros((voxel_grid_2d.num_divisions[0], voxel_grid_2d.num_divisions[2]))
    occupancy_map[voxel_indices[:, 0], voxel_indices[:, 1]] = 1
    occupancy_map = np.flip(occupancy_map.transpose(), axis=0)
    voxel_indices_flipped = np.vstack((np.where(occupancy_map == 1)[1], np.where(occupancy_map == 1)[0])).T
    return voxel_indices_flipped, voxel_grid_2d


def compute_num_points_within_rect(voxel_indices, x_corners, z_corners):

    num_pts = 0

    for px, py in zip(voxel_indices[:, 0], voxel_indices[:, 1]):

        # M of coordinates (x,y) is inside the rectangle iff:
        # (0 < AM dot AB < AB dot AB) and (0<AM dot AD<AD dot AD)
        # (scalar product of vectors)
        m = np.array([px, py])
        a = np.array([x_corners[0], z_corners[0]])
        b = np.array([x_corners[1], z_corners[1]])
        # c = np.array([x_corners[2], z_corners[2]])
        d = np.array([x_corners[3], z_corners[3]])
        am = np.array([m[0] - a[0], m[1] - a[1]])
        ab = np.array([b[0] - a[0], b[1] - a[1]])
        ad = np.array([d[0] - a[0], d[1] - a[1]])

        if (0 < np.dot(am, ab) < np.dot(ab, ab)) and (0 < np.dot(am, ad) < np.dot(ad, ad)):
            num_pts += 1

    return num_pts


def compute_bev_box_corners(obj_label, area_extents, bev_shape):
    x_range = area_extents[0, 1] - area_extents[0, 0]
    z_range = area_extents[2, 1] - area_extents[2, 0]
    x_resolution = x_range / bev_shape[1]
    z_resolution = z_range / bev_shape[0]
    corners3d = obj_panoptic_utils.compute_box_corners_3d(obj_label)

    x_corners = []
    z_corners = []
    if len(corners3d) > 0:
        for i in range(4):
            x_corners = np.append(x_corners, (corners3d[0, i] - area_extents[0, 0]) / x_resolution)
            z_corners = np.append(z_corners, (area_extents[2, 1] - corners3d[2, i]) / z_resolution)
    return x_corners, z_corners


def draw_ptcloud_within_bev_boxes(voxel_indices, mask_bev_x_corners, mask_bev_z_corners, x_corners, z_corners,
                                  obj_label, x_resolution, z_resolution, bev_shape,
                                  show_orientation=False, color_table=None, line_width=3, box_color=None):
    plt.figure(figsize=(15, 15))

    # Plot projected bev pointcloud
    if voxel_indices is not None:
        plt.plot(voxel_indices[:, 0], voxel_indices[:, 1], '.')


    # define colors
    if color_table:
        if len(color_table) != 4:
            raise ValueError('Invalid color table length, must be 4')
    else:
        color_table = ["#00cc00", 'y', 'r', 'w']

    trun_style = ['solid', 'dashed', 'dotted']
    if obj_label is not None:
        trc = int(obj_label.truncation > 0.1)
    else:
        trc = 2

    # Plot all bev boxes
    if x_corners is not None and z_corners is not None:
        for b in range(int(len(x_corners)/4)):
            for i in range(4):
                x = np.append(x_corners[i + 4*b], x_corners[(i + 1) % 4 + 4*b])
                z = np.append(z_corners[i + 4*b], z_corners[(i + 1) % 4 + 4*b])

                # Draw the boxes
                if box_color is None and obj_label is not None:
                    box_color = color_table[int(obj_label.occlusion)]

                plt.plot(x, z, linewidth=line_width, color=box_color, linestyle=trun_style[trc])

    # Plot selected bev box with thinner lines inside
    if mask_bev_x_corners is not None and mask_bev_x_corners is not None:
        for i in range(4):
            x = np.append(mask_bev_x_corners[i], mask_bev_x_corners[(i + 1) % 4])
            z = np.append(mask_bev_z_corners[i], mask_bev_z_corners[(i + 1) % 4])

            plt.plot(x, z, linewidth=line_width, color=box_color, linestyle=trun_style[trc])

            # Draw a thinner second line inside
            plt.plot(x, z, linewidth=line_width / 3.0, color='b')



    # Plot orientation of selected object
    if show_orientation:
        # Compute orientation 3D
        orientation = [0.5*np.cos(obj_label.ry), 0.5*np.sin(obj_label.ry)]  # 0.5 is the length of orientation vector(meters)
        if orientation is not None:
            x = np.append((obj_label.t[0] - area_extents[0, 0])/x_resolution, (obj_label.t[0] + orientation[0] - area_extents[0, 0])/x_resolution)
            z = np.append((area_extents[2, 1] - obj_label.t[2])/z_resolution, (area_extents[2, 1] - obj_label.t[2] + orientation[1])/z_resolution)

            # draw the boxes
            plt.plot(x, z, linewidth=4, color='w')
            # ax.plot(x, y, linewidth=2, color='k')
            plt.plot(x, z, linewidth=2, color=box_color)

    axes = plt.gca()
    axes.set_xlim([0, bev_shape[1]])
    axes.set_ylim([0, bev_shape[0]])

    plt.gca().invert_yaxis()
    plt.show()


def gen_orient_dataset(seq_dir, dataset_path, mini_batch_dir, seq_name, classes_name, idk, seq_uid, kcolor_index_list, viz_flag):

    for k_idx in kcolor_index_list:

        sample_name = '{0:02d}{1:02d}{2:02d}{3:06d}'.format(50, idk, seq_uid, k_idx)
        ptcloud_dir = seq_dir + seq_name + '/training/lidar_ptclouds/'
        label_dir = seq_dir + seq_name + '/training/label_2/'
        planes_dir = seq_dir + seq_name + '/training/planes/'
        rgb_img_dir = seq_dir + seq_name + '/training/rgb_images/'

        # Here is the MaskRCNN result for image 500100008677:
        # mrcnn_result = read_mrcnn_from_file(classes_name, sample_name)
        # rois = mrcnn_result.item().get('rois')  #[y1,x1,y2,x2]
        # print('rois = ', rois)
        # rois =  [[ 313  339  931  605]
        #  [ 206  631 1080 1025]
        #  [ 345 1044  836 1216]
        #  [   0 1224 1080 1815]]

        mrcnn_result = read_mrcnn_from_file(mini_batch_dir, classes_name, 'mrcnn', sample_name)

        if mrcnn_result is None: # no npy is found
            continue

        ####################################### Corner cases ##################################################
        #  0. no person in image, mrcnn_result doesn't even have .item()
        #  1. persons not detected by maskrcnn but may or may not have labels: save the persons orientation at the end.
        #  2. person detected by maskrcnn or if background wrongly detected as person but no label or too/no few pointcloud for that person, save np.nan in correct order
        #  Improvement: maskrcnn bad quality or too few pointcloud for a person, adjust the threshold
        #######################################################################################################
        if not mrcnn_result or mrcnn_result.size == 0:  # this is for case 0.
            print(sample_name + ':  maskrcnn result is not valid.')
            results = {}
            save_numpy_to_file(mini_batch_dir, classes_name, 'orient', sample_name, results)
            continue

        full_masks = mrcnn_result.item().get('full_masks')

        if full_masks is None:  # this is for case 0.
            print(sample_name + ':  full_masks is None.')
            results = {}
            save_numpy_to_file(mini_batch_dir, classes_name, 'orient', sample_name, results)
            continue

        ground_plane = obj_panoptic_utils.get_road_plane(int(sample_name), planes_dir)

        # Read label file
        obj_labels = obj_panoptic_utils.read_labels(label_dir, int(sample_name))

        # Read pointcloud file
        ptcloud = read_lidar_xyz(ptcloud_dir, sample_name)  # (3, 33877)

        # Plot all pointcloud on BEV
        if viz_flag:
            ptcloud_indices, voxel_grid = project_ptcloud_onto_bev(ground_plane, ptcloud)
            bev_shape = (voxel_grid.num_divisions[2], voxel_grid.num_divisions[0])
            draw_ptcloud_within_bev_boxes(ptcloud_indices, None, None, None, None, None, None, None, bev_shape)

            ptcloud_2d = project_points(dataset_path, seq_name, idk, ptcloud)
            plot_on_image(rgb_img_dir, sample_name, ptcloud_2d)


        # Project all pointcloud onto image
        ptcloud_2d = project_points(dataset_path, seq_name, idk, ptcloud)

        ptcloud_2d = np.rint(ptcloud_2d).astype(np.int32)  # round off

        # Filter point outside (1920, 1080) due to rounding off effect.
        for i in range(len(ptcloud_2d[0, :])):
            if ptcloud_2d[1, i] >= 1080:
                ptcloud_2d[1, i] = 1079

            if ptcloud_2d[0, i] >= 1920:
                ptcloud_2d[0, i] = 1919

        ptcloud_img = np.zeros(full_masks[:, :, 0].T.shape)  # (1080, 1920) -> (1920, 1080)
        ptcloud_img[ptcloud_2d[0, :], ptcloud_2d[1, :]] = 1  # set 1s at where the pointcloud is

        orients_gt = []  # output: holds orientations in maskrcnn order.
        orients_gt_indices = []  # holds all orientations that are detected by maskrcnn

        # For each mask from MaskRCNN
        for i in range(0, full_masks.shape[2]):
            full_mask = full_masks[:, :, i]
            full_mask = np.swapaxes(full_mask, 1, 0)
            # TODO: plot full_mask
            if viz_flag:
                full_mask_xy = np.where(full_mask == 1)  # get x y location
                full_mask_2d = np.vstack((full_mask_xy[0], full_mask_xy[1]))

                plot_on_image(rgb_img_dir, sample_name, full_mask_2d)

            masked_ptcloud_img = np.multiply(full_mask, ptcloud_img)  # get pointcloud within mask

            mxy = np.where(masked_ptcloud_img == 1)  # get x y location
            # what if np.where found nothing => mxy contains empty indices x, y => masked_ptcloud is None.

            masked_ptcloud = None
            for mx, my in zip(mxy[0], mxy[1]):
                m_idx = np.where((ptcloud_2d[0, :] == mx) & (ptcloud_2d[1, :] == my))[0]  # get index of masked pointcloud
                if m_idx.any():
                    if masked_ptcloud is None:
                        masked_ptcloud = ptcloud[:, m_idx]
                    else:
                        masked_ptcloud = np.hstack([masked_ptcloud, ptcloud[:, m_idx]])

            if masked_ptcloud is not None:
                if viz_flag:
                    ptcloud_masked_2d = project_points(dataset_path, seq_name, idk, masked_ptcloud)
                    plot_on_image(rgb_img_dir, sample_name, ptcloud_masked_2d)

                # project masked pointcloud onto BEV
                masked_voxel_indices, voxel_grid_2d = project_ptcloud_onto_bev(ground_plane, masked_ptcloud)
                bev_shape = (voxel_grid_2d.num_divisions[2], voxel_grid_2d.num_divisions[0])
                x_range = area_extents[0, 1] - area_extents[0, 0]
                z_range = area_extents[2, 1] - area_extents[2, 0]
                x_resolution = x_range / bev_shape[1]
                z_resolution = z_range / bev_shape[0]

                # Get gt label and compute BEV bounding boxes
                # TODO: set to a higher number as threshold to overcome the noise. Not work well for case 2 => use percentage instead.
                #  use number of pixels instead of number of pointcloud for better performance.
                pts_within_percent_max = 0
                mask_obj_label = None
                mask_bev_x_corners = []
                mask_bev_z_corners = []
                mask_obj_idx = -1
                all_bev_x_corners = []  # for drawing
                all_bev_z_corners = []  # for drawing

                # For each pedestrian entry, find bev bounding box that fits masked pointcloud

                for obj_idx, obj_label in enumerate(obj_labels):

                    # Compute bev corners
                    bev_x_corners, bev_z_corners = compute_bev_box_corners(obj_label, area_extents, bev_shape)

                    all_bev_x_corners.append(bev_x_corners)
                    all_bev_z_corners.append(bev_z_corners)

                    # Compute no. of pointcloud within bev box
                    num_pts = compute_num_points_within_rect(masked_voxel_indices, bev_x_corners, bev_z_corners)
                    pts_within_percent = num_pts / len(masked_voxel_indices[:, 0]) * 100.0
                    # Select no. of points that is highest
                    if pts_within_percent > pts_within_percent_max and pts_within_percent > ptcloud_threshold:
                        mask_obj_label = obj_label
                        pts_within_percent_max = pts_within_percent
                        mask_bev_x_corners = bev_x_corners
                        mask_bev_z_corners = bev_z_corners
                        mask_obj_idx = obj_idx

                print('pts_within_percent_max=', pts_within_percent_max)

                # Plot points and chosen box
                if mask_obj_label and viz_flag:
                    all_bev_x_corners = np.concatenate(all_bev_x_corners).ravel().tolist()
                    all_bev_z_corners = np.concatenate(all_bev_z_corners).ravel().tolist()
                    draw_ptcloud_within_bev_boxes(masked_voxel_indices, mask_bev_x_corners, mask_bev_z_corners,
                                                  all_bev_x_corners, all_bev_z_corners, mask_obj_label, x_resolution, z_resolution, bev_shape, show_orientation=True)

                # Save current orientation
                if mask_obj_label is not None:
                    orients_gt.append([mask_obj_label.x1, mask_obj_label.y1, mask_obj_label.x2, mask_obj_label.y2,
                                       mask_obj_label.h, mask_obj_label.w, mask_obj_label.l,
                                       mask_obj_label.t[0], mask_obj_label.t[1], mask_obj_label.t[2],
                                       mask_obj_label.ry])
                    orients_gt_indices.append(mask_obj_idx)
                else:
                    # this is case 2, save np.nan at the same order
                    print('CASE 2! To little BEV pointcloud overlap. Or no label for detected person.')
                    orients_gt.append(
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

                    if viz_flag:
                        all_bev_x_corners = np.concatenate(all_bev_x_corners).ravel().tolist()
                        all_bev_z_corners = np.concatenate(all_bev_z_corners).ravel().tolist()
                        draw_ptcloud_within_bev_boxes(masked_voxel_indices, None, None,
                                                      all_bev_x_corners, all_bev_z_corners, None, x_resolution,
                                                      z_resolution, bev_shape)
            else:
                print('CASE 2! No pointcloud overlap on image (masked_ptcloud is None).')
                orients_gt.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

        results = {}

        # if len(obj_labels) > len(orients_gt):
        for idx, _ in enumerate(obj_labels):
            if idx not in orients_gt_indices: # this is the case 1, save remaining orientations at the end
                print('CASE 1! Or just saving missed labels as not able to find association.')
                orients_gt.append([obj_labels[idx].x1, obj_labels[idx].y1, obj_labels[idx].x2, obj_labels[idx].y2,
                                   obj_labels[idx].h, obj_labels[idx].w, obj_labels[idx].l,
                                   obj_labels[idx].t[0], obj_labels[idx].t[1], obj_labels[idx].t[2],
                                   obj_labels[idx].ry])

        results['boxes_3d'] = np.array(orients_gt)

        save_numpy_to_file(mini_batch_dir, classes_name, 'orient', sample_name, results)

        # [y1, x1, y2, x2, height, width, length, x,y,z, ry]
        # TODO: save visualization image


def main():


    parser = argparse.ArgumentParser("Generate orientation dataset.")

    # Path 1: /home/trinhle/panoptic-toolbox, /mnt/fcav/projects/bodypose2dsim/
    parser.add_argument("-p1", "--path1", help="Specify path to pointcloud, labels, imgs are stored.", type=str, required=True)
    # Path 2: dataset path /mnt/fcav/datasets/panoptic/
    parser.add_argument("-p2", "--path2", help="Specify path to sequence calibration file.", type=str, required=True)
    # Orientation dataset output path: /mnt/fcav/projects/lpbev_mini_batches/iou_2d/panoptic/train/lidar/
    parser.add_argument("-p3", "--path3", help="Specify output path to store orientation dataset.", type=str, required=True)
    # Sequence name: 160422_haggling1,  160422_ultimatum1
    parser.add_argument("-s", "--seq", help="Specify sequence name", type=str, required=True)
    # Sequence length in terms of no of images
    parser.add_argument("-rb", "--rbegin", help="Specify sample number to start", type=int, default=None)
    parser.add_argument("-re", "--rend", help="Specify sample number to end", type=int, default=None)
    # Unique number (up to 2 digits) to distinguish samples from different sequences.
    # This will be part of sample filename.
    parser.add_argument("-n", "--num",
                        help="Specify unique number (up to 2 digits) to distinguish sequences. This will be part of sample filename.",
                        type=int, choices=range(0, 100), required=True)
    parser.add_argument("-viz", "--viz", default=False, action='store_true')
    parser.add_argument("-list", "--list", help="Specify list of sample numbers", nargs="*", type=int, default=[], )

    args = parser.parse_args()

    viz_flag = args.viz
    classes_name = 'Pedestrian'

    if not args.list and (args.rbegin is None or args.rend is None):
        print("Please specify list of sample indices or range of indices!")
        return

    if args.list:
        kcolor_index_list = args.list
    else:
        kcolor_index_list = range(args.rbegin, args.rend + 1)
    idk = 1

    gen_orient_dataset(args.path1, args.path2, args.path3, args.seq, classes_name, idk, args.num, kcolor_index_list, viz_flag)


if __name__ == "__main__":
    main()
