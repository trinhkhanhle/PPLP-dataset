#!/usr/bin/python

import os
import argparse


def gen_val_txt(kitti_label2_path, processed_kinect_img_path, output_path, seq_name):
    trainval_txt = output_path + seq_name + '/trainval.txt'
    train_txt = output_path + seq_name + '/train.txt'
    val_txt = output_path + seq_name + '/val.txt'

    kinect_imgs_samples = []
    for (_, _, filenames) in os.walk(processed_kinect_img_path):
        kinect_imgs_samples = [filename[-17:-4] for filename in filenames]

    label_2_samples = []
    for (_, _, filenames) in os.walk(kitti_label2_path):
        label_2_samples = [filename[-17:-4] for filename in filenames]

    val_samples = [val for val in kinect_imgs_samples if val in label_2_samples]

    val_samples.sort()

    with open(val_txt, 'w') as vfile:
        vfile.write('\n'.join(val_samples))

    with open(trainval_txt, 'w') as vfile:
        vfile.write('\n'.join(val_samples))

    with open(train_txt, 'w') as vfile:
        vfile.write('')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate val.txt based on no. of ground labels available.")
    # Sequence output path: /home/trinhle/panoptic-toolbox/, /mnt/fcav/projects/bodypose2dsim/
    parser.add_argument("-o", "--output", help="Specify output path", type=str, required=True)
    # Sequence name
    parser.add_argument("-s", "--seq", help="Specify sequence name", type=str, required=True)

    args = parser.parse_args()

    kitti_label2_path = args.output + args.seq + '/training/label_2/'
    processed_kinect_img_path = args.output + args.seq + '/training/rgb_images/'

    gen_val_txt(kitti_label2_path, processed_kinect_img_path, args.output, args.seq)

