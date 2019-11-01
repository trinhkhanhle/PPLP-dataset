#!/usr/bin/python

import os
import random
import argparse


def gen_txts(kitti_label2_path, processed_kinect_img_path, output_path, seq_name):
    trainval_txt = output_path + seq_name + '/trainval.txt'
    train_txt = output_path + seq_name + '/train.txt'
    val_txt = output_path + seq_name + '/val.txt'

    kinect_imgs_samples = []
    for (_, _, filenames) in os.walk(processed_kinect_img_path):
        kinect_imgs_samples = [filename[-17:-4] for filename in filenames]

    label_2_samples = []
    for (_, _, filenames) in os.walk(kitti_label2_path):
        label_2_samples = [filename[-17:-4] for filename in filenames]

    valid_samples = [valid_sample for valid_sample in kinect_imgs_samples if valid_sample in label_2_samples]

    valid_samples.sort()

    with open(trainval_txt, 'w') as vfile:
        vfile.write('\n'.join(valid_samples))

    train_samples = random.sample(valid_samples, int(len(valid_samples) // 2))
    train_samples.sort()

    with open(train_txt, 'w') as vfile:
        vfile.write('\n'.join(train_samples))

    val_samples = [val for val in valid_samples if val not in train_samples]
    with open(val_txt, 'w') as vfile:
        vfile.write('\n'.join(val_samples))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate trainval, train, val.txt based on no. of rgb images.")
    # Sequence output path: /home/trinhle/panoptic-toolbox/, /mnt/fcav/projects/bodypose2dsim/
    parser.add_argument("-o", "--output", help="Specify output path", type=str, required=True)
    # Sequence name: 160422_haggling1,  160422_ultimatum1
    parser.add_argument("-s", "--seq", help="Specify sequence name", type=str, required=True)

    args = parser.parse_args()

    kitti_label2_path = args.output + args.seq + '/training/label_2/'
    processed_kinect_img_path = args.output + args.seq + '/training/rgb_images/'

    gen_txts(kitti_label2_path, processed_kinect_img_path, args.output, args.seq)

