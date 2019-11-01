#!/usr/bin/python

import os
import random
import argparse


# set_file = self.dataset_dir + '/' + data_split + '.txt'
# with open(set_file, 'r') as f:
#     sample_names = f.read().splitlines()
#
# return np.array(sample_names)

def gen_txts(input_path, output_path, seq_names):

    train_sample_names = []
    val_sample_names = []
    trainval_sample_names = []

    for seq_name in seq_names:
        with open(input_path + seq_name + '/trainval.txt') as f:
            seq_sample_names = f.read().splitlines()
            seq_train_sample_names = random.sample(seq_sample_names, int(len(seq_sample_names) // 4) * 3)
            seq_val_sample_names = [val for val in seq_sample_names if val not in seq_train_sample_names]

            train_sample_names += seq_train_sample_names
            val_sample_names += seq_val_sample_names
            trainval_sample_names += seq_sample_names

    trainval_txt = output_path + '/trainval.txt'
    train_txt = output_path + '/train.txt'
    val_txt = output_path + '/val.txt'

    train_sample_names.sort()
    val_sample_names.sort()
    trainval_sample_names.sort()

    with open(trainval_txt, 'w') as vfile:
        vfile.write('\n'.join(trainval_sample_names))

    with open(train_txt, 'w') as vfile:
        vfile.write('\n'.join(train_sample_names))

    with open(val_txt, 'w') as vfile:
        vfile.write('\n'.join(val_sample_names))


# python3 gen_kitti_combined_txts.py -i /mnt/fcav/projects/bodypose2dsim/ -o /mnt/fcav/projects/bodypose2dsim/panoptic_kittized_dataset/ -list 160422_ultimatum1 160226_haggling1 171204_pose3 160422_haggling1
# python3 gen_kitti_combined_txts.py -i /home/trinhle/panoptic-toolbox/ -o /home/trinhle/panoptic-toolbox/ -list 160422_haggling1 160422_ultimatum1
def main():
    print('Hello')
    parser = argparse.ArgumentParser("Generate trainval, train, val.txt based on all generated sequences.")
    parser.add_argument("-i", "--input", help="Specify (common) input path to sequences", type=str, required=True)
    # Sequence output path: /home/trinhle/panoptic-toolbox/, /mnt/fcav/projects/bodypose2dsim/
    parser.add_argument("-o", "--output", help="Specify output path to save combined txts.", type=str, required=True)
    # Sequence name: 160422_haggling1,  160422_ultimatum1
    parser.add_argument("-list", "--list", help="Specify list of sequence names", nargs="*", type=str, default=[], )

    args = parser.parse_args()
    print(args.input, args.output, args.list)
    if not (args.list or args.input or args.output):
        print("Please specify required input, output path or list of sequence names!")
        return

    gen_txts(args.input, args.output, args.list)


if __name__ == "__main__":
    main()
