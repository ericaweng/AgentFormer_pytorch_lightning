import argparse
import os

import numpy as np


def get_data_split(dataset_path='./datasets/trajnet_split', mode='train'):
    if mode == 'train':
        dataset_dir = os.path.join(dataset_path, 'train')
    elif mode == 'val':
        dataset_dir = os.path.join(dataset_path, 'val')
    elif mode == 'test':
        dataset_dir = os.path.join(dataset_path, 'test')
    else:
        raise NotImplementedError

    all_files = os.listdir(dataset_dir)
    all_files = [os.path.join(dataset_dir, _path) for _path in all_files]

    data = []

    for file in all_files:
        gt = np.genfromtxt(file, delimiter=' ', dtype=str)
        data.append(gt)

    data = np.concatenate(data)
    np.savetxt(os.path.join(dataset_path, f'{mode}.txt'), data, fmt="%s")


if __name__ == '__main__':
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'val', 'test'])
    args = parser.parse_args()
    # yapf: enable

    get_data_split(mode=args.mode)
