#!/usr/bin/env python
from __future__ import print_function, division

import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
try:
    import cPickle as pickle  # python 2
except ImportError:
    import pickle  # python 3
import socket
import argparse
from random import randint

from matplotlib import pyplot as plt
from numpy import unravel_index, log, maximum

from cryoio import mrc


def plot_projs(mrcs_files, log_scale=True):
    plot_randomly = True

    for mrcs in mrcs_files:
        image_stack = mrc.readMRCimgs(mrcs, 0)
        size = image_stack.shape
        N = size[0]
        print('image size: {0}x{1}, number of images: {2}'.format(*size))
        print('Select indices randomly:', plot_randomly)
        fig, axes = plt.subplots(3, 3, figsize=(12.9, 9.6))
        for i, ax in enumerate(axes.flat):
            row, col = unravel_index(i, (3, 3))
            if plot_randomly:
                num = randint(0, size[2])
            else:
                num = i
            print('index:', num)
            if log_scale:
                img = log(maximum(image_stack[:, :, num], 1e-6))
            else:
                img = image_stack[:, :, num]
            im = ax.imshow(img, origin='lower')  # cmap='Greys'
            if row == 2:
                ax.set_xticks([0, int(N/4.0), int(N/2.0), int(N*3.0/4.0), int(N-1)])
            else:
                ax.set_xticks([])
            if col == 0:
                ax.set_yticks([0, int(N/4.0), int(N/2.0), int(N*3.0/4.0), int(N-1)])
            else:
                ax.set_yticks([])

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        fig.suptitle('{} before normalization'.format(mrcs))
        # fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mrcs_files", help="list of mrcs files.", nargs='+')
    parser.add_argument("-l", "--log_scale", help="show image in log scale.",
                        action="store_true")
    args = parser.parse_args()

    log_scale = args.log_scale
    mrcs_files = args.mrcs_files
    print('mrcs_files:', mrcs_files)
    print('log_scale', log_scale)
    plot_projs(mrcs_files, log_scale=log_scale)
