#!/usr/bin/env python
from __future__ import print_function, division

import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
import pickle
import argparse
from random import randint

from matplotlib import pyplot as plt
from numpy import unravel_index

from cryoio import mrc


def plot_projs(mrcs_files):
    plot_randomly = True

    for mrcs in mrcs_files:
        image_stack = mrc.readMRCimgs(mrcs, 0)
        size = image_stack.shape
        N = size[0]
        print('image size: {0}x{1}, number of images: {2}'.format(*size))
        print('Select indices randomly:', plot_randomly)
        fig, axes = plt.subplots(3, 3)
        for i, ax in enumerate(axes.flat):
            row, col = unravel_index(i, (3, 3))
            if plot_randomly:
                num = randint(0, size[2])
            else:
                num = i
            print('index:', num)
            img = image_stack[:, :, num]
            im = ax.imshow(img, cmap='Greys', origin='lower')
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

    plt.show()


if __name__ == '__main__':

    print(sys.argv)
    if len(sys.argv) >= 2:
        mrcs_files = sys.argv[1:]
    else:
        assert False, 'Need mrc file as argument'

    plot_projs(mrcs_files)
