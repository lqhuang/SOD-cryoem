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

from numpy import unravel_index, log, maximum
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from cryoio import mrc
from geometry import gen_dense_beamstop_mask


def plot_projs(mrcs_files, log_scale=True, plot_randomly=True):
    for mrcs in mrcs_files:
        image_stack = mrc.readMRCimgs(mrcs, 0)
        size = image_stack.shape
        N = size[0]
        mask = gen_dense_beamstop_mask(N, 2, 0.015, psize=2.8)
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
                img = log(maximum(image_stack[:, :, num], 1e-6)) * mask
            else:
                img = image_stack[:, :, num] * mask
            im = ax.imshow(img, origin='lower')  # cmap='Greys'

            ticks = [0, int(N/4.0), int(N/2.0), int(N*3.0/4.0), int(N-1)]
            if row == 2:
                ax.set_xticks()
            else:
                ax.set_xticks([ticks])
            if col == 0:
                ax.set_yticks()
            else:
                ax.set_yticks([ticks])

        fig.subplots_adjust(right=0.8)
        cbarar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbarar_ax)
        fig.suptitle('{} before normalization'.format(mrcs))
        # fig.tight_layout()
    plt.show()


def plot_projs_with_slider(mrcs_files, log_scale=True):
    for mrcs in mrcs_files:
        image_stack = mrc.readMRCimgs(mrcs, 0)
        size = image_stack.shape
        N = size[0]
        print('image size: {0}x{1}, number of images: {2}'.format(*size))

        # plot projections
        fig = plt.figure(figsize=(8, 8))
        gs = GridSpec(2, 2, width_ratios=[1, 0.075], height_ratios=[1, 0.075], )
        # original
        ax = fig.add_subplot(gs[0, 0])
        curr_img = image_stack[:, :, 0]
        if log_scale:
            curr_img = log(curr_img)
        im = ax.imshow(curr_img, origin='lower')
        ticks = [0, int(N/4.0), int(N/2.0), int(N/4.0*3), int(N-1)]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_title('Slice Viewer (log scale: {}) for {}'.format(log_scale, os.path.basename(mrcs)))
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="7%", pad="2%")
        cbar = fig.colorbar(im, cax=cax)  # colorbar

        # slider
        ax_slider = fig.add_subplot(gs[1, 0])
        idx_slider = Slider(ax_slider, 'index:', 0, size[2]-1, valinit=0, valfmt='%d')

        def update(val):
            idx = int(idx_slider.val)
            curr_img = image_stack[:, :, idx]
            if log_scale:
                curr_img = log(curr_img)
            im.set_data(curr_img)
            cbar.set_clim(vmin=curr_img.min(), vmax=curr_img.max())
            cbar.draw_all()
            fig.canvas.draw_idle()
        idx_slider.on_changed(update)

        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mrcs_files", help="list of mrcs files.", nargs='+')
    parser.add_argument("-l", "--log_scale", help="show image in log scale.",
                        action="store_true")
    parser.add_argument("-r", "--plot_randomly", help="plot image with random index.",
                        action="store_true")
    args = parser.parse_args()

    log_scale = args.log_scale
    mrcs_files = args.mrcs_files
    plot_randomly = args.plot_randomly
    print('mrcs_files:', mrcs_files)
    print('log_scale:', log_scale)
    print('plot_randomly:', plot_randomly)
    if plot_randomly:
        plot_projs(mrcs_files, log_scale=log_scale)
    else:
        plot_projs_with_slider(mrcs_files, log_scale=log_scale)
