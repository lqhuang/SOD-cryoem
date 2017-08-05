from __future__ import print_function, division

import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))

import numpy as np
from scipy.ndimage.interpolation import rotate as imrotate
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from cryoio import mrc
import geometry
from notimplemented import correlation, projector

import pyximport; pyximport.install(
    setup_args={"include_dirs": np.get_include()}, reload_support=True)
import sincint


def vis(imgs, difference, original_img=None, ):
    num_imgs = imgs.shape[0]
    assert imgs.shape[0] == len(difference)

    # plot projections
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(3, 2, height_ratios=[1, 0.075, 0.8])
    # original
    if original_img is None:
        original_img = imgs[0]
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(original_img, origin='lower')
    ax0_divider = make_axes_locatable(ax0)
    cax0 = ax0_divider.append_axes("right", size="7%", pad="2%")
    cb0 = fig.colorbar(im0, cax=cax0)
    title0 = ax0.set_title('original image')
    # rotated
    ax1 = fig.add_subplot(gs[0, 1])
    init_idx = np.random.randint(num_imgs)
    im1 = ax1.imshow(imgs[init_idx],
                     origin='lower')
    ax1.yaxis.set_ticks([])
    ax1_divider = make_axes_locatable(ax1)
    cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
    cb1 = fig.colorbar(im1, cax=cax1)
    title1 = ax1.set_title('index for image: {}'.format(init_idx))
    # slider
    ax_slider = fig.add_subplot(gs[1, 0:])
    idx_slider = Slider(ax_slider, 'index:', 0, num_imgs-1, valinit=init_idx)

    ax_diff = fig.add_subplot(gs[2, :])
    ax_diff.plot(range(num_imgs), difference)
    pointer, = ax_diff.plot(init_idx, difference[init_idx], 'r.', markersize=5)
    # per = np.percentile(difference, 95)
    # ax_diff.set_ylim([min(difference), per])

    def update(val):
        idx = int(idx_slider.val)
        curr_img = imgs[idx]
        im1.set_data(curr_img)
        cb1.draw_all()
        title1.set_text('index for image: {}'.format(idx))
        pointer.set_data(idx, difference[idx])
        fig.canvas.draw_idle()
    idx_slider.on_changed(update)
    
    plt.show()


def realspace_benchmark(model, num_inplane_angles=360, calc_corr_img=False):

    N = M.shape[0]

    pt = np.random.randn(3)
    pt /= np.linalg.norm(pt)
    ea = geometry.genEA(pt)[0]
    euler_angles = np.vstack([np.repeat(ea[0], num_inplane_angles),
                              np.repeat(ea[1], num_inplane_angles),
                              np.linspace(0, 2*np.pi, num_inplane_angles, endpoint=False)]).T

    proj = projector.project(model, ea)
    rot_projs = projector.project(M, euler_angles)
    if calc_corr_img:
        proj = correlation.calc_corr_img(proj)
        rot_projs = correlation.get_corr_imgs(rot_projs)

    diff = np.zeros(num_inplane_angles)
    for i, img in enumerate(rot_projs):
        diff[i] = np.mean( ( 1 - img / proj ) ** 2 )

    vis(rot_projs, diff, original_img=proj)


if __name__ == '__main__':
    M = mrc.readMRC('/Users/lqhuang/Git/SOD-cryoem/particle/EMD-6044-cropped.mrc')
    M[M<30.4] = 0
    # M = mrc.readMRC('/Users/lqhuang/Git/SOD-cryoem/particle/1AON.mrc')
    
    realspace_benchmark(M, calc_corr_img=True)


