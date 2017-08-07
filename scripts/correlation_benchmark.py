from __future__ import print_function, division

import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
import time

import numpy as np
from scipy.ndimage.interpolation import rotate as imrotate
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from cryoio import mrc
import density
import geometry
from notimplemented import correlation, projector


def calc_difference(single_slice, slices, only_real=False):
    num_slices = slices.shape[0]
    img_shape = [-1] + [np.prod(slices.shape[1:])]
    reps = [num_slices] + [1] * (slices.ndim - 1)
    
    broadcast_slice = np.tile(single_slice, reps)

    difference_real = (1 - broadcast_slice.real / slices.real) ** 2
    nan_inf = np.logical_xor(np.isnan(difference_real), np.isinf(difference_real))
    difference_real[nan_inf] = 0
    if only_real:
        return difference_real.reshape(img_shape).mean(axis=1)
    else:
        difference_imag = (1 - broadcast_slice.imag / slices.imag) ** 2
        nan_inf = np.logical_xor(np.isnan(difference_imag), np.isinf(difference_imag))
        difference_imag[nan_inf] = 0
        return difference_real.reshape(img_shape).mean(axis=1), \
               difference_imag.reshape(img_shape).mean(axis=1)


def gen_EAs_randomly(num_inplane_angles=360, ea=None):
    if ea is None:
        pt = np.random.randn(3)
        pt /= np.linalg.norm(pt)
        ea = geometry.genEA(pt)[0]
    euler_angles = np.vstack([np.repeat(ea[0], num_inplane_angles),
                              np.repeat(ea[1], num_inplane_angles),
                              np.linspace(0, 2*np.pi, num_inplane_angles, endpoint=False)]).T
    return ea, euler_angles


def imshow(fig, gs_view, img, title_str, xticks=True, yticks=True):
    ax = fig.add_subplot(gs_view)
    len_y, len_x = img.shape
    im = ax.imshow(img, origin='lower')
    if xticks is False:
        ax.set_xticks([])
    else:
        ax.set_xticks([0, int(len_x/4.0), int(len_x/2.0), int(len_x/4.0*3), len_x-1])
    if yticks is False:
        ax.set_yticks([])
    else:
        ax.set_yticks([0, int(len_y/4.0), int(len_y/2.0), int(len_y/4.0*3), len_y-1])
    if img.shape[0] != img.shape[1]:
        ax.set_aspect('auto')
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="7%", pad="2%")
    cb = fig.colorbar(im, cax=cax)
    if (title_str is not None) and (title_str is not []):
        title = ax.set_title(title_str)
    else:
        title = None
    return ax, im, cb, title


def plot_diff(fig, gs_view, difference, corr_difference, init_idx):
    num_imgs = len(difference)

    ax_diff = fig.add_subplot(gs_view)
    ax_diff.plot(range(num_imgs), difference, label='original image')
    ax_diff.plot(range(num_imgs), corr_difference, label='angular correlation')
    ax_diff.legend(frameon=False)
    ax_diff.set_xlabel('inplane rotation (degree)')
    ax_diff.set_ylabel('difference')
    
    pointer, = ax_diff.plot(init_idx, difference[init_idx], 'r.', markersize=7)
    corr_pointer, = ax_diff.plot(init_idx, corr_difference[init_idx], 'r.', markersize=7)
    comp, = ax_diff.plot([init_idx, init_idx], ax_diff.get_ylim(), 'r--')
    # per = np.percentile(difference, 90)
    # ax_diff.set_ylim([-per/3, per])

    return ax_diff, pointer, corr_pointer, comp


def plot_two_diffs(fig, gs_view, second_gs_view, difference, corr_difference, init_idx):
    num_imgs = len(difference)

    ax_diff, pointer, corr_pointer, comp = plot_diff(fig, gs_view, difference, corr_difference, init_idx)

    ax_corr_diff = fig.add_subplot(second_gs_view)
    ax_corr_diff.plot(range(num_imgs), corr_difference, color='#ff7f0e', label='angular correlation')
    ax_corr_diff.legend(frameon=False)
    ax_corr_diff.set_xlabel('inplane rotation (degree)')
    another_corr_pointer, = ax_corr_diff.plot(init_idx, corr_difference[init_idx], 'r.', markersize=7)

    return ax_diff, pointer, corr_pointer, comp, ax_corr_diff, another_corr_pointer


def vis_real_space(imgs, difference, original_img=None):
    num_imgs = imgs.shape[0]
    assert imgs.shape[0] == len(difference)
    if original_img is None:
        original_img = imgs[0]

    # plot projections
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(3, 2, height_ratios=[1, 0.075, 0.8])
    # original
    ax0, im0, cb0, title0 = imshow(fig, gs[0, 0], original_img, 'original image')
    # rotated
    init_idx = np.random.randint(num_imgs)
    ax1, im1, cb1, title1 = imshow(fig, gs[0, 1], imgs[init_idx],
        'index of rotated images: {}'.format(init_idx), yticks=False)
    # slider
    ax_slider = fig.add_subplot(gs[1, 0:])
    idx_slider = Slider(ax_slider, 'index slider:', 0, num_imgs-1, valinit=init_idx)

    ax_diff = fig.add_subplot(gs[2, :])
    ax_diff.plot(range(num_imgs), difference)
    pointer, = ax_diff.plot(init_idx, difference[init_idx], 'r.', markersize=7)
    # per = np.percentile(difference, 95)
    # ax_diff.set_ylim([min(difference), per])

    def update(val):
        idx = int(idx_slider.val)
        curr_img = imgs[idx]
        im1.set_data(curr_img)
        cb1.draw_all()
        title1.set_text('index of rotated images: {}'.format(idx))
        pointer.set_data(idx, difference[idx])
        fig.canvas.draw_idle()
    idx_slider.on_changed(update)
    
    plt.show()


def vis_real_space_comparison(imgs, corr_imgs, difference, corr_difference,
                              original_img=None, original_corr_img=None):
    num_imgs = imgs.shape[0]
    assert imgs.shape[0] == len(difference)
    assert corr_imgs.shape[0] == len(corr_difference)

    if original_img is None:
        original_img = imgs[0]
    if original_corr_img is None:
        original_corr_img = corr_imgs[0]

    if original_corr_img.shape == original_img.shape:
        xticks = False
        yticks = False
    else:
        xticks = True
        yticks = True

    # plot projections
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(3, 4, height_ratios=[1, 0.075, 0.8])
    
    init_idx = np.random.randint(num_imgs)
    # none correlation
    # original
    ax0, im0, cb0, title0 = imshow(fig, gs[0, 0], original_img,
        'original image', xticks=True, yticks=True)
    # rotated
    ax1, im1, cb1, title1 = imshow(fig, gs[0, 1], imgs[init_idx],
        'index of rotated images: {}'.format(init_idx), yticks=False)
    # correlation
    # original
    ax2, im2, cb2, title2 = imshow(fig, gs[0, 2], original_corr_img,
        'corr image for original image', xticks=True, yticks=yticks)
    # rotated
    init_idx = np.random.randint(num_imgs)
    ax3, im3, cb3, title3 = imshow(fig, gs[0, 3], corr_imgs[init_idx],
        'index of rotated corr images: {}'.format(init_idx), yticks=False)

    # slider
    ax_slider = fig.add_subplot(gs[1, :])
    idx_slider = Slider(ax_slider, 'index slider:', 0, num_imgs-1, valinit=init_idx)

    # difference comparison    
    ax_diff, pointer, corr_pointer, comp, ax_corr_diff, another_corr_pointer = \
        plot_two_diffs(fig, gs[2, 0:2], gs[2, 2:4], difference, corr_difference, init_idx)


    def update(val):
        idx = int(idx_slider.val)

        curr_img = imgs[idx]
        curr_corr_img = corr_imgs[idx]

        im1.set_data(curr_img)
        cb1.draw_all()
        title1.set_text('index of rotated images: {}'.format(idx))
        im3.set_data(curr_corr_img)
        cb3.draw_all()
        title3.set_text('index of rotated corr images: {}'.format(idx))
        
        pointer.set_data(idx, difference[idx])
        corr_pointer.set_data(idx, corr_difference[idx])
        another_corr_pointer.set_data(idx, corr_difference[idx])
        comp.set_data([idx, idx], ax_diff.get_ylim())
        
        fig.canvas.draw_idle()
    idx_slider.on_changed(update)

    plt.show()


def vis_fourier_space_comparison(imgs, corr_imgs,
                                 difference_real, difference_imag,
                                 corr_difference_real, corr_difference_imag,
                                 original_img=None, original_corr_img=None):
    imgs.real = np.log(imgs.real)
    corr_imgs.real = np.log(corr_imgs.real)    
    imgs.imag = np.log(imgs.imag)
    corr_imgs.imag = np.log(corr_imgs.imag)

    num_imgs = imgs.shape[0]
    assert imgs.shape[0] == len(difference_real)
    assert corr_imgs.shape[0] == len(corr_difference_real)

    if original_img is None:
        original_img = imgs[0]
    else:
        original_img.real = np.log(original_img.real)
        original_img.imag = np.log(original_img.imag)
    if original_corr_img is None:
        original_corr_img = corr_imgs[0]
    else:
        original_corr_img.real = np.log(original_corr_img.real)
        original_corr_img.imag = np.log(original_corr_img.imag)

    if original_corr_img.shape == original_img.shape:
        xticks = False
        yticks = False
    else:
        xticks = True
        yticks = True

    # plot projections
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(5, 4, height_ratios=[1, 1, 0.075, 0.4, 0.4])
    
    init_idx = np.random.randint(num_imgs)

    # real part
    # none correlation
    # original
    ax0, im0, cb0, title0 = imshow(fig, gs[0, 0], original_img.real,
        'original image', xticks=False)
    # rotated
    ax1, im1, cb1, title1 = imshow(fig, gs[0, 1], imgs[init_idx].real,
        'index of rotated images: {}'.format(init_idx), xticks=False, yticks=False)
    # correlation
    # original
    ax2, im2, cb2, title2 = imshow(fig, gs[0, 2], original_corr_img.real,
        'corr image for original image', xticks=False, yticks=yticks)
    # rotated
    ax3, im3, cb3, title3 = imshow(fig, gs[0, 3], corr_imgs[init_idx].real, 
        'index of rotated corr images: {}'.format(init_idx), xticks=False, yticks=False)

    # imag part
    # original
    ax0_imag, im0_imag, cb0_imag, title0_imag = imshow(fig, gs[1, 0], original_img.imag,
        None, xticks=True, yticks=True)
    # rotated
    ax1_imag, im1_imag, cb1_imag, title1_imag = imshow(fig, gs[1, 1], imgs[init_idx].imag,
        None, yticks=False)
    # correlation
    # original
    ax2_imag, im2_imag, cb2_imag, title2_imag = imshow(fig, gs[1, 2], original_corr_img.imag,
        None, xticks=True, yticks=yticks)
    # rotated
    ax3_imag, im3_imag, cb3_imag, title3_imag = imshow(fig, gs[1, 3], corr_imgs[init_idx].imag,
        None, yticks=False)

    # slider
    ax_slider = fig.add_subplot(gs[2, :])
    idx_slider = Slider(ax_slider, 'index slider:', 0, num_imgs-1, valinit=init_idx)

    # difference comparison
    # real part
    ax_diff_real, pointer_real, corr_pointer_real, comp_real, ax_corr_diff_real, another_corr_pointer_real = \
        plot_two_diffs(fig, gs[3, 0:2], gs[3, 2:4], difference_real, corr_difference_real, init_idx)
    # imag part
    ax_diff_imag, pointer_imag, corr_pointer_imag, comp_imag, ax_corr_diff_imag, another_corr_pointer_imag = \
        plot_two_diffs(fig, gs[4, 0:2], gs[4, 2:4], difference_imag, corr_difference_imag, init_idx)

    def update(val):
        idx = int(idx_slider.val)

        curr_img_real = imgs[idx].real
        curr_corr_img_real = corr_imgs[idx].real
        curr_img_imag = imgs[idx].imag
        curr_corr_img_imag = corr_imgs[idx].imag

        im1.set_data(curr_img_real)
        cb1.draw_all()
        title1.set_text('index of rotated images: {}'.format(idx))
        im3.set_data(curr_corr_img_real)
        cb3.draw_all()
        title3.set_text('index of rotated corr images: {}'.format(idx))

        im1_imag.set_data(curr_img_imag)
        cb1_imag.draw_all()
        im3_imag.set_data(curr_corr_img_imag)
        cb3_imag.draw_all()

        pointer_real.set_data(idx, difference_real[idx])
        corr_pointer_real.set_data(idx, corr_difference_real[idx])
        another_corr_pointer_real.set_data(idx, corr_difference_real[idx])
        comp_real.set_data([idx, idx], ax_diff_real.get_ylim())

        pointer_imag.set_data(idx, difference_imag[idx])
        corr_pointer_imag.set_data(idx, corr_difference_imag[idx])
        another_corr_pointer_imag.set_data(idx, corr_difference_imag[idx])
        comp_imag.set_data([idx, idx], ax_diff_imag.get_ylim())

        fig.canvas.draw_idle()
    idx_slider.on_changed(update)
    
    plt.show()


def realspace_benchmark(model, num_inplane_angles=360, calc_corr_img=False):

    N = M.shape[0]
    ea, euler_angles = gen_EAs_randomly(num_inplane_angles)

    proj = projector.project(model, ea)
    rot_projs = projector.project(M, euler_angles)
    if calc_corr_img:
        proj = correlation.get_corr_img(proj)
        rot_projs = correlation.get_corr_imgs(rot_projs)

    diff = calc_difference(proj, rot_projs)

    vis_real_space(rot_projs, diff, original_img=proj)


def realspace_benchmark_new_algorithm(model, num_inplane_angles=360, rad=0.6, calc_corr_img=False):

    N = M.shape[0]
    ea, euler_angles = gen_EAs_randomly(num_inplane_angles)

    sl = projector.project(model, ea, rad=rad, truncate=True)
    rot_slice = projector.project(M, euler_angles, rad=rad, truncate=True)
    full_slice = projector.trunc_to_full(sl, N, rad)
    full_rot_slices = projector.trunc_to_full(rot_slice, N, rad)

    proj = density.fspace_to_real(full_slice)
    full_rot_projs = np.zeros_like(full_rot_slices, dtype=density.real_t)
    for i, fs in enumerate(full_rot_slices):
        full_rot_projs[i] = density.fspace_to_real(fs)
    
    trunc_proj = projector.full_to_trunc(proj, rad)
    trunc_rot_projs = projector.full_to_trunc(full_rot_projs, rad)

    if calc_corr_img:
        trunc_proj = correlation.calc_angular_correlation(trunc_proj, N, rad)
        trunc_rot_projs = correlation.calc_angular_correlation(trunc_rot_projs, N, rad)

    diff = calc_difference(trunc_proj, trunc_rot_projs, only_real=True)

    vis_real_space(projector.trunc_to_full(trunc_rot_projs, N, rad), diff,
        original_img=projector.trunc_to_full(trunc_proj, N, rad))


def realspace_benchmark_comparison(model, num_inplane_angles=360):

    N = M.shape[0]
    ea, euler_angles = gen_EAs_randomly(num_inplane_angles)

    proj = projector.project(model, ea)
    rot_projs = projector.project(M, euler_angles)
    corr_proj = correlation.get_corr_img(proj)
    corr_rot_projs = correlation.get_corr_imgs(rot_projs)

    diff = calc_difference(proj, rot_projs, only_real=True)
    corr_diff = calc_difference(corr_proj, corr_rot_projs, only_real=True)

    vis_real_space_comparison(rot_projs, corr_rot_projs, diff, corr_diff,
        original_img=proj, original_corr_img=corr_proj)


def realspace_benchmark_new_algorithm_comparison(model, num_inplane_angles=360, rad=0.6):

    N = M.shape[0]
    ea, euler_angles = gen_EAs_randomly(num_inplane_angles)

    sl = projector.project(model, ea, rad=rad, truncate=True)
    rot_slice = projector.project(M, euler_angles, rad=rad, truncate=True)
    full_slice = projector.trunc_to_full(sl, N, rad)
    full_rot_slices = projector.trunc_to_full(rot_slice, N, rad)

    proj = density.fspace_to_real(full_slice)
    full_rot_projs = np.zeros_like(full_rot_slices, dtype=density.real_t)
    for i, fs in enumerate(full_rot_slices):
        full_rot_projs[i] = density.fspace_to_real(fs)
    
    trunc_proj = projector.full_to_trunc(proj, rad)
    trunc_rot_projs = projector.full_to_trunc(full_rot_projs, rad)
    corr_trunc_proj = correlation.calc_angular_correlation(trunc_proj, N, rad)
    corr_trunc_rot_projs = correlation.calc_angular_correlation(trunc_rot_projs, N, rad)

    diff = calc_difference(trunc_proj, trunc_rot_projs, only_real=True)
    corr_diff = calc_difference(corr_trunc_proj, corr_trunc_rot_projs, only_real=True)

    vis_real_space_comparison(
        full_rot_projs,
        projector.trunc_to_full(corr_trunc_rot_projs, N, rad),
        diff, corr_diff,
        original_img=proj,
        original_corr_img=projector.trunc_to_full(corr_trunc_proj, N, rad))


def fourierspace_benchmark_comparison(model, num_inplane_angles=360, modulus=False):

    N = M.shape[0]
    ea, euler_angles = gen_EAs_randomly(num_inplane_angles)

    proj = projector.project(model, ea)
    one_slice = density.real_to_fspace(proj)
    rot_projs = projector.project(M, euler_angles)
    rot_slices = np.zeros_like(rot_projs, dtype=density.complex_t)
    for i, pj in enumerate(rot_projs):
        rot_slices[i] = density.real_to_fspace(pj)
    if modulus:
        one_slice = np.abs(one_slice)
        rot_slices = np.abs(rot_slices)

    if modulus:
        corr_slice = correlation.get_corr_img(one_slice)
        corr_rot_slices = correlation.get_corr_imgs(rot_slices)

        diff = calc_difference(one_slice, rot_slices, only_real=True)
        corr_diff = calc_difference(corr_slice, corr_rot_slices, only_real=True)
        
        vis_real_space_comparison(
            rot_slices,
            corr_rot_slices,
            diff, corr_diff,
            original_img=one_slice,
            original_corr_img=corr_slice
            )
    else:
        corr_slice = np.zeros((int(N/2.0), 360), dtype=density.complex_t)
        corr_slice.imag = correlation.get_corr_img(one_slice.imag)
        corr_slice.real = correlation.get_corr_img(one_slice.real)
        corr_rot_slices = np.zeros((num_inplane_angles, int(N/2.0), 360), dtype=density.complex_t)
        corr_rot_slices.real = correlation.get_corr_imgs(rot_slices.real)
        corr_rot_slices.imag = correlation.get_corr_imgs(rot_slices.imag)

        diff_real, diff_imag = calc_difference(one_slice, rot_slices)
        corr_diff_real, corr_diff_imag = calc_difference(corr_slice, corr_rot_slices)

        vis_fourier_space_comparison(
            rot_slices,
            corr_rot_slices,
            diff_real, diff_imag,
            corr_diff_real, corr_diff_imag,
            original_img=one_slice,
            original_corr_img=corr_slice
            )


def fourierspace_benchmark_new_algorithm_comparison(model, num_inplane_angles=360, rad=0.6, modulus=False):

    N = M.shape[0]
    ea, euler_angles = gen_EAs_randomly(num_inplane_angles)

    one_slice = projector.project(model, ea, rad=rad, truncate=True)
    rot_slices = projector.project(M, euler_angles, rad=rad, truncate=True)
    if modulus:
        one_slice = np.abs(one_slice)
        rot_slices = np.abs(rot_slices)

    corr_slice = correlation.calc_angular_correlation(one_slice, N, rad)
    corr_rot_slices = correlation.calc_angular_correlation(rot_slices, N, rad)

    if modulus:
        diff = calc_difference(one_slice, rot_slices, only_real=True)
        corr_diff = calc_difference(corr_slice, corr_rot_slices, only_real=True)
        
        vis_real_space_comparison(
            np.log(projector.trunc_to_full(rot_slices, N, rad)),
            np.log(projector.trunc_to_full(corr_rot_slices, N, rad)),
            diff, corr_diff,
            original_img=np.log(projector.trunc_to_full(one_slice, N, rad)),
            original_corr_img=np.log(projector.trunc_to_full(corr_slice, N, rad))
            )
    else:
        diff_real, diff_imag = calc_difference(one_slice, rot_slices)
        corr_diff_real, corr_diff_imag = calc_difference(corr_slice, corr_rot_slices)

        vis_fourier_space_comparison(
            projector.trunc_to_full(rot_slices, N, rad),
            projector.trunc_to_full(corr_rot_slices, N, rad),
            diff_real, diff_imag,
            corr_diff_real, corr_diff_imag,
            original_img=projector.trunc_to_full(one_slice, N, rad),
            original_corr_img=projector.trunc_to_full(corr_slice, N, rad)
            )


    def speed_benchmark_comparison(M):
        pass


if __name__ == '__main__':
    M = mrc.readMRC('/Users/lqhuang/Git/SOD-cryoem/particle/EMD-6044-cropped.mrc')
    M[M<30.4] = 0
    # M = mrc.readMRC('/Users/lqhuang/Git/SOD-cryoem/particle/1AON.mrc')

    # realspace_benchmark(M, calc_corr_img=True)
    # realspace_benchmark_comparison(M)
    # realspace_benchmark_new_algorithm(M, calc_corr_img=False)
    # realspace_benchmark_new_algorithm_comparison(M)

    fourierspace_benchmark_new_algorithm_comparison(M, rad=0.6)
    # fourierspace_benchmark_new_algorithm_comparison(M, modulus=True)

    # fourierspace_benchmark_comparison(M)
    # fourierspace_benchmark_comparison(M, modulus=True)
