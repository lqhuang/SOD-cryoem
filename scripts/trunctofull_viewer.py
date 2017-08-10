from __future__ import print_function, division

import os
import sys
root_dir = os.path.dirname(sys.path[0])
sys.path.append(root_dir)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from cryoio import mrc
import density, cryoops
import geometry
import cryoem
from notimplemented import correlation

import pyximport; pyximport.install(
    setup_args={"include_dirs": np.get_include()}, reload_support=True)
import sincint


def demo(N=128, rad=0.5):
    TtoF = sincint.gentrunctofull(N=N, rad=rad)
    xy, trunc_xy, truncmask = geometry.gencoords(N, 2, rad, True)
    print('shape of TtoF:', TtoF.shape)
    print('slice shape:', trunc_xy.shape[0])
    trunc_slice = np.arange(trunc_xy.shape[0])
    sliced_image = TtoF.dot(trunc_slice).reshape(N, N)

    trunc_xy_idx = np.int_(trunc_xy + int(N/2))
    # Compare speed for getting slices in this way
    new_trunc_slice = sliced_image[trunc_xy_idx[:, 0], trunc_xy_idx[:, 1]]
    print('error:', sum(trunc_slice - new_trunc_slice))
    pol_trunc_xy = correlation.cart2pol(trunc_xy)

    # inside of rad
    # sort trunc_xy coordinates 
    sorted_idx = np.lexsort((pol_trunc_xy[:, 1], pol_trunc_xy[:, 0]))  # lexsort; first, sort rho; second, sort theta
    sorted_pol_trunc_xy = pol_trunc_xy[sorted_idx]
    # reconstuct sorted coordinates into original state
    reco_pol_trunc_xy = sorted_pol_trunc_xy[sorted_idx.argsort()]
    print('error for reconstructed coordinates:', sum(correlation.pol2cart(reco_pol_trunc_xy) - trunc_xy))
    reco_trunc_slice = trunc_slice[sorted_idx.argsort()]
    bingo_sliced_image = TtoF.dot(reco_trunc_slice).reshape(N, N)

    # outside of rad
    xy_outside = xy[~truncmask]
    sliced_image_outside_rad = np.zeros((N, N))
    sliced_image_outside_rad[~truncmask.reshape(N, N)] = np.arange(xy_outside.shape[0])
    pol_xy_outside = correlation.cart2pol(xy_outside)
    outside_sorted_idx = np.lexsort((pol_xy_outside[:, 1], pol_xy_outside[:, 0]))  # lexsort; first, sort rho; second, sort theta
    sorted_pol_xy_outside = pol_xy_outside[outside_sorted_idx]
    reco_pol_xy_outside = np.arange(xy_outside.shape[0])[outside_sorted_idx.argsort()]

    bingo_sliced_image_outside_rad = np.zeros((N, N))
    bingo_sliced_image_outside_rad[~truncmask.reshape(N, N)] = reco_pol_xy_outside

    fig, axes = plt.subplots(2, 2)
    ax = axes.flatten()
    ax[0].imshow(sliced_image)
    ax[1].imshow(bingo_sliced_image)
    ax[2].imshow(sliced_image_outside_rad)
    ax[3].imshow(bingo_sliced_image_outside_rad)
    plt.show()


def compare_interpolation(N=128, rad=1):
    _, trunc_xy, _ = geometry.gencoords(N, 2, rad, True)
    pol_trunc_xy = correlation.cart2pol(trunc_xy)
    sorted_idx = np.lexsort((pol_trunc_xy[:, 1], pol_trunc_xy[:, 0]))  # lexsort; first, sort rho; second, sort theta
    sorted_pol_trunc_xy = pol_trunc_xy[sorted_idx]

    interpolation = ['none', 'nearest', 'nearest_decimal_1', 'nearest_half']
    fig, ax = plt.subplots(nrows=len(interpolation), sharex=True)
    # fig, ax = plt.subplots()

    def round_to(n, precision):
        # correction = 0.5 if n >= 0 else -0.5
        correction = np.ones_like(n) * 0.5
        correction[n < 0] = -0.5
        return np.int_(n / precision + correction) * precision

    def round_half(n):
        return round_to(n, 0.5)

    def get_ip_func(ip_method):
        if 'none' == ip_method.lower():
            return lambda x: x
        elif 'nearest' == ip_method.lower():
            return np.round
        elif 'nearest_decimal_1' == ip_method.lower():
            return lambda x: np.round(x, 1)
        elif 'nearest_half' == ip_method.lower():
            return round_half
        else:
            raise ValueError('please input correct interpolation method.')

    for i, ip in enumerate(interpolation):
        ip_func = get_ip_func(ip)
        ip_pol_xy = ip_func(sorted_pol_trunc_xy[:, 0])
        unique_value, unique_index, unique_inverse, unique_counts = np.unique(ip_pol_xy,
            return_index=True, return_inverse=True, return_counts=True)
        ax[i].plot(unique_value, unique_counts, label='interpolation: {}'.format(ip))
        ax[i].legend(frameon=False)
        ax[i].set_ylabel('counts')
    
    ax[-1].set_xlabel('radius')
    plt.show()


def correlation_trunc_example(M):
    N = M.shape[0]
    rad = 1
    proj = M.sum(axis=0)
    FtoT = sincint.genfulltotrunc(N=N, rad=rad)
    TtoF = sincint.gentrunctofull(N=N, rad=rad)
    trunc = FtoT.dot(proj.flatten())
    corr_trunc = correlation.calc_angular_correlation(trunc, N, 1)
    corr_proj = TtoF.dot(corr_trunc).reshape(N, N)
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(proj)
    ax[1].imshow(corr_proj)
    plt.show()


def view_rad_range(N=128):
    fig, axes = plt.subplots(4, 5, figsize=(12.8, 8))  # , sharex=True, sharey=True, squeeze=False)

    rad_list = np.arange(0.1, 1.1, step=0.2)
    for i, rad in enumerate(rad_list):
        TtoF = sincint.gentrunctofull(N, rad)
        xy, trunc_xy, truncmask = geometry.gencoords(N, 2, rad, True)
        N_T = trunc_xy.shape[0]
        trunc = np.arange(0, N_T)

        image = TtoF.dot(trunc).reshape(N, N)
        axes[0, i].imshow(image, origin='lower')
        axes[0, i].set_title('radius: {:.2f}'.format(rad))

        # outside of radius
        xy_outside = xy[~truncmask]
        image_outside_rad = np.zeros((N, N))
        image_outside_rad[~truncmask.reshape(N, N)] = np.arange(xy_outside.shape[0])
        axes[1, i].imshow(image_outside_rad, origin='lower')
        
        # sort trunc_xy coordinates 
        pol_trunc_xy = correlation.cart2pol(trunc_xy)
        sorted_idx = np.lexsort((pol_trunc_xy[:, 1], pol_trunc_xy[:, 0]))  # lexsort; first, sort rho; second, sort theta
        pol_trunc = trunc[sorted_idx.argsort()]
        pol_image = TtoF.dot(pol_trunc).reshape(N, N)
        axes[2, i].imshow(pol_image, origin='lower')

        # pol coordinate in outside part
        pol_xy_outside = correlation.cart2pol(xy_outside)
        outside_sorted_idx = np.lexsort((pol_xy_outside[:, 1], pol_xy_outside[:, 0]))
        pol_image_outside = np.zeros((N, N))
        pol_image_outside[~truncmask.reshape(N, N)] = np.arange(pol_xy_outside.shape[0])[outside_sorted_idx.argsort()]
        axes[3, i].imshow(pol_image_outside, origin='lower')

    for i, ax in enumerate(axes.flat):
        m, n = np.unravel_index(i, (4, 5))
        if m != 3:
            ax.set_xticks([])
        else:
            ax.set_xticks([0, int(N/4), int(N/2), int(N*3/4), int(N-1)])
        if n != 0:
            ax.set_yticks([])
        else:
            ax.set_yticks([0, int(N/4), int(N/2), int(N*3/4), int(N-1)])
    # fig.savefig('cart_coordinate_view_to_polar_coordinate_view.png', dpi=300)
    plt.show()


if __name__ == '__main__':

    compare_interpolation()

    # M = mrc.readMRC(os.path.join(root_dir, 'particle/EMD-6044-cropped.mrc'))
    # M[M<30.4] = 0
    # correlation_trunc_example(M)

    # view_rad_range()
