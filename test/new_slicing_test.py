from __future__ import print_function, division

import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))

import numpy as np
from matplotlib import pyplot as plt

import geometry
from cryoio import mrc
import cryoops
import cryoem
from quadrature import SK97Quadrature

import pyximport; pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=True)
import sincint


if __name__ == '__main__':
    psize = 2.8 * 6
    freq = 0.05
    rad = 0.5 * 2.0 * psize
    mask_freq = 0.01
    mask_rad = mask_freq * 2.0 * psize

    fM = mrc.readMRC('particle/1AON_fourier.mrc')
    N = fM.shape[0]

    TtoF = sincint.gentrunctofull(N=N, rad=rad, mask_rad=mask_rad)

    theta = np.arange(0, 2*np.pi, 2*np.pi/12)
    degree_R, resolution_R = SK97Quadrature.compute_degree(N, 0.3, 1.0)
    dirs, weights = SK97Quadrature.get_quad_points(degree_R, None)
    Rs = np.vstack([geometry.rotmat3D_dir(vec)[:, 0:2].reshape((1, 3, 2)) for vec in dirs])

    N_R = dirs.shape[0]
    N_I = theta.shape[0]
    N_T = TtoF.shape[1]

    # generate slicing operators
    dir_slice_interp = {'projdirs': dirs, 'N': N, 'kern': 'lanczos', 'kernsize': 6, 
                        'projdirtype': 'dirs', 'onlyRs': True, 'rad': rad, 'sym': None}  # 'zeropad': 0, 'dopremult': True
    R_slice_interp = {'projdirs': Rs, 'N': N, 'kern': 'lanczos', 'kernsize': 6,
                    'projdirtype': 'rots', 'onlyRs': True, 'rad': rad, 'sym': None}  # 'zeropad': 0, 'dopremult': True
    inplane_interp = {'thetas': theta, 'N': N, 'kern': 'lanczos', 'kernsize': 6,
                    'onlyRs': True, 'rad': rad}  # 'zeropad': 1, 'dopremult': True
    inplane_interp['N_src'] = N  # zp_N

    slice_ops = cryoops.compute_projection_matrix(**dir_slice_interp)
    inplane_ops = cryoops.compute_inplanerot_matrix(**inplane_interp)

    # generate slices and inplane-rotated slices
    slices_sampled = cryoem.getslices_interp(fM, slice_ops, rad, mask_rad=mask_rad).reshape((N_R, N_T))
    curr_img = TtoF.dot(slices_sampled[0]).reshape(N, N)
    rotd_sampled = cryoem.getslices_interp(curr_img, inplane_ops, rad, mask_rad=mask_rad).reshape((N_I, N_T))

    ## plot figures
    fig, axes = plt.subplots(3, 4, figsize=(9.6, 7.2))
    for i, ax in enumerate(axes.flatten()):
        img = TtoF.dot(slices_sampled[i]).reshape(N, N)
        ax.imshow(img, origin='lower')
    fig.suptitle('slices_sampled')

    fig, axes = plt.subplots(3, 4, figsize=(9.6, 7.2))
    for i, ax in enumerate(axes.flatten()):
        img = TtoF.dot(slices_sampled[i]).reshape(N, N)
        ax.imshow(img, origin='lower')
    fig.suptitle('rotd_sampled')
    plt.show()
