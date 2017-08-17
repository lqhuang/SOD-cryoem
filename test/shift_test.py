import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))

import numpy as np
import matplotlib.pyplot as plt

from cryoio import mrc
import density, cryoops
import geometry
from quadrature import healpix
import cryoem

import pyximport; pyximport.install(
    setup_args={"include_dirs": np.get_include()}, reload_support=True)
import sincint


def shift_vis(model):
    N = model.shape[0]
    rad = 0.8
    kernel = 'lanczos'
    kernsize = 4

    xy, trunc_xy, truncmask = geometry.gencoords(N, 2, rad, True)
    N_T = trunc_xy.shape[0]
    premult = cryoops.compute_premultiplier(N, kernel=kernel, kernsize=kernsize)
    TtoF = sincint.gentrunctofull(N=N, rad=rad)

    fM = density.real_to_fspace(model)
    prefM = density.real_to_fspace(premult.reshape(
            (1, 1, -1)) * premult.reshape((1, -1, 1)) * premult.reshape((-1, 1, 1)) * model)

    pt = np.random.randn(3)
    pt /= np.linalg.norm(pt)
    psi = 2 * np.pi * np.random.rand()
    ea = geometry.genEA(pt)[0]
    ea[2] = psi
    print('project model for Euler angel: ({:.2f}, {:.2f}, {:.2f}) degree'.format(*np.rad2deg(ea)))

    rot_matrix = geometry.rotmat3D_EA(*ea)[:, 0:2]
    slop = cryoops.compute_projection_matrix([rot_matrix], N, kernel, kernsize, rad, 'rots')
    # trunc_slice = slop.dot(prefM.reshape((-1,)))
    trunc_slice = cryoem.getslices(prefM, slop)
    fourier_slice = TtoF.dot(trunc_slice).reshape(N, N)
    real_proj = density.fspace_to_real(fourier_slice)

    fig, axes = plt.subplots(4, 4, figsize=(12.8, 8))

    im_real = axes[0, 0].imshow(real_proj)
    im_fourier = axes[1, 0].imshow(np.log(np.abs(fourier_slice)))

    for i, ax in enumerate(axes[:, 1:].T):
        shift = np.random.randn(2) * (N/4.0)
        S = cryoops.compute_shift_phases(shift.reshape(1,2), N, rad)[0]

        shift_trunc_slice = S * trunc_slice
        shift_fourier_slice = TtoF.dot(shift_trunc_slice).reshape(N, N)
        shift_real_proj = density.fspace_to_real(shift_fourier_slice)
        
        ax[0].imshow(shift_real_proj)
        ax[1].imshow(np.log(np.abs(shift_fourier_slice)))
        ax[2].imshow(np.log(shift_fourier_slice.real))
        ax[3].imshow(np.log(shift_fourier_slice.imag))

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # M = mrc.readMRC('./particle/EMD-6044.mrc')
    # M = M[:124, :124, :124]
    # mrc.writeMRC('./particle/EMD-6044-cropped.mrc', M, psz=3.0)
    print(sys.argv)
    M = mrc.readMRC(sys.argv[1])
    shift_vis(M)
