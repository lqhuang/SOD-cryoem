import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))

import numpy as np
import matplotlib.pyplot as plt

import cryoem
import geometry
import density
from cryoio import mrc
import cryoops

import pyximport; pyximport.install(
    setup_args={"include_dirs": np.get_include()}, reload_support=True)
import sincint


def premult_vis(N=128):
    kernel_function = ['lanczos', 'sinc']
    for kernel in kernel_function:
        fig, ax = plt.subplots()
        kernel_size = np.arange(3, 11)
        for ksize in kernel_size:
            premult = cryoops.compute_premultiplier(N, kernel, ksize)
            ax.plot(range(premult.shape[0]), premult, label='kernel size: {}'.format(ksize))
        ax.set_title('kernel function: {}'.format(kernel))
        ax.set_xlabel('N')
        ax.legend(frameon=False)
    plt.show()


def premult_test(model, kernel='lanczos', kernsize=6):
    if isinstance(model, str):
        M = mrc.readMRC(model)
    elif isinstance(model, np.ndarray):
        M = model
    
    shape = np.asarray(M.shape)
    assert (shape - shape.mean()).sum() == 0
    
    N = M.shape[0]
    rad = 0.6

    premult = cryoops.compute_premultiplier(N, kernel, kernsize)
    TtoF = sincint.gentrunctofull(N=N, rad=rad)
    premulter =   premult.reshape((1, 1, -1)) \
                * premult.reshape((1, -1, 1)) \
                * premult.reshape((-1, 1, 1))

    fM = density.real_to_fspace(M)
    prefM = density.real_to_fspace(premulter * M)

    pt = np.random.randn(3)
    pt /= np.linalg.norm(pt)
    psi = 2 * np.pi * np.random.rand()
    ea = geometry.genEA(pt)[0]
    ea[2] = psi
    print('project model for Euler angel: ({:.2f}, {:.2f}, {:.2f}) degree'.format(*np.rad2deg(ea)))

    rot_matrix = geometry.rotmat3D_EA(*ea)[:, 0:2]
    slop = cryoops.compute_projection_matrix([rot_matrix], N, kernel, kernsize, rad, 'rots')
    trunc_slice = slop.dot(fM.reshape((-1,)))
    premult_trunc_slice = slop.dot(prefM.reshape((-1,)))
    proj = density.fspace_to_real(TtoF.dot(trunc_slice).reshape(N, N))
    premult_proj = density.fspace_to_real(TtoF.dot(premult_trunc_slice).reshape(N, N))

    fig, ax = plt.subplots(1, 3, figsize=(14.4, 4.8))
    im_proj = ax[0].imshow(proj, origin='lower')
    fig.colorbar(im_proj, ax=ax[0])
    ax[0].set_title('no premulter')
    im_pre = ax[1].imshow(premult_proj, origin='lower')
    fig.colorbar(im_pre, ax=ax[1])
    ax[1].set_title('with premulter')
    im_diff = ax[2].imshow(proj - premult_proj, origin='lower')
    fig.colorbar(im_diff, ax=ax[2])
    ax[2].set_title('difference of two image')
    fig.tight_layout()    
    plt.show()


if __name__ == '__main__':
    model = sys.argv[1]
    premult_vis()
    premult_test(model)
