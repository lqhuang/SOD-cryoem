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


def premult_test(model):
    if isinstance(model, str):
        M = mrc.readMRC(model)
    elif isinstance(model, np.ndarray):
        M = model
    
    shape = np.asarray(M.shape)
    assert (shape - shape.mean()).sum() == 0
    
    N = M.shape[0]
    kernel = 'lanczos'
    ksize = 6
    rad = 0.6

    premult = cryoops.compute_premultiplier(N, kernel, ksize)
    TtoF = sincint.gentrunctofull(N=N, rad=rad)
    premulter =   premult.reshape((1, 1, -1)) \
                * premult.reshape((1, -1, 1)) \
                * premult.reshape((-1, 1, 1))
    # premulter = 1
    fM = density.real_to_fspace(premulter * M)

    pt = np.random.randn(3)
    pt /= np.linalg.norm(pt)
    psi = 2 * np.pi * np.random.rand()
    ea = geometry.genEA(pt)[0]
    ea[2] = psi

    rot_matrix = geometry.rotmat3D_EA(*ea)[:, 0:2]
    slop = cryoops.compute_projection_matrix([rot_matrix], N, kernel, ksize, rad, 'rots')
    trunc_slice = slop.dot(fM.reshape((-1,)))
    proj = density.fspace_to_real(TtoF.dot(trunc_slice).reshape(N, N))

    fig, ax = plt.subplots()
    ax.imshow(proj)
    plt.show()


if __name__ == '__main__':
    model = sys.argv[1]
    # premult_test(model)
    premult_vis()
