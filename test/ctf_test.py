from __future__ import print_function, division

import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))

import numpy as np
from matplotlib import pyplot as plt

from cryoio import mrc
from cryoio import ctf
import geometry
import cryoops as coops

from correlation_benchmark import vis_real_space_comparison, calc_difference
from notimplemented.correlation import calc_angular_correlation

cython_build_dirs = os.path.expanduser('~/.pyxbld/angular_correlation')
import pyximport; pyximport.install(build_dir=cython_build_dirs, setup_args={"include_dirs": np.get_include()}, reload_support=True)
import sincint


def envelope_vis(N=128, rad=1):
    trunc_xy = geometry.gencoords(N, 2, rad)
    psize = 2.8
    bfactor = 500

    freqs = np.sqrt(np.sum(trunc_xy**2, axis=1)) / (psize * N)
    envelope = ctf.envelope_function(freqs, bfactor)

    fig, ax = plt.subplots()
    ax.plot(freqs, envelope, '.', label='bfactor: %d' % bfactor)
    ax.legend(frameon=False)
    ax.set_title('envelope')
    plt.show()


def ctf_vis():
    import cryoops as coops

    fcoords = np.random.randn(10, 2)
    rots = np.array([np.pi / 3.0])
    R = np.array([[np.cos(rots), -np.sin(rots)],
                 [np.sin(rots), np.cos(rots)]]).reshape((2, 2))
    rotfcoords = np.dot(fcoords, R.T)
    akv = 200
    wgh = 0.07
    cs = 2.0
    df1, df2, angast = 44722, 49349, 45.0 * (np.pi / 180.0)
    dscale = 1.0
    v1 = ctf.compute_ctf(fcoords, rots, akv, cs, wgh, df1,
                     df2, angast, dscale).reshape((-1,))
    v2 = ctf.compute_ctf(rotfcoords, None, akv, cs, wgh, df1,
                     df2, angast, dscale).reshape((-1,))

    # This being small confirms that using the rots parameter is equivalent to
    # rotating the coordinates
    print(np.abs(v1 - v2).max())

    # N = 512
    N = 128
    psz = 5.6
    rad = 0.25
    fcoords = geometry.gencoords(N, 2, rad) / (N * psz)
    ctf1_rot = ctf.compute_full_ctf(
        rots, N, psz, akv, cs, wgh, df1, df2, angast, dscale, None)
    ctf2_full = ctf.compute_full_ctf(
        None, N, psz, akv, cs, wgh, df1, df2, angast, dscale, None)

    print(ctf1_rot.shape)
    
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(ctf1_rot.reshape(N, N))
    ax[1].imshow(ctf2_full.reshape(N, N))
    ax[0].set_title('inplane-rotation: 0')
    ax[1].set_title('inplane-rotation: 60 degree')

    fig, ax = plt.subplots()
    x = np.arange(int(-N/2.0), int(N/2.0))
    ax.plot(x, ctf1_rot.reshape(N, N)[int(N/2), :], label='0 degree')
    ax.plot(x, ctf2_full.reshape(N, N)[int(N/2), :], label='60 degree')
    ax.set_xlabel('pixel')
    ax.set_ylabel('CTF value')
    ax.legend()

    plt.show()

    P_rot = coops.compute_inplanerot_matrix(rots, N, 'lanczos', 10, rad)
    ctf2_rot = P_rot.dot(ctf2_full).reshape((-1,))

    P_null = coops.compute_inplanerot_matrix(np.array([0]), N, 'linear', 2, rad)
    ctf1_rot = P_null.dot(ctf1_rot).reshape((-1,))

    print(fcoords.shape)
    print(P_rot.shape)
    print(ctf2_rot.shape)

    roterr = ctf1_rot - ctf2_rot
    relerr = np.abs(roterr) / np.maximum(np.abs(ctf1_rot), np.abs(ctf2_rot))

    # This being small confirms that compute_inplane_rotmatrix and rots use
    # the same rotation convention
    print(relerr.max(), relerr.mean())


def ctf_correlation_benchmark(N=128):
    akv = 200
    wgh = 0.07
    cs = 2.0
    df1, df2, angast = 44722, 49349, 45.0 * (np.pi / 180.0)
    dscale = 1.0
    psz = 5.6
    rad = 0.99

    rots = np.arange(0, 2 * np.pi, 2*np.pi/360)

    ctf_map = ctf.compute_full_ctf(
        None, N, psz, akv, cs, wgh, df1, df2, angast, dscale, None)
    ctf_rots = ctf.compute_full_ctf(
        rots, N, psz, akv, cs, wgh, df1, df2, angast, dscale, None)
    
    ctf_map = ctf_map.reshape(N, N)
    ctf_rots = ctf_rots.reshape(N, N, -1).transpose((2, 0, 1))

    FtoT = sincint.genfulltotrunc(N=N, rad=rad)
    TtoF = FtoT.T

    corr_ctf_map = TtoF.dot(calc_angular_correlation(
        FtoT.dot(ctf_map.flatten()), N=N, rad=rad
        )).reshape(N, N)

    corr_ctf_rots = np.zeros_like(ctf_rots, dtype=ctf_rots.dtype)
    for i, curr_ctf in enumerate(ctf_rots):
        corr_ctf_rots[i] = TtoF.dot(calc_angular_correlation(
            FtoT.dot(curr_ctf.flatten()), N=N, rad=rad
            )).reshape(N, N)
    
    difference = calc_difference(ctf_map, ctf_rots, only_real=True)
    corr_difference = calc_difference(corr_ctf_map, corr_ctf_rots, only_real=True)

    vis_real_space_comparison(ctf_rots, corr_ctf_rots, difference, corr_difference,
                              original_img=ctf_map, original_corr_img=corr_ctf_map,
                              save_animation=True, animation_name='ctf_correlation_benchmark')


if __name__ == '__main__':
    envelope_vis()
    ctf_vis()
    ctf_correlation_benchmark()
