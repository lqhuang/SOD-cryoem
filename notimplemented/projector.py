from __future__ import print_function, division

import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))

import numpy as np

from cryoio import mrc
import density, cryoops
import geometry
import cryoem

cython_build_dirs = os.path.expanduser('~/.pyxbld/angular_correlation')
import pyximport; pyximport.install(
    build_dir=cython_build_dirs, setup_args={"include_dirs": np.get_include()}, reload_support=True)
import sincint


def full_to_trunc(full_samples, rad):
    """convert samples in full shape to samples in truncation"""
    dtype = full_samples.dtype
    if full_samples.ndim == 3:
        num_samples = full_samples.shape[0]
        N = full_samples.shape[1]
    elif full_samples.ndim == 2:
        N = full_samples.shape[0]
        num_samples = 1
        full_samples = (full_samples,)
    
    FtoT = sincint.genfulltotrunc(N=N, rad=rad)
    N_T = FtoT.shape[0]
    
    trunc_samples = np.zeros((num_samples, N_T), dtype=dtype)
    for i, sample in enumerate(full_samples):
        trunc_samples[i, :] = FtoT.dot(sample.flatten())

    return trunc_samples

def trunc_to_full(trunc_samples, N, rad):
    """convert truncated samples to samples in full shape"""
    num_samples = trunc_samples.shape[0]
    TtoF = sincint.gentrunctofull(N=N, rad=rad)
    
    full_samples = np.zeros((num_samples, N, N), dtype=trunc_samples.dtype)
    for i, sample in enumerate(trunc_samples):
        full_samples[i] = TtoF.dot(sample).reshape((N,N))

    if num_samples == 1:
        full_samples = full_samples.reshape((N, N))

    return full_samples


def project(model, euler_angles, rad=0.95, truncate=False):
    if isinstance(model, str):
        M = mrc.readMRC(model)
    elif isinstance(model, np.ndarray):
        M = model
    
    N = M.shape[0]
    kernel = 'lanczos'
    ksize = 6

    premult = cryoops.compute_premultiplier(N, kernel, ksize)
    TtoF = sincint.gentrunctofull(N=N, rad=rad)
    premulter =   premult.reshape((1, 1, -1)) \
                * premult.reshape((1, -1, 1)) \
                * premult.reshape((-1, 1, 1))
    # premulter = 1
    fM = density.real_to_fspace(premulter * M)

    euler_angles = euler_angles.reshape((-1, 3))
    num_projs = euler_angles.shape[0]
    if truncate:
        projs = np.zeros((num_projs, TtoF.shape[1]), dtype=fM.dtype)
    else:
        projs = np.zeros((num_projs, N, N), dtype=M.dtype)
    for i, ea in enumerate(euler_angles):
        rot_matrix = geometry.rotmat3D_EA(*ea)[:, 0:2]
        slop = cryoops.compute_projection_matrix([rot_matrix], N, kernel, ksize, rad, 'rots')
        trunc_slice = slop.dot(fM.reshape((-1,)))
        if truncate:
            projs[i, :] = trunc_slice
        else:
            projs[i, :, :] = density.fspace_to_real(TtoF.dot(trunc_slice).reshape(N, N))

    if num_projs == 1 and not truncate:
        projs = projs.reshape((N, N))

    return projs
