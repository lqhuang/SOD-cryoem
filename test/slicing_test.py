from __future__ import print_function, division

import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))

import numpy as np
from matplotlib import pyplot as plt

from cryoio import mrc
from cryoio import ctf
import geometry
import cryoops
import cryoem
import density

import pyximport; pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=True)
import sincint


N = 128
psize = 2.8
rad_freq = 0.50
rad = rad_freq * 2.0 * psize
beamstop_freq = 0.001

mask_2d = geometry.gen_trunc_mask(N, 2, rad_freq, beamstop_freq, psize=2.8)

TtoF = sincint.gentrunctofull(N=N, rad=rad)
FtoT = sincint.genfulltotrunc(N=N, rad=rad)


theta = np.arange(0, 2*np.pi, 2*np.pi/36)
EAs = np.asarray([[i, j] for i in np.arange(0, np.pi, np.pi/12.0) for j in np.arange(0, 2*np.pi, 2*np.pi/6.0)])
print(EAs.shape)
EAs = np.vstack((EAs.T, np.zeros(EAs.shape[0]))).T
dirs = geometry.genDir(EAs)

N_R = dirs.shape[0]
N_I = theta.shape[0]
N_T = FtoT.shape[0]
print("length of truncation slice", N_T)


# M = cryoem.generate_phantom_density(N, 0.95*N/2.0, 5 * N / 128.0, 30, seed=1)
M = mrc.readMRC('particle/1AON.mrc') 

zeropad = 1
zeropad_size = int(zeropad * (N / 2))
zp_N = zeropad_size * 2 + N
zp_M_shape = (zp_N,) * 3
ZP_M = np.zeros(zp_M_shape, dtype=density.real_t)
zp_M_slicer = (slice( zeropad_size, (N + zeropad_size) ),) * 3

M_totalmass = 5000
M *= M_totalmass / M.sum()

ZP_M[zp_M_slicer] = M

N = M.shape[0]
kernel = 'lanczos'
ksize = 6
premult = cryoops.compute_premultiplier(zp_N, kernel, ksize)
V = density.real_to_fspace(premult.reshape((1, 1, -1)) * premult.reshape((1, -1, 1)) * premult.reshape((-1, 1, 1)) * ZP_M)
# V = density.real_to_fspace(ZP_M)
ZP_fM = V.real ** 2 + V.imag ** 2

fM = ZP_fM[zp_M_slicer]

# mask_3d_outlier = geometry.gen_dense_beamstop_mask(N, 3, 0.015, psize=2.8)
# fM *= mask_3d_outlier

# fM = mrc.readMRC('particle/1AON_fM_totalmass_5000.mrc') * mask_3d_outlier

imgdata = mrc.readMRCimgs('data/1AON_xfel_5000_totalmass_05000/imgdata.mrc', 420, 1)
curr_img = imgdata[:, :, 0]
zp_img = np.zeros((zp_N,)*2, dtype=density.real_t)
zp_img[zp_M_slicer[0:2]] = curr_img
# curr_img = zp_img

slice_interp = {'projdirs': dirs, 'N': N, 'kern': 'lanczos', 'kernsize': 4, 'rad': rad, 'sym': None}  # 'zeropad': 0, 'dopremult': True
inplane_interp = {'thetas': theta, 'N': N, 'kern': 'lanczos', 'kernsize': 4, 'rad': rad, 'onlyRs': False}  # 'zeropad': 1, 'dopremult': True
inplane_interp['N_src'] = N  # zp_N

slice_ops = cryoops.compute_projection_matrix(**slice_interp)
inplane_ops = cryoops.compute_inplanerot_matrix(**inplane_interp)

slices_sampled = cryoem.getslices(fM, slice_ops).reshape((N_R, N_T))
rotd_sampled = cryoem.getslices(curr_img, inplane_ops).reshape((N_I, N_T))

slices_sampled += 1.0

print(slices_sampled[0])
print("slices_sampled max():", slices_sampled.max(axis=1))
print("slices_sampled min():", slices_sampled.min(axis=1))
print("slices_sampled < 1.0:", (slices_sampled<1.0).sum(axis=1))
print((rotd_sampled < 1.0).sum(axis=1))


# fig, ax = plt.subplots()
# im = ax.imshow( ( curr_img ) )
# fig.colorbar(im, ax=ax)

fig, axes = plt.subplots(3, 4)
for cim, ax in zip(rotd_sampled, axes.flatten()):
    dense_cim = ( TtoF.dot( mask_2d * cim ).reshape(N, N) )
    im = ax.imshow(dense_cim)
    fig.colorbar(im, ax=ax)


fig, axes = plt.subplots(3, 4)
for cim, ax in zip(slices_sampled, axes.flatten()):
    dense_cim = ( TtoF.dot( mask_2d * cim ).reshape(N, N) )
    im = ax.imshow(dense_cim)
    fig.colorbar(im, ax=ax)

plt.show()




### plot mask

# N = 128
# mask = np.zeros((N, N))
# # for i in np.arange(0.005, 0.04, 0.005):
# #     print(i)
# #     new = geometry.gen_dense_beamstop_mask(N, 2, i, psize=2.8)
# #     mask += new


# mask_2d = geometry.gen_dense_beamstop_mask(N, 2, 0.015, psize=2.8)
# mask_3d = geometry.gen_dense_beamstop_mask(N, 3, 0.015, psize=2.8)

# plt.imshow(mask_2d*2 + mask_3d[:, :, int(N/2)])
# plt.colorbar()
# plt.show()
