from __future__ import print_function, division

import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
import pickle

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
rad = 0.5

fM = mrc.readMRC('particle/1AON_fM_totalmass_20000_oversampling_6.mrc')
TtoF = sincint.gentrunctofull(N=N, rad=rad)

theta = np.arange(0, 2*np.pi, 2*np.pi/12)
EAs = np.asarray([[i, j] for i in np.arange(0, np.pi, np.pi/6.0) for j in np.arange(0, 2*np.pi, 2*np.pi/2.0)])
EAs = np.vstack((EAs.T, np.zeros(EAs.shape[0]))).T

euler_angles = []
data_dir = 'data/1AON_xfel_5000_totalmass_20000_oversampling_3'
with open(os.path.join(data_dir, 'ctf_gt.par')) as par:
    par.readline()
    # 'C                 PHI      THETA        PSI        SHX        SHY       FILM        DF1        DF2     ANGAST'
    while True:
        try:
            line = par.readline().split()
            euler_angles.append([float(line[1]), float(line[2]), float(line[3])])
        except:
            break
euler_angles = np.deg2rad(np.asarray(euler_angles))
EAs = euler_angles[0:12, :]

dirs = geometry.genDir(EAs)
Rs = np.vstack([geometry.rotmat3D_EA(*ea)[:, 0:2].reshape((1, 3, 2)) for ea in EAs])
print(EAs.shape)

N_R = dirs.shape[0]
N_I = theta.shape[0]
N_T = TtoF.shape[1]

dir_slice_interp = {'projdirs': dirs, 'N': N, 'kern': 'lanczos', 'kernsize': 6,  'projdirtype': 'dirs', 'rad': rad, 'sym': None}  # 'zeropad': 0, 'dopremult': True
R_slice_interp = {'projdirs': Rs, 'N': N, 'kern': 'lanczos', 'kernsize': 6, 'projdirtype': 'rots', 'rad': rad, 'sym': None}  # 'zeropad': 0, 'dopremult': True
inplane_interp = {'thetas': theta, 'N': N, 'kern': 'lanczos', 'kernsize': 6, 'rad': rad}  # 'zeropad': 1, 'dopremult': True
inplane_interp['N_src'] = N  # zp_N

slice_ops = cryoops.compute_projection_matrix(**R_slice_interp)
inplane_ops = cryoops.compute_inplanerot_matrix(**inplane_interp)

print(slice_ops.shape)

slices_sampled = cryoem.getslices(fM, slice_ops).reshape((N_R, N_T))
curr_img = TtoF.dot(slices_sampled[0]).reshape(N, N)
rotd_sampled = cryoem.getslices(curr_img, inplane_ops).reshape((N_I, N_T))

### 

# with open(os.path.join(data_dir, 'pardata.pkl'), 'rb') as pkl:
#     pardata = pickle.load(pkl)
# for i, ea in enumerate(EAs):
#     R = geometry.rotmat3D_EA(*ea)[:, 0:2]
#     print('ea R, ', R)
#     d = geometry.genDir([ea])
#     R = geometry.rotmat3D_dir(d)[:, 0:2]
#     print('dir R', R)
#     # R = pardata['R'][i]
#     slop = cryoops.compute_projection_matrix([R], N, 'lanczos', 6, rad, 'rots')
#     slices_sampled[i] = slop.dot(fM.reshape(-1))


# fig, axes = plt.subplots(3, 3, sharex=True, sharey=True,  figsize=(9.6, 7.2))
# for i, ax in enumerate(axes.flatten()):
#     img = TtoF.dot(slices_sampled[i]).reshape(N, N)
#     ax.imshow(img, origin='lower')
# fig.suptitle('slices_sampled')

# fig, axes = plt.subplots(3, 4)
# for i, ax in enumerate(axes.flatten()):
#     img = TtoF.dot(slices_sampled[i]).reshape(N, N)
#     ax.imshow(img, origin='lower')
# fig.suptitle('rotd_sampled')

# plt.show()

###

# new slicing method


dir_slice_interp = {'projdirs': dirs, 'N': N, 'kern': 'lanczos', 'kernsize': 6, 
                    'projdirtype': 'dirs', 'onlyRs': True, 'rad': rad, 'sym': None}  # 'zeropad': 0, 'dopremult': True
R_slice_interp = {'projdirs': Rs, 'N': N, 'kern': 'lanczos', 'kernsize': 6,
                  'projdirtype': 'rots', 'onlyRs': True, 'rad': rad, 'sym': None}  # 'zeropad': 0, 'dopremult': True
inplane_interp = {'thetas': theta, 'N': N, 'kern': 'lanczos', 'kernsize': 6,
                  'onlyRs': True, 'rad': rad}  # 'zeropad': 1, 'dopremult': True
inplane_interp['N_src'] = N  # zp_N

slice_ops = cryoops.compute_projection_matrix(**R_slice_interp)
inplane_ops = cryoops.compute_inplanerot_matrix(**inplane_interp)

slices_sampled = cryoem.getslices_interp(fM, slice_ops, rad).reshape((N_R, N_T))
curr_img = TtoF.dot(slices_sampled[0]).reshape(N, N)
rotd_sampled = cryoem.getslices_interp(curr_img, inplane_ops, rad).reshape((N_I, N_T))

###

fig, axes = plt.subplots(3, 3, sharex=True, sharey=True,  figsize=(9.6, 7.2))
for i, ax in enumerate(axes.flatten()):
    img = TtoF.dot(slices_sampled[i]).reshape(N, N)
    ax.imshow(img, origin='lower')
fig.suptitle('slices_sampled')

fig, axes = plt.subplots(3, 4)
for i, ax in enumerate(axes.flatten()):
    img = TtoF.dot(slices_sampled[i]).reshape(N, N)
    ax.imshow(img, origin='lower')
fig.suptitle('rotd_sampled')

plt.show()

###
