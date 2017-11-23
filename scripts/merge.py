from __future__ import print_function, division

import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
try:
    import cPickle as pickle  # python 2
except ImportError:
    import pickle  # python 3

import numpy as np

import geometry
import density
from cryoio import mrc
import cryoem

import pyximport; pyximport.install(
    setup_args={"include_dirs": np.get_include()}, reload_support=True)
import sincint


N = 128
model = np.zeros((N,) * 3, dtype=density.real_t)
model_weight = np.zeros((N,) * 3)


data_dir = 'data/1AON_xfel_5000_totalmass_20000_oversampling_3'
img_file = os.path.join(data_dir, 'imgdata.mrc')
par_file = os.path.join(data_dir, 'pardata.pkl')

# with open(par_file, 'rb') as pkl_file:
#     pardata = pickle.load(pkl_file)
euler_angles = list()
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
imgdata = mrc.readMRC(img_file)

num_data = imgdata.shape[2]

rad = 0.8
FtoT = sincint.genfulltotrunc(N=N, rad=rad)

N_T = FtoT.shape[0]

slices = np.zeros((num_data, N_T), dtype=np.float32)

for i in range(num_data):
    curr_img = imgdata[:, :, i]
    slices[i] = FtoT.dot(curr_img.reshape(-1, 1)).reshape(-1)

Rs = np.vstack([geometry.rotmat3D_EA(*ea)[:, 0:2].reshape((1, 3, 2)) for ea in euler_angles])

# cx, cy = int(N/2), int(N/2)
# nx, ny = N, N

# for i, ea in enumerate(euler_angles):
#     print(i)
#     R_matrix = geometry.rotmat3D_EA(*ea)
#     curr_img = imgdata[:, :, i]
    
#     num_valid_voxel = 0

#     for xx in range(N):
#         for yy in range(N):
#             voxel_intensity = curr_img[xx, yy]

#             x = xx - cx
#             y = yy - cy
#             z = 0
#             coord = np.asarray([x, y, z]).reshape(-1, 1)
#             rot_coord = np.dot(R_matrix, coord).reshape(1, -1)[0]
#             rot_coord = np.int_(np.round(rot_coord)) + cx

#             in_x = rot_coord[0] >= 0 and rot_coord[0] < N
#             in_y = rot_coord[1] >= 0 and rot_coord[1] < N
#             in_z = rot_coord[2] >= 0 and rot_coord[2] < N

#             if in_x and in_y and in_z:
#                 num_valid_voxel += 1

#                 model_voxel_intensity = model[tuple(rot_coord)]
#                 model_weight[tuple(rot_coord)] += 1
#                 voxel_weight = model_weight[tuple(rot_coord)]
#                 delta_intensity = voxel_intensity - model_voxel_intensity
#                 model_voxel_intensity += delta_intensity / voxel_weight
#                 model[tuple(rot_coord)] = model_voxel_intensity



model = cryoem.merge_slices(slices, Rs, N, rad, beamstop_rad=None)

mrc.writeMRC('particle/merge.mrc', model, psz=2.8)
