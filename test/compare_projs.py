import numpy as np
import matplotlib.pyplot as plt

from cryoio import mrc
import density, cryoops
import geometry
from quadrature import healpix
from cryoem import cryoem

import pyximport; pyximport.install(
    setup_args={"include_dirs": np.get_include()}, reload_support=True)
import sincint

M = mrc.readMRC('./particle/EMD-6044.mrc')
# M = mrc.readMRC('./particle/1AON.mrc')
# M = M / np.sum(M)
M = M[:124, :124, :124]

mrc.writeMRC('./particle/EMD-6044-cropped.mrc', M, psz=3.0)

N = M.shape[0]
print(M.shape)
rad = 1
kernel = 'lanczos'
ksize = 4

xy, trunc_xy, truncmask = geometry.gencoords(N, 2, rad, True)
# premult = cryoops.compute_premultiplier(N, kernel='lanczos', kernsize=6)
premult = cryoops.compute_premultiplier(N, kernel, ksize)
TtoF = sincint.gentrunctofull(N=N, rad=rad)

fM = density.real_to_fspace(M)
prefM = density.real_to_fspace(premult.reshape(
        (1, 1, -1)) * premult.reshape((1, -1, 1)) * premult.reshape((-1, 1, 1)) * M)

EAs_grid = healpix.gen_EAs_grid(nside=2, psi_step=360)
Rs = [geometry.rotmat3D_EA(*EA)[:, 0:2] for EA in EAs_grid]
slice_ops = cryoops.compute_projection_matrix(Rs, N, kern='lanczos', kernsize=ksize, rad=rad, projdirtype='rots')

slices_sampled = cryoem.getslices(fM, slice_ops).reshape((EAs_grid.shape[0], trunc_xy.shape[0]))

premult_slices_sampled = cryoem.getslices(prefM, slice_ops).reshape((EAs_grid.shape[0], trunc_xy.shape[0]))

S = cryoops.compute_shift_phases(np.asarray([100, -20]).reshape((1,2)), N, rad)[0]

trunc_slice = slices_sampled[0]
premult_trunc_slice = premult_slices_sampled[0]
premult_trunc_slice_shift = S * premult_slices_sampled[0]

fourier_slice = TtoF.dot(trunc_slice).reshape((N, N))
real_proj = density.fspace_to_real(fourier_slice)
premult_fourier_slice = TtoF.dot(premult_trunc_slice).reshape((N, N))
premult_fourier_slice_shift = TtoF.dot(premult_trunc_slice_shift).reshape((N, N))
premult_real_proj = density.fspace_to_real(premult_fourier_slice)
premult_real_proj_shift = density.fspace_to_real(premult_fourier_slice_shift)

fig, axes = plt.subplots(2, 2)
ax = axes.flatten()
ax[0].imshow(real_proj)
ax[1].imshow(np.log(np.abs(fourier_slice)))
ax[2].imshow(premult_real_proj)
ax[3].imshow(np.log(np.abs(premult_fourier_slice)))

fig, axes = plt.subplots(2, 2)
ax = axes.flatten()
ax[0].imshow(premult_real_proj)
ax[1].imshow(np.log(np.abs(premult_fourier_slice)))
ax[2].imshow(premult_real_proj_shift)
ax[3].imshow(np.log(np.abs(premult_fourier_slice_shift)))

fig, ax = plt.subplots()
im = ax.imshow(np.log(np.abs(premult_fourier_slice_shift - premult_fourier_slice)))
fig.colorbar(im)

plt.show()
