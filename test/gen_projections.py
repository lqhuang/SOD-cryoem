from __future__ import print_function, division

import numpy as np

from cryoio import mrc
import cryoem, density, cryoops, geom

import pyximport; pyximport.install(setup_args={"include_dirs":np.get_include()},reload_support=True)
import sincint

from time import time

phantompath = './particle/1AON.mrc'

N = 128
rad = 0.95
shift_sigma = 3.0
sigma_noise = 25.0

M_totalmass = 80000

kernel = 'lanczos'
ksize = 6

M = mrc.readMRC(phantompath)

tic = time()
premult = cryoops.compute_premultiplier(N, kernel, ksize)

TtoF = sincint.gentrunctofull(N=N, rad=rad)

cryoem.window(M, 'circle') # apply a sphere mask
M[M<0] = 0
if M_totalmass is not None:
    M *= M_totalmass / M.sum()

V = density.real_to_fspace( premult.reshape((1,1,-1)) * premult.reshape((1,-1,1)) * premult.reshape((-1,1,1)) * M )

# print('preprocessing time: ', time()-tic)
pardata = {'R':[],'t':[]}

tic = time()
# Randomly generate the viewing direction/shift
pt = np.random.randn(3)
pt /= np.linalg.norm(pt)
psi = 2*np.pi*np.random.rand()
EA = geom.genEA(pt)[0]
# print(EA)
EA[2] = psi
shift = np.random.randn(2) * shift_sigma

R = geom.rotmat3D_EA(*EA)[:,0:2]
print(R)
slop = cryoops.compute_projection_matrix([R], N, kernel, ksize, rad, 'rots')
shift_phases = cryoops.compute_shift_phases(shift.reshape((1,2)), N, rad)[0]

D = slop.dot( V.reshape((-1,)) )
D *= shift_phases

# imgdata = density.fspace_to_real( (TtoF.dot(D)).reshape((N,N)) ) + np.require(np.random.randn(N, N)*sigma_noise, dtype=density.real_t)
imgdata = density.fspace_to_real( (TtoF.dot(D)).reshape((N,N)) )

# genctf_stack.add_img(genctfI,
#                         PHI=EA[0]*180.0/np.pi,THETA=EA[1]*180.0/np.pi,PSI=EA[2]*180.0/np.pi,
#                         SHX=shift[0],SHY=shift[1])
print('generate a projection: ', time()-tic)
pardata['R'].append(R)
pardata['t'].append(shift)