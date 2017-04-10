import os
import numpy as np
from matplotlib import pyplot as plt

from cryoio import mrc
from cryoem import cryoem

plt.rc('font', size=17)

# working_directory = './EMD-2325-dataset1-gaussian'
working_directory = './EMD-2325-dataset1-noise-1000'
angpix = 2.08

dataset1_mrc = os.path.join(working_directory, 'dataset1.mrc')
# dataset1_mrc = './particle/EMD-2325.mrc'
dataset2_mrc = os.path.join(working_directory, 'dataset2.mrc')

V1 = mrc.readMRC(dataset1_mrc)
V2 = mrc.readMRC(dataset2_mrc)
N = V1.shape[0]

# test_V = cryoem.generate_phantom_density(N, 1*N/2.0, 5*N/128.0, 30)

VF1 = np.fft.fftshift(np.fft.fftn(V1))
VF2 = np.fft.fftshift(np.fft.fftn(V2))
# test_VF = np.fft.fftshift(np.fft.fftn(test_V))
maxrad = 1

rads, fsc, thresholds, resolutions = cryoem.compute_fsc(VF1, VF2, maxrad)

plt.figure(1)
plt.plot( (1 - rads) * N/2 * angpix , fsc)
plt.gca().invert_xaxis()
print('thresholds:', thresholds, 'resolutions:',  (1 - np.asarray(resolutions)) * angpix * N/2)
plt.xlabel(r'Resolution($\AA$)')
plt.ylabel('FSC')
plt.tight_layout()
plt.show()