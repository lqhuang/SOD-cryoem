import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec

from cryoio import mrc
from cryoem import cryoem

plt.rc('font', size=17)

WD = 'data/job2'
figure_dir = os.path.join(WD, 'Figures')
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)


def calc_fsc(model_path1, model_path2):
    V1 = mrc.readMRC(model_path1)
    V2 = mrc.readMRC(model_path2)
    N = V1.shape[0]

    V1, _ = cryoem.align_density(V1)
    V2, _ = cryoem.align_density(V2)

    VF1 = np.fft.fftshift(np.fft.fftn(V1))
    VF2 = np.fft.fftshift(np.fft.fftn(V2))
    # test_VF = np.fft.fftshift(np.fft.fftn(test_V))
    maxrad = 1

    rads, fsc, thresholds, resolutions = cryoem.compute_fsc(VF1, VF2, maxrad)
    return rads, fsc, thresholds, resolutions


angpix = 2.08
N = 125
correct_model = 'particle/EMD-6044.mrc'

plt.figure(0)
first_model = os.path.join(WD, 'it000', 'it000_reco.mrc')
rads, fsc, thresholds, resolutions = calc_fsc(correct_model, first_model)
plt.plot((1 - rads) * N / 2 * angpix, fsc)
plt.gca().invert_xaxis()
print('thresholds:', thresholds, 'resolutions:',
      (1 - np.asarray(resolutions)) * angpix * N / 2)
plt.xlabel(r'Resolution($\AA$)')
plt.ylabel('FSC')
plt.title('Compare with correct model')
plt.tight_layout()
plt.savefig(os.path.join(figure_dir, 'it000_fsc'), dpi=150)

for i in range(1, 20):
    last_model = os.path.join(
        WD, 'it' + str(i - 1).zfill(3), 'it' + str(i - 1).zfill(3) + '_reco.mrc')
    now_model = os.path.join(
        WD, 'it' + str(i).zfill(3), 'it' + str(i).zfill(3) + '_reco.mrc')

    fig = plt.figure(num=i, figsize=(16, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    plt.suptitle('Iteration: {0}'.format(i))

    plt.subplot(gs[0])
    rads, fsc, thresholds, resolutions = calc_fsc(correct_model, now_model)
    plt.plot((1 - rads) * N / 2 * angpix, fsc)
    plt.gca().invert_xaxis()
    print('thresholds:', thresholds, 'resolutions:',
          (1 - np.asarray(resolutions)) * angpix * N / 2)
    plt.xlabel(r'Resolution($\AA$)')
    plt.ylabel('FSC')
    plt.title('Compare with correct model')

    plt.subplot(gs[1])
    rads, fsc, thresholds, resolutions = calc_fsc(last_model, now_model)
    plt.plot((1 - rads) * N / 2 * angpix, fsc)
    plt.gca().invert_xaxis()
    print('thresholds:', thresholds, 'resolutions:',
          (1 - np.asarray(resolutions)) * angpix * N / 2)
    plt.xlabel(r'Resolution($\AA$)')
    plt.ylabel('FSC')
    plt.title('Compare with last model')
    plt.savefig(os.path.join(figure_dir, 'it' + str(i).zfill(3) +
                             '_fsc'), dpi=150, bbox_inches='tight')
