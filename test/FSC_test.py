from __future__ import print_function, division

import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
import glob
import re
import shutil

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec

from cryoio import mrc
import cryoem

plt.rc('font', size=17)


def calc_fsc(model_path1, model_path2):
    V1 = mrc.readMRC(model_path1)
    V2 = mrc.readMRC(model_path2)
    N = V1.shape[0]

    alignedV1, R1 = cryoem.align_density(V1)
    alignedV1 = cryoem.rotate_density(alignedV1, R1)
    alignedV2, R1 = cryoem.align_density(V2)
    alignedV2 = cryoem.rotate_density(alignedV2, R1)

    VF1 = np.fft.fftshift(np.fft.fftn(alignedV1))
    VF2 = np.fft.fftshift(np.fft.fftn(alignedV2))

    maxrad = 1
    rads, fsc, thresholds, resolutions = cryoem.compute_fsc(VF1, VF2, maxrad)
    return rads, fsc, thresholds, resolutions


def compare_two_exps(exp_dir_1, exp_dir_2, angpix, savepath='.'):
    mrc_list_1 = glob.glob(os.path.join(exp_dir_1, 'model') + '-*.mrc')
    mrc_list_2 = glob.glob(os.path.join(exp_dir_2, 'model') + '-*.mrc')
    num_models = min(len(mrc_list_1), len(mrc_list_2))

    for i, model1, model2 in zip(range(num_models), mrc_list_1, mrc_list_2):
        rads, fsc, thresholds, resolutions = calc_fsc(model1, model2)

        fig, ax = plt.subplots(figsize=(9.6, 4.8))
        ax.plot(rads * 1 / angpix, fsc)
        print( 'thresholds:', thresholds, 'resolutions:', 1 / (np.asarray(resolutions) * (1 / angpix)) )
        ax.set_xlabel(r'Resolution($1/\AA$)')
        ax.set_ylabel('FSC')
        ax.set_title('Iteration: {0}'.format(i))
        fig.savefig(os.path.join(savepath, str(i) + '_fsc'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    print(sys.argv)
    exp_dir_1 = sys.argv[1]
    exp_dir_2 = sys.argv[2]
    angpix = 2.8
    compare_two_exps(exp_dir_1, exp_dir_2, angpix)
