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
    # test_VF = np.fft.fftshift(np.fft.fftn(test_V))
    maxrad = 1

    rads, fsc, thresholds, resolutions = cryoem.compute_fsc(VF1, VF2, maxrad)
    return rads, fsc, thresholds, resolutions


def main(correct_model_file, angpix, exp_dir):
    M = mrc.readMRC(correct_model_file)
    N = M.shape[0]

    figure_dir = os.path.join(exp_dir, 'Figures')
    if not os.path.exists(figure_dir):
        os.mkdir(figure_dir)
    else:
        shutil.rmtree(figure_dir)
        os.mkdir(figure_dir)

    fig = plt.figure(num=0, figsize=(8, 6))
    init_model = os.path.join(exp_dir, 'logs', 'init_model.mrc')
    rads, fsc, thresholds, resolutions = calc_fsc(correct_model_file, init_model)
    plt.plot(rads * 1 / angpix, fsc)
    print( 'thresholds:', thresholds, 'resolutions:', 1 / (np.asarray(resolutions) * (1 / angpix)) )
    plt.xlabel(r'Resolution($1/\AA$)')
    plt.ylabel('FSC')
    plt.title('Compare with correct model')
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, 'init_fsc'), dpi=150)
    plt.close(fig)

    mrc_list = glob.glob(os.path.join(exp_dir, 'logs', 'model') + '-*.mrc')

    for i, now_model in enumerate(mrc_list):
        if i == 0:
            last_model = init_model
        else:
            last_model = os.path.join(mrc_list[i-1])
        
        iteration = re.findall(''.join(['\d' for i in range(6)]), now_model)[0]

        fig = plt.figure(num=i, figsize=(16, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
        plt.suptitle('Iteration: {0}'.format(iteration))

        plt.subplot(gs[0])
        rads, fsc, thresholds, resolutions = calc_fsc(correct_model_file, now_model)
        plt.plot(rads * 1 / angpix, fsc)
        print( 'thresholds:', thresholds, 'resolutions:', 1 / (np.asarray(resolutions) * (1 / angpix)) )
        plt.xlabel(r'Resolution($1/\AA$)')
        plt.ylabel('FSC')
        plt.title('Compare with correct model')

        plt.subplot(gs[1])
        rads, fsc, thresholds, resolutions = calc_fsc(last_model, now_model)
        plt.plot(rads * 1 / angpix, fsc)
        print( 'thresholds:', thresholds, 'resolutions:', 1 / (np.asarray(resolutions) * (1 / angpix)) )
        plt.xlabel(r'Resolution($1/\AA$)')
        plt.ylabel('FSC')
        plt.title('Compare with last model')
        plt.savefig(os.path.join(figure_dir, 'it' + str(iteration) +
                                '_fsc'), dpi=150, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    exp_dir = sys.argv[1]

    particle = '../particle/1AON.mrc'
    angpix = 2.8
    main(particle, angpix, exp_dir)
