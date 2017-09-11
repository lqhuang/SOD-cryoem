from __future__ import print_function, division

import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
import time
import pickle

import numpy as np
import matplotlib.pyplot as plt

from notimplemented import py_objective_kernels


def plot_difference(output_dict, new_output_dict):
    for i, key in enumerate(output_dict.keys()):
        if isinstance(output_dict[key], np.ndarray):
            fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))
            ax[0].plot(output_dict[key], label='pre-%s' % key)
            ax[0].plot(new_output_dict[key], label='now-%s' % key)
            ax[0].legend()
            ax[1].plot(output_dict[key]- new_output_dict[key], label='difference')
            ax[1].legend()
        else:
            print('comparison for %s' % key)
            print('previous value:', output_dict[key])
            print('new value:', new_output_dict[key])
            try:
                print('difference:', new_output_dict[key] - output_dict[key])
            except TypeError:
                pass
    plt.show()


def load_RI(prep_file, output_file):
    with open(prep_file, 'rb') as pkl:
        input_tuple = pickle.load(pkl)
        g, sigma2, slice_ops, envelope, \
            W_R_sampled, sampleinfo_R, slices_sampled, slice_inds, \
            W_I_sampled, sampleinfo_I, rotd_sampled, rotc_sampled = input_tuple

    N_R, N_T = slices_sampled.shape
    log_W_R = np.log(W_R_sampled)
    log_W_I = np.log(W_I_sampled)

    workspace = None

    with open(output_file, 'rb') as pkl:
        output_tuple = pickle.load(pkl)
        like, (cphi_I, cphi_R), csigma2_est, ccorrelation, cpower, g, workspace = output_tuple

    output_dict = {
        'like': like,
        'cphi_I': cphi_I,
        'cphi_R': cphi_R,
        'csigma2_est': csigma2_est,
        'ccorrelation': ccorrelation,
        'cpower': cpower,
        'g': g,
    }

    tic = time.time()
    like, (cphi_I, cphi_R), csigma2_est, ccorrelation, cpower, workspace = \
        py_objective_kernels.doimage_RI(slices_sampled, envelope, \
            rotc_sampled, rotd_sampled, \
            log_W_I, log_W_R, \
            sigma2, g, workspace)
    print(time.time() - tic)

    new_output_dict = {
        'like': like,
        'cphi_I': cphi_I,
        'cphi_R': cphi_R,
        'csigma2_est': csigma2_est,
        'ccorrelation': ccorrelation,
        'cpower': cpower,
        'g': g,
    }

    plot_difference(output_dict, new_output_dict)


def load_RIS(prep_file, output_file):
    with open(prep_file, 'rb') as pkl:
        input_tuple = pickle.load(pkl)
        g, sigma2, slice_ops, envelope, \
            W_R_sampled, sampleinfo_R, slices_sampled, slice_inds, \
            W_I_sampled, sampleinfo_I, rotd_sampled, rotc_sampled, \
            W_S_sampled, sampleinfo_S, S_sampled = input_tuple

    N_R, N_T = slices_sampled.shape
    log_W_R = np.log(W_R_sampled)
    log_W_I = np.log(W_I_sampled)
    log_W_S = np.log(W_S_sampled)

    workspace = None

    with open(output_file, 'rb') as pkl:
        output_tuple = pickle.load(pkl)
        like, (cphi_S, cphi_I, cphi_R), csigma2_est, ccorrelation, cpower, g, workspace = output_tuple

    output_dict = {
        'like': like,
        'cphi_S': cphi_S,
        'cphi_I': cphi_I,
        'cphi_R': cphi_R,
        'csigma2_est': csigma2_est,
        'ccorrelation': ccorrelation,
        'cpower': cpower,
        'g': g,
    }

    tic = time.time()
    like, (cphi_S, cphi_I, cphi_R), csigma2_est, ccorrelation, cpower, workspace = \
        py_objective_kernels.doimage_RIS(slices_sampled, S_sampled, envelope, \
            rotc_sampled, rotd_sampled, \
            log_W_S, log_W_I, log_W_R, \
            sigma2, g, workspace)
    print(time.time() - tic)

    new_output_dict = {
        'like': like,
        'cphi_S': cphi_S,
        'cphi_I': cphi_I,
        'cphi_R': cphi_R,
        'csigma2_est': csigma2_est,
        'ccorrelation': ccorrelation,
        'cpower': cpower,
        'g': g,
    }
    plot_difference(output_dict, new_output_dict)


if __name__ == '__main__':
    # prep_file = 'exp/RI_random_density/minibatch_idx_0_prep.pkl'
    # output_file = 'exp/RI_random_density/minibatch_idx_0_output.pkl'
    # load_RI(prep_file, output_file)

    prep_file = 'exp/RIS_random_density/minibatch_idx_0_prep.pkl'
    output_file = 'exp/RIS_random_density/minibatch_idx_0_output.pkl'
    load_RIS(prep_file, output_file)
