from __future__ import print_function, division

import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
import time
import pickle

import numpy as np
import matplotlib.pyplot as plt

from cryoio import mrc
import cryoops
import density
from notimplemented import py_objective_kernels
from test.likelihood_test import load_kernel

import pyximport; pyximport.install(setup_args={"include_dirs":np.get_include()},reload_support=True)
from objectives import objective_kernels


def plot_difference(cython_output_dict, py_output_dict):
    for key, cython_result in cython_output_dict.items():
        python_result = py_output_dict[key]
        # if key == 'g' or key == 'cphi_I' or key == 'cphi_R' or key == 'like':
        if True:
            if isinstance(cython_result, np.ndarray):
                if np.iscomplexobj(cython_result):
                    cython_result = np.abs(np.copy(cython_result)).flatten()
                    python_result = np.abs(np.copy(python_result)).flatten()
                fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))
                ax[0].plot(cython_result, label='cython-%s' % key)
                ax[0].plot(python_result, label='python-%s' % key)
                ax[0].legend()
                ax[1].plot(cython_result - python_result, label='difference')
                ax[1].legend()
            else:
                print('Comparison for %s' % key)
                print('  cython value:', cython_result)
                print('  python value:', python_result)
                try:
                    print('  difference:', np.abs(cython_result - python_result))
                except TypeError:
                    pass
        # if key == 'g':
        #     fig, ax = plt.subplots()
        #     a = np.abs(cython_result).flatten()
        #     b = np.abs(python_result).flatten()
        #     ax.hist(a[a != 0.0], bins=100, label='cython', alpha=0.6)
        #     ax.hist(b[b != 0.0], bins=100, label='python', alpha=0.6)
        #     ax.legend()
    plt.show()


def doimage_test(data_dir, model_file, use_angular_correlation=False):
    kernel = load_kernel(data_dir, model_file, use_angular_correlation)

    # for idx in range(kernel.minibatch['N_M']):
    for idx in [1]:
        tic = time.time()
        if kernel.sampler_S is not None:
            slice_ops, envelope, \
            W_R_sampled, sampleinfo_R, slices_sampled, slice_inds, \
            W_I_sampled, sampleinfo_I, rotd_sampled, rotc_sampled, \
            W_S_sampled, sampleinfo_S, S_sampled = \
                kernel.prep_operators(kernel.fM, idx)
        else:
            slice_ops, envelope, \
            W_R_sampled, sampleinfo_R, slices_sampled, slice_inds, \
            W_I_sampled, sampleinfo_I, rotd_sampled, rotc_sampled = \
                kernel.prep_operators(kernel.fM, idx)
        print("prep operators timing:", time.time()-tic)

        tic = time.time()
        if kernel.use_angular_correlation:
            ac_slices_sampled, ac_data_sampled = kernel.get_angular_correlation(
                slices_sampled, rotd_sampled, rotc_sampled, envelope, W_I_sampled)
        print("angular correlation timing:", time.time()-tic)

        # Cython part
        sigma2 = 1.0
        log_W_R = np.log(W_R_sampled)
        log_W_I = np.log(W_I_sampled)
        if kernel.sampler_S is not None:
            log_W_S = np.log(W_S_sampled)
        workspace = None
        g = np.zeros_like(slices_sampled, dtype=kernel.G_datatype)

        print(slices_sampled.dtype)

        tic = time.time()
        if use_angular_correlation:
            like, (cphi_I, cphi_R), csigma2_est, ccorrelation, cpower, workspace = \
                objective_kernels.doimage_ACRI(slices_sampled, envelope, \
                    rotc_sampled, rotd_sampled, \
                    ac_slices_sampled, ac_data_sampled, \
                    log_W_I, log_W_R, \
                    sigma2, g, workspace)
        else:
            like, (cphi_I, cphi_R), csigma2_est, ccorrelation, cpower, workspace = \
                objective_kernels.doimage_RI(slices_sampled, envelope, \
                    rotc_sampled, rotd_sampled, \
                    log_W_I, log_W_R, \
                    sigma2, g, workspace)
        print("Cython:")
        print("  cython objective kernel timing:", time.time() - tic)
        print("  cython like:", like)
        print("  cython g sum:", g.sum())
        print("  cython g max:", g.max())
        print("  cython g min:", g.min())
        cython_output_dict = {
            'like': like,
            'cphi_I': cphi_I,
            'cphi_R': cphi_R,
            'csigma2_est': csigma2_est,
            'ccorrelation': ccorrelation,
            'cpower': cpower,
            'g': g,
        }

        # Python part
        sigma2 = 1.0
        log_W_R = np.log(W_R_sampled)
        log_W_I = np.log(W_I_sampled)
        if kernel.sampler_S is not None:
            log_W_S = np.log(W_S_sampled)
        workspace = None
        g = np.zeros_like(slices_sampled, dtype=kernel.G_datatype)

        tic = time.time()
        if use_angular_correlation:
            like, (cphi_I, cphi_R), csigma2_est, ccorrelation, cpower, workspace = \
                py_objective_kernels.doimage_ACRI(slices_sampled, envelope, \
                    rotc_sampled, rotd_sampled, \
                    ac_slices_sampled, ac_data_sampled, \
                    log_W_I, log_W_R, \
                    sigma2, g, workspace)
        else:
            like, (cphi_I, cphi_R), csigma2_est, ccorrelation, cpower, workspace = \
                py_objective_kernels.doimage_RI(slices_sampled, envelope, \
                    rotc_sampled, rotd_sampled, \
                    log_W_I, log_W_R, \
                    sigma2, g, workspace)
        print("Python:")
        print("  python objective kernel timing:", time.time() - tic)
        print("  python like:", like)
        print("  python g sum:", g.sum())
        print("  python g max:", g.max())
        print("  python g min:", g.min())
        py_output_dict = {
            'like': like,
            'cphi_I': cphi_I,
            'cphi_R': cphi_R,
            'csigma2_est': csigma2_est,
            'ccorrelation': ccorrelation,
            'cpower': cpower,
            'g': g,
        }

        # plot_difference(cython_output_dict, py_output_dict)


if __name__ == '__main__':
    data_dir = sys.argv[1]
    model_file = sys.argv[2]
    doimage_test(data_dir, model_file)
