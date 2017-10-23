from __future__ import print_function, division

import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
import matplotlib.pyplot as plt


class StatAnalyzer(object):
    NUM_PLOT = 0

    def __init__(self, stat_file, show_together=True):
        if isinstance(stat_file, str):
            with open(stat_file, 'rb') as pkl:
                self.stat = pickle.load(pkl)
        elif isinstance(stat_file, dict):
            self.stat = stat_file
        self.iteration = self.stat['train_iteration']
        self.num_iter = len(self.iteration)
        self.show_together = show_together

    def keys(self):
        return self.stat.keys()

    def plot_kern_timing(self, start=0, end=100):
        # timing_key = [key for key in stat.keys() if 'timing' in key and 'train' in key]
        # timing_key = [key for key in stat.keys() if 'kern' in key and 'train' in key]
        timing_key = ['train_kern_timing_prep', 'train_kern_timing_work']  #, 'train_kern_timing_proc']
        if 'train_angular_correlation_timing' in self.keys():
            timing_key.append('train_angular_correlation_timing')

        sl = slice(start, end)
        bottom = np.zeros_like(self.iteration[sl])

        fig, ax = plt.subplots(figsize=(12.8, 4.8))
        self.NUM_PLOT += 1
        for key in timing_key:
            ax.bar(self.iteration[sl], self.stat[key][sl], bottom=bottom, label=key)
            bottom = bottom + np.asarray(self.stat[key][sl])
        ax.set_xlabel('iteration: #')
        ax.set_ylabel('kernel time (s)')
        lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False).draggable()
        fig.tight_layout()
        fig.subplots_adjust(right=0.75)

        if not self.show_together:
            plt.show()

    def plot_timer(self):
        total_runtime = (np.asarray(self.stat['time']) - self.stat['time'][0]) / 3600
        minibatch_time = self.stat['minibatch_time']

        fig, ax = plt.subplots()
        self.NUM_PLOT += 1
        ax.plot(self.iteration, total_runtime, label='total runtime')
        ax.set_xlabel('iteration: #')
        ax.set_ylabel('total runtime (hours)')
        ylim = ax.get_ylim()
        ax.set_ylim([0, ylim[1]])
        for i in range(0, self.num_iter, 999):
            ax.plot([i, i], [0, total_runtime[i]], linestyle='--', color='grey')
            ax.text(i-self.num_iter/200, total_runtime[i]+1, '{:.2f}'.format(total_runtime[i]))
        lgd = ax.legend(frameon=False)
        fig.tight_layout()

        fig, ax = plt.subplots()
        self.NUM_PLOT += 1
        ax.plot(self.iteration[:minibatch_time.__len__()], minibatch_time, label='minibatch time')
        ax.set_xlabel('iteration: #')
        ax.set_ylabel('time for each batch (seconds)')
        lgd = ax.legend(frameon=False)
        fig.tight_layout()

        if not self.show_together:
            plt.show()

    def plot_sagd(self):
        sagd_keys = ['sagd_L', 'sagd_gnorm', 'sagd_eps', 'step_size', 'inc_ratio', 'grad_size', 'logp', 'like', 'sigma']

        fig, axes = plt.subplots(3, 3, figsize=(12.8, 8))
        self.NUM_PLOT += 1
        for ax, key in zip(axes.flat, sagd_keys):
            ax.plot(self.iteration, self.stat[key], label=key)
            lgd = ax.legend(frameon=False)
        fig.tight_layout()

        if not self.show_together:
            plt.show()

    def show_all(self):
        plt.show()


if __name__ == '__main__':
    pkl_file = sys.argv[1]
    print(pkl_file)
    stat_analyzer = StatAnalyzer(pkl_file)
    # stat_analyzer.plot_kern_timing()
    # stat_analyzer.plot_timer()
    stat_analyzer.plot_sagd()
    stat_analyzer.show_all()


"""
stat
dict_keys(['total_density', 'avg_density', 'nonzero_density', 'max_density', 'min_density',

'test_is_speedup_R', 'test_is_speedup_I', 'test_is_speedup_Total',
'test_sigma', 'test_logp', 'test_like', 'test_num_data', 'test_num_data_evals', 'test_iteration', 'test_epoch', 'test_cepoch',
'test_time', 'test_like_timing_setup', 'test_like_timing_slice', 'test_like_timing_queue', 'test_like_timing_join', 'test_like_timing_total',
'test_kern_timing_prep_sample_R', 'test_kern_timing_prep_sample_I', 'test_kern_timing_prep_slice', 'test_kern_timing_prep_rot_img', 'test_kern_timing_prep_rot_ctf', 'test_kern_timing_prep', 'test_kern_timing_work', 'test_kern_timing_proc', 'test_kern_timing_store', 'test_angular_correlation_timing',
'test_full_like_quantiles', 'test_mini_like_quantiles', 'test_num_like_quantiles',

'sagd_L', 'sagd_gnorm', 'sagd_eps', 'step_size', 'inc_ratio', 'grad_size', 'norm_density',

'train_is_speedup_R', 'train_is_speedup_I', 'train_is_speedup_Total',
'train_sigma', 'train_logp', 'train_like', 'train_num_data', 'train_num_data_evals', 'train_iteration', 'train_epoch', 'train_cepoch',
'train_time', 'train_like_timing_setup', 'train_like_timing_slice', 'train_like_timing_queue', 'train_like_timing_join', 'train_like_timing_unslice', 'train_like_timing_record', 'train_like_timing_premult', 'train_like_timing_total',
'train_kern_timing_prep_sample_R', 'train_kern_timing_prep_sample_I', 'train_kern_timing_prep_slice', 'train_kern_timing_prep_rot_img', 'train_kern_timing_prep_rot_ctf', 'train_kern_timing_prep', 'train_kern_timing_work', 'train_kern_timing_proc', 'train_kern_timing_store', 'train_angular_correlation_timing',
'train_full_like_quantiles', 'train_mini_like_quantiles', 'train_num_like_quantiles',

'num_data', 'num_data_evals', 'iteration', 'epoch', 'cepoch', 'logp', 'like', 'sigma', 'time', 'minibatch_time'])


diag
dict_keys(['test_CV2_R', 'test_CV2_I', 'test_idxs', 'test_sigma2_est', 'test_correlation', 'test_power',
'global_phi_R', 'global_phi_I', 'iteration', 'epoch', 'cepoch',
'train_CV2_R', 'train_CV2_I', 'train_idxs', 'train_sigma2_est', 'train_correlation', 'train_power',
'params', 'envelope_mle', 'sigma2_mle', 'hostname'])


like
dict_keys(['img_likes', 'train_idxs', 'test_idxs'])
"""
