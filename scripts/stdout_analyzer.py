from __future__ import print_function, division

import sys
import re

import numpy as np
import matplotlib.pyplot as plt

# reversed
FACTOR = [1.0, 60.0, 3600.0, 86400.0]

def time_converter(time_string):
    time_string = time_string.lower().rstrip()
    if time_string.endswith('s'):
        time_string = time_string[:-1]
    time_list = time_string.split(':')
    time = 0
    for i, value in enumerate(reversed(time_list)):
        time += float(value) * FACTOR[i]
    return time


def timer(stdout_file):
    pattern = 'Total Runtime - '
    compiled_pattern = re.compile(pattern)

    runtime = []

    with open(stdout_file) as stdout_file:
        for line in stdout_file:
            splited_line = compiled_pattern.split(line)
            if len(splited_line) >= 2:
                time_str = splited_line[-1]
                time = time_converter(time_str)
                runtime.append(time)

    total_runtime = np.asarray(runtime) / 3600.0
    iteration = np.arange(total_runtime.shape[0])

    fig, ax = plt.subplots()
    ax.plot(iteration + 1, total_runtime)
    ax.set_xlabel('Iteration: #')
    ax.set_ylabel('Total Runtime (hours)')
    ylim = ax.get_ylim()
    ax.set_ylim([0, ylim[1]])
    for i in range(0, 10000, 999):
        ax.plot([i, i], [0, total_runtime[i]], linestyle='--', color='grey')
        ax.text(i-500, total_runtime[i]+1, '{:.2f}'.format(total_runtime[i]))
    plt.show()


def is_eff(stdout_file):
    """    IS Speedup R / I / Total: 1239.74 (30 of 37465) / 40.43 (8 of 338) / 48094.08 (263 of 12663170)"""
    pattern = '    IS Speedup R / I / Total:'
    compiled_pattern = re.compile(pattern)

    grid_r = []
    grid_i = []
    grid_total = []

    is_r = []
    is_i = []
    is_total = []

    speedup_r = []
    speedup_i = []
    speedup_total = []

    with open(stdout_file) as stdout_file:
        for line in stdout_file:
            splited_line = compiled_pattern.split(line)
            if len(splited_line) >= 2:
                speedup = splited_line[1].rstrip()
                speedup = speedup.replace('(', ' ').replace(')', ' ')
                speed_data = speedup.split()
                # print(speed_data)

                speedup_r.append(float(speed_data[0]))
                is_r.append(float(speed_data[1]))
                grid_r.append(float(speed_data[3]))

                speedup_i.append(float(speed_data[5]))
                is_i.append(float(speed_data[6]))
                grid_i.append(float(speed_data[8]))

                speedup_total.append(float(speed_data[10]))
                is_total.append(float(speed_data[11]))
                grid_total.append(float(speed_data[13]))


    iteration = np.arange(len(speedup_r))

    fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(8, 9))
    ax[0].plot(iteration, is_r, label='grid by importance sampling')
    ax[0].plot(iteration, grid_r, label='size of grid')
    ax[0].legend()
    ax[0].set_ylabel('number of projections')
    ax[1].plot(iteration, is_r, label='importance sampling')
    ax[2].plot(iteration, speedup_r, label='speedup', color='g')
    ax[2].legend()
    ax[2].set_xlabel('Iteration: #')
    ax[2].set_ylabel('speedup factor')
    fig.suptitle(r'Importance Sampling with $(\phi, \theta)$ (dir on sphere)')

    fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(8, 9))
    ax[0].plot(iteration, is_i, label='grid by importance sampling')
    ax[0].plot(iteration, grid_i, label='size of grid')
    ax[0].legend()
    ax[0].set_ylabel('number of projections')
    ax[1].plot(iteration, is_i, label='importance sampling')
    ax[2].plot(iteration, speedup_i, label='speedup', color='g')
    ax[2].legend()
    ax[2].set_xlabel('Iteration: #')
    ax[2].set_ylabel('speedup factor')
    fig.suptitle(r'Importance Sampling with $(\psi)$ (inplane-rotation)')

    fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(8, 9))
    ax[0].plot(iteration, is_total, label='grid by importance sampling')
    ax[0].plot(iteration, grid_total, label='size of grid')
    ax[0].legend()
    ax[0].set_ylabel('number of projections')
    ax[1].plot(iteration, is_total, label='importance sampling')
    ax[2].plot(iteration, speedup_total, label='speedup', color='g')
    ax[2].legend()
    ax[2].set_xlabel('Iteration: #')
    ax[2].set_ylabel('speedup factor')
    fig.suptitle(r'Importance Sampling (Total)')
    
    plt.show()

if __name__ == '__main__':
    stdout_file = sys.argv[1]
    timer(stdout_file)
    is_eff(stdout_file)