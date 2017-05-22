import os
import glob
import argparse

import numpy as np
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
from matplotlib import gridspec

from cryoio import star


def calc_rmse(a, b):
    """
    [[11, 12, 13],
     [21, 22, 23],
     ...,
     [n1, n2, n3]]
    """
    return np.sqrt(np.mean((a - b)**2, axis=1))


def plot_rmse(working_directory):

    figure_dir = os.path.join(working_directory, 'Figures')
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir, exist_ok=True)

    x_grid = np.arange(0, 360, 10)
    correct = star.get_EAs_from_star(os.path.join(
        working_directory, 'exp_projections.star'))

    plt.figure(0)
    first = star.get_EAs_from_star(os.path.join(
        working_directory, 'it000', 'orientations.star'))
    correct_rmse = calc_rmse(correct, first)
    # plt.hist(correct_rmse)]
    correct_kde = gaussian_kde(correct_rmse)
    plt.plot(x_grid, correct_kde.evaluate(x_grid))
    # plt.ylim([0, 1])
    plt.xlabel('RMSE for 3 Euler angles')
    plt.ylabel('Count')
    plt.title('Compare with correct angle distribution')
    plt.savefig(os.path.join(figure_dir, 'it000'), dpi=150)

    exp_folder = glob.glob(os.path.join(working_directory, 'it*'))

    for i, folder in enumerate(exp_folder):
        last = star.get_EAs_from_star(os.path.join(
            folder, 'orientations.star'))
        now = star.get_EAs_from_star(os.path.join(
            folder, 'orientations.star'))
        correct_rmse = calc_rmse(correct, now)
        last_rmse = calc_rmse(last, now)

        fig = plt.figure(num=i, figsize=(16, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
        plt.suptitle('Iteration: {0}'.format(i))
        plt.subplot(gs[0])
        # plt.hist(correct_rmse)
        correct_kde = gaussian_kde(correct_rmse)
        plt.plot(x_grid, correct_kde.evaluate(x_grid))
        plt.xlabel('RMSE for 3 Euler angles')
        plt.ylabel('Count')
        plt.title('Compare with correct angle distribution')
        plt.subplot(gs[1])
        # plt.hist(last_rmse)
        last_kde = gaussian_kde(last_rmse)
        plt.plot(x_grid, last_kde.evaluate(x_grid))
        plt.xlabel('RMSE for 3 Euler angles')
        plt.ylabel('Count')
        plt.title('Compare with angle distribution of last iteration')
        plt.savefig(os.path.join(figure_dir, 'it' + str(i).zfill(3)),
                    dpi=150, bbox_inches='tight')


def main():
    # working_directory = '/home/lqhuang/Git/SOD-cryoem/data/job1'
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--working_directory', type=str)
    args = parser.parse_args()
    working_directory = args.working_directory
    WD = os.path.realpath(working_directory)
    plot_rmse(WD)


if __name__ == '__main__':
    main()
