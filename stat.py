import os

import numpy as np
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
from matplotlib import gridspec

from cryoio import star

WD = '/home/lqhuang/Git/SOD-cryoem/data/job1'
figure_dir = os.path.join(WD, 'Figures')
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)

def calc_rmse(a, b):
    """
    [[11, 12, 13],
     [21, 22, 23],
     ...,
     [n1, n2, n3]]
    """
    return np.sqrt(np.mean((a - b)**2, axis=1))

x_grid = np.arange(0, 360, 10)

correct = star.get_EAs_from_star(os.path.join(WD, 'exp_projections.star'))

plt.figure(0)
first = star.get_EAs_from_star(os.path.join(WD, 'it000', 'orientations.star'))
correct_rmse = calc_rmse(correct, first)
# plt.hist(correct_rmse)]
correct_kde = gaussian_kde(correct_rmse)
plt.plot(x_grid, correct_kde.evaluate(x_grid))
# plt.ylim([0, 1])
plt.xlabel('RMSE for 3 Euler angles')
plt.ylabel('Count')
plt.title('Compare with correct angle distribution')
plt.savefig(os.path.join(figure_dir, 'it000'), dpi=150)

for i in range(1, 20):
    last = star.get_EAs_from_star(os.path.join(WD, 'it'+str(i-1).zfill(3), 'orientations.star'))
    now = star.get_EAs_from_star(os.path.join(WD, 'it'+str(i).zfill(3), 'orientations.star'))
    correct_rmse = calc_rmse(correct, now)
    last_rmse = calc_rmse(last, now)

    fig = plt.figure(num=i, figsize=(16, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    fig = plt.figure(i)
    plt.suptitle('Iteration: {0}'.format(i))
    plt.subplot(gs[0])
    # plt.hist(correct_rmse)
    correct_kde = gaussian_kde(correct_rmse)
    plt.plot(x_grid, correct_kde.evaluate(x_grid))
    # plt.ylim([0, 1])
    plt.xlabel('RMSE for 3 Euler angles')
    plt.ylabel('Count')
    plt.title('Compare with correct angle distribution')
    plt.subplot(gs[1])
    # plt.hist(last_rmse)
    last_kde = gaussian_kde(last_rmse)
    plt.plot(x_grid, last_kde.evaluate(x_grid))
    # plt.ylim([0, 1])
    plt.xlabel('RMSE for 3 Euler angles')
    plt.ylabel('Count')
    plt.title('Compare with angle distribution of last iteration')
    plt.savefig(os.path.join(figure_dir, 'it'+str(i).zfill(3)), dpi=150, bbox_inches='tight')
