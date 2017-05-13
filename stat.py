import os

import numpy as np
from matplotlib import pyplot as plt

from cryoio import star

WD = '/home/lqhuang/Git/SOD-cryoem/data/job'
ori = star.get_EAs_from_star(os.path.join(WD, 'exp_projections.star'))
correct = np.asarray(ori).T

ori = star.get_EAs_from_star(os.path.join(WD, 'it000', 'orientations.star'))
rec = np.asarray(ori).T


def rmse(a, b):
    """
    [[11, 12, 13],
     [21, 22, 23],
     ...,
     [n1, n2, n3]]
    """
    return np.sqrt(np.mean((a - b)**2, axis=1))


it000_rmse = rmse(correct, rec)

print(it000_rmse)

plt.hist(it000_rmse)
plt.show()