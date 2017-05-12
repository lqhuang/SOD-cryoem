import os
from time import time
import pickle

import numpy as np
from skimage.measure import block_reduce
import sklearn.preprocessing as prep

from cryoem import relion
from cryoio import mrc, star
from geom import geom
from geom import healpix as hp


WD = '/Users/lqhuang/Git/SOD-cryoem'

# phantompath = os.path.join(WD, 'particle', '1AON.mrc')
phantompath = os.path.join(WD, 'particle', 'EMD-6044.mrc')
recom_contour = 17.0

N = 128
sigma_noise = 25.0

M_totalmass = 80000
M = mrc.readMRC(phantompath)
M[M<recom_contour] = 0
if M_totalmass is not None:
    M *= M_totalmass / M.sum()

tic = time()

def gen_euler_angles(num_EAs):
    EAs = list()
    for i in range(num_EAs):
        # Randomly generate the viewing direction/shift
        pt = np.random.randn(3)
        pt /= np.linalg.norm(pt)
        psi = 2 * np.pi * np.random.rand()
        EA = geom.genEA(pt)[0]
        EA[2] = psi
        EAs.append(np.rad2deg(EA))
        # print(EA)
    return EAs

# generate train dataset
num_train = 10000
ori_train = gen_euler_angles(num_train)
star.easy_writeSTAR(os.path.join(WD, 'data', 'EMD-6044', 'train.star'), EAs=ori_train)
train_star, train_mrcs = relion.project(phantompath, os.path.join(WD, 'data', 'EMD-6044', 'EMD6044_train'),
                                        ang=os.path.join(WD, 'data', 'EMD-6044', 'train.star'))
# generate test dataset
num_test = 1000
ori_test = gen_euler_angles(num_test)
star.easy_writeSTAR(os.path.join(WD, 'data', 'EMD-6044', 'test.star'), EAs=ori_test)
test_star, test_mrcs = relion.project(phantompath, os.path.join(WD, 'data', 'EMD-6044', 'EMD6044_test'),
                                      ang=os.path.join(WD, 'data', 'EMD-6044', 'test.star'))

# train_star, train_mrcs = './data/1AON_train.star', './data/1AON_train.mrcs'
# test_star, test_mrcs = './data/1AON_train.star', './data/1AON_train.mrcs'

# train_imgs = mrc.readMRC(train_mrcs, 0)
# test_imgs = mrc.readMRC(test_mrcs, 0)

# def dump_imgset(images, fname):
#     assert images.shape[0] == images.shape[1]
#     imgset = np.zeros([images.shape[2], int(images.shape[0]/2 * images.shape[1]/2)], dtype=np.float32)
#     for i in range(images.shape[2]):
#         img = block_reduce(images[:, :, i], block_size=(2, 2), func=np.mean)
#         # imgset_scaled = prep.scale(img, axis=1)
#         imgset[i] = img.flatten()
#         print('mean: ', np.mean(imgset[i]), '\n std:', np.std(imgset[i]))
#     with open('./data/'+fname, 'wb') as pkl_file:
#         pickle.dump(imgset, pkl_file)

# dump_imgset(train_imgs, '1AON_train.pkl')
# dump_imgset(test_imgs, '1AON_test.pkl')
