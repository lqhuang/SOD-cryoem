from __future__ import print_function, division

import os
import time

import numpy as np

from cryoio import mrc
from cryoio import star

# from util import run_system_command
"""
loop_ 
_rlnImageName #1 
_rlnAngleRot #2
_rlnAngleTilt #3
_rlnAnglePsi #4 
_rlnOriginX #5
_rlnOriginY #6 
"""

# def timetest():
if __name__ == '__main__':
    fname = '{0}/test_profiles.star'.format('./particle')
    # label_names = ['ImageName', 'AngleRot', 'AngleTilt', 'AnglePsi', 'OriginX', 'OriginY']
    label_names = ['ImageName', 'AngleRot', 'AngleTilt', 'AnglePsi']
    # label_names = ['AngleRot', 'AngleTilt', 'AnglePsi']

    step = 30.0
    rot = np.arange(0, 360.0, step)
    tilt = np.arange(0, 180.0+step, step)
    psi = np.arange(0, 360.0, step)

    imgs_length = rot.shape[0] * tilt.shape[0] * psi.shape[0]

    xshift = np.zeros(imgs_length)
    yshift = np.zeros(imgs_length)

    metadata = dict()
    for label in label_names:
        metadata[label] = list()
    for idx in range(imgs_length):
        i, j, k = np.unravel_index(idx, [rot.shape[0], tilt.shape[0], psi.shape[0]])
        metadata['ImageName'].append(idx)
        metadata['AngleRot'].append(rot[i])
        metadata['AngleTilt'].append(tilt[j])
        metadata['AnglePsi'].append(psi[k])
        # metadata['OriginX'].append(xshift[idx])
        # metadata['OriginY'].append(yshift[idx])

    a = metadata['ImageName']
    print(a[0])
    print(metadata['ImageName'][0])
    b = metadata['AnglePsi']
    star.writeSTAR('./particle/test_profiles.star', \
                   imgs_path='Users/lqhuang/Git/Test/particle/projections.mrcs', AnglePsi=b, ImageName=a)
    print('a', a[0])
    print(metadata['ImageName'][0])
    # print('Calculate projections')
    # tic = time.time()
    # project_cmd = 'relion_project --i {0}/1AON.mrc --o {0}/projections --ang {0}/test_profiles.star --angpix 3.0'.format('/Users/lqhuang/Git/Test/particle')
    # info = run_system_command(project_cmd)
    # print(info)
    # print('projection times: ', time.time()-tic)

    # data = mrc.readMRCimgs('./particle/projections.mrcs', 0)

    # x, y, leng = data.shape
    # cx = int(x/2)
    # cy = int(y/2)

    # resave = np.zeros_like(data)

    # xarr = np.arange(0, x) - float(cx)
    # yarr = np.arange(0, y) - float(cy)
    # xx, yy = np.meshgrid(xarr,yarr)
    # mask = ((xx**2 + yy*2) < 0) * 1.0

    # for idx in range(leng):

    #     mean_img = np.mean(data[:,:,idx].flatten())
    #     var_img = np.var(data[:,:,idx].flatten())
    #     resave[:,:,idx] = (data[:,:,idx] - mean_img) / np.sqrt(var_img)
    #     resave[:,:,idx] *= mask

    # mrc.writeMRC('./particle/projections.mrcs', resave)


    # print('Calculate reconstruction')
    # tic = time.time()
    # reconstruct_cmd = 'relion_reconstruct --i {0}/projections.star --o {0}/test.mrc --angpix 3.0 --j 8 --fsc True'.format('/Users/lqhuang/Git/Test/particle')
    # info = run_system_command(reconstruct_cmd)
    # # print(info)
    # print('reconstruction times: ', time.time()-tic)