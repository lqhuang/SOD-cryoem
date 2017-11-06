#!/usr/bin/env python
from __future__ import print_function, division

import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
import argparse
from random import randint

from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from cryoio import mrc
import density
import geometry


def gen_slices(model_files, fspace=False, log_scale=True):
    for model in model_files:
        M = mrc.readMRC(model)
        N = M.shape[0]
        mask_freq = 0.010
        mask = geometry.gen_dense_mask(N, 2, mask_freq, psize=2.8)
        print('model size: {0}x{0}x{0}'.format(N))

        if fspace:
            fM = M
        else:
            M_totalmass = 500000
            M *= M_totalmass / M.sum()

            zeropad = 1
            zeropad_size = int(zeropad * (N / 2))
            zp_N = zeropad_size * 2 + N
            zpm_shape = (zp_N,) * 3
            zp_M = np.zeros(zpm_shape, dtype=density.real_t)
            zpm_slices = (slice( zeropad_size, (N + zeropad_size) ),) * 3
            zp_M[zpm_slices] = M
            zp_fM = density.real_to_fspace(zp_M)
            fM = zp_fM[zpm_slices]
            fM = fM.real ** 2 + fM.imag ** 2

            mask_3D = geometry.gen_dense_mask(N, 3, mask_freq, psize=2.8)
            fM *= mask_3D
            mrc.writeMRC('particle/1AON_fM_totalmass_{}_oversampling_{}.mrc'.format(str(int(M_totalmass)).zfill(5), zeropad), fM, psz=2.8)

        slicing_func = RegularGridInterpolator([np.arange(N),]*3, fM, bounds_error=False, fill_value=0.0)
        coords = geometry.gencoords_base(N, 2)

        fig, axes = plt.subplots(3, 3, figsize=(12.9, 9.6))
        for i, ax in enumerate(axes.flat):
            row, col = np.unravel_index(i, (3, 3))
            
            # Randomly generate the viewing direction/shift
            pt = np.random.randn(3)
            pt /= np.linalg.norm(pt)
            psi = 2 * np.pi * np.random.rand()
            EA = geometry.genEA(pt)[0]
            EA[2] = psi
            R = geometry.rotmat3D_EA(*EA)[:, 0:2]
            rotated_coords = R.dot(coords.T).T + int(N/2)
            img = slicing_func(rotated_coords).reshape(N, N)
            img = np.require(np.random.poisson(img), dtype=np.float32)

            if log_scale:
                img = np.log(np.maximum(img, 0)) * mask
            else:
                img *= mask

            im = ax.imshow(img, origin='lower')  # cmap='Greys'
            if row == 2:
                ax.set_xticks([0, int(N/4.0), int(N/2.0), int(N*3.0/4.0), int(N-1)])
            else:
                ax.set_xticks([])
            if col == 0:
                ax.set_yticks([0, int(N/4.0), int(N/2.0), int(N*3.0/4.0), int(N-1)])
            else:
                ax.set_yticks([])
            fig.colorbar(im, ax=ax)

        # fig.subplots_adjust(right=0.8)
        # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        # fig.colorbar(im, cax=cbar_ax)
        fig.suptitle('simulated experimental data of XFEL for'.format(model))
        # fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_files", help="list of mrcs files.", nargs='+')
    parser.add_argument("-l", "--log_scale", help="show image in log scale.",
                        action="store_true")
    parser.add_argument('-f', '--fspace', help='input model has been already in Fourier space.', action='store_true')
    args = parser.parse_args()

    log_scale = args.log_scale
    model_files = args.model_files
    fspace = args.fspace
    print('model_files:', model_files)
    print('fspace:', fspace)
    print('log_scale', log_scale)
    gen_slices(model_files, fspace=fspace, log_scale=log_scale)
