import time, os, sys
sys.path.append(os.path.dirname(sys.path[0]))

from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

from cryoio.imagestack import MRCImageStack, CombinedImageStack, FourierStack
from cryoio.ctfstack import CTFStack, CombinedCTFStack
from cryoio.dataset import CryoDataset

from cryoio.mrc import writeMRC, readMRC
import cryoem
import cryoops
import density
import geometry
from symmetry import get_symmetryop


from noise_test import plot_noise_histogram, plot_stack_noise


def load_dataset_test(data_dir):
    params = {
        'dataset_name': "1AON",
        'inpath': os.path.join(data_dir, 'imgdata.mrc'),
        'ctfpath': os.path.join(data_dir, 'defocus.txt'),
        'microscope_params': {'akv': 200, 'wgh': 0.07, 'cs': 2.0},
        'resolution': 2.8,
        'sigma': 'noise_std',
        'sigma_out': 'data_std',
        'minisize': 200,
        'test_imgs': 200,
        'random_seed': 1
    }
    cparams = {}

    imgpath = params['inpath']
    psize = params['resolution']
    imgstk = MRCImageStack(imgpath, psize)

    if params.get('float_images', True):
        imgstk.float_images()

    ctfpath = params['ctfpath']
    mscope_params = params['microscope_params']

    ctfstk = CTFStack(ctfpath, mscope_params)

    cryodata = CryoDataset(imgstk, ctfstk)
    cryodata.compute_noise_statistics()
    if params.get('window_images',True):
        imgstk.window_images()

    cryodata.set_datasign(params.get('datasign', 'auto'))
    if params.get('normalize_data',True):
        cryodata.normalize_dataset()

    voxel_size = cryodata.pixel_size

    fspace_stack = FourierStack(cryodata.imgstack,
                                caching = True, zeropad=1)

    N = cryodata.imgstack.get_num_pixels()
    premult = cryoops.compute_premultiplier(N + 2 * int(1 * (N/2)),
                                            'lanczos', 8)
    premult = premult.reshape((-1,1)) * premult.reshape((1,-1))
    fspace_stack.set_transform(premult, 1)


    N = cryodata.imgstack.get_num_pixels()
    rad = 0.99
    coords = geometry.gencoords(N, 2).reshape((N**2, 2))
    Cs = np.sum(coords**2, axis=1).reshape((N, N)) > (rad * N / 2.0 - 1.5)**2

    idx = np.random.randint(cryodata.imgstack.num_images)
    normalized = cryodata.imgstack.get_image(1)
    f_normalized = fspace_stack.get_image(1)

    plot_noise_histogram(normalized, f_normalized,
        rmask=~Cs, fmask=None, plot_unmask=False)
    plt.show()


if __name__ == '__main__':
    print(sys.argv)
    dataset_dir = sys.argv[1]
    load_dataset_test(dataset_dir)
