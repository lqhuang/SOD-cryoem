from __future__ import print_function, division

import time, os, sys
sys.path.append(os.path.dirname(sys.path[0]))

from copy import copy
try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
from matplotlib import pyplot as plt

from cryoio.imagestack import MRCImageStack, FourierStack
from cryoio.ctfstack import CTFStack
from cryoio.dataset import CryoDataset
from cryoio import ctf
import cryoops
import density
import geometry
from symmetry import get_symmetryop

cython_build_dirs = os.path.expanduser('~/.pyxbld/angular_correlation')
import pyximport; pyximport.install(
    build_dir=cython_build_dirs, setup_args={"include_dirs": np.get_include()}, reload_support=True)
import sincint

from noise_test import plot_noise_histogram, plot_stack_noise


class SimpleDataset():
    def __init__(self, model, dataset_params, ctf_params,
                 interp_params={'kern': 'lanczos', 'kernsize': 8.0, 'zeropad': 1, 'dopremult': True},
                 try_load_cache=True):
        assert isinstance(model, np.ndarray), "Unexpected data type for input model"

        self.dataset_params = dataset_params
        self.num_pixels = model.shape[0]
        N = self.num_pixels
        self.num_images = dataset_params['num_images']
        assert self.num_images > 1, "it's better to make num_images larger than 1."
        self.pixel_size = float(dataset_params['pixel_size'])
        euler_angles = dataset_params['euler_angles']
        self.is_sym = get_symmetryop(dataset_params.get('symmetry', None))

        if euler_angles is None and self.is_sym is None:
            pt = np.random.randn(self.num_images, 3)
            pt /= np.linalg.norm(pt, axis=1, keepdims=True)
            euler_angles = geometry.genEA(pt)
            euler_angles[:, 2] = 2 * np.pi * np.random.rand(self.num_images)
        elif euler_angles is None and self.is_sym is not None:
            euler_angles = np.zeros((self.num_images, 3))
            for i, ea in enumerate(euler_angles):
                while True:
                    pt = np.random.randn(3)
                    pt /= np.linalg.norm(pt)
                    if self.is_sym.in_asymunit(pt.reshape(-1, 3)):
                        break
                ea[0:2] = geometry.genEA(pt)[0][0:2]
                ea[2] = 2 * np.pi * np.random.rand()
        self.euler_angles = euler_angles.reshape((-1, 3))

        if ctf_params is not None:
            self.use_ctf = True
            ctf_map = ctf.compute_full_ctf(None, N, ctf_params['psize'],
                ctf_params['akv'], ctf_params['cs'], ctf_params['wgh'],
                ctf_params['df1'], ctf_params['df2'], ctf_params['angast'],
                ctf_params['dscale'], ctf_params.get('bfactor', 500))
            self.ctf_params = copy(ctf_params)
            if 'bfactor' in self.ctf_params.keys():
                self.ctf_params.pop('bfactor')
        else:
            self.use_ctf = False
            ctf_map = np.ones((N**2,), dtype=density.real_t)

        sigma_noise = float(dataset_params.get('sigma_noise', 25.0))
        kernel = 'lanczos'
        ksize = 6
        rad = 0.95
        premult = cryoops.compute_premultiplier(N, kernel, ksize)
        TtoF = sincint.gentrunctofull(N=N, rad=rad)
        premulter =   premult.reshape((1, 1, -1)) \
                    * premult.reshape((1, -1, 1)) \
                    * premult.reshape((-1, 1, 1))
        fM = density.real_to_fspace(premulter * model)

        # if try_load_cache:
        #     try:
        print("Generating Dataset ... :")
        tic = time.time()
        imgdata = np.empty((self.num_images, N, N), dtype=density.real_t)
        for i, ea in enumerate(self.euler_angles):
            R = geometry.rotmat3D_EA(*ea)[:, 0:2]
            slop = cryoops.compute_projection_matrix(
                [R], N, kernel, ksize, rad, 'rots')
            D = slop.dot(fM.reshape((-1,)))
            imgdata[i] = density.fspace_to_real((ctf_map * TtoF.dot(D)).reshape((N, N))) + np.require(
                np.random.randn(N, N) * sigma_noise, dtype=density.real_t)
            from multiprocessing import pool, cpu_count
        self.imgdata = imgdata
        print("  cost {} seconds.".format(time.time()-tic))


        self.set_transform(interp_params)
        self.prep_processing()

    def __iter__(self):
        return self.imgdata.__iter__()

    def get_pixel_size(self):
        return self.pixel_size

    def get_num_images(self):
        return self.num_images

    def get_num_pixels(self):
        return self.num_pixels

    def scale_images(self, scale):
        self.imgdata *= scale

    def scale_ctfs(self, scale):
        if self.use_ctf:
            self.ctf_params['dscale'] *= scale

    def prep_processing(self):
        self.compute_noise_statistics()
        self.normalize_dataset()

    def compute_variance(self):
        vals = []
        for img in self:
            vals.append(np.mean(img**2, dtype=np.float64))
        return np.mean(vals, dtype=np.float64)

    def estimate_noise_variance(self, esttype='robust', zerosub=False, rad=1.0):
        N = self.get_num_pixels()
        Cs = np.sum(geometry.gencoords(N, 2).reshape((N**2, 2))**2,
                    axis=1).reshape((N, N)) > (rad * N / 2.0 - 1.5)**2
        vals = []
        for img in self:
            cvals = img[Cs]
            vals.append(cvals)

        if esttype == 'robust':
            if zerosub:
                var = (
                    1.4826 * np.median(np.abs(np.asarray(vals) - np.median(vals))))**2
            else:
                var = (1.4826 * np.median(np.abs(vals)))**2
        elif esttype == 'mle':
            var = np.mean(np.asarray(vals)**2, dtype=np.float64)
            if zerosub:
                var -= np.mean(vals, dtype=np.float64)**2
        return var

    def compute_noise_statistics(self):
        self.noise_var = self.estimate_noise_variance()
        self.data_var = self.compute_variance()

        print('Dataset noise profile')
        print('  Noise: {0:.3g}'.format(np.sqrt(self.noise_var)))
        print('  Data: {0:.3g}'.format(np.sqrt(self.data_var)))
        assert self.data_var > self.noise_var
        self.signal_var = self.data_var - self.noise_var
        print('  Signal: {0:.3g}'.format(np.sqrt(self.signal_var)))
        print('  Signal-to-Noise Ratio: {0:.1f}% ({1:.1f}dB)'.format(100 * self.signal_var / self.noise_var, 10 * np.log10(self.signal_var / self.noise_var)))

    def normalize_dataset(self):
        self.real_noise_var = self.noise_var
        self.scale_images(1.0 / np.sqrt(self.noise_var))
        self.scale_ctfs(1.0 / np.sqrt(self.noise_var))

        self.data_var = self.data_var / self.noise_var
        self.signal_var = self.signal_var / self.noise_var
        self.noise_var = 1.0

    def get_image(self, idx):
        return self.imgdata[idx]

    def set_transform(self, interp_params, caching=True):
        self.caching = caching
        self.transformed = {}
        self.interp_params = interp_params

        zeropad = interp_params.get('zeropad', 1)
        kernel = interp_params.get('kern', 'lanczos')
        kernsize = interp_params.get('kernsize', 8)

        self.zeropad = int(zeropad * (self.get_num_pixels() / 2))
        Nzp = 2 * self.zeropad + self.num_pixels
        self.zpimg = np.zeros((Nzp, Nzp), dtype=density.real_t)
    
        if interp_params.get('dopremult', True):
            premult = cryoops.compute_premultiplier(Nzp, kernel, kernsize)
            reshape = ((-1, 1), (1, -1))
            self.premult = np.prod([premult.reshape(rs) for rs in reshape])
        else:
            self.premult = None
        if self.premult is not None:
            assert self.premult.shape[0] == Nzp
            assert self.premult.shape[1] == Nzp

    def get_fft_image(self, idx):
        if not self.caching:
            self.transformed = {}
        if idx not in self.transformed:
            N = self.get_num_pixels()
            zpimg = self.zpimg
            slice_indices = [slice(self.zeropad, N+self.zeropad)] * 2
            zpimg[slice_indices] = self.get_image(idx)

            if self.premult is not None:
                zpimg = self.premult * zpimg
            self.transformed[idx] = density.real_to_fspace(zpimg)

        return self.transformed[idx]
    
    def get_ctf(self, idx):
        if self.use_ctf:
            self.cCTF = ctf.ParametricCTF(self.ctf_params)
            return self.cCTF
        else:
            raise NotImplementedError("CTF is disable here.")


def dataset_loading_test(params, visualize=False):
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
    cryodata.divide_dataset(params['minisize'], params['test_imgs'],
                            params['partition'], params['num_partitions'], params['random_seed'])
    cryodata.set_datasign(params.get('datasign', 'auto'))
    if params.get('normalize_data',True):
        cryodata.normalize_dataset()

    # voxel_size = cryodata.pixel_size
    N = cryodata.imgstack.get_num_pixels()

    fspace_stack = FourierStack(cryodata.imgstack,
                                caching = True, zeropad=1)
    premult = cryoops.compute_premultiplier(N + 2 * int(1 * (N/2)), 'lanczos', 8)
    premult = premult.reshape((-1,1)) * premult.reshape((1,-1))
    fspace_stack.set_transform(premult, 1)

    if visualize:
        rad = 0.99
        coords = geometry.gencoords(N, 2).reshape((N**2, 2))
        Cs = np.sum(coords**2, axis=1).reshape((N, N)) > (rad * N / 2.0 - 1.5)**2

        idx = np.random.randint(cryodata.imgstack.num_images)
        normalized = cryodata.imgstack.get_image(1)
        f_normalized = fspace_stack.get_image(1)

        plot_noise_histogram(normalized, f_normalized,
            rmask=~Cs, fmask=None, plot_unmask=False)
        plt.show()
    
    return cryodata, fspace_stack


if __name__ == '__main__':
    print(sys.argv)
    dataset_dir = sys.argv[1]
    data_params = {
        'dataset_name': "1AON",
        'inpath': os.path.join(dataset_dir, 'imgdata.mrc'),
        'ctfpath': os.path.join(dataset_dir, 'defocus.txt'),
        'microscope_params': {'akv': 200, 'wgh': 0.07, 'cs': 2.0},
        'resolution': 2.8,
        'sigma': 'noise_std',
        'sigma_out': 'data_std',
        'minisize': 150,
        'test_imgs': 20,
        'partition': 0,
        'num_partitions': 0,
        'random_seed': 1
    }
    cryodata, fstack = dataset_loading_test(data_params, visualize=True)
