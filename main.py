import os
import time
import argparse
import multiprocessing as mp

import numpy as np
import healpy as hp

from cryoio import star, mrc
from cryoem import relion, xmipp, cryoem
from geom import geom


"""
lessons:
1. test ctypes array whether support indice slicing (support 1d array but nd array)

undone:
2. standard image preprocessing
"""

arr_type = type(np.ctypeslib.as_ctypes(np.float32()))

def gen_exp_samples(num_EAs, phantompath, data_dst):
    EAs = list()
    for i in range(num_EAs):
        # Randomly generate the viewing direction/shift
        pt = np.random.randn(3)
        pt /= np.linalg.norm(pt)
        psi = 2 * np.pi * np.random.rand()
        EA = geom.genEA(pt)[0]
        EA[2] = psi
        EAs.append(np.rad2deg(EA))

    # EAs_grid = gen_ref_EAs_grid()
    # grid_size = EAs_grid.shape[0]
    # EAs = EAs_grid[np.random.randint(0, grid_size, size=num_EAs)]
    ang_star = data_dst + '_gen.star'
    star.easy_writeSTAR_relion(ang_star, EAs=EAs)
    projs_star, projs_mrcs = relion.project(phantompath, data_dst,
                                            ang=ang_star)
    return projs_star, projs_mrcs


def gen_ref_EAs_grid(nside=8, psi_step=10):

    npix = hp.nside2npix(nside)
    resol = np.rad2deg(hp.nside2resol(nside))
    theta, phi = hp.pix2ang(nside, range(npix))
    theta, phi = np.rad2deg(theta), np.rad2deg(phi)
    psi = np.arange(0, 360, psi_step)
    # indexing = 'xy'
    grid_theta, grid_psi = np.meshgrid(theta, psi)
    grid_phi, _ = np.meshgrid(phi, psi)
    EAs_tuple = (grid_phi.flatten(), grid_theta.flatten(), grid_psi.flatten())
    EAs_grid = np.vstack(EAs_tuple).T
    print('number of points on shpere: {0}. \n'
          'resolution: {1:.2f} degree, step of inplane-rotation: {2} degree\n'
          'total grid size: {3}'.format(npix, resol, psi_step, npix*360/psi_step))
    return EAs_grid


def gen_mrcs_from_EAs(EAs, phantompath, dstpath):

    ang_star = os.path.join(dstpath, 'EAs.star')
    star.easy_writeSTAR_relion(ang_star, EAs=EAs)
    projs_star, projs_mrcs = relion.project(phantompath, dstpath, ang=ang_star)
    model_projs = mrc.readMRCimgs(projs_mrcs, idx=0)

    return model_projs


def standard_processing(img_or_imgstack):
    pass


def find_similar(exp_proj):
    tic = time.time()
    grid_size = model_proj_imgs.shape[2]
    likelihood = np.zeros(grid_size)
    for i in range(grid_size):
        ref_proj = model_proj_imgs[:, :, i]
        p = np.mean((exp_proj - ref_proj) ** 2 / (-2 * exp_proj ** 2))
        likelihood[i] = p
    idx = np.argmax(likelihood)
    toc = time.time() - tic
    if toc > 3:
        print('\r{0} forloops cost {1:.4f} seconds.'.format(grid_size, toc), end='')
    return idx


def projection_matching(input_model, projs, nside, dir_suffix=None, **kwargs):
    """
    Parameters:
    -------
    input_model: path of input mrc file
    projections: numpy array
    """
    try:
        WD = kwargs['WD']
    except KeyError as error:
        error.__doc__
    EAs_grid = gen_ref_EAs_grid(nside=nside, psi_step=10)
    if dir_suffix:
        dstpath = os.path.join(WD, dir_suffix, 'model_projections')
    else:
        dstpath = os.path.join(WD, 'model_projections')

    N = projs.shape[0]
    num_projs = projs.shape[2]
    num_model_imgs = EAs_grid.shape[0]
    num_threads = mp.cpu_count()
    tic = time.time()
    global model_proj_imgs
    model_proj_imgs = gen_mrcs_from_EAs(EAs_grid, input_model, dstpath)
    print('Time to recover projections from mrcs file: {0:.4f} s'.format(
        time.time() - tic))

    print('Projection matching: multiprocess start')
    with mp.Pool(processes=num_threads) as pool:
        idx_orientations = pool.map(
            find_similar, (projs[:, :, i] for i in range(num_projs)))
    print('\nFinish orientation!')
    orientations = EAs_grid[idx_orientations]
    return orientations


def reconstruct(projs_path, nside, psi_step, **kwargs):

    try:
        WD = kwargs['WD']
    except KeyError as error:
        error.__doc__
    exp_samples = mrc.readMRCimgs(projs_path, idx=0)
    input_shape = exp_samples.shape
    print('size of input images: {0}x{1}, number of input images: {2}'.format(*input_shape))
    print('generating random phantom density')
    N = input_shape[0]
    M = cryoem.generate_phantom_density(N, 0.95 * N / 2.0, 5 * N / 128.0, 30)
    output_mrc = os.path.join(WD, 'initial_random_model.mrc')
    mrc.writeMRC(output_mrc, M)
    print('Start projection matching')
    total_iteration = 20
    for it in range(total_iteration):
        print('Iteration {0}'.format(it))
        dir_suffix = 'it' + str(it).zfill(3)
        iter_dir = os.path.join(WD, dir_suffix)
        os.makedirs(iter_dir, exist_ok=True)

        tic = time.time()

        input_model = output_mrc
        ori = projection_matching(input_model, exp_samples, nside=nside, dir_suffix=dir_suffix)

        ori_star = os.path.join(WD, dir_suffix, 'orientations.star')
        star.easy_writeSTAR_xmipp(ori_star, EAs=ori, imgs_path=projs_path)

        output_mrc = os.path.join(WD, dir_suffix, dir_suffix + '_reco.mrc')
        xmipp.reconstruct_fourier(ori_star, output_mrc, thr=4)

        toc = time.time() - tic

        print('Iteration {0} finished, spread time: {1:.4f} seconds'.format(it, toc))
        print('Total iteration is {0}, remian {1:.2f} minutes'.format(
            total_iteration, (total_iteration-it)*toc/60))
        print('---------------------------------------------------------------------------------')


def main():
    paser = argparse.ArgumentParser()
    paser.add_argument('-j', '--job_id', help='ID for Job', type=int)
    paser.add_argument('-n', '--num_projs', help='generate N projections for simulation', type=int)
    paser.add_argument('--nside', help='nside for healpix to generate reference grid.',
                       type=float)
    paser.add_argument('--psi_step', help='step for generating psi grids', type=int)
    paser.add_argument('--print_to_file', type=bool, default=False)

    args = paser.parse_args()
    job_id = args.job_id
    num = args.num_projs
    nside = args.nside
    psi_step = args.psi_step
    print_to_file = args.print_to_file

    WD = os.path.join(os.path.dirname(__file__), 'data', 'job'+str(job_id))
    os.makedirs(WD, exist_ok=True)

    if print_to_file:
        import sys
        sys.stdout = open(os.path.join(WD, 'log.txt'), 'w')
    exp_star, exp_mrcs = gen_exp_samples(num,
                                         os.path.join(__file__, 'particle/EMD-6044.mrc'),
                                         os.path.join(WD, 'exp_projections'))
    reconstruct(exp_mrcs, nside=nside, psi_step=psi_step, **{'WD': WD})
    if print_to_file:
        sys.stdout.close()


if __name__ == '__main__':
    main()
