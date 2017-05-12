import os.path
import tempfile
import multiprocessing as mp

import numpy as np
import healpy as hp

from cryoio import star, mrc
from cryoem import relion, cryoem


"""
lessons:
1. test ctypes array whether support indice slicing (support 1d array but nd array)

undone:
2. standard image preprocessing
"""

WD = '/Users/lqhuang/Git/SOD-cryoem/data/job'
# temp_directory = tempfile.mkdtemp()
temp_directory = os.path.join(WD, 'tmp')

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

    exp_star = data_dst + '_gen.star'
    star.easy_writeSTAR(exp_star, EAs=EAs)
    projs_star, projs_mrcs = relion.project(phantompath, data_dst,
                                            ang=exp_star)
    return projs_star, projs_mrcs


def gen_ref_EAs_grid(nside=8, psi_step=10):

    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, range(npix))
    theta, phi = np.rad2deg(theta), np.rad2deg(phi)
    psi = np.arange(0, 360, psi_step)
    # indexing = 'xy'
    grid_theta, grid_psi = np.meshgrid(theta, psi)
    grid_phi, _ = np.meshgrid(phi, psi)
    EAs_tuple = (grid_phi.flatten(), grid_theta.flatten(), grid_psi.flatten())
    EAs_grid = np.vstack(EAs_tuple).T

    return EAs_grid


def gen_mrcs_from_EAs(EAs, phantompath, dstpath):

    ang_star = os.path.join(temp_directory, 'EAs.star')
    star.easy_writeSTAR(ang_star, EAs=EAs)
    projs_star, projs_mrcs = relion.project(phantompath, dstpath, ang=ang_star)
    model_proj_imgs = mrc.readMRCimgs(projs_mrcs, idx=0)

    return model_proj_imgs


def find_similar(input_proj, ref_projs):
    grid_size = 0
    likelihood = np.zeros(grid_size)
    for i in range(grid_size):
        ref_proj = np.ctypeslib.as_array(ref_proj_stack[:, :, i])
        p = np.sum((input_proj - ref_proj) ** 2 / (-2 * input_proj))
        likelihood[i] = p

    idx = np.argmin(likelihood)
    return idx


def projection_matching(input_model, projs, dir_suffix=None):
    """
    Parameters:
    -------
    input_model: path of input mrc file
    projections: numpy array
    """
    EAs_grid = gen_ref_EAs_grid(nside=8)
    if dir_suffix:
        dstpath = os.path.join(WD, dir_suffix, 'model_projections')
    else:
        dstpath = os.path.join(WD, 'model_projections')

    N = projs.shape[0]
    num_projs = projs.shape[2]
    num_model_imgs = EAs_grid.shape[0]
    num_threads = 4
    model_proj_imgs = gen_mrcs_from_EAs(EAs_grid, input_model, dstpath)

    def find_similar(input_proj):
        grid_size = model_proj_imgs.shape[2]
        likelihood = np.zeros(grid_size)
        for i in range(grid_size):
            ref_proj = model_proj_imgs[:, :, i]
            p = np.sum((exp_proj - ref_proj) ** 2 / (-2 * exp_proj))
            likelihood[i] = p
        idx = np.argmin(likelihood)
        return idx

    pool = mp.Pool(processes=num_threads)
    idx_orientations = np.zeros(num_projs)
    idx_orientations = pool.map(
        find_similar, (projs[:, :, i] for i in range(num_projs)))
    orientations = EAs_grid[idx_orientations]
    return orientations


def reconstruct(projs_path):

    exp_samples = mrc.readMRCimgs(projs_path, idx=0)
    N, _, num_samples = exp_samples.shape

    M = cryoem.generate_phantom_density(N, 0.95 * N / 2.0, 5 * N / 128.0, 30)
    output_mrc = os.path.join(WD, 'initial_random_model.mrc')
    mrc.writeMRC(output_mrc, M)

    total_iteration = 20
    for it in range(total_iteration):

        input_model = output_mrc
        ori = projection_matching(input_model, exp_samples)

        dir_suffix = 'it' + str(it).zfill(3)
        ori_star = os.path.join(WD, dir_suffix, 'orientations.star')
        output_mrc = os.path.join(WD, dir_suffix, dir_suffix + '_reco.mrc')

        star.easy_writeSTAR(ori_star, EAs=ori, imgs_path=projs_path)
        relion.reconstruct(ori_star, output_mrc)


if __name__ == '__main__':
    exp_star, exp_mrcs = gen_exp_samples(10000,
                                         '/Users/lqhuang/Git/SOD-cryoem/particle/EMD-6044.mrc',
                                         '/Users/lqhuang/Git/SOD-cryoem/data/job/exp_projections')
    reconstruct(exp_mrcs)
