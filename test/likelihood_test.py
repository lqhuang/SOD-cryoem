from __future__ import print_function, division

import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
import time

from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np
from scipy.stats import entropy

from cryoio import mrc
from cryoio import ctf
from cryoio.dataset import CryoDataset

import geometry
import density
import cryoem
import cryoops
from objectives.cpu_kernel import UnknownRSThreadedCPUKernel
from importancesampler.fisher import FixedFisherImportanceSampler
from importancesampler.gaussian import FixedGaussianImportanceSampler
from symmetry import get_symmetryop
import quadrature

from notimplemented import correlation, projector, py_objective_kernels
from test.dataset_test import dataset_loading_test, SimpleDataset

cython_build_dirs = os.path.expanduser('~/.pyxbld/angular_correlation')
import pyximport; pyximport.install(build_dir=cython_build_dirs, setup_args={"include_dirs":np.get_include()},reload_support=True)
from objectives import objective_kernels
import sincint


# ------------------------ utils------------------------------------- #
def plot_directions(dirs,vals,vmin=None,vmax=None):
    if vmin != None or vmax != None:
        vals = np.clip(vals,vmin,vmax)
    mlab.points3d(dirs[:,0],dirs[:,1],dirs[:,2],np.log(1e-10+vals),scale_mode='none',scale_factor=5.0,opacity=0.2)

def winkeltriple(t,ph):
    ph1 = np.arccos(2.0/np.pi)
    a = np.arccos(np.cos(ph)*np.cos(t/2.0))
    x = 0.5*( t*np.cos(ph1) + 2.0*np.cos(ph)*np.sin(t/2.0) / np.sinc(a/np.pi) )
    y = 0.5*( ph + np.sin(ph)/np.sinc(a/np.pi) )
    return x,y

def plotwinkeltriple(dirs, values, spot=None, others=None, vmin=None, vmax=None, lognorm=True, axes=None):
    """ Plots a winkel projection of a function on a sphere evaluated at directions d
    v - values
    """
    # phi = np.arctan2(d[:,2],np.linalg.norm(d[:,0:2],axis=1)).reshape((-1,))
    phi = np.arctan2(dirs[:,2],np.linalg.norm(dirs[:,0:2],axis=1)).reshape((-1,))
    theta = np.arctan2(dirs[:,1],dirs[:,0]).reshape((-1,))
    x,y = winkeltriple(theta,phi)
    
    t_border = np.concatenate( [ np.linspace(np.pi,-np.pi,50), np.ones(50)*-np.pi, np.linspace(-np.pi,np.pi,50), np.ones(50)*np.pi ] )
    ph_border = np.concatenate( [ np.ones(50)*-np.pi/2.0, np.linspace(-np.pi/2,np.pi/2.0,50), np.ones(50)*np.pi/2.0, np.linspace(np.pi/2.0,-np.pi/2.0,50) ] )
    x_border,y_border = winkeltriple(t_border,ph_border)

    if lognorm:
        norm = colors.LogNorm()
    else:
        norm = colors.Normalize()
    
    if axes is not None:
        ax = axes
    else:
        ax = plt.subplot(111)
    tripim = ax.tripcolor(x, y, 1e-10 + values, shading='gouraud',vmin=vmin+1e-10,vmax=vmax+1e-10,norm=norm)
    plt.colorbar(tripim)
    ax.plot(x_border,y_border,'-k')

    if spot is not None:
        phi_s = np.arctan2(spot[:,2],np.linalg.norm(spot[:,0:2],axis=1)).reshape((-1,))
        theta_s = np.arctan2(spot[:,1],spot[:,0]).reshape((-1,))
        x_s, y_s = winkeltriple(theta_s, phi_s)
        ax.plot(x_s, y_s, '.r', markersize=5)

    if others is not None:
        for i, vec in enumerate(others):
            vec = vec.reshape((1, -1))
            phi_o = np.arctan2(vec[:,2],np.linalg.norm(vec[:,0:2],axis=1)).reshape((-1,))
            theta_o = np.arctan2(vec[:,1],vec[:,0]).reshape((-1,))
            x_o, y_o = winkeltriple(theta_o, phi_o)
            ax.text(x_o, y_o, str(i))

    if axes is None:
        plt.show()
# ------------------------------------------------------------------- #


def load_kernel(data_dir, model_file, use_angular_correlation=False, sample_shifts=False):
    data_params = {
        'dataset_name': "1AON",
        'inpath': os.path.join(data_dir, 'imgdata.mrc'),
        'ctfpath': os.path.join(data_dir, 'defocus.txt'),
        'microscope_params': {'akv': 200, 'wgh': 0.07, 'cs': 2.0},
        'resolution': 2.8,
        'sigma': 'noise_std',
        'sigma_out': 'data_std',
        'minisize': 20,
        'test_imgs': 20,
        'partition': 0,
        'num_partitions': 0,
        'random_seed': 1,
        # 'symmetry': 'C7'
    }
    print("Loading dataset %s" % data_dir)
    cryodata, _ = dataset_loading_test(data_params)
    # mleDC, _, mleDC_est_std = cryodata.get_dc_estimate()
    # modelscale = (np.abs(mleDC) + 2*mleDC_est_std)/cryodata.N
    modelscale = 1.0

    if model_file is not None:
        print("Loading density map %s" % model_file)
        M = mrc.readMRC(model_file)
    else:
        print("Generating random initial density map ...")
        M = cryoem.generate_phantom_density(cryodata.N, 0.95 * cryodata.N / 2.0, \
                                            5 * cryodata.N / 128.0, 30, seed=0)
        M *= modelscale/M.sum()
    slice_interp = {'kern': 'lanczos', 'kernsize': 4, 'zeropad': 0, 'dopremult': True}
    fM = M

    minibatch = cryodata.get_next_minibatch(shuffle_minibatches=False)

    is_sym = get_symmetryop(data_params.get('symmetry',None))
    sampler_R = FixedFisherImportanceSampler('_R', is_sym)
    sampler_I = FixedFisherImportanceSampler('_I')
    sampler_S = None

    cparams = {
        'use_angular_correlation': use_angular_correlation,

        'iteration': 0,
        'pixel_size': cryodata.pixel_size,
        'max_frequency': 0.02,

        'interp_kernel_R': 'lanczos',
        'interp_kernel_size_R':	4,
        'interp_zeropad_R':	0,
        'interp_premult_R':	True,

        'interp_kernel_I': 'lanczos',
        'interp_kernel_size_I':	4,
        'interp_zeropad_I':	0.0,
        'interp_premult_I':	True,

        'quad_shiftsigma': 10,
        'quad_shiftextent': 60,

        'sigma': cryodata.noise_var,
        # 'symmetry': 'C7'
    }
    kernel = UnknownRSThreadedCPUKernel()
    kernel.setup(cparams, None, None, None)
    kernel.set_samplers(sampler_R, sampler_I, sampler_S)
    kernel.set_dataset(cryodata)
    kernel.precomp_slices = None
    kernel.set_data(cparams, minibatch)
    kernel.using_precomp_slicing = False
    kernel.using_precomp_inplane = False
    kernel.M = M
    kernel.fM = fM
    return kernel


def kernel_test(data_dir, model_file, use_angular_correlation=False):
    kernel = load_kernel(data_dir, model_file, use_angular_correlation)

    euler_angles = []
    with open(os.path.join(data_dir, 'ctf_gt.par')) as par:
        par.readline()
        # 'C                 PHI      THETA        PSI        SHX        SHY       FILM        DF1        DF2     ANGAST'
        while True:
            try:
                line = par.readline().split()
                euler_angles.append([float(line[1]), float(line[2]), float(line[3])])
            except:
                break
    euler_angles = np.asarray(euler_angles)

    for idx in range(kernel.minibatch['N_M']):
        img_idx = kernel.minibatch['img_idxs'][idx]
        ea = euler_angles[img_idx]
        ea_vec = geometry.genDir([ea]).reshape(1, -1)
        slice_ops, envelope, \
        W_R_sampled, sampleinfo_R, slices_sampled, slice_inds, \
        W_I_sampled, sampleinfo_I, rotd_sampled, rotc_sampled = \
            kernel.prep_operators(kernel.fM, img_idx)

        if kernel.use_angular_correlation:
            tic = time.time()
            ac_slices_sampled, ac_data_sampled = kernel.get_angular_correlation(
                slices_sampled, rotd_sampled, rotc_sampled, envelope, W_I_sampled)

        sigma2 = kernel.inlier_sigma2_trunc
        workspace = None
        g = np.zeros(slices_sampled.shape, dtype=np.float32)
        log_W_R = np.log(W_R_sampled)
        log_W_I = np.log(W_I_sampled)
        if kernel.sampler_S is not None:
            log_W_S = np.log(W_S_sampled)

        if kernel.use_angular_correlation:
            like, (cphi_I, cphi_R), csigma2_est, ccorrelation, cpower, workspace = \
                objective_kernels.doimage_ACRI(slices_sampled, envelope, \
                    rotc_sampled, rotd_sampled, \
                    ac_slices_sampled, ac_data_sampled, \
                    log_W_I, log_W_R, \
                    sigma2, g, workspace)
        else:
            like, (cphi_I, cphi_R), csigma2_est, ccorrelation, cpower, workspace = \
                objective_kernels.doimage_RI(slices_sampled, envelope, \
                rotc_sampled, rotd_sampled, \
                log_W_I, log_W_R, \
                sigma2, g, workspace)

        workspace['cphi_R'] = cphi_R
        workspace['cphi_I'] = cphi_I
        if kernel.sampler_S is not None:
            workspace['cphi_S'] = cphi_S
    
        SimpleKernel.plot_distribution(workspace, kernel.quad_domain_R, kernel.quad_domain_I, correct_ea=ea)

        isw = 1
        logspace_phis = True
        testImg = kernel.minibatch['test_batch']
        kernel.sampler_R.record_update(idx, sampleinfo_R[1], cphi_R, sampleinfo_R[2], isw, testImg, logspace = logspace_phis)
        kernel.sampler_I.record_update(idx, sampleinfo_I[1], cphi_I, sampleinfo_I[2], isw, testImg, logspace = logspace_phis)

    return kernel


class SimpleKernel():
    def __init__(self, cryodata, use_angular_correlation=False):
        self.cryodata = cryodata
        self.N = self.cryodata.get_num_pixels()
        self.psize = self.cryodata.get_pixel_size()

        symmetry = self.cryodata.dataset_params.get('symmetry', None)
        self.is_sym = get_symmetryop(symmetry)
        self.sampler_R = FixedFisherImportanceSampler('_R', self.is_sym)
        self.sampler_I = FixedFisherImportanceSampler('_I')

        self.use_cached_slicing = True
        self.use_angular_correlation = use_angular_correlation

        self.G_datatype = np.float32

        self.cached_workspace = dict()

    def get_angular_correlation(self, slices_sampled, rotd_sampled, rotc_sampled, envelope, log_W_I):
        N_R, N_T = slices_sampled.shape
        assert rotd_sampled[0].shape == rotc_sampled[0].shape
        assert rotd_sampled[0].shape[0] == N_T

        max_W_I = log_W_I.argmax()
        if envelope is not None:
            assert envelope.shape[0] == slices_sampled.shape[1], "wrong length for envelope"
            slices_sampled = np.tile(envelope, (N_R, 1)) * np.tile(rotc_sampled[max_W_I], (N_R, 1)) \
                             * slices_sampled
        else:
            slices_sampled = np.tile(rotc_sampled[max_W_I], (N_R, 1)) * slices_sampled
        
        ac_slices = correlation.calc_angular_correlation(np.abs(slices_sampled), self.N, self.rad, self.psize)
        ac_data = correlation.calc_angular_correlation(np.abs(rotd_sampled[max_W_I]), self.N, self.rad, self.psize)

        # check zeros
        ac_slices[ac_slices == 0.0] + 1e-16
        # calculating K-L divergence
        ac_e_R = entropy(np.tile(ac_data, (N_R, 1)).T, ac_slices.T)  # qk is used to approximate pk, qk
        ac_indices = ac_e_R.argsort()
        cutoff = 7

        return ac_indices[0:cutoff]

    def set_data(self, model, cparams):

        assert isinstance(model, np.ndarray), "Unexpected input numpy array."
        self.model = model

        self.slices_sampled = None
        self.rotd_sampled = None
        self.rad = None

        cparams['iteration'] = 0
        max_freq = cparams['max_frequency']
        rad_cutoff = cparams.get('rad_cutoff', 1.0)
        rad = min(rad_cutoff, max_freq * 2.0 * self.psize)

        self.xy, self.trunc_xy, self.truncmask = geometry.gencoords(self.N, 2, rad, True)
        self.trunc_freq = np.require(self.trunc_xy / (self.N * self.psize), dtype=np.float32) 
        self.N_T = self.trunc_xy.shape[0]

        # set CTF and envelope
        if self.cryodata.use_ctf:
            radius_freqs = np.sqrt(np.sum(self.trunc_xy**2,axis=1))/(self.psize*self.N)
            self.envelope = ctf.envelope_function(radius_freqs, cparams.get('learn_like_envelope_bfactor', None))
        else:
            self.envelope = np.ones(self.N_T, dtype=np.float32)
        
        print("Iteration {0}: freq = {3}, rad = {1:.4f}, N_T = {2}".format(cparams['iteration'], rad, self.N_T, max_freq))
        self.set_quad(rad)

        # Setup inlier model
        self.inlier_sigma2 = 1.0  # cparams['sigma']**2
        base_sigma2 = 1.0  # self.cryodata.noise_var
        self.inlier_sigma2_trunc = self.inlier_sigma2 
        # self.inlier_const = (self.N_T/2.0)*np.log(2.0*np.pi*self.inlier_sigma2)

        # # Compute the likelihood for the image content outside of rad
        # _,_,fspace_truncmask = gencoords(self.fspace_stack.get_num_pixels(), 2, rad*self.fspace_stack.get_num_pixels()/self.N, True)
        # self.imgpower = np.empty((self.minibatch['N_M'],),dtype=density.real_t)
        # self.imgpower_trunc = np.empty((self.minibatch['N_M'],),dtype=density.real_t)
        # for idx,Idx in enumerate(self.minibatch['img_idxs']):
        #     Img = self.fspace_stack.get_image(Idx)
        #     self.imgpower[idx] = np.sum(Img.real**2) + np.sum(Img.imag**2)

        #     Img_trunc = Img[fspace_truncmask.reshape(Img.shape) == 0]
        #     self.imgpower_trunc[idx] = np.sum(Img_trunc.real**2) + np.sum(Img_trunc.imag**2)
        # like_trunc = 0.5*self.imgpower_trunc/base_sigma2
        # self.inlier_like_trunc = like_trunc
        # self.inlier_const += ((self.N**2 - self.N_T)/2.0)*np.log(2.0*np.pi*base_sigma2)

    def set_quad(self, rad):
        if self.rad == rad:
            print("Using previous quadurature schem")
        else:
            tic = time.time()
            # set slicing interpolation parameters
            self.slice_params = {'quad_type': 'sk97'}
            self.slice_interp = {'N': self.N, 'kern': 'lanczos', 'kernsize': 6, 'rad': rad, 'zeropad': 0, 'dopremult': True, }# 'onlyRs': True}
            # set slicing quadrature
            usFactor_R = 1.0
            quad_R = quadrature.quad_schemes[('dir', self.slice_params['quad_type'])]
            degree_R, resolution_R = quad_R.compute_degree(self.N, rad, usFactor_R)
            slice_quad = {}
            slice_quad['resolution'] = max(0.5*quadrature.compute_max_angle(self.N, rad), resolution_R)
            slice_quad['dir'], slice_quad['W'] = quad_R.get_quad_points(degree_R, self.is_sym)
            self.slice_quad = slice_quad
            self.quad_domain_R = quadrature.FixedSphereDomain(slice_quad['dir'], slice_quad['resolution'], sym=self.is_sym)
            # generate slicing operators (points on sphere)
            self.slice_ops = self.quad_domain_R.compute_operator(self.slice_interp)
            self.N_R = len(self.quad_domain_R)
            print("  Slice Ops: %d, resolution: %.2f degree, generated in: %.4f seconds" \
                % (self.N_R, np.rad2deg(self.quad_domain_R.resolution), time.time()-tic))

            tic = time.time()
            # set inplane interpolation parameters
            self.inplane_interp = {'N': self.N, 'kern': 'lanczos', 'kernsize': 6, 'rad': rad, 'zeropad': 0, 'dopremult': True, }# 'onlyRs': True}
            # set inplane quadrature
            usFactor_I = 1.0
            maxAngle = quadrature.compute_max_angle(self.N, rad, usFactor_I)
            degree_I = np.uint32(np.ceil(2.0 * np.pi / maxAngle))
            resolution_I = max(0.5*quadrature.compute_max_angle(self.N, rad), 2.0*np.pi / degree_I)
            inplane_quad = {}
            inplane_quad['resolution'] = resolution_I
            inplane_quad['thetas'] = np.linspace(0, 2.0*np.pi, degree_I, endpoint=False)
            inplane_quad['thetas'] += inplane_quad['thetas'][1]/2.0
            inplane_quad['W'] = np.require((2.0*np.pi/float(degree_I))*np.ones((degree_I,)), dtype=np.float32)
            self.inplane_quad = inplane_quad
            self.quad_domain_I = quadrature.FixedCircleDomain(inplane_quad['thetas'],
                                                            inplane_quad['resolution'])
            # generate inplane operators
            self.inplane_ops = self.quad_domain_I.compute_operator(self.inplane_interp)
            self.N_I = len(self.quad_domain_I)
            print("  Inplane Ops: %d, resolution: %.2f degree, generated in: %.4f seconds." \
                % (self.N_I, np.rad2deg(self.quad_domain_I.resolution), time.time()-tic))

            transform_change = self.cryodata.interp_params['kern'] != self.inplane_interp['kern'] or \
                            self.cryodata.interp_params['kernsize'] != self.inplane_interp['kernsize'] or \
                            self.cryodata.interp_params['zeropad'] != self.inplane_interp['zeropad'] or \
                            self.cryodata.interp_params['dopremult'] != self.inplane_interp['dopremult']
            if transform_change:
                self.cryodata.set_transform(self.inplane_interp)

            self.rad = rad

    def prep_operators(self, idx):
        N_R = self.N_R
        N_I = self.N_I
        N_T = self.N_T

        # ctf samples
        envelope = self.envelope
        if self.cryodata.use_ctf:
            cCTF = self.cryodata.get_ctf(idx)
            rotc_sampled = cCTF.compute(self.trunc_freq, self.quad_domain_I.theta).T
        else:
            rotc_sampled = np.ones_like(rotd_sampled, dtype=density.real_t)

        # compute operators and slices
        # slicing samples
        W_R = self.slice_quad['W']
        W_R_sampled = np.require(W_R, dtype=density.real_t)
        samples_R = np.arange(N_R)
        sampleinfo_R = None  # N_R_sampled, samples_R, sampleweights_R
        # self.slice_ops = self.quad_domain_R.compute_operator(self.slice_interp, self.samples_R)
        slice_ops = self.slice_ops
        if self.use_cached_slicing and self.slices_sampled is not None:
            slices_sampled = self.slices_sampled
        else:
            fM = self.model
            slices_sampled = cryoem.getslices_interp(fM, self.slice_ops, self.rad).reshape((N_R, N_T))
            slices_sampled *= np.tile(rotc_sampled[0], (N_R, 1))
            # slices_sampled += 1.0
            self.slices_sampled = slices_sampled

        # inplane samples
        W_I = self.inplane_quad['W']
        W_I_sampled = np.require(W_I, dtype=density.real_t)
        sampleinfo_I = None  # N_I_sampled, samples_I, sampleweights_I
        # self.inplane_ops = self.quad_domain_I.compute_operator(self.interp_params_I, self.samples_I)
        curr_fft_image = self.cryodata.get_image(idx)
        print('(min, max) intensity for fft proj: ({}, {})'.format(curr_fft_image.min(), curr_fft_image.max()))
        rotd_sampled = cryoem.getslices_interp(curr_fft_image, self.inplane_ops, self.rad).reshape((N_I, N_T))
        rotd_sampled *= rotc_sampled

        # rotd_sampled += 1.0

        print('total photons of slices_sampled: {}'.format(slices_sampled.sum(axis=1)))
        print('total photons of rotd_sampled: {}'.format(rotd_sampled.sum(axis=1)))

        np.maximum(slices_sampled, 1e-4, out=slices_sampled)
        np.maximum(rotd_sampled, 1e-4, out=rotd_sampled)
        # np.maximum(slices_sampled, 1.0, out=slices_sampled)
        # np.maximum(rotd_sampled, 1.0, out=rotd_sampled)

        return slice_ops, envelope, \
            W_R_sampled, sampleinfo_R, slices_sampled, samples_R, \
            W_I_sampled, sampleinfo_I, rotd_sampled, rotc_sampled

    def worker(self, idx):
        g_size = (self.N_R, self.N_T)
        g = np.zeros(g_size, dtype=self.G_datatype)
        workspace = None
        sigma2 = self.inlier_sigma2_trunc 

        slice_ops, envelope, \
        W_R_sampled, sampleinfo_R, slices_sampled, slice_inds, \
        W_I_sampled, sampleinfo_I, rotd_sampled, rotc_sampled = \
            self.prep_operators(idx)

        TtoF = sincint.gentrunctofull(N=self.N, rad=self.rad)
        slicing = TtoF.dot(slices_sampled[11]).reshape(self.N, self.N)
        rotd = TtoF.dot(rotd_sampled[0]).reshape(self.N, self.N)
        fig, ax = plt.subplots(ncols=2)
        im_slicing = ax[0].imshow(slicing)
        fig.colorbar(im_slicing, ax=ax[0])
        im_rotd = ax[1].imshow(rotd)
        fig.colorbar(im_rotd, ax=ax[1])
        # plt.show()

        print('(min, max) intensity for slices_sampled: ({}, {})'.format(slices_sampled.min(), slices_sampled.max()))
        print('(min, max) intensity for rotd_sampled: ({}, {})'.format(rotd_sampled.min(), rotd_sampled.max()))
        if self.use_angular_correlation:
            # print('max intensity for ac_slices_sampled:', ac_slices_sampled.max())
            # print('max intensity for ac_data_sampled:', ac_data_sampled.max())
            ac_indices = self.get_angular_correlation(
                slices_sampled, rotd_sampled, rotc_sampled, envelope, W_I_sampled)

        log_W_I = np.log(W_I_sampled)
        log_W_R = np.log(W_R_sampled)

        if self.use_angular_correlation:
            like, (cphi_I, cphi_R), csigma2_est, ccorrelation, cpower, workspace = \
                py_objective_kernels.doimage_ACRI(slices_sampled, envelope, \
                    rotc_sampled, rotd_sampled, \
                    # ac_slices_sampled, ac_data_sampled, \
                    ac_indices,
                    log_W_I, log_W_R, \
                    sigma2, g, workspace)
        else:
            like, (cphi_I, cphi_R), csigma2_est, ccorrelation, cpower, workspace = \
                objective_kernels.doimage_RI(slices_sampled, envelope, \
                    rotc_sampled, rotd_sampled, \
                    log_W_I, log_W_R, \
                    sigma2, g, workspace)

        print("Like", like)
        workspace['like'] = like
        workspace['cphi_I'] = cphi_I
        workspace['cphi_R'] = cphi_R
        
        g = np.zeros(g_size, dtype=self.G_datatype)

        # like, (cphi_I, cphi_R), csigma2_est, ccorrelation, cpower, _ = \
        #     objective_kernels.doimage_RI(slices_sampled, envelope, \
        #         rotc_sampled, rotd_sampled, \
        #         log_W_I, log_W_R, \
        #         sigma2, g, None)

        # print("full like:", like)
        # workspace['full_like_cphi_I'] = cphi_I
        # workspace['full_like_cphi_R'] = cphi_R

        self.cached_workspace[idx] = workspace
        return workspace

    def plot_quadrature(self):
        import healpy as hp
        fig_slicing = plt.figure(num=0)
        theta, phi = hp.pixelfunc.vec2ang(self.quad_domain_R.dirs)
        hp.visufunc.orthview(fig=0, title='slicing quadrature')
        hp.visufunc.projscatter(theta, phi)

        fig_inplane, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(self.quad_domain_I.theta, np.ones(self.N_I), '.')
        ax.set_rmax(1.5)
        ax.set_rticks([0.5, 1, 1.5])
        ax.set_yticklabels([])
        ax.set_title('inplane quadrature')
        ax.grid(True)

        plt.show()

    def plot_rmsd(self):
        processed_idxs = self.cached_workspace.keys()
        quad_domain_R = self.quad_domain_R

        num_idxs = len(processed_idxs)
        top1_rmsd = np.zeros(num_idxs)
        topN_rmsd = np.zeros(num_idxs)
        cutoff_R = 7
        topN_weight = np.exp(-np.arange(cutoff_R) / 2)
        topN_weight /= topN_weight.sum()

        for i, idx in enumerate(processed_idxs):
            ea = self.cryodata.euler_angles[idx][0:2]
            tiled_ea = np.tile(ea, (cutoff_R, 1))
            cphi_R = self.cached_workspace[idx]['cphi_R']
            sorted_indices_R = (-cphi_R).argsort()
            potential_R = quad_domain_R.dirs[sorted_indices_R[0:cutoff_R]]
            eas_of_dirs = geometry.genEA(potential_R)[:, 0:2]
            top1_rmsd[i] = np.sqrt(((ea[0:2] - eas_of_dirs[0])**2).mean())
            topN_rmsd[i] = (np.sqrt(((tiled_ea - eas_of_dirs) ** 2).mean(axis=1)) * topN_weight).sum()

        print("Slicing quadrature scheme, resolution {}, num of points {}".format(
            quad_domain_R.resolution, len(quad_domain_R.dirs)))
        print("Top 1 RMSD:", top1_rmsd.mean())
        print("Top {} RMSD: {}".format(cutoff_R, topN_rmsd.mean()))

    @staticmethod
    def plot_distribution(workspace, quad_domain_R, quad_domain_I, correct_ea=None, lognorm=False):
        phi_R = np.asarray(workspace['cphi_R'])
        phi_I = np.asarray(workspace['cphi_I'])

        sorted_indices_R = (-phi_R).argsort()
        sorted_indices_I = (-phi_I).argsort()

        cutoff_idx_R = 7
        cutoff_idx_I = 1
        # cutoff_idx_R = np.diff((-phi_R)[sorted_indices_R]).argmax() + 1
        # cutoff_idx_I = np.diff((-phi_I)[sorted_indices_I]).argmax() + 1
        potential_R = quad_domain_R.dirs[sorted_indices_R[0:cutoff_idx_R]]
        potential_I = quad_domain_I.theta[sorted_indices_I[0:cutoff_idx_I]]

        if correct_ea is not None:
            spot = geometry.genDir([correct_ea])
        else:
            spot = None

        fig = plt.figure(figsize=(12.8, 4.8))
        gs = gridspec.GridSpec(1, 3)

        ax_slicing = fig.add_subplot(gs[0:2])
        plotwinkeltriple(quad_domain_R.dirs, -phi_R, spot=spot, others=potential_R,
                         vmin=-phi_R.max(), vmax=-phi_R.min(), lognorm=lognorm, axes=ax_slicing)
        ax_slicing.set_title('slicing distribution for likelihood')

        ax_inplane = fig.add_subplot(gs[-1], projection='polar')
        ax_inplane.plot(quad_domain_I.theta, -phi_I)
        if correct_ea is not None:
            ax_inplane.plot([np.rad2deg(correct_ea[2]), np.rad2deg(correct_ea[2])], [-phi_I.max(), -phi_I.min()], 'r')
        if 'potential_I' in locals():
            for i, th in enumerate(potential_I):
                ax_inplane.text(np.rad2deg(th), (-phi_I)[sorted_indices_I[i]], str(i))
        ax_inplane.set_ylim([(-phi_I).min(), (-phi_I).max()])
        ax_inplane.set_yticklabels([])
        ax_inplane.set_title('inplane distribution for likelihood\nmax:{0:.2f}, min:{1:.2f}'.format(
            (-phi_I).max(), (-phi_I).min())
        )
        ax_inplane.grid(True)
        
        plt.show()

# ---------------------------------- main --------------------------- #
def likelihood_estimate(model, refined_model, use_angular_correlation=False, add_ctf=True):
    # model, _ = cryoem.align_density(model)
    # refined_model, _ = cryoem.align_density(refined_model)

    # model[model < 1.0] = 1.0
    # refined_model[refined_model < 1.0] = 1.0
    # density_totalmass = 80000
    # if density_totalmass is not None:
    #     model *= density_totalmass / model.sum()

    euler_angles = []
    data_dir = 'data/1AON_xfel_5000_totalmass_10000_oversampling_3'
    with open(os.path.join(data_dir, 'ctf_gt.par')) as par:
        par.readline()
        # 'C                 PHI      THETA        PSI        SHX        SHY       FILM        DF1        DF2     ANGAST'
        while True:
            try:
                line = par.readline().split()
                euler_angles.append([float(line[1]), float(line[2]), float(line[3])])
            except:
                break
    euler_angles = np.asarray(euler_angles)

    data_params = {'num_images': 20, 'pixel_size': 2.8,
                   'sigma_noise': 5.0,
                   # 'euler_angles': np.deg2rad(euler_angles[0:20]),
                   # 'symmetry': 'C7',
                   'euler_angles': None
                  }
    if add_ctf:
        ctf_params = {'akv': 200, 'wgh': 0.07, 'cs': 2.0, 'psize': 2.8, 'dscale': 1, 'bfactor': 500.0,
                      'df1': 31776.1, 'df2': 31280.3, 'angast': 0.82}
    else:
        ctf_params = None
    cryodata = SimpleDataset(model, data_params, ctf_params)

    cparams = {'max_frequency': 0.04, 'learn_like_envelope_bfactor': 500}
    print("Using angular correlation patterns:", use_angular_correlation)
    sk = SimpleKernel(cryodata, use_angular_correlation=use_angular_correlation)
    sk.set_data(refined_model, cparams)

    for i in range(data_params['num_images']):
        tic = time.time()
        workspace = sk.worker(i)
        print("idx: {}, time per worker: {}".format(i, time.time()-tic))
        sk.plot_distribution(workspace, sk.quad_domain_R, sk.quad_domain_I,
                             correct_ea=cryodata.euler_angles[i], lognorm=False)

        # workspace['cphi_R'] = workspace['full_like_cphi_R']
        # workspace['cphi_I'] = workspace['full_like_cphi_I']
        # sk.plot_distribution(workspace, sk.quad_domain_R, sk.quad_domain_I,
        #                      correct_ea=cryodata.euler_angles[i], lognorm=False)

    # sk.plot_rmsd()


if __name__ == '__main__':
    model_file = sys.argv[1]
    # refined_model_file = sys.argv[2]
    print("Loading 3D density: %s" % model_file)
    # print("Loading refined 3D density: %s" % refined_model_file)

    # data_dir = 'data/1AON_xfel_5000_totalmass_05000_oversampling'
    # kernel = kernel_test(data_dir, model_file, use_angular_correlation=False)

    model = mrc.readMRC(model_file)
    refined_model = model # = mrc.readMRC(refined_model_file)
    likelihood_estimate(model, refined_model, use_angular_correlation=False)
