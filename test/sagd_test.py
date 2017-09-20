from __future__ import print_function, division

import os, sys, time
sys.path.append(os.path.dirname(sys.path[0]))

import numpy as np

from cryoio.mrc import readMRC
import cryoem
from optimizers.sagd import SAGDStep
from util import Output, OutputStream
from objectives import eval_objective, SumObjectives
from reconstructor import ObjectiveWrapper
from importancesampler.fisher import FixedFisherImportanceSampler
from symmetry import get_symmetryop

from likelihood_test import kernel_test, SimpleKernel
from dataset_test import dataset_loading_test

if os.path.exists('exp/sagd_L0.pkl'):
    os.remove('exp/sagd_L0.pkl')
# for diagnostics and parameters, # for stats (per image etc), # for likelihoods of individual images
diagout = Output(os.path.join('exp', 'diag'), runningout=False)
statout = Output(os.path.join('exp', 'stat'), runningout=True)
likeout = Output(os.path.join('exp', 'like'), runningout=False)
ostream = OutputStream(None)

if len(sys.argv) > 1:
    data_dir = sys.argv[1]
else:
    data_dir = 'data/1AON_no_shifts_200'
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
# Setup dataset
print("Loading dataset %s" % data_dir)
cryodata, _ = dataset_loading_test(data_params)
mleDC, _, mleDC_est_std = cryodata.get_dc_estimate()
modelscale = (np.abs(mleDC) + 2*mleDC_est_std)/cryodata.N

if len(sys.argv) > 2:
    model_file = sys.argv[2]
    print("Loading density map %s" % model_file)
    M = readMRC(model_file)
else:
    print("Generating random initial density map ...")
    M = cryoem.generate_phantom_density(cryodata.N, 0.95 * cryodata.N / 2.0, \
                                        5 * cryodata.N / 128.0, 30, seed=0)
M *= modelscale/M.sum()
slice_interp = {'kern': 'lanczos', 'kernsize': 4, 'zeropad': 0, 'dopremult': True}
fM = SimpleKernel.get_fft(M, slice_interp)

cparams = {
    'likelihood': 'UnknownRSLikelihood()',
    'kernel': 'multicpu',

    'prior_name': "'Sparsity'",
    'sparsity_lambda': 0.9,
    'prior': 'SparsityPrior(True)',

    # 'prior_name': "'CAR'",
    # 'prior': 'CARPrior()',
    # 'car_type': 'gauss0.5',
    # 'car_tau': 75.0,

    'iteration': 0,'pixel_size': cryodata.pixel_size, 'max_frequency': 0.02,
    'num_batches': cryodata.N_batches,

    'interp_kernel_R': 'lanczos',
    'interp_kernel_size_R':	4,
    'interp_zeropad_R':	0,
    'interp_premult_R':	True,

    'interp_kernel_I': 'lanczos',
    'interp_kernel_size_I':	8,
    'interp_zeropad_I':	1.0,
    'interp_premult_I':	True,

    'quad_shiftsigma': 10,
    'quad_shiftextent': 60,

    'sigma': cryodata.noise_var,
    'modelscale': modelscale,
    # 'symmetry': 'C7'
}

is_params = {
    # importance sampling
    # Ignore the first 50 iterations entirely
    'is_prior_prob': max(0.05,2**(-0.005*max(0,cparams['iteration']-50))),
    'is_temperature': max(1.0,2**(750.0/max(1,cparams['iteration']-50))),
    'is_ess_scale': 10,

    'is_fisher_chirality_flip': cparams['iteration'] < 2500,

    'is_on_R':  True,
    'is_global_prob_R': 0.9,

    'is_on_I':  True,
    'is_global_prob_I': 1e-10,

    'is_on_S':  True,
    'is_global_prob_S': 0.9,
    'is_gaussian_sigmascale_S': 0.67,
}

cparams.update(is_params)

sagd_params = {
    'iteration': cparams['iteration'],
    'exp_path': 'exp/',

    'num_batches': cryodata.N_batches,

    'sagd_linesearch':          'max_frequency_changed or ((iteration%5 == 0) if iteration < 2500 else \
                                                        (iteration%3 == 0) if iteration < 5000 else \
                                                        True)',
    'sagd_linesearch_accuracy': 1.01 if cparams['iteration'] < 10 else \
                                1.10 if cparams['iteration'] < 2500 else \
                                1.25 if cparams['iteration'] < 5000 else \
                                1.50,
    # 'sagd_linesearch_maxits':   "5 if cparams['iteration'] < 2500 else 3",
    'sagd_incL':                1.0,

    'sagd_momentum':      1 - 1.0/(1.0 + 0.1 * cparams['iteration']),
    # 'sagd_learnrate':     '1.0/min(16.0,2**(num_max_frequency_changes))',

    'shuffle_minibatches': 'iteration >= 1000',
}


# Setup SAGD optimizer
step = SAGDStep()
step.setup(cparams, diagout, statout, ostream)

# Objective function setup 
param_type = cparams.get('parameterization','real')
cplx_param = param_type in ['complex','complex_coeff','complex_herm_coeff']
like_func = eval_objective(cparams['likelihood'])
prior_func = eval_objective(cparams['prior'])

obj = SumObjectives(cplx_param, [like_func, prior_func], [None,None])
obj.setup(cparams, diagout, statout, ostream)
obj.set_dataset(cryodata)
obj_wrapper = ObjectiveWrapper(param_type)

is_sym = get_symmetryop(cparams.get('is_symmetry',cparams.get('symmetry',None)))
sampler_R = FixedFisherImportanceSampler('_R',is_sym)
sampler_I = FixedFisherImportanceSampler('_I')
# sampler_S = FixedGaussianImportanceSampler('_S')
sampler_S = None
like_func.set_samplers(sampler_R=sampler_R,sampler_I=sampler_I,sampler_S=sampler_S)

# Start iteration
num_data_evals = 0
num_iterations = 10
for i in range(num_iterations):
    cparams['iteration'] = i
    sagd_params['iteration'] = i
    print('Iteration #:', i)
    minibatch = cryodata.get_next_minibatch(True)
    num_data_evals += minibatch['N_M']

    # setup the wrapper for the objective function 
    obj.set_data(cparams, minibatch)
    obj_wrapper.set_objective(obj)
    x0 = obj_wrapper.set_density(M, fM)
    evalobj = obj_wrapper.eval_obj

    # Get step size
    trainLogP, dlogP, v, res_train, extra_num_data = \
        step.do_step(x0, sagd_params, cryodata, evalobj, batch=minibatch)

    # print('trainLogP:', trainLogP)
    # print('dlogP:', dlogP.shape)
    # print('v:', v.shape)
    # print('res_train:', res_train.keys())  # dict_keys(['CV2_R', 'CV2_I', 'Evar_like', 'Evar_prior', 'sigma2_est', 'correlation', 'power', 'like', 'N_R_sampled', 'N_I_sampled', 'N_Total_sampled', 'totallike_logscale', 'kern_timing', 'angular_correlation_timing', 'like_timing', 'N_R', 'N_I', 'N_Total', 'N_R_sampled_total', 'N_I_sampled_total', 'N_Total_sampled_total', 'L', 'all_logPs', 'all_dlogPs'])
    # print('res_train L:', res_train['L'])
    # print('res_train like:', res_train['like'])
    # print('extra_num_data:', extra_num_data)

    # Apply the step
    x = x0 + v

    # Convert from parameters to value
    prevM = np.copy(M)
    M, fM = obj_wrapper.convert_parameter(x, comp_real=True)
    
    # Compute net change
    dM = prevM - M

    # Compute step statistics
    step_size = np.linalg.norm(dM)
    grad_size = np.linalg.norm(dlogP)
    M_norm = np.linalg.norm(M)

    num_data_evals += extra_num_data
    inc_ratio = step_size / M_norm

    # Update import sampling distributions
    sampler_R.perform_update()
    sampler_I.perform_update()
    if sampler_S is not None:
        sampler_S.perform_update()

