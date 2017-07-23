from .fixed import FixedImportanceSampler
from util import logsumexp
import numpy as np

class FixedGaussianImportanceSampler(FixedImportanceSampler):
    def __init__(self,suffix):
        FixedImportanceSampler.__init__(self,suffix)

    def evaluate_kernel(self,inds,vals,odomain,params,logspace=False):
        sqdist_mat = self.domain.get_sqdist_mat(odomain,curr_inds = None, other_inds = inds)

        sigma_scale = params.get('is_gaussian_sigmascale'+self.suffix,params.get('is_gaussian_sigmascale',1.0))

#         sigma = sigma_scale*odomain.resolution
        sigma = sigma_scale*np.reshape(odomain.get_pt_resolution(inds),(1,-1))
        if logspace:
            logK = (-0.5/sigma**2)*sqdist_mat
            ret = logsumexp(logK + vals.reshape((1,-1)),axis=1)
            ret -= logsumexp(ret)
        else:
            K = np.exp((-0.5/sigma**2)*sqdist_mat)
            ret = np.dot(K,vals)
            ret /= ret.sum()

        return ret

 
