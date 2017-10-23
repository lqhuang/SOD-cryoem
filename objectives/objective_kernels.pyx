#cython: boundscheck=False

# from six.moves import xrange

import numpy as n
cimport numpy as n
import pyximport; pyximport.install(setup_args={"include_dirs":n.get_include()},reload_support=True)

from libc.string cimport memset
from libc.math cimport exp, log, log1p, isfinite

cdef double my_logsumexp(unsigned int N, double *a) nogil:
    cdef double a_max
    cdef double a_sum
    # cdef double diff
    cdef unsigned int i

    a_max = a[0]
    for i in range(N):
        if a[i] > a_max:
            a_max = a[i]

    a_sum = 0
    for i in range(N):
        # diff = a[i] - a_max
        # if diff > -72.0:  # exp(-72) = 5.3801861600211382e-32
        a_sum += exp(a[i] - a_max)

    return a_max + log(a_sum)

cdef double my_logaddexp(double a, double b) nogil:
    cdef double tmp
    if a == b:
        return a + 0.69314718055994529 # This is the numerical value of ln(2)
    else:
        tmp = a-b
        
        if tmp > 0:
            # if tmp > 72.0:
            #     return a
            # else:
            return a + log1p(exp(-tmp))
        elif tmp <= 0:
            # if tmp < -72.0:
            #     return b
            # else:
            return b + log1p(exp(tmp))
        else:
            return tmp

def update_workspace(workspace, N_R, N_I, N_S, N_T):
    if workspace is None:
        workspace = {'N_R':0,'N_I':0,'N_S':0,'N_T':0}

    if N_R is not None and workspace['N_R'] < N_R or workspace['N_T'] != N_T:
        workspace['sigma2_R'] = n.empty((N_R,N_T), dtype=n.float64)
        workspace['correlation_R'] = n.empty((N_R,N_T), dtype=n.float64)
        workspace['power_R'] = n.empty((N_R,N_T), dtype=n.float64)
        if workspace['N_R'] < N_R:
            workspace['e_R'] = n.empty((N_R,), dtype=n.float64)
            workspace['avgphi_R'] = n.empty((N_R,), dtype=n.float64)
        workspace['N_R'] = N_R
        
    if N_I is not None and (workspace['N_I'] < N_I or workspace['N_T'] != N_T):
        workspace['sigma2_I'] = n.empty((N_I, N_T), dtype=n.float64)
        workspace['correlation_I'] = n.empty((N_I, N_T), dtype=n.float64)
        workspace['power_I'] = n.empty((N_I, N_T), dtype=n.float64)
        workspace['g_I'] = n.empty((N_I,N_T), dtype=n.float32)
        if workspace['N_I'] < N_I:
            workspace['e_I'] = n.empty((N_I,), dtype=n.float64)
            workspace['avgphi_I'] = n.empty((N_I,), dtype=n.float64)
        workspace['N_I'] = N_I

    if workspace['N_T'] != N_T:
        workspace['sigma2_est']  = n.zeros((N_T,), dtype=n.float64)
        workspace['correlation'] = n.zeros((N_T,), dtype=n.float64)
        workspace['power'] = n.zeros((N_T,), dtype=n.float64)
        workspace['nttmp'] = n.empty((N_T,), dtype=n.float64)
    else:
        workspace['sigma2_est'][:] = 0
        workspace['correlation'][:] = 0
        workspace['power'][:] = 0

    workspace['N_T'] = N_T
        
    return workspace


"""
This function computes the negative log likelihood of a single image an
its gradient.  It also computes the phi vectors needed for importance sampling
and an estimate of the noise.

no shifts
""" 
def doimage_RI(n.ndarray[n.float32_t, ndim=2] slices, # Slices of 3D volume (N_R x N_T)
               n.ndarray[n.float32_t, ndim=1] envelope, # (Experimental) envelope (N_T)
               n.ndarray[n.float32_t, ndim=2] ctf, # CTF operators (rotated) (N_I x N_T)
               n.ndarray[n.float32_t, ndim=2] d, # Image data (rotated) (N_I x N_T)
               n.ndarray[n.float32_t, ndim=1] logW_I, # Inplane weights
               n.ndarray[n.float32_t, ndim=1] logW_R, # Slice weights 
               sigma2, # Inlier noise, can be a scalar or an N_T length vector
               n.ndarray[n.float32_t, ndim=2] g, # Where to store gradient output
               workspace):

    # cdef unsigned int N_S = S.shape[0] # Number of shifts 
    # assert logW_S.shape[0] == N_S

    cdef unsigned int N_I = ctf.shape[0] # Number of inplane rotations 
    assert logW_I.shape[0] == N_I
    assert d.shape[0] == N_I

    cdef unsigned int N_R = slices.shape[0] # Number of slices (projections)
    assert logW_R.shape[0] == N_R

    cdef unsigned int N_T = slices.shape[1] # Number of (truncated) fourier coefficients
    # assert S.shape[1] == N_T
    assert ctf.shape[1] == N_T
    assert d.shape[1] == N_T

    workspace = update_workspace(workspace,N_R,N_I,None,N_T)

    cdef n.ndarray[n.float32_t, ndim=2] g_I = workspace['g_I']

    cdef n.ndarray[n.float64_t, ndim=1] e_R  = workspace['e_R']
    cdef n.ndarray[n.float64_t, ndim=2] sigma2_R = workspace['sigma2_R']
    cdef n.ndarray[n.float64_t, ndim=2] correlation_R  = workspace['correlation_R']
    cdef n.ndarray[n.float64_t, ndim=2] power_R  = workspace['power_R']
    cdef n.ndarray[n.float64_t, ndim=1] avgphi_R  = workspace['avgphi_R']

    cdef n.ndarray[n.float64_t, ndim=1] e_I  = workspace['e_I']
    cdef n.ndarray[n.float64_t, ndim=2] sigma2_I = workspace['sigma2_I']
    cdef n.ndarray[n.float64_t, ndim=2] correlation_I  = workspace['correlation_I']
    cdef n.ndarray[n.float64_t, ndim=2] power_I = workspace['power_I']
    cdef n.ndarray[n.float64_t, ndim=1] avgphi_I  = workspace['avgphi_I']

    # cdef n.ndarray[n.float64_t, ndim=1] e_S  = workspace['e_S']
    # cdef n.ndarray[n.float64_t, ndim=2] sigma2_S  = workspace['sigma2_S']
    # cdef n.ndarray[n.float64_t, ndim=2] correlation_S  = workspace['correlation_S']
    # cdef n.ndarray[n.float64_t, ndim=2] power_S  = workspace['power_S']
    # cdef n.ndarray[n.float64_t, ndim=1] avgphi_S  = workspace['avgphi_S']

    cdef n.ndarray[n.float64_t, ndim=1] sigma2_est  = workspace['sigma2_est']
    cdef n.ndarray[n.float64_t, ndim=1] correlation = workspace['correlation']
    cdef n.ndarray[n.float64_t, ndim=1] power = workspace['power']
    
    cdef n.ndarray[n.float64_t, ndim=1] nttmp = workspace['nttmp']

    # cdef n.float64_t e, ei, er, es, div_in, lse_in, tmp, phitmp, etmp
    cdef n.float64_t e, ei, er, div_in, lse_in, tmp, phitmp, etmp
    cdef n.float64_t cproj, cim
    
    cdef n.float64_t sigma2_white
    cdef n.ndarray[n.float64_t, ndim=1] sigma2_coloured = None

    cdef unsigned int use_envelope
    cdef unsigned int use_whitenoise
    cdef unsigned int computeGrad

    cdef unsigned int i, r, s, t

    use_envelope = envelope is not None
    use_whitenoise = not isinstance(sigma2,n.ndarray)
    computeGrad = g is not None
    # avgphi_S.fill(-n.inf)
    avgphi_I.fill(-n.inf)

    if use_whitenoise:
        sigma2_white = sigma2
        # div_in = -1.0/(2.0*sigma2)
        div_in = 1.0
    else:
        sigma2_coloured = sigma2
        assert sigma2_coloured.shape[0] == N_T
        # div_in = -0.5
        div_in = 1.0

    if use_envelope:
        assert envelope.shape[0] == N_T

    if computeGrad:
        assert g.shape[0] == N_R
        assert g.shape[1] == N_T

    with nogil:
        for r in xrange(N_R):
            for i in xrange(N_I):
                # Compute the error at each frequency
                tmp = 0  # temp variable for summation of sigma2_I
                for t in xrange(N_T):
                    cproj = ctf[i,t]*slices[r,t] + 1.0 - ctf[i,t]
                    cim = ctf[i,t]*d[i,t] + 1.0 - ctf[i,t]

                    correlation_I[i,t] = cproj * cim  # + cproj.imag*cim.imag
                    power_I[i,t] = cproj * cproj  # + cproj.imag*cproj.imag

                    # Compute the gradient
                    if computeGrad:
                        if use_envelope:
                            # g_I[i,t] = cim / (envelope[t]*cproj) - 1.0
                            g_I[i,t] = (envelope[t]*cproj) / cim  - 1.0
                        else:
                            # g_I[i,t] = cim / cproj - 1.0
                            g_I[i,t] = cproj / cim - 1.0
                        g_I[i,t] = - ctf[i,t] * g_I[i,t]
                        
                    # Compute the log likelihood
                    if use_whitenoise:
                        # sigma2_I[i,t] = ctf[i,t] * (cim * log(cproj) - cproj)
                        sigma2_I[i,t] = ctf[i,t] * (cproj * log(cim) - cim)
                        tmp += sigma2_I[i,t]
                    else:
                        # sigma2_I[i,t] = ctf[i,t] * (cim * log(cproj) - cproj)
                        sigma2_I[i,t] = ctf[i,t] * (cproj * log(cim) - cim)
                        tmp += sigma2_I[i,t] / sigma2_coloured[t]
                e_I[i] = div_in*tmp + logW_I[i]                        

            etmp = my_logsumexp(N_I, <double*>e_I.data)
            e_R[r] = etmp + logW_R[r]

            # Noise estimate
            for t in xrange(N_T):
                sigma2_R[r,t] = 0
                correlation_R[r,t] = 0
                power_R[r,t] = 0

            tmp = logW_R[r]
            for i in xrange(N_I):
                phitmp = exp(e_I[i] - etmp)
                avgphi_I[i] = my_logaddexp(avgphi_I[i], tmp + e_I[i])
                for t in xrange(N_T):
                    correlation_R[r,t] += phitmp * correlation_I[i,t]
                    power_R[r,t] += phitmp * power_I[i,t]
                    sigma2_R[r,t] += phitmp * sigma2_I[i,t]

                if computeGrad:
                    for t in xrange(N_T):
                        g[r,t] = g[r,t] + phitmp*g_I[i,t]

        e = my_logsumexp(N_R,<double*>e_R.data)
        lse_in = -e

        for t in xrange(N_T):
            nttmp[t] = 1.0

        # Noise estimate
        for r in xrange(N_R):
            phitmp = e_R[r] - e
            avgphi_R[r] = phitmp
            phitmp = exp(phitmp)
            for t in xrange(N_T):
                sigma2_est[t] += phitmp*sigma2_R[r,t]
            for t in xrange(N_T):
                correlation[t] += phitmp*correlation_R[r,t]
            for t in xrange(N_T):
                power[t] += phitmp*power_R[r,t]

            if computeGrad:
                if use_envelope or not use_whitenoise:
                    for t in xrange(N_T):
                        g[r,t] = phitmp*nttmp[t]*g[r,t]
                else:
                    # phitmp *= -2.0*div_in
                    for t in xrange(N_T):
                        g[r,t] = phitmp*g[r,t]

        ei = my_logsumexp(N_I, <double*>avgphi_I.data)
        for i in xrange(N_I):
            avgphi_I[i] = avgphi_I[i] - ei
        
    return lse_in, (avgphi_I[:N_I], avgphi_R[:N_R]), sigma2_est, correlation, power, workspace
