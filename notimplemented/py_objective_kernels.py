from __future__ import print_function, division

import numpy as np


def my_logsumexp(N , a):
    """
    return log(sum_i(exp(a_i)))
    unsigned int N, double *a:
    https://en.wikipedia.org/wiki/LogSumExp
    """
    a_max = a[0]
    for i in range(N):
        if a[i] > a_max:
            a_max = a[i]

    a_sum = 0
    for i in range(N):
        a_sum += np.exp(a[i] - a_max)

    return a_max + np.log(a_sum)


def my_logaddexp(a, b):
    """
    return: log(exp(a) + exp(b))
    double a, double b
    https://en.wikipedia.org/wiki/Log_probability
    """
    if a == b:
        return a + 0.69314718055994529 # This is the numerical value of ln(2)
    else:
        tmp = a-b
        
        if tmp > 0:
            return a + np.log1p(np.exp(-tmp))
        elif tmp <= 0:
            return b + np.log1p(np.exp(tmp))
        else:
            return tmp


def check(slices, shifts, envelope,
          ctf, data,
          logW_S, logW_I, logW_R,
          sigma2, g):
    # type checking
    if isinstance(slices, np.ndarray):
        assert slices.dtype == np.complex64
    else:
        raise TypeError(
            'Wrong type ({}}) for slices. '.format(type(slices)) + 
            'Numpy ndarray with shape (N_R, N_T) is expected')
    if shifts is None:
        pass
    elif isinstance(shifts, np.ndarray):
        assert shifts.dtype == np.complex64
    else:
        raise TypeError(
            'Wrong type ({}) for shifts. '.format(type(shifts)) +
            'Numpy ndarray with shape (N_S, N_T) is expected')
    if isinstance(envelope, np.ndarray):
        assert envelope.dtype == np.float32
    else:
        raise TypeError(
            'Wrong type ({}) for envelope. '.format(type(envelope)) +
            'Numpy ndarry with sahpe (N_T,) is expected.')
    if isinstance(ctf, np.ndarray):
        assert ctf.dtype == np.float32
    else:
        raise TypeError(
            'Wrong type ({}) for ctf. '.format(type(ctf)) +
            'Numpy ndarry with sahpe (N_I, N_T) is expected.')
    if isinstance(data, np.ndarray):
        assert data.dtype == np.complex64
    else:
        raise TypeError(
            'Wrong type ({}) for data. '.format(type(data)) +
            'Numpy ndarry with sahpe (N_I, N_T) is expected.')
    if logW_S is None:
        pass
    elif isinstance(logW_S, np.ndarray):
        assert logW_S.dtype == np.float32
    else:
        raise TypeError(
            'Wrong type ({}) for logW_S. '.format(type(logW_S)) +
            'Numpy ndarry with sahpe (N_S,) is expected.')
    if isinstance(logW_I, np.ndarray):
        assert logW_I.dtype == np.float32
    else:
        raise TypeError(
            'Wrong type ({}) for logW_I. '.format(type(logW_I)) +
            'Numpy ndarry with sahpe (N_I,) is expected.')
    if isinstance(logW_R, np.ndarray):
        assert logW_R.dtype == np.float32
    else:
        raise TypeError(
            'Wrong type ({}) for logW_R. '.format(type(logW_R)) +
            'Numpy ndarry with sahpe (N_R,) is expected.')
    if isinstance(sigma2, np.ndarray):
        assert sigma2.dtype == np.float32
    elif isinstance(sigma2, float) or isinstance(sigma2, int):
        pass
    else:
        raise TypeError(
            'Wrong type ({}) for sigma2. '.format(type(sigma2)) +
            'Scalar or a numpy ndarry with sahpe (N_T,) is expected.')
    if isinstance(g, np.ndarray):
        assert g.dtype == np.complex64
    elif g is None:
        pass
    else:
        raise TypeError(
            'Wrong type ({}) for g. '.format(type(g)) +
            'None type or a numpy ndarry with sahpe (N_T,) is expected.')

    # size checking
    N_R, N_T = slices.shape
    N_I = data.shape[0]
    if shifts is not None:
        N_S = shifts.shape[0]
    else:
        N_S = None

    assert logW_R.shape[0] == N_R
    assert logW_I.shape[0] == N_I
    if logW_S is not None:
        assert logW_S.shape[0] == N_S
    assert ctf.shape[0] == N_I

    assert envelope.shape[0] == N_T
    assert data.shape[1] == N_T
    assert ctf.shape[1] == N_T
    if shifts is not None:
        assert shifts.shape[1] == N_T
    
    return N_R, N_I, N_S, N_T


def update_workspace(workspace, N_R, N_I, N_S, N_T):
    if workspace is None:
        workspace = {'N_R':0,'N_I':0,'N_S':0,'N_T':0}

    if N_R is not None and workspace['N_R'] < N_R or workspace['N_T'] != N_T:
        workspace['sigma2_R'] = np.empty((N_R,N_T), dtype=np.float64)
        workspace['correlation_R'] = np.empty((N_R,N_T), dtype=np.float64)
        workspace['power_R'] = np.empty((N_R,N_T), dtype=np.float64)
        workspace['g_R'] = np.empty((N_R,N_T), dtype=np.float32)
        if workspace['N_R'] < N_R:
            workspace['e_R'] = np.empty((N_R,), dtype=np.float64)
            workspace['avgphi_R'] = np.empty((N_R,), dtype=np.float64)
        workspace['N_R'] = N_R

    if N_I is not None and (workspace['N_I'] < N_I or workspace['N_T'] != N_T):
        workspace['sigma2_I'] = np.empty((N_I, N_T), dtype=np.float64)
        workspace['correlation_I'] = np.empty((N_I, N_T), dtype=np.float64)
        workspace['power_I'] = np.empty((N_I, N_T), dtype=np.float64)
        workspace['g_I'] = np.empty((N_I,N_T), dtype=np.complex64)
        if workspace['N_I'] < N_I:
            workspace['e_I'] = np.empty((N_I,), dtype=np.float64)
            workspace['avgphi_I'] = np.empty((N_I,), dtype=np.float64)
        workspace['N_I'] = N_I

    if N_S is not None and (workspace['N_S'] < N_S or workspace['N_T'] != N_T):
        workspace['sigma2_S'] = np.empty((N_S, N_T), dtype=np.float64)
        workspace['correlation_S'] = np.empty((N_S, N_T), dtype=np.float64)
        workspace['power_S'] = np.empty((N_S, N_T), dtype=np.float64)
        workspace['g_S'] = np.empty((N_S,N_T), dtype=np.complex64)
        if workspace['N_S'] < N_S:
            workspace['e_S'] = np.empty((N_S,), dtype=np.float64)
            workspace['avgphi_S'] = np.empty((N_S,), dtype=np.float64)
        workspace['N_S'] = N_S

    if workspace['N_T'] != N_T:
        workspace['sigma2_est']  = np.zeros((N_T,), dtype=np.float64)
        workspace['correlation'] = np.zeros((N_T,), dtype=np.float64)
        workspace['power'] = np.zeros((N_T,), dtype=np.float64)
        workspace['nttmp'] = np.empty((N_T,), dtype=np.float64)
    else:
        workspace['sigma2_est'][:] = 0
        workspace['correlation'][:] = 0
        workspace['power'][:] = 0

    workspace['N_T'] = N_T

    return workspace


def doimage_RIS(slices, # np.ndarray[np.complex64_t, ndim=2] slices,  # Slices of 3D volume (N_R x N_T)
                shifts, # np.ndarray[n.complex64_t, ndim=2] S, # Shift operators (N_S X N_T)
                envelope, # np.ndarray[np.float32_t, ndim=1] envelope,  # (Experimental) envelope (N_T)
                ctf, # np.ndarray[np.float32_t, ndim=1] ctf,  # CTF operator (rotated) (N_I X N_T)
                data, # np.ndarray[np.complex64_t, ndim=1] d,  # Image data (rotated) (N_I X N_T)
                logW_S, # np.ndarray[n.float32_t, ndim=1] logW_S, # Shift weights
                logW_I, # np.ndarray[np.float32_t, ndim=1] logW_I, # Inplane weights
                logW_R, # np.ndarray[np.float32_t, ndim=1] logW_R, # Slice weights 
                sigma2, # Inlier noise, can be a scalar or an N_T length vector
                g, # np.ndarray[np.complex64_t, ndim=2] g, # Where to store gradient output
                workspace):
    # type checking and size checking
    N_R, N_I, N_S, N_T = check(slices, shifts, envelope, ctf, data,
                               logW_S, logW_I, logW_R, sigma2, g)

    # update working space
    workspace = update_workspace(workspace, N_R, N_I, N_S, N_T)
    g_I = workspace['g_I']
    g_S = workspace['g_S']

    e_R = workspace['e_R']
    sigma2_R = workspace['sigma2_R']
    correlation_R = workspace['correlation_R']
    power_R = workspace['power_R']
    avgphi_R = workspace['avgphi_R']

    e_I = workspace['e_I']
    sigma2_I = workspace['sigma2_I']
    correlation_I = workspace['correlation_I']
    power_I = workspace['power_I']
    avgphi_I = workspace['avgphi_I']

    e_S = workspace['e_S']
    sigma2_S = workspace['sigma2_S']
    correlation_S = workspace['correlation_S']
    power_S = workspace['power_S']
    avgphi_S = workspace['avgphi_S']

    sigma2_est = workspace['sigma2_est']
    correlation = workspace['correlation']
    power = workspace['power']

    nttmp = workspace['nttmp']

    # set params
    use_envelope = envelope is not None
    use_whitenoise = not isinstance(sigma2, np.ndarray)
    computerGrad = g is not None
    avgphi_S.fill(-np.inf)
    avgphi_I.fill(-np.inf)

    if use_whitenoise:
        sigma2_white = sigma2
        div_in = -1.0 / (2.0 * sigma2)
    else:
        sigma2_coloured = sigma2
        assert sigma2_coloured.shape[0] == N_T
        tiled_sigma2_coloured = np.tile(sigma2_coloured, (N_I, 1))
        div_in = -0.5
    
    if use_envelope:
        assert envelope.shape[0] == N_T
        tiled_envelope = np.tile(envelope, (N_I, 1))

    if computerGrad:
        assert g.shape[0] == N_R
        assert g.shape[1] == N_T

    # computing
    for r, cproj in enumerate(slices):
        for s, curr_shift in enumerate(shifts):
            cprojs = ctf * np.tile(cproj, (N_I, 1))
            cimg = np.tile(curr_shift, (N_I, 1)) * data
            # compute the error at each frequency
            correlation_I[:] = cprojs.real * data.real + cprojs.imag * cimg.imag
            power_I[:] = cprojs.real ** 2 + cprojs.imag ** 2

            if use_envelope:
                g_I[:] = tiled_envelope * cprojs - cimg
            else:
                g_I[:] = cprojs - cimg
        
            # compute the log likelihood
            sigma2_I[:] = g_I.real ** 2 + g_I.imag ** 2
            if use_whitenoise:
                tmp = sigma2_I
            else:
                tmp = sigma2_I / tiled_sigma2_coloured
            e_I[:] = div_in * tmp.sum(axis=1) + logW_I

            # compute the gradient
            if computerGrad:
                g_I[:] = ctf * g_I[:]

            etmp = my_logsumexp(N_I, e_I)
            e_S[s] = etmp + logW_S[s]

            # Noise estimate
            correlation_S.fill(0.0)
            power_S.fill(0.0)
            sigma2_S.fill(0.0)
            if computerGrad:
                g_S.fill(0.0)
            tmp = logW_S[s] + logW_R[r]
            phitmp = np.exp(e_I - etmp)
            tiled_phitmp = np.tile(phitmp, (N_T, 1)).T
            avgphi_I[:] = np.asarray([my_logaddexp(avgphi_I[i], tmp + e_I[i]) for i in range(N_I)], dtype=np.float64)
            correlation_S[s] = (tiled_phitmp * correlation_I).sum(axis=0)
            power_S[s] = (tiled_phitmp * power_I).sum(axis=0)
            sigma2_S[s] = (tiled_phitmp * sigma2_I).sum(axis=0)

            if computerGrad:
                g[s] = g[s] + (tiled_phitmp * g_I).sum(axis=0)
            
        etmp = my_logsumexp(N_S, e_S)
        e_R[r] = etmp + logW_R[r]

        # Noise estimate
        sigma2_R.fill(0)
        correlation_R.fill(0)
        power_R.fill(0)
        tmp = logW_R[r]
        phitmp = np.exp(e_S - etmp)
        tiled_phitmp = np.tile(phitmp, (N_T, 1)).T
        avgphi_S[:] = np.asarray([my_logaddexp(avgphi_S[s], tmp + e_S[s]) for s in range(N_S)], dtype=np.float64)
        correlation_R[r] = (tiled_phitmp * correlation_S).sum(axis=0)
        power_R[r] = (tiled_phitmp * power_S).sum(axis=0)
        sigma2_R[r] = (tiled_phitmp * sigma2_S).sum(axis=0)

        if computerGrad:
            g[r] = g[r] + (tiled_phitmp * g_S).sum(axis=0)
    
    e = my_logsumexp(N_R, e_R)
    lse_in = -e

    if computerGrad:
        tmp = -2.0 * div_in
        if not use_whitenoise:
            if use_envelope:
                nttmp[:] = tmp * envelope / sigma2_coloured
            else:
                nttmp[:] = tmp / sigma2_coloured
        else:
            if use_envelope:
                nttmp[:] = tmp * envelope
            # else:
            #     nttmp[:] = tmp

    # Noise estimate
    phitmp = e_R - e
    avgphi_R = phitmp
    phitmp = np.exp(phitmp)
    tiled_phitmp = np.tile(phitmp, (N_T, 1)).T
    sigma2_est[:] = (tiled_phitmp * sigma2_R).sum(axis=0)
    correlation[:] = (tiled_phitmp * correlation_R).sum(axis=0)
    power[:] = (tiled_phitmp * power_R).sum(axis=0)

    if computerGrad:
        if use_envelope or not use_whitenoise:
            g[:] = tiled_phitmp * np.tile(nttmp, (N_R, 1)) * g
        else:
            tiled_phitmp *= -2.0 * div_in
            g[:] = tiled_phitmp * g
            
    es = my_logsumexp(N_S, avgphi_S)
    avgphi_S[:] = avgphi_S - es
    ei = my_logsumexp(N_I, avgphi_I)
    avgphi_I[:] = avgphi_I - ei

    return lse_in, (avgphi_S[:N_S], avgphi_I[:N_I], avgphi_R[:N_R]), sigma2_est, correlation, power, workspace


def doimage_RI(slices, # np.ndarray[np.complex64_t, ndim=2] slices,  # Slices of 3D volume (N_R x N_T)
               envelope, # np.ndarray[np.float32_t, ndim=1] envelope,  # (Experimental) envelope (N_T)
               ctf, # np.ndarray[np.float32_t, ndim=1] ctf,  # CTF operators (rotated) (N_I X N_T)
               data, # np.ndarray[np.complex64_t, ndim=1] d,  # Image data (rotated) (N_I X N_T)
               logW_I, # np.ndarray[np.float32_t, ndim=1] logW_I, # Inplane weights
               logW_R, # np.ndarray[np.float32_t, ndim=1] logW_R, # Slice weights 
               sigma2, # Inlier noise, can be a scalar or an N_T length vector
               g, # np.ndarray[np.complex64_t, ndim=2] g, # Where to store gradient output
               workspace):
    # type checking and size checking
    N_R, N_I, N_S, N_T = check(slices, None, envelope, ctf, data,
                               None, logW_I, logW_R, sigma2, g)

    # update working space
    workspace = update_workspace(workspace, N_R, N_I, None, N_T)

    g_I = workspace['g_I']

    e_R = workspace['e_R']
    sigma2_R = workspace['sigma2_R']
    correlation_R = workspace['correlation_R']
    power_R = workspace['power_R']
    avgphi_R = workspace['avgphi_R']

    e_I = workspace['e_I']
    sigma2_I = workspace['sigma2_I']
    correlation_I = workspace['correlation_I']
    power_I = workspace['power_I']
    avgphi_I = workspace['avgphi_I']

    sigma2_est = workspace['sigma2_est']
    correlation = workspace['correlation']
    power = workspace['power']

    nttmp = workspace['nttmp']

    # set params
    use_envelope = envelope is not None
    use_whitenoise = not isinstance(sigma2, np.ndarray)
    computerGrad = g is not None
    avgphi_I.fill(-np.inf)

    if use_whitenoise:
        sigma2_white = sigma2
        div_in = -1.0 / (2.0 * sigma2)
    else:
        sigma2_coloured = sigma2
        assert sigma2_coloured.shape[0] == N_T
        tiled_sigma2_coloured = np.tile(sigma2_coloured, (N_I, 1))
        div_in = -0.5
    
    if use_envelope:
        assert envelope.shape[0] == N_T
        tiled_envelope = np.tile(envelope, (N_I, 1))

    if computerGrad:
        assert g.shape[0] == N_R
        assert g.shape[1] == N_T

    for r, cproj in enumerate(slices):
        cprojs = ctf * np.tile(cproj, (N_I, 1))
        # compute the error at each frequency
        correlation_I[:] = cprojs.real * data.real + cprojs.imag * data.imag
        power_I[:] = cprojs.real ** 2 + cprojs.imag ** 2

        if use_envelope:
            g_I[:] = tiled_envelope * cprojs - data
        else:
            g_I[:] = cprojs - data
        
        # compute the log likelihood
        sigma2_I[:] = g_I.real ** 2 + g_I.imag ** 2
        if use_whitenoise:
            tmp = sigma2_I
        else:
            tmp = sigma2_I / tiled_sigma2_coloured
        e_I[:] = div_in * tmp.sum(axis=1) + logW_I

        # compute the gradient
        if computerGrad:
            g_I[:] = ctf * g_I

        etmp = my_logsumexp(N_I, e_I)
        e_R[r] = etmp + logW_R[r]
    
        # Noise estimate
        sigma2_R[r].fill(0.0)
        correlation_R[r].fill(0.0)
        power_R[r].fill(0.0)

        tmp = logW_R[r]
        phitmp = np.exp(e_I - etmp)
        tiled_phitmp = np.tile(phitmp, (N_T, 1)).T
        avgphi_I[:] = np.asarray([my_logaddexp(avgphi_I[i], tmp + e_I[i]) for i in range(N_I)], dtype=np.float64)
        correlation_R[r] = (tiled_phitmp * correlation_I).sum(axis=0)
        power_R[r] = (tiled_phitmp * power_I).sum(axis=0)
        sigma2_R[r] = (tiled_phitmp * sigma2_I).sum(axis=0)

        if computerGrad:
            g[r] = g[r] + (tiled_phitmp * g_I).sum(axis=0)
        
    e = my_logsumexp(N_R, e_R)
    lse_in = -e

    if computerGrad:
        tmp = -2.0 * div_in
        if not use_whitenoise:
            if use_envelope:
                nttmp[:] = tmp * envelope / sigma2_coloured
            else:
                nttmp[:] = tmp / sigma2_coloured
        else:
            if use_envelope:
                nttmp[:] = tmp * envelope
            # else:
            #     nttmp[:] = tmp

    # Noise estimate
    phitmp = e_R - e
    avgphi_R = phitmp
    phitmp = np.exp(phitmp)
    tiled_phitmp = np.tile(phitmp, (N_T, 1)).T
    sigma2_est[:] = (tiled_phitmp * sigma2_R).sum(axis=0)
    correlation[:] = (tiled_phitmp * correlation_R).sum(axis=0)
    power[:] = (tiled_phitmp * power_R).sum(axis=0)

    if computerGrad:
        if use_envelope or not use_whitenoise:
            g[:] = tiled_phitmp * np.tile(nttmp, (N_R, 1)) * g
        else:
            tiled_phitmp *= -2.0 * div_in
            g[:] = tiled_phitmp * g

    ei = my_logsumexp(N_I, avgphi_I)
    avgphi_I[:] = avgphi_I - ei

    return lse_in, (avgphi_I[:N_I], avgphi_R[:N_R]), sigma2_est, correlation, power, workspace


def doimage_ACRI(slices, # np.ndarray[np.complex64_t, ndim=2] slices,  # Slices of 3D volume (N_R x N_T)
                 envelope, # np.ndarray[np.float32_t, ndim=1] envelope,  # (Experimental) envelope (N_T)
                 ctf, # np.ndarray[np.float32_t, ndim=1] ctf,  # CTF operators (rotated) (N_I X N_T)
                 data, # np.ndarray[np.complex64_t, ndim=1] d,  # Image data (rotated) (N_I X N_T)
                 # ac_slices, # n.ndarray[n.complex64_t, ndim=2] ac_slices, # Angular correlation slices of 3D volume (N_R x N_T)
                 # ac_data, # n.ndarray[n.complex64_t, ndim=1] ac_d, # Image data (1 x N_T)
                 ac_indices, # n.ndarray[n.int64_t, ndim=1]
                 logW_I, # np.ndarray[np.float32_t, ndim=1] logW_I, # Inplane weights
                 logW_R, # np.ndarray[np.float32_t, ndim=1] logW_R, # Slice weights 
                 sigma2, # Inlier noise, can be a scalar or an N_T length vector
                 g, # np.ndarray[np.complex64_t, ndim=2] g, # Where to store gradient output
                 workspace):
    # type checking and size checking
    N_R, N_I, N_S, N_T = check(slices, None, envelope, ctf, data,
                               None, logW_I, logW_R, sigma2, g)

    # update working space
    workspace = update_workspace(workspace, N_R, N_I, None, N_T)

    g_I = workspace['g_I']
    g_R = workspace['g_R']

    e_R = workspace['e_R']
    sigma2_R = workspace['sigma2_R']
    correlation_R = workspace['correlation_R']
    power_R = workspace['power_R']
    avgphi_R = workspace['avgphi_R']

    e_I = workspace['e_I']
    sigma2_I = workspace['sigma2_I']
    correlation_I = workspace['correlation_I']
    power_I = workspace['power_I']
    avgphi_I = workspace['avgphi_I']

    sigma2_est = workspace['sigma2_est']
    correlation = workspace['correlation']
    power = workspace['power']

    nttmp = workspace['nttmp']

    # set params
    use_envelope = envelope is not None
    use_whitenoise = not isinstance(sigma2, np.ndarray)
    computerGrad = g is not None
    
    avgphi_R.fill(-np.inf)
    avgphi_I.fill(-np.inf)
    e_R.fill(-np.inf)
    e_I.fill(-np.inf)
    
    sigma2_R.fill(0.0)
    correlation_R.fill(0.0)
    power_R.fill(0.0)
    sigma2_I.fill(0.0)
    correlation_I.fill(0.0)
    power_I.fill(0.0)

    if use_whitenoise:
        sigma2_white = sigma2
        div_in = -1.0 / (2.0 * sigma2)
    else:
        sigma2_coloured = sigma2
        assert sigma2_coloured.shape[0] == N_T
        tiled_sigma2_coloured = np.tile(sigma2_coloured, (N_I, 1))
        div_in = -0.5

    if use_envelope:
        assert envelope.shape[0] == N_T
        tiled_envelope = np.tile(envelope, (N_I, 1))

    if computerGrad:
        assert g.shape[0] == N_R
        assert g.shape[1] == N_T

    for r in ac_indices:
        cproj = slices[r]
        cprojs = ctf * np.tile(cproj, (N_I, 1))
        # compute the error at each frequency
        correlation_I[:] = cprojs.real * data.real + cprojs.imag * data.imag
        power_I[:] = cprojs.real ** 2 + cprojs.imag ** 2

        if use_envelope:
            g_I[:] = tiled_envelope * cprojs - data
        else:
            g_I[:] = cprojs - data
        
        # compute the log likelihood
        sigma2_I[:] = g_I.real ** 2 + g_I.imag ** 2
        if use_whitenoise:
            tmp = sigma2_I
        else:
            tmp = sigma2_I / tiled_sigma2_coloured
        e_I[:] = div_in * tmp.sum(axis=1) + logW_I

        # compute the gradient
        if computerGrad:
            g_I[:] = ctf * g_I[:]

        etmp = my_logsumexp(N_I, e_I)
        e_R[r] = etmp + logW_R[r]
    
        # Noise estimate
        sigma2_R[r].fill(0.0)
        correlation_R[r].fill(0.0)
        power_R[r].fill(0.0)

        tmp = logW_R[r]
        phitmp = np.exp(e_I - etmp)
        tiled_phitmp = np.tile(phitmp, (N_T, 1)).T
        avgphi_I[:] = np.asarray([my_logaddexp(avgphi_I[i], tmp + e_I[i]) for i in range(N_I)], dtype=np.float64)
        correlation_R[r] = (tiled_phitmp * correlation_I).sum(axis=0)
        power_R[r] = (tiled_phitmp * power_I).sum(axis=0)
        sigma2_R[r] = (tiled_phitmp * sigma2_I).sum(axis=0)

        if computerGrad:
            g[r] = g[r] + (tiled_phitmp * g_I).sum(axis=0)
        
    e = my_logsumexp(N_R, e_R)
    lse_in = -e

    if computerGrad:
        tmp = -2.0 * div_in
        if not use_whitenoise:
            if use_envelope:
                nttmp[:] = tmp * envelope / sigma2_coloured
            else:
                nttmp[:] = tmp / sigma2_coloured
        else:
            if use_envelope:
                nttmp[:] = tmp * envelope
            # else:
            #     nttmp[:] = tmp

    # Noise estimate
    phitmp = e_R - e
    avgphi_R = phitmp
    phitmp = np.exp(phitmp)
    tiled_phitmp = np.tile(phitmp, (N_T, 1)).T
    sigma2_est[:] = (tiled_phitmp * sigma2_R).sum(axis=0)
    correlation[:] = (tiled_phitmp * correlation_R).sum(axis=0)
    power[:] = (tiled_phitmp * power_R).sum(axis=0)

    if computerGrad:
        if use_envelope or not use_whitenoise:
            g[:] = tiled_phitmp * np.tile(nttmp, (N_R, 1)) * g
        else:
            tiled_phitmp *= -2.0 * div_in
            g[:] = tiled_phitmp * g

    ei = my_logsumexp(N_I, avgphi_I)
    avgphi_I[:] = avgphi_I - ei

    return lse_in, (avgphi_I[:N_I], avgphi_R[:N_R]), sigma2_est, correlation, power, workspace
