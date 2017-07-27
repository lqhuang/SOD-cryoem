#!/usr/bin/env python
from __future__ import print_function, division

import sys
import os
import inspect
# This file is run from a subdirectory of the package.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), os.pardir))

import time
from cryoio import mrc
from cryoio.ctfstack import CTFStack, GeneratedCTFStack
import cryoem
import density
import cryoops
import geometry
from util import format_timedelta

try:
    import cPickle as pickle  # python 2
except ImportError:
    import pickle  # python 3

import numpy as np
import pyximport; pyximport.install(
    setup_args={"include_dirs": np.get_include()}, reload_support=True)
import sincint


def genphantomdata(N_D, phantompath, ctfparfile):
    # mscope_params = {'akv': 200, 'wgh': 0.07,
    #                  'cs': 2.0, 'psize': 2.8, 'bfactor': 500.0}
    mscope_params = {'akv': 200, 'wgh': 0.07,
                     'cs': 2.0, 'psize': 3.0, 'bfactor': 500.0}
    
    M = mrc.readMRC(phantompath)

    N = M.shape[0]
    rad = 0.95
    shift_sigma = 3.0
    sigma_noise = 25.0
    M_totalmass = 80000
    kernel = 'lanczos'
    ksize = 6

    premult = cryoops.compute_premultiplier(N, kernel, ksize)

    tic = time.time()

    N_D = int(N_D)
    N = int(N)
    rad = float(rad)
    psize = mscope_params['psize']
    bfactor = mscope_params['bfactor']
    shift_sigma = float(shift_sigma)
    sigma_noise = float(sigma_noise)
    M_totalmass = float(M_totalmass)

    srcctf_stack = CTFStack(ctfparfile, mscope_params)
    genctf_stack = GeneratedCTFStack(mscope_params, parfields=[
                                     'PHI', 'THETA', 'PSI', 'SHX', 'SHY'])

    TtoF = sincint.gentrunctofull(N=N, rad=rad)
    Cmap = np.sort(np.random.random_integers(
        0, srcctf_stack.get_num_ctfs() - 1, N_D))

    cryoem.window(M, 'circle')
    M[M < 0] = 0
    if M_totalmass is not None:
        M *= M_totalmass / M.sum()

    V = density.real_to_fspace(
        premult.reshape((1, 1, -1)) * premult.reshape((1, -1, 1)) * premult.reshape((-1, 1, 1)) * M)

    print("Generating data...")
    sys.stdout.flush()
    imgdata = np.empty((N_D, N, N), dtype=density.real_t)

    pardata = {'R': [], 't': []}

    prevctfI = None
    for i, srcctfI in enumerate(Cmap):
        ellapse_time = time.time() - tic
        remain_time = float(N_D - i) * ellapse_time / max(i, 1)
        print("\r%.2f Percent.. (Elapsed: %s, Remaining: %s)" % (i / float(N_D)
                                                                 * 100.0, format_timedelta(ellapse_time), format_timedelta(remain_time)))
        sys.stdout.flush()

        # Get the CTF for this image
        cCTF = srcctf_stack.get_ctf(srcctfI)
        if prevctfI != srcctfI:
            genctfI = genctf_stack.add_ctf(cCTF)
            C = cCTF.dense_ctf(N, psize, bfactor).reshape((N**2,))
            prevctfI = srcctfI

        # Randomly generate the viewing direction/shift
        pt = np.random.randn(3)
        pt /= np.linalg.norm(pt)
        psi = 2 * np.pi * np.random.rand()
        EA = geometry.genEA(pt)[0]
        EA[2] = psi
        shift = np.random.randn(2) * shift_sigma

        R = geometry.rotmat3D_EA(*EA)[:, 0:2]
        slop = cryoops.compute_projection_matrix(
            [R], N, kernel, ksize, rad, 'rots')
        S = cryoops.compute_shift_phases(shift.reshape((1, 2)), N, rad)[0]

        D = slop.dot(V.reshape((-1,)))
        D *= S

        imgdata[i] = density.fspace_to_real((C * TtoF.dot(D)).reshape((N, N))) + np.require(
            np.random.randn(N, N) * sigma_noise, dtype=density.real_t)

        genctf_stack.add_img(genctfI,
                             PHI=EA[0] * 180.0 / np.pi, THETA=EA[1] * 180.0 / np.pi, PSI=EA[2] * 180.0 / np.pi,
                             SHX=shift[0], SHY=shift[1])

        pardata['R'].append(R)
        pardata['t'].append(shift)

    print("\rDone in ", time.time() - tic, " seconds.")
    return imgdata, genctf_stack, pardata, mscope_params


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("""Wrong Number of Arguments. Usage:
%run genphantomdata num inputmrc inputpar output
%run genphantomdata 5000 finalphantom.mrc Data/thermus/thermus_Nature2012_128x128.par Data/phantom_5000
""")
        sys.exit()

    imgdata, ctfstack, pardata, mscope_params = genphantomdata(sys.argv[1], sys.argv[2], sys.argv[3])
    outpath = sys.argv[4]
    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    tic = time.time()
    print("Dumping data..."); sys.stdout.flush()
    def_path = os.path.join(outpath,'defocus.txt')
    ctfstack.write_defocus_txt(def_path)

    par_path = os.path.join(outpath,'ctf_gt.par')
    ctfstack.write_pardata(par_path)

    mrc_path = os.path.join(outpath,'imgdata.mrc')
    print(os.path.realpath(mrc_path))
    mrc.writeMRC(mrc_path, np.transpose(imgdata,(1,2,0)), mscope_params['psize'])

    pard_path = os.path.join(outpath,'pardata.pkl')
    print(os.path.realpath(par_path))
    with open(pard_path,'wb') as fi:
        pickle.dump(pardata, fi, protocol=2)
    print("Done in ", time.time()-tic, " seconds.")

    sys.exit()
