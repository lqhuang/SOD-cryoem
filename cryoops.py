from __future__ import print_function, division

import numpy as np
import pyximport; pyximport.install(
    setup_args={"include_dirs": np.get_include()}, reload_support=True)
import sincint
import geometry


precomputed_Rs = {}
def compute_projection_matrix(projdirs, N, kern, kernsize, rad, projdirtype='dirs', sym=None, onlyRs=False, **kwargs):
    projdirs = np.asarray(projdirs, dtype=np.float32)
    if projdirtype == 'dirs':
        # Input is a set of projection directions
        dirhash = hash(projdirs.tostring())
        if onlyRs and dirhash in precomputed_Rs:
            Rs = precomputed_Rs[dirhash]
        else:
            Rs = np.vstack(
                [geometry.rotmat3D_dir(d)[:, 0:2].reshape((1, 3, 2)) for d in projdirs])
            if onlyRs:
                precomputed_Rs[dirhash] = Rs
    elif projdirtype == 'rots':
        # Input is a set of rotation matrices mapping image space to protein
        # space
        Rs = projdirs
    else:
        assert False, 'Unknown projdirtype, must be either dirs or rots'

    if sym is None:
        symRs = None
    else:
        symRs = np.vstack([np.require(R, dtype=np.float32).reshape(
            (1, 3, 3)) for R in sym.get_rotations()])

    if onlyRs:
        return Rs
    else:
        return sincint.compute_interpolation_matrix(Rs, N, N, rad, kern, kernsize, symRs)


precomputed_RIs = {}
def compute_inplanerot_matrix(thetas, N, kern, kernsize, rad, N_src=None, onlyRs=False):
    dirhash = hash(thetas.tostring())
    if N_src is None:
        N_src = N
        scale = 1
    else:
        scale = float(N_src) / N
    if onlyRs and dirhash in precomputed_RIs:
        Rs = precomputed_RIs[dirhash]
    else:
        Rs = np.vstack(
            [scale * geometry.rotmat2D(np.require(th, dtype=np.float32)).reshape((1, 2, 2)) \
            for th in thetas])
        if onlyRs:
            precomputed_RIs[dirhash] = Rs
    if onlyRs:
        return Rs
    else:
        return sincint.compute_interpolation_matrix(Rs, N, N_src, rad, kern, kernsize, None)


def compute_shift_phases(pts, N, rad):
    xy = geometry.gencoords(N, 2, rad)
    N_T = xy.shape[0]
    N_S = pts.shape[0]

    shifts = np.empty((N_S, N_T), dtype=np.complex64)
    for (i, (sx, sy)) in enumerate(pts):
        shifts[i] = np.exp(2.0j * np.pi / N * (xy[:, 0] * sx + xy[:, 1] * sy))

    return shifts


def compute_premultiplier(N, kernel, kernsize, scale=512):
    krange = int(N / 2)
    koffset = int((N / 2) * scale)

    x = np.arange(-scale * krange, scale * krange) / float(scale)
    if kernel == 'lanczos':
        a = kernsize / 2
        k = np.sinc(x) * np.sinc(x / a) * (np.abs(x) <= a)
    elif kernel == 'sinc':
        a = kernsize / 2.0
        k = np.sinc(x) * (np.abs(x) <= a)
    elif kernel == 'linear':
        assert kernsize == 2
        k = np.maximum(0.0, 1 - np.abs(x))
    elif kernel == 'quad':
        assert kernsize == 3
        k = (np.abs(x) <= 0.5) * (1 - 2 * x**2) + \
            ((np.abs(x) < 1) * (np.abs(x) > 0.5)) * 2 * (1 - np.abs(x))**2
    else:
        assert False, 'Unknown kernel type'

    sk = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(k))).real
    if N % 2 == 0:
        premult = 1.0 / (N * sk[(koffset - krange):(koffset + krange)])
    else:
        premult = 1.0 / (N * sk[(koffset - krange - 1):(koffset + krange)])

    return premult


if __name__ == '__main__':

    kern = 'sinc'
    kernsize = 3

    N = 128

    pm1 = compute_premultiplier(N, kern, kernsize, 512)
    pm2 = compute_premultiplier(N, kern, kernsize, 8192)

    print(np.max(np.abs(pm1 - pm2)))

    premult = compute_premultiplier(125, 'lanczos', 4)
    print(premult.shape)
