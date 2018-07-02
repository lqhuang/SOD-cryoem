from __future__ import print_function, division

import numpy as np
from util import memoize
try:
    import pyfftw
    fftmod = pyfftw.interfaces.numpy_fft
    pyfftw.interfaces.cache.enable()
    # install like so: https://dranek.com/blog/2014/Feb/conda-binstar-and-fftw/
    # print("LOADED FFTW")
    USINGFFTW = True
    import multiprocessing
    fft_threads = multiprocessing.cpu_count()
except:
    fftmod = np.fft
    USINGFFTW = False
    print("ERROR LOADING FFTW! USING NUMPY")
    fft_threads = None

real_t = np.float32
complex_t = np.complex64


def empty_like(x):
    sz = x.shape
    if USINGFFTW:
        return pyfftw.n_byte_align_empty(sz, 32, x.dtype)
    else:
        return np.empty(sz, dtype=x.dtype)


def zeros_like(x):
    sz = x.shape
    if USINGFFTW:
        ret = pyfftw.n_byte_align_empty(x.shape, 32, x.dtype)
        ret[:] = 0
        return ret
    else:
        return np.zeros(sz, dtype=x.dtype)


def empty_real(sz):
    if USINGFFTW:
        return pyfftw.n_byte_align_empty(sz, 32, real_t)
    else:
        return np.empty(sz, dtype=real_t)


def empty_cplx(sz):
    if USINGFFTW:
        return pyfftw.n_byte_align_empty(sz, 32, complex_t)
    else:
        return np.empty(sz, dtype=complex_t)


def real_to_fspace(M, axes=None, threads=None):
    """ Convert real-space M to (unitary) Fourier space """
    if USINGFFTW:
        if threads is None:
            threads = fft_threads
        ret = np.require(np.fft.fftshift(fftmod.fftn(np.fft.fftshift(M, axes=axes),
                                                     axes=axes, threads=threads),
                                         axes=axes),
                         dtype=complex_t)
    else:
        ret = np.require(np.fft.fftshift(fftmod.fftn(np.fft.fftshift(M, axes=axes),
                                                     axes=axes),
                                         axes=axes),
                         dtype=complex_t)
    # nrm is the scaling factor needed to make an unnormalized FFT a
    # unitary transform
    if axes is None:
        nrm = 1.0 / np.sqrt(np.prod(M.shape))
    else:
        nrm = 1.0 / np.sqrt(np.prod(np.array(M.shape)[np.array(axes)]))
    ret *= nrm
    return ret


def fspace_to_real(fM, axes=None, threads=None):
    """ Convert unitary Fourier space fM to real space """
    if USINGFFTW:
        if threads is None:
            threads = fft_threads
        ret = np.require(np.fft.ifftshift(fftmod.ifftn(np.fft.ifftshift(fM, axes=axes),
                                                       axes=axes, threads=threads),
                                          axes=axes).real,
                         dtype=real_t)
    else:
        ret = np.require(np.fft.ifftshift(fftmod.ifftn(np.fft.ifftshift(fM, axes=axes),
                                                       axes=axes),
                                          axes=axes).real,
                         dtype=real_t)
    # nrm is the scaling factor needed to make an unnormalized FFT a
    # unitary transform
    if axes is None:
        nrm = np.sqrt(np.prod(fM.shape))
    else:
        nrm = np.sqrt(np.prod(np.array(fM.shape)[np.array(axes)]))
    ret *= nrm
    return ret


def make_hermitian(fM):
    N = fM.shape[0]
    startFreq = 1 - (N % 2)
    if startFreq:
        fM += np.roll(np.roll(np.roll(fM[::-1, ::-1, ::-1],
                                      1, axis=0), 1, axis=1), 1, axis=2).conj()
    else:
        fM += fM[::-1, ::-1, ::-1].conj()
    fM *= 0.5
    return fM


def check_hermitian(fM):
    N = fM.shape[0]
    startFreq = 1 - (N % 2)
    if startFreq:
        E = fM - \
            np.roll(
                np.roll(np.roll(fM[::-1, ::-1, ::-1], 1, axis=0), 1, axis=1), 1, axis=2).conj()
    else:
        E = fM - fM[::-1, ::-1, ::-1].conj()

    return np.linalg.norm(np.absolute(E))



def real_to_fspace_with_oversampling(M, oversampling_factor, do_premult=False, kernel='lanczos', ksize=6):
    N = M.shape[0]
    ndim = M.ndim
    zeropad = oversampling_factor - 1
    zeropad_size = int(zeropad * (N / 2.0))
    zp_N = zeropad_size * 2 + N
    zp_M_shape = (zp_N,) * ndim
    zp_M = np.zeros(zp_M_shape, dtype=real_t)
    zp_M_slicer = (slice( zeropad_size, (N + zeropad_size) ),) * ndim
    zp_M[zp_M_slicer] = M

    if do_premult:
        from cryoops import compute_premultiplier
        premult = compute_premultiplier(zp_N, kernel, ksize)
        V = real_to_fspace(
                premult.reshape((1, 1, -1))
                * premult.reshape((1, -1, 1))
                * premult.reshape((-1, 1, 1))
                * zp_M)
    else:
        V = real_to_fspace(zp_M)

    fM = V[zp_M_slicer]
    return fM


# These functons are used in GPU OTF slicing and unslicing.
# They either take a volume and put it into 2x2x2 cell format, or take
# a 2x2x2 cell format and accumulate back into a volume.


def to_cell_3d(V, c=2):
    N = V.shape[0]
    p_b, rem = divmod(c - 1, 2)
    p_a = c - p_b - 2
    Np = N + p_b + p_a
    Vp = np.zeros([Np] * 3, dtype=V.dtype)
    Vp[p_b:Np - p_a, p_b:Np - p_a, p_b:Np - p_a] = V
    N_C = N - 1
    C = np.zeros((N_C**3, c**3), dtype=V.dtype).reshape(N_C, N_C, N_C, -1)
    for xo in range(c):
        for yo in range(c):
            for zo in range(c):
                Vo = Vp[xo:xo + (Np - xo) / c * c, yo:yo +
                        (Np - yo) / c * c, zo:zo + (Np - zo) / c * c]
                No = Vo.shape
                Vo_cells = Vo.reshape(No[0] / c, c,
                                      No[1] / c, c,
                                      No[2] / c, c, ) \
                    .transpose(0, 2, 4, 1, 3, 5) \
                    .reshape(No[0] / c, No[1] / c, No[2] / c, c**3)
                C[xo::c, yo::c, zo::c] = Vo_cells
    return C


def from_cell_3d(C, c=2):
    N_C = C.shape[0]
    N = N_C + 1
    V = np.zeros((N, N, N), dtype=C.dtype)
    p_b, rem = divmod(c - 1, 2)
    p_a = c - p_b - 2
    Np = N + p_b + p_a
    Vp = np.zeros([Np] * 3, dtype=V.dtype)
    for xo in range(c):
        for yo in range(c):
            for zo in range(c):
                Vo_cells = C[xo::c, yo::c, zo::c]
                No = Vo_cells.shape[:3]; No = (No[0] * c, No[1] * c, No[2] * c)
                Vo = Vo_cells.reshape(No[0] / c,
                                      No[1] / c,
                                      No[2] / c, c, c, c) \
                    .transpose(0, 3, 1, 4, 2, 5) \
                    .reshape(No[0], No[1], No[2])
                Vp[xo:xo + (Np - xo) / c * c, yo:yo + (Np - yo) /
                   c * c, zo:zo + (Np - zo) / c * c] += Vo
    V = Vp[p_b:Np - p_a, p_b:Np - p_a, p_b:Np - p_a]
    return V


def to_cell_2d(V, c=2):
    N = V.shape[0]
    p_b, rem = divmod(c - 1, 2)
    p_a = c - p_b - 2
    Np = N + p_b + p_a
    Vp = np.zeros([Np] * 2, dtype=V.dtype)
    Vp[p_b:Np - p_a, p_b:Np - p_a] = V
    N_C = N - 1
    C = np.zeros((N_C**2, c**2), dtype=V.dtype).reshape(N_C, N_C, -1)
    for xo in range(c):
        for yo in range(c):
            Vo = Vp[xo:xo + (Np - xo) / c * c, yo:yo + (Np - yo) / c * c]
            No = Vo.shape
            Vo_cells = Vo.reshape(No[0] / c, c,
                                  No[1] / c, c) \
                .transpose(0, 2, 1, 3) \
                .reshape(No[0] / c, No[1] / c, c**2)
            C[xo::c, yo::c] = Vo_cells
    return C


def from_cell_2d(C, c=2):
    N_C = C.shape[0]
    N = N_C + 1
    V = np.zeros((N, N), dtype=C.dtype)
    p_b, rem = divmod(c - 1, 2)
    p_a = c - p_b - 2
    Np = N + p_b + p_a
    Vp = np.zeros([Np] * 2, dtype=V.dtype)
    for xo in range(c):
        for yo in range(c):
            Vo_cells = C[xo::c, yo::c]
            No = Vo_cells.shape[:2]; No = (No[0] * c, No[1] * c)
            Vo = Vo_cells.reshape(No[0] / c,
                                  No[1] / c, c, c) \
                .transpose(0, 2, 1, 3) \
                .reshape(No[0], No[1])
            Vp[xo:xo + (Np - xo) / c * c, yo:yo + (Np - yo) / c * c] += Vo
    V = Vp[p_b:Np - p_a, p_b:Np - p_a]
    return V


def to_cell(V, c=2):
    assert(c >= 2)
    ndim = V.ndim
    if ndim == 2:
        return to_cell_2d(V, c)
    elif ndim == 3:
        return to_cell_3d(V, c)
    else:
        raise NotImplementedError('Only 2d and 3d cells right now')


def from_cell(C, c=None):
    ndim = C.ndim - 1
    if c is None:
        c = int(C.shape[-1] ** (1 / float(ndim)))
    assert (c**ndim == C.shape[-1])
    if ndim == 2:
        return from_cell_2d(C, c)
    elif ndim == 3:
        return from_cell_3d(C, c)
    else:
        raise NotImplementedError('Only 2d and 3d cells right now')

# def to_cell(V, c=2):
#     assert(c >= 2)
#     N = V.shape[0]
#     p_b, rem = divmod(c-1,2)
#     p_a = c - p_b - 2
#     Np = N+p_b+p_a
#     Vp = n.zeros([Np]*3, dtype=V.dtype)
#     Vp[p_b:Np-p_a, p_b:Np-p_a, p_b:Np-p_a] = V
#     N_C = N - 1
#     C = n.zeros((N_C**3, c**3), dtype=V.dtype).reshape(N_C, N_C, N_C, -1)
#     for xo in range(c):
#         for yo in range(c):
#             for zo in range(c):
#                 Vo = Vp[xo:xo+(Np-xo)/c*c, yo:yo+(Np-yo)/c*c, zo:zo+(Np-zo)/c*c]
#                 No = Vo.shape
#                 Vo_cells = Vo.reshape(No[0]/c, c,
#                                       No[1]/c, c,
#                                       No[2]/c, c, ) \
#                              .transpose(0,2,4,1,3,5) \
#                              .reshape(No[0]/c, No[1]/c, No[2]/c, c**3)
#                 C[xo::c, yo::c, zo::c] = Vo_cells
#     return C

# def from_cell(C, c=2):
#     N_C = C.shape[0]
#     N = N_C + 1
#     V = n.zeros((N,N,N), dtype=C.dtype)
#     p_b, rem = divmod(c-1,2)
#     p_a = c - p_b - 2
#     Np = N+p_b+p_a
#     Vp = n.zeros([Np]*3, dtype=V.dtype)
#     for xo in range(c):
#         for yo in range(c):
#             for zo in range(c):
#                 Vo_cells = C[xo::c, yo::c, zo::c]
#                 No = Vo_cells.shape[:3]; No = (No[0]*c, No[1]*c, No[2]*c)
#                 Vo = Vo_cells.reshape(No[0]/c,
#                                       No[1]/c,
#                                       No[2]/c, c, c, c ) \
#                              .transpose(0,3,1,4,2,5) \
#                              .reshape(No[0], No[1], No[2])
#                 Vp[xo:xo+(Np-xo)/c*c, yo:yo+(Np-yo)/c*c, zo:zo+(Np-zo)/c*c] += Vo
#     V = Vp[p_b:Np-p_a, p_b:Np-p_a, p_b:Np-p_a]
#     return V
