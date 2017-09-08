from __future__ import print_function, division

import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))

import numpy as np
from scipy import interpolate
from scipy.ndimage import interpolation as spinterp
from scipy.stats import threshold

import geometry
import density


def cart2pol(*coords):
    """Convert cartesian coordinates to polar coordinates.
    rho, theta = cart2pol(x, y)"""
    if len(coords) == 1:
        cart = coords[0]
        assert cart.shape[1] == 2
        rho = np.sqrt(np.sum(cart ** 2, 1))
        theta = np.arctan2(cart[:, 1], cart[:, 0])
        return np.vstack((rho, theta)).T
    elif len(coords) == 2:
        x, y = coords
        assert x.shape == y.shape
        rho = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return rho, theta
    else:
        raise ValueError('inappropriate arguments')


def pol2cart(*coords):
    """Convert polar coordinates to cartesian coordinates.
    x, y = pol2cart(rho, theta)"""
    if len(coords) == 1:
        pol = coords[0]
        assert pol.shape[1] == 2
        x = pol[:, 0] * np.cos(pol[:, 1])
        y = pol[:, 0] * np.sin(pol[:, 1])
        return np.vstack((x, y)).T
    elif len(coords) == 2:
        rho, theta = coords
        assert rho.shape == theta.shape
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        return x, y
    else:
        raise ValueError('inappropriate arguments')


# Image center:
# The center of rotation of a 2D image of dimensions xdim x ydim is defined by 
# ((int)xdim/2, (int)(ydim/2)) (with the first pixel in the upper left being (0,0).
# Note that for both xdim=ydim=65 and for xdim=ydim=64, the center will be at (32,32).
# This is the same convention as used in SPIDER and XMIPP. Origin offsets reported 
# for individual images translate the image to its center and are to be applied 
# BEFORE rotations.
def imgpolarcoord(img, rad=1.0):
    """
    Convert a given image from cartesian coordinates to polar coordinates.
    """
    row, col = img.shape
    cx = int(col/2)
    cy = int(row/2)
    radius = int(min([row-cy, col-cx, cx, cy]) * rad)
    angle = 360.0
    # Interpolation: Nearest
    pcimg = np.zeros((int(radius), int(angle)))
    radius_range = np.arange(0, radius, 1)
    angle_range = np.arange(0, 2*np.pi, 2*np.pi/angle)
    i = 0
    for r in radius_range:
        j = 0
        for a in angle_range:
            pcimg[i, j] = img[int(cy+round(r*np.sin(a))), int(cx+round(r*np.cos(a)))]
            j = j + 1
        i = i + 1
    return pcimg


def imgpolarcoord3(img, rad=1.0):
    """
    converts a given image from cartesian coordinates to polar coordinates.
    """
    row, col = img.shape
    cx = int(col/2)
    cy = int(row/2)
    radius = float(min([row-cy, col-cx, cx, cy])) * rad
    angle = 360.0
    # Interpolation: Linear
    rho_range = np.arange(0, radius, 1)
    theta_range = np.arange(0, 2*np.pi, 2*np.pi/angle)
    theta_grid, rho_grid = np.meshgrid(theta_range, rho_range)
    new_x_grid, new_y_grid = pol2cart(rho_grid, theta_grid)

    pcimg = spinterp.map_coordinates(img, (new_x_grid + radius, new_y_grid + radius))
    return pcimg


def get_corr_img(img, rad=1.0, pcimg_interpolation='nearest'):
    """
    get a angular correlation image
    """
    if 'nearest' in pcimg_interpolation.lower():
        pcimg = imgpolarcoord(img, rad=rad)
    elif 'linear' in pcimg_interpolation.lower():
        pcimg = imgpolarcoord3(img, rad=rad)

    pcimg_fourier = np.fft.fftshift(np.fft.fft(pcimg, axis=1))
    corr_img = np.fft.ifft(np.fft.ifftshift(pcimg_fourier*np.conjugate(pcimg_fourier)), axis=1)
    return np.require(corr_img.real, dtype=density.real_t)


def get_corr_imgs(imgs, rad=1.0, pcimg_interpolation='nearest'):
    num_imgs = imgs.shape[0]
    N = imgs.shape[1]
    assert N == imgs.shape[2]
    corr_imgs = np.zeros((num_imgs, int(N/2.0), 360), dtype=density.real_t)
    for i, img in enumerate(imgs):
        corr_imgs[i, :, :] = get_corr_img(img, rad=rad, pcimg_interpolation=pcimg_interpolation)

    return corr_imgs


def gencoords_outside(N, d, rad=None, truncmask=False, trunctype='circ'):
    """ generate coordinates of all points in an NxN..xN grid with d dimensions 
    coords in each dimension are [-N/2, N/2) 
    N should be even"""
    if not truncmask:
        _, truncc, _ = gencoords_outside(N, d, rad, True)
        return truncc
    
    c = geometry.gencoords_base(N, d)

    if rad is not None:
        if trunctype == 'circ':
            r2 = np.sum(c**2, axis=1)
            trunkmask = r2 > (rad*N/2.0)**2
        elif trunctype == 'square':
            r = np.max(np.abs(c), axis=1)
            trunkmask = r > (rad*N/2.0)
            
        truncc = c[trunkmask, :]
    else:
        trunkmask = np.ones((c.shape[0],), dtype=np.bool8)
        truncc = c
 
    return c, truncc, trunkmask


def calc_angular_correlation(trunc_slices, N, rad, interpolation='nearest',
                             sort_theta=True, clip=True, outside=False,):
    """compute angular correlation for input array
    outside: True or False (default: False)
        calculate angular correlation in radius or outside of radius
    sort_theta: True or False (default: True)
        sort theta when slicing the same rho in trunc array
    """
    # 1. get a input (single: N_T or multi: N_R x N_T) with normal sequence.
    # 2. sort truncation array by rho value of polar coordinates
    # 3. apply angular correlation function to sorted slice for both real part and imaginary part
    # 4. deal with outlier beyond 3 sigma (no enough points to do sampling via fft)
    #    (oversampling is unavailable, hence dropout points beyond 3 sigma)
    # 5. return angluar correlation slice with normal sequence.

    # 1.
    iscomplex = np.iscomplexobj(trunc_slices)
    if outside:
        trunc_xy = gencoords_outside(N, 2, rad)
    else:
        trunc_xy = geometry.gencoords(N, 2, rad)
    if trunc_slices.ndim < 2:
        assert trunc_xy.shape[0] == trunc_slices.shape[0], "wrong length of trunc slice or wrong radius"
    else:
        assert trunc_xy.shape[0] == trunc_slices.shape[1], "wrong length of trunc slice or wrong radius"

    # 2.
    pol_trunc_xy = cart2pol(trunc_xy)
    if sort_theta:
        # lexsort; first, sort rho; second, sort theta
        sorted_idx = np.lexsort((pol_trunc_xy[:, 1], pol_trunc_xy[:, 0]))
    else:
        sorted_idx = np.argsort(pol_trunc_xy[:, 0])
    axis = trunc_slices.ndim - 1
    sorted_rho = np.take(pol_trunc_xy[:, 0], sorted_idx)
    sorted_slice = np.take(trunc_slices, sorted_idx, axis=axis)

    # 3.
    if 'none' in interpolation:
        pass
    elif 'nearest' in interpolation:
        sorted_rho = np.round(sorted_rho)
    elif 'linear' in interpolation:
        raise NotImplementedError()
    else:
        raise ValueError('unsupported method for interpolation')

    _, unique_idx, unique_counts = np.unique(sorted_rho, return_index=True, return_counts=True)
    indices = [slice(None)] * trunc_slices.ndim
    angular_correlation = np.zeros_like(trunc_slices, dtype=trunc_slices.dtype)
    for i, count in enumerate(unique_counts):
        indices[axis] = slice(unique_idx[i], unique_idx[i] + count)
        if count < 4:
            angular_correlation[indices]  = np.copy(sorted_slice[indices])
        else:
            # use view (slicing) or copy (fancy indexing, np.take(), np.put())?
            same_rho = np.copy(sorted_slice[indices])
            fpcimg_real = density.real_to_fspace(same_rho.real, axes=(axis,))  # polar image in fourier sapce
            angular_correlation[indices].real = density.fspace_to_real(
                fpcimg_real * fpcimg_real.conjugate(), axes=(axis,)).real
            if iscomplex:  # FIXME: stupid way. optimize this
                fpcimg_fourier = density.real_to_fspace(same_rho.imag, axes=(axis,))  # polar image in fourier sapce
                angular_correlation[indices].imag = density.fspace_to_real(
                    fpcimg_fourier * fpcimg_fourier.conjugate(), axes=(axis,)).real

    # 4.
    if clip:
        factor = 3
        for i, count in enumerate(unique_counts):
            if count <= 10:
                pass
            else:
                indices[axis] = slice(unique_idx[i], unique_idx[i] + count)
                mean = angular_correlation[indices].mean(axis)
                std = angular_correlation[indices].std(axis)
                vmin = mean - std * factor
                vmax = mean + std * factor
                # angular_correlation[indices] = np.clip(angular_correlation[indices].T, vmin, vmax).T  # set outlier to nearby boundary
                angular_correlation[indices] = threshold(angular_correlation[indices].T, vmin, vmax, 0).T  # set outlier to 0

    # 5.
    corr_trunc_slices = np.take(angular_correlation, sorted_idx.argsort(), axis=axis)
    return corr_trunc_slices


def calc_full_ac(image, rad, outside=True, **ac_kwargs):
    cython_build_dirs=os.path.expanduser('~/.pyxbld/angular_correlation')
    import pyximport; pyximport.install(build_dir=cython_build_dirs, setup_args={"include_dirs": np.get_include()}, reload_support=True)
    import sincint

    assert image.ndim == 2, "wrong dimension"
    assert image.shape[0] == image.shape[1]

    N = image.shape[0]
    FtoT = sincint.genfulltotrunc(N, rad)
    TtoF = FtoT.T
    trunc = FtoT.dot(image.flatten())
    corr_trunc = calc_angular_correlation(trunc, N, rad, **ac_kwargs)
    full_angular_correlation = TtoF.dot(corr_trunc)
    
    if outside:
        _, _, outside_mask = gencoords_outside(N, 2, rad, True)
        corr_trunc_outside = calc_angular_correlation(image[outside_mask.reshape(N, N)].flatten(),
                                                    N, rad, outside=True, **ac_kwargs)
        full_angular_correlation[outside_mask] = corr_trunc_outside

    return full_angular_correlation.reshape(N, N)


if __name__ == '__main__':
    from cryoio import mrc
    from matplotlib import pyplot as plt
    map_file = '../particle/1AON.mrc'
    model = mrc.readMRC(map_file)
    proj = np.sum(model, axis=2)
    c2_img_nearest = get_corr_img(proj, pcimg_interpolation='nearest')
    c2_img_linear = get_corr_img(proj, pcimg_interpolation='linear')
    plt.figure(1)
    plt.imshow(proj)
    plt.figure(2)
    plt.imshow(c2_img_linear)
    plt.show()
