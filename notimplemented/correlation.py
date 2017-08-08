from __future__ import print_function, division

import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))

import numpy as np
from scipy import interpolate

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
        x = pol[:, 1] * np.cos(pol[:, 0])
        y = pol[:, 1] * np.sin(pol[:, 0])
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
    # Interpolation: Linear (undone)
    x_range = np.arange(-radius+1, radius, 1)
    y_range = np.arange(-radius+1, radius, 1)
    z = img[1:int(cy+radius),1:int(cx+radius)]
    # f = interpolate.interp2d(x_range, y_range, img[1:int(cy+radius),1:int(cx+radius)], kind='linear')
    # f = interpolate.RectBivariateSpline(x_range, y_range, img[1:int(cy+radius),1:int(cx+radius)])
    f = interpolate.interp2d(x_range, y_range, z, kind='linear')

    rho_range = np.arange(0, radius, 1)
    theta_range = np.arange(0, 2*np.pi, 2*np.pi/angle)
    rho_grid, theta_grid = np.meshgrid(rho_range, theta_range)
    new_x_grid, new_y_grid = pol2cart(rho_grid, theta_grid)

    plt.figure(3)
    plt.imshow(new_x_grid)
    plt.figure(4)
    plt.imshow(new_y_grid)
    plt.show()

    # pcimg = f(new_x_grid.ravel(), new_y_grid.ravel())
    # pcimg = pcimg.reshape((int(radius+1),-1))
    pcimg = np.zeros((int(radius), int(angle)))
    print(pcimg.shape)
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


def calc_angular_correlation(trunc_slices, N, rad, interpolation='nearest', sort_theta=False):
    """compute angular correlation for input array"""
    # 1. get a input (single: N_T or multi: N_R x N_T) with normal sequence.
    # 2. sort truncation array by rho value of polar coordinates
    # 3. apply angular correlation function to sorted slice for both real part and imaginary part
    # 4. return angluar correlation slice with normal sequence (need to do this?).

    # 1.
    iscomplex = np.iscomplexobj(trunc_slices)
    _, trunc_xy, _ = geometry.gencoords(N, 2, rad, True)
    if trunc_slices.ndim < 2:
        assert trunc_xy.shape[0] == trunc_slices.shape[0]
    else:
        assert trunc_xy.shape[0] == trunc_slices.shape[1]

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
        if count < 2:
            angular_correlation[indices]  = np.copy(sorted_slice[indices])
        else:
            # use view (slicing) or copy (fancy indexing, np.take(), np.put())?
            same_rho = np.copy(sorted_slice[indices])
            fpcimg_real = density.real_to_fspace(same_rho.real, axes=(axis,))  # polar image in fourier sapce
            angular_correlation[indices].real = density.fspace_to_real(
                fpcimg_real * fpcimg_real.conjugate(), axes=(axis,))
            if iscomplex:  # FIXME: stupid way. optimize this
                fpcimg_fourier = density.real_to_fspace(same_rho.imag, axes=(axis,))  # polar image in fourier sapce
                angular_correlation[indices].imag = density.fspace_to_real(
                    fpcimg_fourier * fpcimg_fourier.conjugate(), axes=(axis,))

    # 4. 
    corr_trunc_slices = np.take(angular_correlation, sorted_idx.argsort(), axis=axis)
    return corr_trunc_slices


if __name__ == '__main__':
    from cryoio import mrc
    from matplotlib import pyplot as plt
    map_file = '../particle/1AON.mrc'
    model = mrc.readMRC(map_file)
    proj = np.sum(model, axis=2)
    c2_img = get_corr_img(proj, pcimg_interpolation='linear')
    plt.figure(1)
    plt.imshow(proj)
    # plt.figure(2)
    # plt.imshow(c2_img)
    plt.show()
