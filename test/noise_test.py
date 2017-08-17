from __future__ import print_function, division

import time, os, sys
sys.path.append(os.path.dirname(sys.path[0]))

from datetime import datetime

import numpy as np
from scipy.stats import chisquare
from scipy.stats import chi2
from matplotlib import pyplot as plt 

import cryoops
import density
import geometry

from notimplemented import correlation


def chisquare_pdf(x, df=2):
    p = chi2.pdf(x, df)
    return p


def complex_gaussian(x, mu=0, sigma=1):
    p = 1 / (2 * np.pi * sigma**2) * np.exp(-(x - mu)**2 / (2 * sigma**2))
    return p


def std_gaussian(x, mu=0, sigma=1):
    p = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2))
    return p


def compute_statistics(x, pdf_func):
    if pdf_func is std_gaussian:
        mu = x.mean()
        sigma = x.std()
        return mu, sigma
    elif pdf_func is chisquare_pdf:
        df = 2
        return (df, )
    else:
        raise ValueError('wrong pdf function')

def get_stat_str(pdf_func, *args):
    if pdf_func is std_gaussian:
        return 'mu:{:.3f}, sigma:{:.3f}'.format(*args)
    elif pdf_func is chisquare_pdf:
        return 'df:{}'.format(*args)
    else:
        raise ValueError('wrong pdf function')


def plot_noise_histogram(real_image, fourier_image, rmask=None, fmask=None, plot_unmask=True):
    """
    rmask: mask for image of real space
    fmask: mask for image of fourier space
    """
    if rmask is None:
        rmask = np.ones_like(real_image, dtype=np.bool).flatten()
    else:
        rmask = rmask.flatten()
    if fmask is None:
        fmask = np.ones_like(fourier_image, dtype=np.bool).flatten()
    else:
        fmask = fmask.flatten()

    def subplot(axs, arr, mask, title, pdf_func=std_gaussian):
        im = axs[0].imshow(arr * mask.reshape(arr.shape))
        fig.colorbar(im, ax=axs[0])
        axs[0].set_title('masked ' + str(title))
        _, bins, _ = axs[1].hist(arr.flatten()[mask], bins=100, normed=True)
        stat_vars = compute_statistics(arr.flatten()[mask], pdf_func)
        stat = get_stat_str(pdf_func, *stat_vars)
        axs[1].plot(bins, pdf_func(bins, *stat_vars))
        axs[1].set_title(stat)
        axs[1].set_xlabel('value of pixel')
        
        if np.all(mask) != True and plot_unmask:
            unmasked_im = axs[2].imshow(arr * ~mask.reshape(arr.shape))
            fig.colorbar(unmasked_im, ax=axs[2])
            axs[2].set_title('unmasked ' + str(title))
            _, um_bins, _ = axs[3].hist(arr.flatten()[~mask], bins=100, normed=True)
            um_stat_vars = compute_statistics(arr.flatten()[~mask], pdf_func)
            um_stat = get_stat_str(pdf_func, *um_stat_vars)
            axs[3].plot(um_bins, pdf_func(um_bins, *um_stat_vars))
            axs[3].set_title(um_stat)
            axs[3].set_xlabel('value of pixel')
    
    if plot_unmask:
        fig, ax = plt.subplots(4, 4, figsize=(12.8, 8))
    else:
        fig, ax = plt.subplots(2, 4, figsize=(12.8, 4.8))
    subplot(ax[:, 0], real_image, rmask, 'real space')
    subplot(ax[:, 1], np.abs(fourier_image), fmask, 'modulus', pdf_func=chisquare_pdf)
    subplot(ax[:, 2], fourier_image.real, fmask, 'real part')
    subplot(ax[:, 3], fourier_image.imag, fmask, 'imaginary part')
    fig.tight_layout()
    plt.show()


def plot_stack_noise(real, fourier):
    def subplot(ax, arr, title, pdf_func=std_gaussian):
        _, bins, _ = ax.hist(arr, bins=100, normed=True)
        stat_vars = compute_statistics(arr.flatten(), pdf_func)
        stat = get_stat_str(pdf_func, *stat_vars)
        ax.plot(bins, std_gaussian(bins, *stat_vars))
        ax.set_xlabel('value of pixel')
        ax.set_title(str(title)+'\n'+stat)

    fig, ax = plt.subplots(1, 4, figsize=(12.8, 4))
    subplot(ax[0], real, 'real space')
    subplot(ax[1], np.abs(fourier), 'modulus', pdf_func=chisquare_pdf)
    subplot(ax[2], fourier.real, 'real part')
    subplot(ax[3], fourier.imag, 'imaginary part')
    fig.tight_layout()
    plt.show()


def no_correlation(num_images=1000, N=128, rad=0.8):
    center = int(N/2)
    x_shift, y_shift = np.random.randint(-center, center, size=2)

    noise_stack = np.require(np.random.randn(num_images, N, N), dtype=density.real_t)
    real_image = noise_stack[np.random.randint(num_images)]
    fourier_noise_stack = density.real_to_fspace(noise_stack, axes=(1, 2))
    fourier_noise = density.real_to_fspace(real_image)

    noise_zoom = noise_stack[:, x_shift, y_shift]
    fourier_noise_zoom = fourier_noise_stack[:, x_shift, y_shift]

    _, _, mask = geometry.gencoords(N, 2, rad, True)
    plot_noise_histogram(real_image, fourier_noise, rmask=mask, fmask=mask)
    plot_stack_noise(noise_zoom, fourier_noise_zoom)


def correlation_noise(num_images=1000, N=128, rad=0.6, stack_noise=False):
    noise_stack = np.require(np.random.randn(num_images, N, N), dtype=density.real_t)
    real_image = noise_stack[np.random.randint(num_images)]
    corr_real_image = correlation.calc_full_ac(real_image, rad=rad)
    fourier_noise = density.real_to_fspace(real_image)
    fourier_corr_image = correlation.calc_full_ac(fourier_noise, rad=rad)

    _, _, mask = geometry.gencoords(N, 2, rad, True)
    plot_noise_histogram(corr_real_image, fourier_corr_image, mask, mask)

    if stack_noise:
        center = int(N/2)
        x_shift, y_shift = np.random.randint(-center, center, size=2)

        fourier_noise_stack = density.real_to_fspace(noise_stack, axes=(1, 2))
        corr_noise_stack = np.zeros_like(noise_stack, dtype=density.real_t)
        fourier_corr_noise_stack = np.zeros_like(fourier_noise_stack, dtype=density.complex_t)

        for i in range(num_images):
            corr_noise_stack[i] = correlation.calc_full_ac(noise_stack[i], rad=rad)
            fourier_corr_noise_stack[i] = correlation.calc_full_ac(fourier_noise_stack[i], rad=rad)

        noise_zoom = noise_stack[:, x_shift, y_shift]
        fourier_noise_zoom = fourier_noise_stack[:, x_shift, y_shift]
        plot_stack_noise(noise_zoom, fourier_noise_zoom)


if __name__ == '__main__':
    no_correlation(N=256)
    correlation_noise(N=256, rad=0.6)
