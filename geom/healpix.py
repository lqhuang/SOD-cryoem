from __future__ import print_function, division

import numpy as np
import healpy as hp


# given nside | number of pixels | resolution (pixel size in degree) | Maximum angular distance (degree) | pixel area (in square degrees)
#      1      |        12        |              58.6323              |              48.1897              |          3437.746771
#      2      |        48        |              29.3162              |              27.5857              |           859.436693
#      4      |       192        |              14.6581              |              14.5722              |           214.859173
#      8      |       768        |              7.3290               |              7.4728               |           53.714793
#     16      |       3072       |              3.6645               |              3.7824               |           13.428698
#     32      |      12288       |              1.8323               |              1.9026               |            3.357175
#     64      |      49152       |              0.9161               |              0.9541               |            0.839294
#     128     |      196608      |              0.4581               |              0.4778               |            0.209823
#     256     |      786432      |              0.2290               |              0.2391               |            0.052456
#     512     |     3145728      |              0.1145               |              0.1196               |            0.013114
#    1024     |     12582912     |              0.0573               |              0.0598               |            0.003278

def gen_EAs_grid(nside=8, psi_step=360, unit='rad'):
    """generate grid of Euler angles"""
    npix = hp.nside2npix(nside)
    resol = np.rad2deg(hp.nside2resol(nside))
    theta, phi = hp.pix2ang(nside, range(npix))
    psi = np.arange(0, 360, psi_step)
    if unit == 'deg':
        theta, phi = np.rad2deg(theta), np.rad2deg(phi)
    elif unit == 'rad':
        psi = np.deg2rad(psi)
    else:
        raise NotImplementedError('unsupport unit of angle')
    # sequence of indexing is equal to 'xy'
    grid_theta, grid_psi = np.meshgrid(theta, psi)
    grid_phi, _ = np.meshgrid(phi, psi)
    EAs_tuple = (grid_phi.flatten(), grid_theta.flatten(), grid_psi.flatten())
    EAs_grid = np.vstack(EAs_tuple).T
    print('number of points on shpere: {0}. \n'
          'resolution: {1:.2f} degree, step of inplane-rotation: {2} degree\n'
          'total grid size: {3}'.format(npix, resol, psi_step, npix * 360 / psi_step))
    return EAs_grid