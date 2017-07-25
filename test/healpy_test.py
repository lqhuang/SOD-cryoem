from __future__ import print_function, division

import numpy as np
import healpy as hp
from matplotlib import pyplot as plt

import geometry

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

def calculate_nside_resolution():
    NSIDE = [2**i for i in range(11)]
    print('given nside | number of pixels | resolution (pixel size in degree) | Maximum angular distance (degree) | pixel area (in square degrees)')
    for nside in NSIDE:
        npix = hp.nside2npix(nside)
        resol = np.rad2deg(hp.nside2resol(nside))
        maxrad = np.rad2deg(hp.max_pixrad(nside))
        pixarea = hp.nside2pixarea(nside, degrees=True)
        print('{0:^11} | {1:^16} | {2:^33.4f} | {3:^33.4f} | {4:^30.6f}'.format(nside, npix, resol, maxrad, pixarea))


if __name__ == '__main__':

    calculate_nside_resolution()

    # generate random distribution of Euler angles
    v = np.random.randn(100,3)
    v = v  / np.linalg.norm(v, axis=1).repeat(3).reshape(-1,3)
    EA = geometry.genEA(v)
    phi = EA[:, 0]
    # phi += 2 * np.pi
    theta = EA[:, 1]
    # visulization
    hp.mollview()
    hp.visufunc.projscatter(theta, phi, 'r.')
    hp.graticule()
    plt.show()
