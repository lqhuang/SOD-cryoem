import numpy as np
from . import compute_max_angle

def gensk97(N):
    # See http://dx.doi.org/10.1016/j.jsb.2006.06.002 and references therein
    N = int(N)
    h = -1.0 + (2.0/(N-1))*np.arange(0,N)
    theta = np.arccos(h)
    phi_base = np.zeros_like(theta)
    phi_base[1:(N-1)] = ((3.6/np.sqrt(N))/np.sqrt(1 - h[1:(N-1)]**2))
    phi = np.cumsum(phi_base)
    phi[0] = 0
    phi[N-1] = 0

    stheta = np.sin(theta)
    dirs = np.vstack([np.cos(phi)*stheta, np.sin(phi)*stheta, np.cos(theta)]).T

    return dirs

class SK97Quadrature:
    @staticmethod
    def compute_degree(N,rad,usFactor):
        ang = compute_max_angle(N,rad,usFactor)
        return SK97Quadrature.get_degree(ang)

    @staticmethod
    def get_degree(maxAngle):
        degree = np.ceil((3.6/maxAngle)**2)
        cmaxAng = 3.6/np.sqrt(degree)

        return degree, cmaxAng

    @staticmethod
    def get_quad_points(degree,sym = None):
        verts = gensk97(degree)
        p = np.array(verts)
        w = (4.0*np.pi/len(verts)) * np.ones(len(verts))

        if sym is None:
            return p, w
        else:
            validPs = sym.in_asymunit(p)
            return p[validPs], w[validPs] * (np.sum(w) / np.sum(w[validPs]))

