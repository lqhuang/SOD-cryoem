import numpy as np

class LegendreShiftQuadrature:
    @staticmethod
    def get_degree(N,rad,shiftsigma,shiftextent,usFactor):
        return 1+2*np.round(shiftextent*rad/usFactor)

    @staticmethod
    def get_quad_points(degree,shiftsigma,shiftextent,trunctype):
        assert trunctype in ['none','circ']
        if degree == 0:
            gpoints = []
            gweights = []
        elif degree == 1:
            gpoints = [0]
            gweights = [1]
        else:
            # Assumes a N(0,shiftsigma^2) distribution over shifts
            # truncated to [-shiftextent,shiftextent]
            gpoints, gweights = np.polynomial.legendre.leggauss(degree)
            gpoints *= shiftextent  
            gweights *= shiftextent # since we're not integrating from -1 to 1 anymore
            if np.isfinite(shiftsigma):
                gweights *= np.exp(-gpoints**2/(2*shiftsigma**2))

        K = len(gpoints)**2
        W = np.empty(K, dtype=np.float32)

        i = 0
        pts = np.empty((K,2),dtype=np.float32)
        for sx, wx in zip(gpoints, gweights):
            for sy, wy in zip(gpoints, gweights):
                if trunctype is None or trunctype == 'none' or (trunctype == 'circ' and sx**2 + sy**2 < shiftextent**2):
                    W[i] = wx*wy
                    pts[i,0] = sx
                    pts[i,1] = sy
                    i += 1

        return pts[0:i],W[0:i]/np.sum(W[0:i])

