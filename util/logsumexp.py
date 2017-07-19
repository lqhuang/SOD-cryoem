import numpy as np

def logsumexp(a, axis=None, b=None, dtype=None):
    a = np.asarray(a)
    if (axis is None and a.size == 1) or (axis is not None and a.shape[axis] == 1):
        if b is not None:
            return a + np.log(b)
        else:
            return a
    if axis is None:
        a = a.ravel()
    else:
        a = np.rollaxis(a, axis)
    a_max = a.max(axis=0)
    if b is not None:
        b = np.asarray(b)
        if axis is None:
            b = b.ravel()
        else:
            b = np.rollaxis(b, axis)
        out = np.log(np.sum(b * np.exp(a - a_max), axis=0, dtype=dtype))
    else:
        out = np.log(np.sum(np.exp(a - a_max), axis=0, dtype=dtype))
    out += a_max
    return out
