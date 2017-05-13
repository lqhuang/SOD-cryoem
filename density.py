import numpy as np
from geom.geom import gencoords


def generate_phantom_density(N, window, sigma, num_blobs, seed=None):
    if seed is not None:
        np.random.seed(seed)
    M = np.zeros((N, N, N), dtype=np.float32)

    coords = gencoords(N, 3).reshape((N**3, 3))
    inside_window = np.sum(coords**2, axis=1).reshape((N, N, N)) < window**2

    curr_c = np.array([0.0, 0.0, 0.0])
    curr_n = 0
    while curr_n < num_blobs:
        csigma = sigma * np.exp(0.25 * np.random.randn())
        radM = np.sum((coords - curr_c.reshape((1, 3))) ** 2, axis=1).reshape((N, N, N))
        inside = np.logical_and(radM < (3 * csigma)**2, inside_window)
#        M[inside] = 1
        M[inside] += np.exp(-0.5 * (radM[inside] / csigma**2))
        curr_n += 1

        curr_dir = np.random.randn(3)
        curr_dir /= np.sum(curr_dir**2)
        curr_c += 2.0 * csigma * curr_dir
        curr_w = np.sqrt(np.sum(curr_c**2))
        while curr_w > window:
            curr_n_dir = curr_c / curr_w
            curr_r_dir = (2 * np.dot(curr_dir, curr_n_dir)) * \
                curr_n_dir - curr_dir
            curr_c = curr_n_dir + (curr_w - window) * curr_r_dir
            curr_w = np.sqrt(np.sum(curr_c**2))

    return M
