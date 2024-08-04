import numpy as np
from numpy import cos
from consts import c_thz


def r_ij_p(n_i, n_j, th_i, th_j):
    num = n_j * cos(th_i) - n_i * cos(th_j)
    den = n_j * cos(th_i) + n_i * cos(th_j)
    return num / den


def r_ij_s(n_i, n_j, th_i, th_j):
    num = n_i * cos(th_i) - n_j * cos(th_j)
    den = n_i * cos(th_i) + n_j * cos(th_j)
    return num / den


def r_1lay(freqs=None, d_list=None, n=None, angle=60, pol="s"):
    th_1 = np.pi * angle / 180
    w = 2 * np.pi * freqs
    n_in, n_f, n_sub, n_out = n[:, 0], n[:, 1], n[:, 2], n[:, 3]
    d_f, d_sub = d_list[1], d_list[2]

    th_f = np.arcsin(n_in * np.sin(th_1) / n_f)
    th_2 = np.arcsin(n_f * np.sin(th_f) / n_sub)

    beta = (n_f * w * d_f / c_thz) * np.cos(th_f)

    phi = np.exp(2 * 1j * beta)

    if pol == "s":
        r_ij = r_ij_s
    elif pol == "p":
        r_ij = r_ij_p
    else:
        raise Exception("Bad pol selection")

    r_1f = r_ij(n_in, n_f, th_1, th_f)
    r_f2 = r_ij(n_f, n_sub, th_f, th_2)

    num = r_1f + r_f2 * phi
    den = 1 + r_1f * r_f2 * phi

    return num / den


