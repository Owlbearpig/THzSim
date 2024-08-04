import numpy as np
from scipy.constants import epsilon_0, physical_constants
from consts import z0


def cond_sheet(freqs=None, d_list=None, n=None, angle=8, pol="s"):
    th_1 = np.pi * angle / 180
    n_in, n_f, n_sub, n_out = n[:, 0], n[:, 1], n[:, 2], n[:, 3]
    d_f, d_sub = d_list[1], d_list[2]
    w = 2 * np.pi * freqs
    th_2 = np.arcsin(n_in * np.sin(th_1) / n_sub)
    s = -1j * (w * 1e12) * (n_f ** 2 - n_sub ** 2) * (d_f * 1e-6) * epsilon_0

    if pol == "p":
        num = ((n_sub + z0 * s * np.cos(th_2)) * np.cos(th_1) - n_in * np.cos(th_2))
        den = ((n_sub + z0 * s * np.cos(th_2)) * np.cos(th_1) + n_in * np.cos(th_2))
    elif pol == "s":
        num = n_in * np.cos(th_1) - n_sub * np.cos(th_2) - z0 * s
        den = n_in * np.cos(th_1) + n_sub * np.cos(th_2) + z0 * s
    else:
        raise Exception("Bad pol")

    return num / den
