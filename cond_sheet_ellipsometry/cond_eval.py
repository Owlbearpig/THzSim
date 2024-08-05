import numpy as np
from numpy import cos

from consts import z0


def eval_rho_analytical(r_p, r_s, n=None, angle=8):
    n_in, n_f, n_sub, n_out = n[:, 0], n[:, 1], n[:, 2], n[:, 3]

    rho = r_p / r_s
    th_1 = np.pi * angle / 180
    th_2 = np.arcsin(n_in * np.sin(th_1) / n_sub)

    A = (n_sub * cos(th_1) - n_in * cos(th_2)) / (cos(th_1) * cos(th_2))
    B = (n_sub * cos(th_1) + n_in * cos(th_2)) / (cos(th_1) * cos(th_2))
    C = n_in * cos(th_1) + n_sub * cos(th_2)
    D = n_in * cos(th_1) - n_sub * cos(th_2)

    k = 1 / (2 * (rho + 1) * z0)

    return k * (-(A + C + rho * (B - D)) + np.sqrt(
        (A + C + (B - D) * rho) ** 2 - 4 * (rho + 1) * (A * C - rho * B * D)))
