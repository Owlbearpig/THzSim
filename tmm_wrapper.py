from numpy import inf, pi, array, nan_to_num
from tmm import coh_tmm
from consts import c_thz


def tmm_package_wrapper(freqs=None, d_list=None, n=None, geom="r", angle=8, pol="s"):
    # freq should be in THz ("between 0 and 10 THz"), d in um (wl is in um)
    # n[freq_idx, n_idx]
    if d_list[0] != inf:
        d_list = [inf, *d_list]
    if d_list[-1] != inf:
        d_list = [*d_list, inf]

    angle_in = angle * pi / 180

    lambda_vacs = c_thz / freqs
    r_list = []
    for i, lambda_vac in enumerate(lambda_vacs):
        n_list = n[i]
        r_list.append(coh_tmm(pol, n_list, d_list, angle_in, lambda_vac)[geom])
    r_arr = -(geom == "r") * array(r_list)

    r_arr = nan_to_num(r_arr)

    return r_arr
