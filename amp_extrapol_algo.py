from signal_sim import *
from numpy import pi
from functions import unwrap, do_fft, do_ifft, polyfit


def algo(ref_fd_, sam_fd_):
    f_axis = ref_fd_[:, 0].real
    w = 2 * pi * f_axis

    phi_ref = unwrap(ref_fd_)
    phi_sam = unwrap(sam_fd_)

    phi = phi_sam[:, 1] - phi_ref[:, 1]

    n_ = 1 + c_THz * phi / (w * d0)

    n_ = np.nan_to_num(n_, nan=n_[10])

    N = 4*n_/(n_+1)**2
    a = np.abs(sam_fd_[:, 1]/ref_fd_[:, 1])
    loga = np.log(a)

    f_min_idx, f_max_idx = np.argmin(np.abs(f_axis - 0.1)), np.argmin(np.abs(f_axis - 0.2))

    ax0_cri.plot(f_axis, n_, label="n extracted")

    x = np.arange(len(n_[f_min_idx:f_max_idx]))
    res = polyfit(x, n_[f_min_idx:f_max_idx])
    n0 = res["polynomial"][1]
    a0_theo = np.log(4 * n0 / (1 + n0) ** 2)

    res = polyfit(x, loga[f_min_idx:f_max_idx])
    a0_exp = res["polynomial"][1]

    print(res["polynomial"])
    axa.plot(f_axis, loga, label="log(A)")

    k_raw = np.log(N / a) * c_THz / (w * d0)

    ax1_cri.plot(f_axis, k_raw, label="k extracted")

    a = a - a0_exp + a0_theo

    k_ = np.log(N / a) * c_THz / (w * d0)

    ax1_cri.plot(f_axis, k_raw, label="k corrected")

    return n_, k_


def main():
    ref_td, ref_fd = ref_signal()

    t_axis = ref_td[:, 0].real
    f_axis = ref_fd[:, 0].real

    t = t_sample_sim(f_axis, en_plot=True)
    phi_shift = np.exp(-1j * d0 * 2 * pi * f_axis / c_THz)

    sam_fd = array([f_axis, t * ref_fd[:, 1] * phi_shift]).T

    sam_td = do_ifft(sam_fd, t_=t_axis)

    ax0_sig.plot(t_axis, sam_td[:, 1], label="Sample")
    ax2_sig.plot(f_axis, to_db(sam_fd[:, 1]), label="Sample")

    phi_s = unwrap(sam_fd, one_d=True)
    phi_r = unwrap(ref_fd, one_d=True)

    ax1_sig.plot(f_axis, phi_r, label="Reference")
    ax1_sig.plot(f_axis, phi_s, label="Sample")

    n, k = algo(ref_fd, sam_fd)






if __name__ == '__main__':
    main()
    ax0_cri.legend()
    ax1_cri.legend()
    ax2_cri.legend()
    ax0_sig.legend()
    axa.legend()
    ax1_sig.legend(loc='upper right')
    ax2_sig.legend(loc='upper right')

    plt.show()
