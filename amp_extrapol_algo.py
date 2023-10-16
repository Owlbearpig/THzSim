from signal_sim import *
from numpy import pi
from functions import unwrap, do_fft, do_ifft


def real_ri(ref_fd_, sam_fd_):
    f = ref_fd_[:, 0].real
    w = 2 * pi * f

    phi_ref = unwrap(ref_fd_)
    phi_sam = unwrap(sam_fd_)

    phi = phi_sam[:, 1] - phi_ref[:, 1]

    n_ = 1 + c_THz * phi / (w * d0)

    return n_


def main():
    ref_td, ref_fd = ref_signal()

    t_axis = ref_td[:, 0].real
    f_axis = ref_fd[:, 0].real

    t = t_sample_sim(f_axis, en_plot=True)
    phi_shift = np.exp(-1j*d0*2*pi*f_axis/c_THz)
    phi_shift *= 1 # np.exp(-1j*2*pi*f_axis*pulse_shift)

    sam_fd = array([f_axis, t * ref_fd[:, 1] * phi_shift]).T

    sam_td = do_ifft(sam_fd, t_=t_axis)

    ax0_sig.plot(t_axis, sam_td[:, 1], label="Sample")
    ax2_sig.plot(f_axis, to_db(sam_fd[:, 1]), label="Sample")


if __name__ == '__main__':
    main()
    ax0_cri.legend()
    ax1_cri.legend()
    ax2_cri.legend()
    ax0_sig.legend()
    ax1_sig.legend(loc='upper right')
    ax2_sig.legend(loc='upper right')

    plt.show()
