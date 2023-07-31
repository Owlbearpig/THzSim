import numpy as np
from numpy import exp, array
import matplotlib.pyplot as plt
from functions import do_fft, do_ifft, unwrap
from scipy.constants import c as c0


def fresnel_(f, en_plot=True):
    cnt = len(f)

    n0 = 1 + 1j*0

    n1_r = np.linspace(1.4, 1.45, cnt)
    n1_i = np.linspace(0.00, 0.05, cnt)

    width = 0.005
    peak = (1/np.pi)*(width/((f-1)**2 + width**2))
    n1_i = n1_i + peak * 0.1

    n1 = n1_r + 1j*n1_i

    r_ = (n0 - n1) / (n0 + n1)
    t_ = 2*n0 / (n0 + n1)

    alpha = (1/100)*4*np.pi*n1.imag*(f*1e12)/c0

    if en_plot:
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
        ax0.plot(f, n1.real, label="Real part of n1")
        ax1.plot(f, n1.imag, label="Imaginary part of n1")
        ax2.plot(f, alpha, label="Absorption coefficient (1/cm)")
        ax0.legend()
        ax1.legend()
        ax2.legend()

        ax0.set_ylabel("Real part")
        ax2.set_xlabel("Frequency (THz)")
        ax1.set_ylabel("Imaginary part")
        ax2.set_ylabel("Absorption coefficient (1/cm)")

    return t_


def shift_signal(y_td, shift):
    # shift in ps (unit of t)
    t_, y = y_td[:, 0].real, y_td[:, 1].real
    dt = np.mean(np.diff(t_))
    shift = int(shift / dt)
    y = np.roll(y, shift)

    return array([t_, y]).T


def correlation(y0_td, y1_td):
    t = y0_td[:, 0].real
    scale = 1/len(t)
    dt_ = np.mean(np.diff(t))

    tau = np.arange(-5, 5, dt_)

    corr_ = np.zeros_like(tau)
    for t_idx in range(-int(5/dt_), int(5//dt_)):
        corr_[t_idx] = scale*np.sum(y0_td[:, 1].real * np.roll(y1_td[:, 1].real, t_idx))

    corr_ = np.abs(corr_)
    min_idx = np.argmin(corr_)
    print(tau[min_idx])

    return array([tau, corr_]).T


def main():
    dt = 0.01
    t = np.arange(-100, 100+dt, dt)
    sigma = 0.8
    y = t*exp(-(t / sigma) ** 2)
    y = y + np.random.random((len(t))) * 1E-10

    ref_td = array([t, y]).T
    ref_fd = do_fft(ref_td)
    f = ref_fd[:, 0].real
    r = fresnel_(f, en_plot=True)

    sam_fd = array([f, r * ref_fd[:, 1]]).T

    sam_td = do_ifft(sam_fd, t_=t)
    sam_shift = 0.100
    sam_td_shifted = shift_signal(sam_td, shift=sam_shift)

    sam_fd_shifted = do_fft(sam_td_shifted)

    phi_ref = unwrap(ref_fd)
    phi_sam = unwrap(sam_fd)
    phi_sam_shifted = unwrap(sam_fd_shifted)

    corr = correlation(ref_td, sam_td_shifted)

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
    ax0.plot(corr[:, 0], corr[:, 1])

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1)

    ax0.plot(ref_td[:, 0], ref_td[:, 1], label="Reference")
    ax0.plot(sam_td[:, 0], sam_td[:, 1], label="Sample dt=0 ps")
    ax0.plot(sam_td_shifted[:, 0], sam_td_shifted[:, 1], label=f"Sample dt={sam_shift} ps")
    ax0.set_xlabel("Time (ps)")
    ax0.set_ylabel("Amplitude (arb. u.)")

    ax1.plot(phi_ref[:, 0], phi_ref[:, 1], label="Reference")
    ax1.plot(phi_sam[:, 0], phi_sam[:, 1], label="Sample dt=0")
    ax1.plot(phi_sam_shifted[:, 0], phi_sam_shifted[:, 1], label=f"Sample dt={sam_shift} ps")
    ax1.set_ylabel("Phase (rad)")

    ax2.plot(ref_fd[:, 0], 20*np.log10(np.abs(ref_fd[:, 1])), label="Reference")
    ax2.plot(sam_fd[:, 0], 20 * np.log10(np.abs(sam_fd[:, 1])), label="Sample dt=0")
    ax2.plot(sam_fd_shifted[:, 0], 20 * np.log10(np.abs(sam_fd_shifted[:, 1])), label=f"Sample dt={sam_shift} ps")
    ax2.set_xlabel("Frequency (THz)")
    ax2.set_ylabel("Amplitude (dB)")

    ax0.legend()
    ax1.legend()
    ax2.legend()



if __name__ == '__main__':
    main()

    plt.show()
