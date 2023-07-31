import numpy as np
from numpy import exp, array
import matplotlib.pyplot as plt
from functions import do_fft
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

    return r_


def main():
    t = np.arange(-100, 100, 0.1)
    sigma = 0.8
    y = t*exp(-(t / sigma) ** 2)
    y = y + np.random.random((len(t))) * 1E-10

    ref_td = array([t, y]).T
    ref_fd = do_fft(ref_td)
    f = ref_fd[:, 0].real
    r = fresnel_(f, en_plot=True)

    sam_fd = array([f, r * ref_fd[:, 1]]).T

    fig, (ax0, ax1) = plt.subplots(2, 1)

    ax0.plot(ref_td[:, 0], ref_td[:, 1], label="Reference")
    ax0.set_xlabel("Time (ps)")
    ax0.set_ylabel("Amplitude (arb. u.)")
    ax0.legend()

    ax1.plot(ref_fd[:, 0], 20*np.log10(np.abs(ref_fd[:, 1])), label="Reference")
    ax1.plot(sam_fd[:, 0], 20 * np.log10(np.abs(sam_fd[:, 1])), label="Sample")
    ax1.set_xlabel("Frequency (THz)")
    ax1.set_ylabel("Amplitude (dB)")
    ax1.legend()


if __name__ == '__main__':
    main()

    plt.show()
