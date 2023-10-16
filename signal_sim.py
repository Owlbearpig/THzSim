import matplotlib.pyplot as plt
import numpy as np
from functions import do_fft, to_db
from numpy import exp, array, cos, sin, arcsin
from mpl_settings import *
from scipy.constants import c as c0

c_THz = c0 * 1e-9  # mm / ps

# TODO write class

# signal
dt = 0.001  # dt = 0.05 # sampling time
frame_size = 100  # sampling window
sigma = 0.2  # pulse width
noise_factor = 1E-10
pulse_shift = 20  # time axis shift (ps)

# sample time shift
sam_shift = 0.105  # PTFE 31.5 um
# sam_shift = 0.138  # RDX 41.4 um
# sam_shift = 0.050
# sam_shift = 0

n_min, n_max = 1.4, 1.45
# k_min, k_max = 0.00, 0.035
k_min, k_max = 0.07, 0.09

# absorption peak
width = 5E-3
abs_peak_scale = 5E-4

# incident angle
theta_i = 8 * np.pi / 180

# slab thickness (mm)
d0 = 10.0

fig, (ax0_cri, ax1_cri, ax2_cri) = plt.subplots(3, 1)
ax0_cri.set_ylabel("Real part")
ax2_cri.set_xlabel("Frequency (THz)")
ax1_cri.set_ylabel("Imaginary part")
ax2_cri.set_ylabel("Absorption coefficient (1/cm)")
ax0_cri.set_xlim((-0.1, 8))
ax1_cri.set_xlim((-0.1, 8))
ax2_cri.set_xlim((-0.1, 8))
ax0_cri.set_ylim((0.5, 3.5))
ax1_cri.set_ylim((-0.01, 0.15))
ax2_cri.set_ylim((-1, 110))

fig_, (ax0_sig, ax1_sig, ax2_sig) = plt.subplots(3, 1)
ax0_sig.set_xlabel("Time (ps)")
ax0_sig.set_ylabel("Amplitude (arb. u.)")
ax0_sig.xaxis.set_label_position('top')
ax0_sig.xaxis.set_ticks_position('top')
ax2_sig.set_xlabel("Frequency (THz)")
ax2_sig.set_ylabel("Amplitude (dB)")
ax1_sig.set_ylabel("Phase (rad)")
ax1_sig.set_xlim((-0.1, 8))
# ax1_sig.set_ylim((-np.pi, np.pi))
ax2_sig.set_xlim((-0.1, 15))

fig_, (axa) = plt.subplots(1, 1)
axa.set_xlabel("Frequency (THz)")
axa.set_xlim((-0.1, 8))

def refl_sample_sim(f, en_plot=True, pol="s"):
    n0 = 1 + 1j * 0

    f_max = int(np.argmin(np.abs(f - 15.0)))
    n1_r, n1_i = np.ones_like(f), np.zeros_like(f)

    n1_r[:f_max] = np.linspace(n_min, n_max, f_max)
    n1_i[:f_max] = np.linspace(k_min, k_max, f_max)

    peak = (1 / np.pi) * (width / ((f - 1) ** 2 + width ** 2))
    n1_i = n1_i + peak * abs_peak_scale

    n1 = n1_r + 1j * n1_i

    theta_t = arcsin((n0 / n1) * sin(theta_i))

    if pol == "s":
        r_ = (n0 * cos(theta_i) - n1 * cos(theta_t)) / (n0 * cos(theta_i) + n1 * cos(theta_t))  # r_s
    else:
        r_ = (n1 * cos(theta_i) - n0 * cos(theta_t)) / (n1 * cos(theta_i) + n0 * cos(theta_t))  # r_p

    # r_ = (n0 - n1) / (n0 + n1)
    # t_ = 2 * n0 / (n0 + n1)

    alpha = (1 / 100) * 4 * np.pi * n1.imag * f / c_THz  # 1 mm

    if en_plot:
        ax0_cri.plot(f, n1.real, label="Real part of n1 (truth)")
        ax1_cri.plot(f, n1.imag, label="Imaginary part of n1 (truth)")
        ax2_cri.plot(f, alpha, label="Absorption coefficient (truth) (1/cm)")

    return r_


def t_sample_sim(f, en_plot=False):
    w = 2*np.pi*f
    n0 = 1 + 1j * 0

    f_max = int(np.argmin(np.abs(f - 15.0)))
    n1_r, n1_i = np.ones_like(f), np.zeros_like(f)

    n1_r[:f_max] = np.linspace(n_min, n_max, f_max)
    n1_i[:f_max] = np.linspace(k_min, k_max, f_max)

    peak = (1 / np.pi) * (width / ((f - 1) ** 2 + width ** 2))
    n1_i = n1_i + peak * abs_peak_scale

    n1 = n1_r + 1j * n1_i

    t01_ = 2*n0 / (n0+n1)
    t12_ = 2 * n1 / (n0 + n1)

    t_ = t01_ * t12_ * exp(1j*d0*w*n1/c_THz)

    alpha = 10 * 4 * np.pi * n1.imag * f / c_THz

    if en_plot:
        ax0_cri.plot(f, n1.real, label="n actual")
        ax1_cri.plot(f, n1.imag, label="k actual")
        ax2_cri.plot(f, alpha, label="Absorption coefficient (truth)")

    return t_


def ref_signal():
    # simulated (reference) signal with noise

    t = np.arange(-frame_size, frame_size, dt)

    t_ = t - pulse_shift
    y = t_ * exp(-(t_ / sigma) ** 2)
    y += np.random.random((len(t))) * noise_factor

    ref_td_ = array([t, y], dtype=float).T

    ref_fd_ = do_fft(ref_td_)
    y_fd_db = to_db(ref_fd_)

    ax0_sig.plot(t, y, label="Reference")
    ax2_sig.plot(ref_fd_[:, 0].real, y_fd_db, label="Reference")

    return ref_td_, ref_fd_


