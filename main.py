import numpy as np
from numpy import exp, array, cos, sin, arcsin
from functions import do_fft, do_ifft, unwrap, annot_max
from scipy.constants import c as c0
from mpl_settings import *

fig, (ax0_cri, ax1_cri, ax2_cri) = plt.subplots(3, 1)
ax0_cri.set_ylabel("Real part")
ax2_cri.set_xlabel("Frequency (THz)")
ax1_cri.set_ylabel("Imaginary part")
ax2_cri.set_ylabel("Absorption coefficient (1/cm)")
ax0_cri.set_xlim((-0.1, 8))
ax1_cri.set_xlim((-0.1, 8))
ax2_cri.set_xlim((-0.1, 8))
ax0_cri.set_ylim((0.5, 3.5))
ax1_cri.set_ylim((-0.01, 0.06))
ax2_cri.set_ylim((-1, 30))

fig_, (ax0_sig, ax1_sig, ax2_sig) = plt.subplots(3, 1)
ax0_sig.set_xlabel("Time (ps)")
ax0_sig.set_ylabel("Amplitude (arb. u.)")
ax0_sig.xaxis.set_label_position('top')
ax0_sig.xaxis.set_ticks_position('top')
ax2_sig.set_xlabel("Frequency (THz)")
ax2_sig.set_ylabel("Amplitude (dB)")
ax1_sig.set_ylabel("Phase (rad)")
ax1_sig.set_xlim((-0.1, 8))

# signal
dt = 0.002  # dt = 0.05 # sampling time
frame_size = 10  # plus minus (2N)
sigma = 0.2  # pulse width
noise_factor = 1E-10
pulse_shift = 0  # time axis shift

# sample time shift
sam_shift = 0.105  # PTFE 31.5 um
sam_shift = 0.138  # RDX 41.4 um
# sam_shift = 0.050
# sam_shift = 0

k_max, n_max = 0.06, 1.64

# absorption peak
width = 0.05
amplitude = 0.005

# incident angle
theta_i = 8 * np.pi / 180


def sample_sim(f, en_plot=True, pol="s"):
    n0 = 1 + 1j * 0

    f_max = int(np.argmin(np.abs(f - 15.0)))
    n1_r, n1_i = np.zeros_like(f), np.zeros_like(f)

    n1_r[:f_max] = np.linspace(1.4, n_max, f_max)
    n1_i[:f_max] = np.linspace(0.00, k_max, f_max)

    peak = (1 / np.pi) * (width / ((f - 1) ** 2 + width ** 2))
    n1_i = n1_i + peak * amplitude

    n1 = n1_r + 1j * n1_i

    theta_t = arcsin((n0 / n1) * sin(theta_i))

    if pol == "s":
        r_ = (n0 * cos(theta_i) - n1 * cos(theta_t)) / (n0 * cos(theta_i) + n1 * cos(theta_t))  # r_s
    else:
        r_ = (n1 * cos(theta_i) - n0 * cos(theta_t)) / (n1 * cos(theta_i) + n0 * cos(theta_t))  # r_p

    # r_ = (n0 - n1) / (n0 + n1)
    # t_ = 2 * n0 / (n0 + n1)

    alpha = (1 / 100) * 4 * np.pi * n1.imag * (f * 1e12) / c0

    if en_plot:
        ax0_cri.plot(f, n1.real, label="Real part of n1 (truth)")
        ax1_cri.plot(f, n1.imag, label="Imaginary part of n1 (truth)")
        ax2_cri.plot(f, alpha, label="Absorption coefficient (truth) (1/cm)")

    return r_


def shift_signal(y_td, shift):
    # shift in ps (unit of t)
    t_, y = y_td[:, 0].real, y_td[:, 1].real
    dt = np.mean(np.diff(t_))
    shift = int(shift / dt)
    y = np.roll(y, shift)

    return array([t_, y], dtype=float).T


def correlation(y0_td, y1_td):
    t = y0_td[:, 0].real
    scale = 1 / len(t)
    dt_ = np.mean(np.diff(t))

    tau = np.arange(-5, 5, dt_)

    corr_ = np.zeros_like(tau)
    for t_idx in range(len(tau)):
        corr_[t_idx] = scale * np.sum(y1_td[:, 1].real * np.roll(y0_td[:, 1].real, t_idx - int(5 / dt_)))

    # corr_ = np.abs(corr_)
    max_idx = np.argmax(corr_)
    print(f"Correlation maximum at {round(tau[max_idx], 5)} ps ({tau[max_idx]})")
    print(f"({round(tau[max_idx], 5) * c0 * 1E-6} um)")

    fig, ax0 = plt.subplots(1, 1)
    ax0.plot(tau, corr_)
    annot_max(tau, corr_, ax0, x_unit="ps")
    ax0.set_xlabel("Time (ps)")
    ax0.set_ylabel("Cross-correlation")
    ax0.set_ylim((np.min(corr_)*1.2, np.max(corr_)*1.5))

    return array([tau, corr_], dtype=float).T


def ref_signal():
    # simulated (reference) signal with noise

    t = np.arange(-frame_size, frame_size + dt, dt)
    if len(t) % 2 == 0:
        t = np.arange(-frame_size, frame_size, dt)

    t += pulse_shift  # shift away from 0
    y = t * exp(-(t / sigma) ** 2)
    y += np.random.random((len(t))) * noise_factor

    return array([t, y], dtype=float).T


def extract_cri(ref_fd_, sam_fd_, corr_=None):
    f = ref_fd_[:, 0].real
    amp_ratio = np.abs(sam_fd_[:, 1] / ref_fd_[:, 1])

    phi_ref = unwrap(ref_fd_)
    phi_sam = unwrap(sam_fd_)
    phi_m = phi_sam[:, 1] - phi_ref[:, 1]

    if (corr_ is not None) and (sam_shift != 0):
        max_idx = np.argmax(corr_[:, 1])
        dtau = corr_[max_idx, 0]
        dtau += dt / 10
        phi_c = 2*np.pi*dtau*f
    else:
        print("No correction required")
        phi_c = np.zeros_like(f)

    phi_a = phi_m - phi_c
    ax1_sig.plot(f, phi_c, label="phi_c")
    ax1_sig.plot(f, phi_a, label="Phase after correction")

    n = (1 - amp_ratio ** 2) / (1 + amp_ratio ** 2 - 2 * amp_ratio * cos(phi_a))
    n[0] = n[1]  # DC
    k = (2 * amp_ratio * sin(phi_a)) / (1 + amp_ratio ** 2 - 2 * amp_ratio * cos(phi_a))
    alpha = (1 / 100) * 4 * np.pi * k * f * 1e12 / c0

    ax0_cri.plot(f, n, label="Real part of n1 (extracted)")
    ax1_cri.plot(f, k, label="Imaginary part of n1 (extracted)")
    ax2_cri.plot(f, alpha, label="Absorption coefficient (extracted) (1/cm)")

    return n + 1j * k


def main():
    """
    Ref_td -> Ref_fd -> r_p * Ref_fd = Sam_fd -> Sam_td ->
    -> Sam_td + noise -> Sam_td(t - shift) = Sam_td_shifted ->
    -> correlation(Ref_td, Sam_td_shifted) = corr -> dtau = argmax(corr) ->
    -> extract RI (Ref_fd, Sam_fd_shifted, dtau)
    """

    ref_td = ref_signal()
    t = ref_td[:, 0].real
    ref_fd = do_fft(ref_td)
    f = ref_fd[:, 0].real
    r = sample_sim(f, en_plot=True, pol="p")

    sam_fd = array([f, r * ref_fd[:, 1]]).T

    sam_td = do_ifft(sam_fd, t_=t)
    sam_td[:, 1] += np.random.random((len(t))) * noise_factor

    sam_td_shifted = shift_signal(sam_td, shift=sam_shift)

    sam_fd_shifted = do_fft(sam_td_shifted)

    phi_ref = unwrap(ref_fd)
    phi_sam = unwrap(sam_fd)
    phi_sam_shifted = unwrap(sam_fd_shifted)

    phi_diff = phi_sam_shifted[:, 1] - phi_ref[:, 1]

    corr = correlation(ref_td, sam_td_shifted)

    cri_extracted = extract_cri(ref_fd, sam_fd_shifted, corr_=corr)

    ax0_sig.plot(t, ref_td[:, 1].real, label="Reference")
    ax0_sig.plot(t, sam_td[:, 1].real, label="Sample dt=0 ps")
    ax0_sig.plot(t, sam_td_shifted[:, 1].real, label=f"Sample dt={sam_shift} ps")

    ax1_sig.plot(phi_ref[:, 0], phi_ref[:, 1], label="Reference")
    ax1_sig.plot(phi_sam[:, 0], phi_sam[:, 1], label="Sample dt=0")
    ax1_sig.plot(phi_sam_shifted[:, 0], phi_sam_shifted[:, 1], label=f"Sample dt={sam_shift} ps")
    ax1_sig.plot(f, phi_diff, label=f"Difference phi_sam_shifted - phi_ref ({sam_shift} ps)")

    ax2_sig.plot(f, 20 * np.log10(np.abs(ref_fd[:, 1])), label="Reference")
    ax2_sig.plot(f, 20 * np.log10(np.abs(sam_fd[:, 1])), label="Sample dt=0")
    ax2_sig.plot(f, 20 * np.log10(np.abs(sam_fd_shifted[:, 1])), label=f"Sample dt={sam_shift} ps")


if __name__ == '__main__':
    main()
    ax0_cri.legend()
    ax1_cri.legend()
    ax2_cri.legend()
    ax0_sig.legend()
    ax1_sig.legend()
    ax2_sig.legend()

    plt.show()
