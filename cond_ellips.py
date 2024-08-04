from tmm import coh_tmm
import numpy as np
from tmm_wrapper import tmm_package_wrapper
from mpl_settings import rcParams
import matplotlib
import matplotlib.pyplot as plt
from fresnel_models import r_1lay
from cond_sheet_model import cond_sheet
from scipy.constants import epsilon_0 as eps0

matplotlib.rcParams = rcParams

freqs = np.linspace(-0.10, 5.15, 1000)

one = np.ones_like(freqs)

# 1 um film, 10 mm substrate
d_list = [np.inf, 1, 10e3, np.inf]

eps_inf = 4
t, s0 = 9e-15, 76500  # s, S/m # ITO values
w = 2*np.pi*freqs*1e12  # 1/s

n1 = one
n_film = 10 * one + 10*1j*one
n_film = np.sqrt(4 + 1j*s0/(eps0*w*(1-1j*t*w)))

n_sub = 3.4 * one + 0.1*1j*one

n = np.array([n1, n_film, n_sub, n_sub[0] * n1], dtype=complex).T

pol = "p"
kwargs = {"freqs": freqs, "d_list": d_list, "n": n, "angle": 60, "pol": pol}

r_tmm_s = tmm_package_wrapper(**kwargs)
r_1lay_s = r_1lay(**kwargs)
r_sheet_s = cond_sheet(**kwargs)

plt.figure()
plt.plot(freqs, n_film.real, label="n_film real")
plt.plot(freqs, n_film.imag, label="n_film imag")
plt.xlabel("Frequency")
plt.ylabel("Refractive index")
plt.legend()

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.set_xlabel("Frequency (THz)")
ax1.set_ylabel(f"$|r_{pol}|$")
ax2.set_ylabel(f"Arg($r_{pol}$)")


def plot_r(freq, r, label):
    r_abs, r_phi = np.abs(r), np.angle(r)
    ax1.plot(freq, r_abs, label=f"$|r|$ {label}")
    ax2.plot(freq, r_phi, "--", label=f"Arg($r$) {label}")


plot_r(freqs, r_tmm_s, label="TMM")
plot_r(freqs, r_1lay_s, label="1 Layer model")
plot_r(freqs, r_sheet_s, label="cond sheet model")

ax1.legend(loc="lower right")
ax2.legend(loc="upper left")
plt.show()
