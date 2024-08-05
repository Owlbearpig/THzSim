import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import epsilon_0 as eps0

from cond_eval import eval_rho_analytical
from cond_sheet_model import cond_sheet
from fresnel_models import r_1lay
from mpl_settings import rcParams
from tmm_wrapper import tmm_package_wrapper

matplotlib.rcParams = rcParams

freqs = np.linspace(0.0, 5, 1000)

one = np.ones_like(freqs)

# 1 um film, 1 mm substrate
d_f, d_sub = 1, 1e3
d_list = [np.inf, d_f, d_sub, np.inf]

eps_inf = 4
t, s0 = 9e-16, 7650  # 9e-15  # 76500  # s, S/m # ITO values
w = 2 * np.pi * freqs * 1e12  # 1/s
s = s0 / (1 - 1j * w * t)
s_sheet = s * (d_f * 1e-6)

n1 = one
# n_film = 5 * one + 0 * 1j * one
n_film = np.sqrt(eps_inf + 1j * s / (eps0 * w))

n_sub = 3.4 * one + 0.0 * 1j * one

n = np.array([n1, n_film, n_sub, n_sub[0] * n1], dtype=complex).T

settings = {"freqs": freqs, "d_list": d_list, "n": n, "angle": 60}

r_s_tmm = tmm_package_wrapper(**settings, pol="s")
r_p_tmm = tmm_package_wrapper(**settings, pol="p")
r_s_1lay = r_1lay(**settings, pol="s")
r_p_1lay = r_1lay(**settings, pol="p")
r_s_sheet = cond_sheet(**settings, pol="s")
r_p_sheet = cond_sheet(**settings, pol="p")

s_exp_1lay = eval_rho_analytical(r_p_1lay, r_s_1lay, n=n, angle=settings["angle"], )
s_exp_sheet = eval_rho_analytical(r_p_sheet, r_s_sheet, n=n, angle=settings["angle"], )
s_exp_tmm = eval_rho_analytical(r_p_tmm, r_s_tmm, n=n, angle=settings["angle"], )

n_exp_sheet = np.sqrt(n_sub**2 + 1j*(s_exp_sheet/(d_f*1e-6*eps0*w)))

plt.figure()
plt.plot(freqs, s_sheet.real, label=r"Truth")
plt.plot(freqs, s_exp_sheet.real, label=r"Sheet eq.")
plt.plot(freqs, s_exp_1lay.real, label=r"1 layer model")
# plt.plot(freqs, s_exp_tmm.real, label=r"TMM")
plt.xlabel("Frequency (THz)")
plt.ylabel("Sheet conductivity Re (S)")
plt.legend()

plt.figure()
plt.plot(freqs, s_sheet.imag, label=r"Truth")
plt.plot(freqs, s_exp_sheet.imag, label=r"Sheet eq.")
plt.plot(freqs, s_exp_1lay.imag, label=r"1 layer model")
# plt.plot(freqs, s_exp_tmm.imag, label=r"TMM")
plt.xlabel("Frequency (THz)")
plt.ylabel("Sheet conductivity Im (S)")
plt.legend()

plt.figure()
plt.plot(freqs, n_film.real, label="Truth $n_f$ real")
plt.plot(freqs, n_film.imag, label="Truth $n_f$ imag")
plt.plot(freqs, n_exp_sheet.real, label="Exp $n_f$ real")
plt.plot(freqs, n_exp_sheet.imag, label="Exp $n_f$ imag")
plt.xlabel("Frequency (THz)")
plt.ylabel("Refractive index")
plt.legend()

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.set_xlabel("Frequency (THz)")
ax1.set_ylabel(f"$|r|$")
ax2.set_ylabel(f"Arg($r$)")


def plot_r(freq, r, label):
    r_abs, r_phi = np.abs(r), np.angle(r)
    ax1.plot(freq, r_abs, label=f"{label}")
    ax2.plot(freq, r_phi, "--", label=f"{label}")


# plot_r(freqs, r_tmm_s, label="TMM (s-pol)")
plot_r(freqs, r_s_1lay, label="1 layer model (s-pol)")
plot_r(freqs, r_s_sheet, label="Cond. sheet model (s-pol)")

ax1.legend(loc="lower right")
ax2.legend(loc="upper left")

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.set_xlabel("Frequency (THz)")
ax1.set_ylabel(f"$|r|$")
ax2.set_ylabel(f"Arg($r$)")


def plot_r(freq, r, label):
    r_abs, r_phi = np.abs(r), np.angle(r)
    ax1.plot(freq, r_abs, label=f"{label}")
    ax2.plot(freq, r_phi, "--", label=f"{label}")


# plot_r(freqs, r_tmm_p, label="TMM (p-pol)")
plot_r(freqs, r_p_1lay, label="1 layer model (p-pol)")
plot_r(freqs, r_p_sheet, label="Cond. sheet model (p-pol)")

ax1.legend(loc="lower right")
ax2.legend(loc="upper left")
plt.show()

plt.show()
