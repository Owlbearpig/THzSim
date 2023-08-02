import numpy as np
from numpy import nan_to_num, array
from numpy.fft import fft, fftfreq, ifft, rfftfreq, irfft, rfft
from mpl_settings import plt


def do_fft(data_td):
    data_td = nan_to_num(data_td)

    dt = float(np.mean(np.diff(data_td[:, 0].real)))
    freqs, data_fd = rfftfreq(n=len(data_td[:, 0]), d=dt), rfft(data_td[:, 1])

    return array([freqs, data_fd]).T


def do_ifft(data_fd, hermitian=False, t_=None):
    freqs, y_fd = data_fd[:, 0].real, data_fd[:, 1]

    y_fd = nan_to_num(y_fd)

    y_td = irfft(y_fd)

    if t_ is None:
        df = np.mean(np.diff(freqs))
        n = len(y_td)
        t_ = np.arange(0, n) / (n * df)

        # t_ = np.linspace(0, len(y_td)*df, len(y_td))
        # t_ += 885

    return array([t_, y_td]).T


def unwrap(data_fd):
    if data_fd.ndim == 2:
        y = nan_to_num(data_fd[:, 1])
    else:
        y = nan_to_num(data_fd)

    phi = np.asarray(np.unwrap(np.angle(y)), dtype=float)

    return array([data_fd[:, 0].real, phi], dtype=float).T


def annot_max(x, y, ax=None, x_unit=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:.4f}".format(xmax)
    if x_unit:
        text += f" ({x_unit})"
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)


