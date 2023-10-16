import numpy as np
from numpy import nan_to_num, array
from numpy.fft import fft, fftfreq, ifft, rfftfreq, irfft, rfft
from mpl_settings import plt


def shift_signal(y_td, shift):
    # shift in ps (unit of t)
    t_, y = y_td[:, 0].real, y_td[:, 1].real
    dt = np.mean(np.diff(t_))
    shift = int(shift / dt)
    y = np.roll(y, shift)

    return array([t_, y], dtype=float).T



# Polynomial Regression
def polyfit(x, y, degree=None, remove_worst_outlier=False):
    if degree is None:
        degree = 1

    def _fit(x_, y_):
        res = {}

        coeffs = np.polyfit(x_, y_, degree)

        # Polynomial Coefficients
        res['polynomial'] = coeffs.tolist()

        # r-squared
        p = np.poly1d(coeffs)
        # fit values, and mean
        yhat = p(x_)  # or [p(z) for z in x]
        ybar = np.sum(y_) / len(y_)  # or sum(y)/len(y)
        ssreg = np.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
        sstot = np.sum((y_ - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])

        res['determination'] = ssreg / sstot

        return res

    results = _fit(x, y)

    return results


def do_fft(data_td):
    data_td = nan_to_num(data_td)

    dt = float(np.mean(np.diff(data_td[:, 0].real)))
    f_axis, data_fd = rfftfreq(n=len(data_td[:, 0]), d=dt), rfft(data_td[:, 1])

    return array([f_axis, data_fd]).T


def do_ifft(data_fd, t_=None):
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


def unwrap(data_fd, one_d=False):
    if data_fd.ndim == 2:
        y = nan_to_num(data_fd[:, 1])
    else:
        y = nan_to_num(data_fd)

    phi = np.asarray(np.unwrap(np.angle(y)), dtype=float)

    if one_d:
        return array(phi, dtype=float)
    else:
        return array([data_fd[:, 0].real, phi], dtype=float).T


def annot_max(x, y, ax=None, x_unit=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text = "x={:.4f}".format(xmax)
    if x_unit:
        text += f" ({x_unit})"
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data', textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94, 0.96), **kw)


def to_db(data_fd):
    if data_fd.ndim == 2:
        return 20 * np.log10(np.abs(data_fd[:, 1]))
    else:
        return 20 * np.log10(np.abs(data_fd))
