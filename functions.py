import numpy as np
from numpy import nan_to_num, array
from numpy.fft import fft, fftfreq, ifft


def do_fft(data_td, pos_freqs_only=True):
    data_td = nan_to_num(data_td)

    dt = float(np.mean(np.diff(data_td[:, 0].real)))
    freqs, data_fd = fftfreq(n=len(data_td[:, 0]), d=dt), np.conj(fft(data_td[:, 1]))

    if pos_freqs_only:
        post_freq_slice = freqs >= 0
        return array([freqs[post_freq_slice], data_fd[post_freq_slice]]).T
    else:
        return array([freqs, data_fd]).T


def do_ifft(data_fd, hermitian=True, shift=0, flip=False, t_=None):
    freqs, y_fd = data_fd[:, 0].real, data_fd[:, 1]

    y_fd = nan_to_num(y_fd)

    if hermitian:
        y_fd = np.concatenate((np.conj(y_fd), np.flip(y_fd[1:])))
        # y_fd = np.concatenate((y_fd, np.flip(np.conj(y_fd[1:]))))
        """
        * ``a[0]`` should contain the zero frequency term,
        * ``a[1:n//2]`` should contain the positive-frequency terms,
        * ``a[n//2 + 1:]`` should contain the negative-frequency terms, in
          increasing order starting from the most negative frequency.
        """

    y_td = ifft(y_fd)

    if t_ is None:
        df = np.mean(np.diff(freqs))
        n = len(y_td)
        t_ = np.arange(0, n) / (n * df)

        # t_ = np.linspace(0, len(y_td)*df, len(y_td))
        # t_ += 885

    dt = np.mean(np.diff(t_))
    shift = int(shift / dt)
    y_td = np.roll(y_td, shift)

    if flip:
        y_td = np.flip(y_td)

    return array([t_, y_td]).T


def unwrap(data_fd):
    if data_fd.ndim == 2:
        y = nan_to_num(data_fd[:, 1])
    else:
        y = nan_to_num(data_fd)
        return np.unwrap(np.angle(y))

    return array([data_fd[:, 0].real, np.unwrap(np.angle(y))]).T
