import numpy as np
from numpy import nan_to_num, array
from numpy.fft import fft, fftfreq


def do_fft(data_td, pos_freqs_only=True):
    data_td = nan_to_num(data_td)

    dt = float(np.mean(np.diff(data_td[:, 0])))
    freqs, data_fd = fftfreq(n=len(data_td[:, 0]), d=dt), np.conj(fft(data_td[:, 1]))

    if pos_freqs_only:
        post_freq_slice = freqs >= 0
        return array([freqs[post_freq_slice], data_fd[post_freq_slice]]).T
    else:
        return array([freqs, data_fd]).T
