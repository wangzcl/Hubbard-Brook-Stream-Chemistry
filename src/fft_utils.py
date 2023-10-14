import numpy as np
from numpy.typing import ArrayLike
import pandas as pd


def spectrum(x: ArrayLike):
    fourier = np.fft.rfft(x)[1:]  # [1:] means neglecting 0 Hz signal (baseline)
    power_specturm = np.abs(fourier) ** 2
    power_specturm /= np.sum(power_specturm)  # normalized power
    f = np.fft.rfftfreq(len(x))[1:]  # frequencies of FFT results
    Ts = (
        1 / f
    )  # Periods as indices of power specturm. The unit is the interval of original series
    return pd.Series(power_specturm, index=Ts)
