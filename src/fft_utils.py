import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt


def spectrum(x: npt.ArrayLike):
    fourier = np.fft.rfft(x)[1:]  # [1:] neglects 0 Hz signal (baseline)
    power_spectrum = np.abs(fourier) ** 2
    power_spectrum /= np.sum(power_spectrum)  # normalized power
    f = np.fft.rfftfreq(len(x))[1:]  # frequencies of real FFT results
    t = 1 / f  # Periods as indices of power spectrum
    # The unit is the interval of the original series
    return pd.Series(power_spectrum, index=t, name="Normalized Power")


class FourierSpectrumPlot:
    def __init__(self, power_spectrum: pd.Series, ax=None) -> None:
        self.power_spectrum = power_spectrum
        if ax is None:
            ax = plt.gca()
        self.ax = ax
        return

    def plot(self):
        self.ax.stem(
            self.power_spectrum.index, self.power_spectrum, markerfmt="", basefmt="C0-"
        )
        self.ax.set_xscale("log")
        self.ax.set_xlabel("Period (Week)")
        self.ax.set_ylabel("Normalized Power")
        return
