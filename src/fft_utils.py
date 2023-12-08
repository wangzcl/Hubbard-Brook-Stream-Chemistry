"""
Utilities for FFT analysis.
"""
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt


def spectrum(x: npt.ArrayLike, interval: float = 1, interval_unit=None) -> pd.Series:
    """
    Relative Fourier power spectrum of a real time series.

    For a evenly sampled time series of size N (even) and frequency f,
    the frequencies of FFT components are [0, 1, ..., N/2]*f/N.

    In this function, the 0 Hz background (mean value) is neglected.

    The power spectrum is normalized to sum to 1.

    The index of the returned series has the unit of the interval of the
    original series.

    Parameters
    ----------

    x : array-like
        Time series data.
    interval : float, default 1
        Interval of the time series data.
    interval_unit : str, default "Week"
        Unit of the interval.

    Returns
    -------

    pd.Series
        Fourier power spectrum data, with a name of "Normalized Power",
        and an index with a name of "Period (``interval_unit``)".
    """
    fourier = np.fft.rfft(x)[1:]  # [1:] neglects 0 Hz signal (background)
    power_spectrum = np.abs(fourier) ** 2
    power_spectrum /= np.sum(power_spectrum)  # normalized power
    f = np.fft.rfftfreq(len(x), interval)[1:]
    t = 1 / f  # Periods as indices of power spectrum
    index_name = "Period"
    if interval_unit is not None:
        index_name += f" ({interval_unit})"
    return pd.Series(
        power_spectrum, index=pd.Index(t, name=index_name), name="Normalized Power"
    )


class FourierSpectrumPlot:
    """
    Stem plot Fourier power spectrum data.
    """

    def __init__(self, ax=None) -> None:
        if ax is None:
            ax = plt.gca()
        self.ax = ax

    def plot(self, power_spectrum: pd.Series, **kwargs):
        """
        Stem plot in ``self.ax``.

        Parameters
        ----------
        power_spectrum : pd.Series
            fourier power spectrum data.

        **kwargs
            Keyword arguments passed to ``matplotlib.axes.Axes.stem``.

        """
        self.ax.stem(
            power_spectrum.index,
            power_spectrum,
            markerfmt="",
            basefmt="C0-",
            **kwargs
        )
        self.ax.set_xscale("log")
        self.ax.set_xlabel(power_spectrum.index.name)
        self.ax.set_ylabel(power_spectrum.name)
        return


def weekly_resample(x: pd.Series) -> pd.Series:
    """
    Resample a time series to weekly data.

    Even though the raw data is weekly, they are not evenly sampled.
    That is why we need this function.

    Every bin interval begins on Monday and ends on Sunday (both inclusive).

    After resampling, indices are set to the first day of the week, which is
    also the most common sampling day in the raw data.

    Parameters
    ----------
    x : pd.Series
        Time series data.

    Returns
    -------
    pd.Series
        Weekly resampled data.
    """
    return x.resample("W-MON", closed="left", label="left").mean()
