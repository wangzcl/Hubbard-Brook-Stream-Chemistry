"""
Utilities for loading and plotting Hubbard Brook data.
"""
import pandas as pd
import matplotlib.pyplot as plt


def load_watershed_data(
    watershed: str | list[str],
    col: str | list[str],
    filepath: str = "../data/HB_weekly_stream_chem/watersheds.xlsx",
) -> pd.Series | pd.DataFrame:
    """
    Load watershed data from an excel file.

    Dates are used as the index.

    You can specify the watershed name(s) and the column name(s).

    If you specify a single watershed name and a single column name,
    the function returns a ``Series``, otherwise a ``DataFrame``.
    """
    index_col = "date"
    usecols = [index_col,]
    if isinstance(col, str):
        usecols.append(col)
    elif isinstance(col, list):
        usecols.extend(col)

    df = pd.read_excel(
        filepath,
        sheet_name=watershed,
        index_col=index_col,
        usecols=usecols,
    )
    if isinstance(watershed, str) and isinstance(col, str):
        df = df.squeeze()
    return df


def load_DIC_series(watershed: str):
    """
    A wrapper of ``load_watershed_data`` for DIC data.
    """
    return load_watershed_data(watershed, "DIC")


def load_pH_series(watershed: str):
    """
    A wrapper of ``load_watershed_data`` for pH data.
    """
    return load_watershed_data(watershed, "pH")


class SeriesPlotWithMissingValues:
    """
    Plot a series with missing values dropped and marked at the bottom
    as red vertical lines.
    """
    def __init__(self, ax=None):
        if ax is None:
            ax = plt.gca()
        self.ax = ax
        self.data = None
        self.missing = None
        return

    def plot(self, data: pd.Series, missing_yloc=None):
        """
        Plot the series. You may specify the y location of the missing values.
        """
        missing = data[data.isna()]
        data.dropna().plot(ax=self.ax)

        self.data = data
        self.missing = missing

        ymin, ymax = self.ax.get_ylim()
        if missing_yloc is None:
            missing_yloc = 0.98 * ymin + 0.02 * ymax
        missing = missing.copy()
        missing[:] = missing_yloc

        missing.plot(
            ax=self.ax,
            linestyle="",
            marker="|",
            markersize=1,
            color="r",
        )
        return
