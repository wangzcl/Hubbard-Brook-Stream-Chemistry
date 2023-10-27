import pandas as pd
import matplotlib.pyplot as plt


def load_watershed_data(
    watershed: str | list, col: str | list[str]
) -> pd.Series | pd.DataFrame:
    usecols = ["date"]
    if isinstance(col, str):
        usecols.append(col)
    elif isinstance(col, list):
        usecols.extend(col)

    df = pd.read_excel(
        "../data/HB_weekly_stream_chem/watersheds.xlsx",
        sheet_name=watershed,
        index_col="date",
        usecols=usecols,
    )
    if isinstance(watershed, str) and isinstance(col, str):
        df = df.squeeze()
    
    return df


def load_DIC_series(watershed: str):
    return load_watershed_data(watershed, "DIC")


def load_pH_series(watershed: str):
    return load_watershed_data(watershed, "pH")


class SeriesPlotWithMissingValues:
    def __init__(self, data: pd.Series, ax=None):
        self.data = data

        missing = data[data.isna()]
        self.missing = missing

        if ax is None:
            ax = plt.gca()
        self.ax = ax
        return

    def plot(self, missing_yloc=None):
        self.data.dropna().plot(ax=self.ax)

        ymin, ymax = self.ax.get_ylim()
        if missing_yloc is None:
            missing_yloc = 0.98*ymin+0.02*ymax
        missing = self.missing.copy()
        missing[:] = missing_yloc
        
        missing.plot(
            ax=self.ax,
            linestyle="",
            marker="|",
            markersize=1,
            color="r",
        )
        return
