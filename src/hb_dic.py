import pandas as pd


def load_dic(watershed: str):
    return pd.read_excel(
        "../data/HB_weekly_stream_chem/watersheds.xlsx",
        sheet_name=watershed,
        index_col="date",
        usecols=["date", "DIC"],
    )


def clean_dic_series(dic_series: pd.Series):
    ds = dic_series.squeeze()
    ds = ds[ds.index.drop_duplicates()]
    ds.interpolate(inplace=True)
    return ds
