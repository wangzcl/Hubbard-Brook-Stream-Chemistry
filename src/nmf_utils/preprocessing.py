# Molar mass data from IUPAC periodic table (May 4 2022)
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.utils import resample


MOLAR_MASS = {
    "Ca": 40.078,
    "Mg": 24.305,
    "K": 39.098,
    "Na": 22.990,
    "SO4": 32.06 + 15.999 * 4,
    "Cl": 35.45,
    "NO3": 14.007 + 15.999 * 3,
    "SiO2": 28.085 + 15.999 * 2,
}


def count_endmember(pca: PCA, thres=0.9):
    """
    We explain the number of principal components that explain 90% of the variance
    as the number of endmembers. You can manually adjust the 90% threshold.

    Parameters
    ----------

    pca : sklearn.decomposition.PCA
        PCA model.
    thres : float, default 0.9
        The threshold of explained variance ratio.
    """
    n_endmember = np.count_nonzero(np.cumsum(pca.explained_variance_ratio_) < thres) + 1
    return n_endmember


class NMFPreprocessor:
    """
    Preprocess data for NMF.

    Parameters
    ----------

    dropna : bool, default True
        Drop missing values.
    convert_weight : bool, default True
        Convert weight to moles.
    weight_ignore : str, default "DIC"
        The column name of the weight to ignore when converting weight to moles.
    normalizer : str or None, default ``None``
        The column name of the normalizer.
        Other columns will be divided by this column. Default is ``None``
        (do not perform normalization).
        Also changes the column names. (e.g. ``Ca`` -> ``Ca/Na``)
    rescale : bool, default True
        Rescale the data to <=1.
    bootstrap : int, default None
        The number of desired bootstrap samples.
        If None, no bootstrap is performed.
    bootstrap_random_state : int, default 42
        Random state for bootstrap.

    Attributes
    ----------

    scaler_ : pd.Series
    """

    def __init__(
        self,
        dropna=True,
        convert_weight=True,
        weight_ignore="DIC",
        normalizer="Na",
        rescale=True,
        bootstrap: int | None = None,
        bootstrap_random_state: int | None = None,
    ) -> None:
        self.dropna = dropna
        self.convert_weight = convert_weight
        self.weight_ignore = weight_ignore
        self.normalizer = normalizer
        self.rescale = rescale
        self.bootstrap = bootstrap
        self.bootstrap_random_state = bootstrap_random_state
        self.scaler_ = None

    def _dropna(self, df: pd.DataFrame) -> None:
        df.dropna(inplace=True)

    def _weight_to_mole(self, df: pd.DataFrame, ignore: str = None) -> None:
        if ignore is None:
            ignore = self.weight_ignore
        for col in df:
            if col != ignore:
                df[col] /= MOLAR_MASS[col]

    def _normalize(self, df: pd.DataFrame, col=None) -> pd.DataFrame:
        """
        It does not operate in place.
        """
        if col is None:
            col = self.normalizer

        normalizer = df[col]
        df = df.drop(columns=col)
        df = df.div(normalizer, axis=0)
        print("Normalizer: " + col)
        print("Normalized matrix columns:\n\t", df.columns.values, sep="")
        df.rename(columns=lambda x: x + "/" + col, inplace=True)
        return df

    def _rescale(self, df: pd.DataFrame) -> None:
        scaler = df.max(axis=0)
        self.scaler_ = scaler
        df /= scaler
        # StandardScaler(with_mean=False).fit_transform(df)

    def _bootstrap(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        It does not operate in place.
        """
        df = resample(
            df, n_samples=self.bootstrap, random_state=self.bootstrap_random_state
        )
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data.
        """
        df = df.copy()
        if self.dropna:
            self._dropna(df)  # Drop missing rows
        if self.convert_weight:
            self._weight_to_mole(df)  # Convert weight to moles
        if self.normalizer is not None:
            df = self._normalize(df)  # Normalized with an assigned normalizer
        if self.rescale:
            self._rescale(df)  # Rescale to <=1
        if self.bootstrap:
            df = self._bootstrap(df)
        return df
