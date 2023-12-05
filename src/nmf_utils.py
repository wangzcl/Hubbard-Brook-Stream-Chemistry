"""
Utilities for NMF.
"""
import math
import warnings
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

MOLAR_MASS = {
    "Ca": 40.078,
    "Mg": 24.305,
    "K": 39.098,
    "Na": 22.990,
    "SO4": 32.06 + 15.999 * 4,
    "Cl": 35.45,
    "NO3": 14.007 + 15.999 * 3,
}


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
    normalizer : str, default "Na"
        The column name of the normalizer.
        Other columns will be divided by this column. Default is sodium.
    rescale : bool, default True
        Rescale the data to <=1.
    bootstrap : int, default None
        The number of desired bootstrap samples.
        If None, no bootstrap is performed.
    bootstrap_random_state : int, default 42
        Random state for bootstrap.

    Attributes
    ----------

    scaler : sklearn.preprocessing.StandardScaler
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
        # Molar mass data from IUPAC periodic table (May 4 2022)
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

    def _bootstrap(self, df: pd.DataFrame, n_samples=None) -> pd.DataFrame:
        """
        It does not operate in place.
        """
        if n_samples is None:
            n_samples = self.bootstrap
        df = resample(df, n_samples=n_samples, random_state=self.bootstrap_random_state)
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


class ChemistryHeatmap:
    """
    Plot a heatmap of endmember chemistry.
    """

    def __init__(self, ax=None) -> None:
        if ax is None:
            ax = plt.gca()
        self.ax = ax

    def plot(self, data: npt.ArrayLike, xlabels: npt.ArrayLike) -> None:
        sns.heatmap(data, annot=True, cmap="YlGnBu", xticklabels=xlabels)


def count_endmember(pca: PCA, thres=0.9):
    n_endmember = np.count_nonzero(np.cumsum(pca.explained_variance_ratio_) < thres) + 1
    return n_endmember


class NMFResizer:
    """
    Resize the NMF result matrix ``w`` and ``h`` with a number ``k``.

    The transformation is:

    w' = w / k

    h' = h * k

    The transformation makes the sum of each row of ``W'`` as close to 1
    as possible.

    Attributes
    ----------

    k : float
        Resizing factor.
    sigma : float
        Standard deviation of ``log(sum(W, axis=1))``.
        We assume the sum of each row of W follows a log-normal distribution.
        By resizing, we make the peak of distribution 1.
    """

    def __init__(self) -> None:
        self.k = None
        self.sigma = None

    def _init_factor(self, W: npt.ArrayLike) -> float:
        proportion_sum = W.sum(axis=1)
        log_proportion_sum = np.log(proportion_sum)
        k = math.e ** log_proportion_sum.mean()
        self.k = k
        self.sigma = np.std(log_proportion_sum, ddof=1)

    def transform(
        self, w, h, inplace=False
    ) -> tuple[npt.ArrayLike, npt.ArrayLike] | None:
        """
        Conduct the transformation.
        """
        self._init_factor(w)
        if inplace:
            w /= self.k
            h *= self.k
            return
        w = w / self.k
        h = h * self.k
        return w, h


class NMFPermuter:
    """
    Permute endmembers in chemistry matrices (H) by KMeans,
    so that the orders are the same in different matrices (or NMF results).

    The permuter collects all the endmembers in all the chemistry matrices,
    and then clusters them into ``n_endmember`` clusters. After that, it
    permutes the endmembers in each chemistry matrix according to the labels,
    so each chemistry matrix has the same order of endmembers (0, 1, ...,
    n_endmember-1).

    A 3d array ``x`` is required for permutation.
    It is a collection of chemistry matrices from different NMF runs. The first
    dimension is the number of NMF runs, the second dimension is the number of
    endmembers, and the third dimension is the number of chemical species.

    Parameters
    ----------
    n_endmember : int
        The number of endmembers.
    **kmeans_kwargs
        Keyword arguments for ``sklearn.cluster.KMeans``.

    Attributes
    ----------

    kmeans : sklearn.cluster.KMeans
        KMeans model.
    labels : np.ndarray
        Labels of the clustered endmembers.

        ``labels[i]`` is the permutation of the i-th chemistry matrix H.

    Note
    ----

    The permutation is not stable, as KMeans may put two endmembers in one
    endmember chemistry matrix H into the same cluster, which is not what we
    want.
    """

    def __init__(self, n_endmember: int, **kmeans_kwargs) -> None:
        self.n_endmember = n_endmember
        # TODO: set default n_init to 5
        self.kmeans = KMeans(n_clusters=n_endmember, **kmeans_kwargs)
        self.labels = None

    def check_labels(self):
        """
        Check if the labels of permuted endmembers are right.

        The labels of each submatrix H should be 0, 1, 2, ..., n_endmember - 1,
        which means that every NMF result (matrix H) has a complete set of
        endmembers.
        """
        return np.array_equiv(np.arange(self.n_endmember), np.sort(self.labels))

    def fit(self, x) -> None:
        """
        Cluster the data and get the permutation labels.

        The first dimension of x should be the number of NMF runs,
        every slice ``x[i,:,:]`` is a chemistry matrix H in the i-th run.
        """
        x = x.reshape(-1, x.shape[-1])
        self.kmeans.fit(x)
        self.labels = self.kmeans.labels_.reshape((-1, self.n_endmember))
        labels_right = self.check_labels()
        if not labels_right:
            warnings.warn("labels not right!")

    def transform(self, x, inplace=False):
        """
        Transform the data. ``fit`` required before calling this method.
        """
        y = x if inplace else x.copy()
        for i in range(y.shape[0]):
            y[i] = x[i, self.labels[i]]

        return y

    def fit_transform(self, x, inplace=False):
        """
        Cluster the data and permute each chemistry matrix H in the same order.

        The first dimension of x should be the number of NMF runs,
        every slice ``x[i,:,:]`` is a chemistry matrix H in the i-th run.

        """
        self.fit(x)
        return self.transform(x, inplace=inplace)
