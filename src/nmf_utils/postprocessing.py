import math
import warnings
import numpy as np
from numpy.typing import ArrayLike
from sklearn.cluster import KMeans


class NMFTrivialResizer:
    # TODO: This class has not been tested.
    """
    Resize the NMF result matrix ``w`` and ``h`` with a number ``k``.

    The transformation is:

    w' = w / k

    h' = h * k

    The transformation makes the sum of each row of ``W'`` as close to 1
    as possible.

    Attributes
    ----------

    k : float or ndarray
        Resizing factor.
    sigma : float or ndarray
        Standard deviation of ``log(sum(W, axis=-1))``.
        We assume the sum of each row of ``w`` follows a log-normal distribution.
        By resizing, we make the peak of distribution 1.
    """

    def __init__(self) -> None:
        self.k = None
        self.sigma = None

    def fit(self, w: ArrayLike) -> tuple[float, float]:
        """
        Find the resizing factor ``k`` and std of log-normal distribution ``sigma``.
        """
        proportion_sum = w.sum(axis=-1)
        log_proportion_sum = np.log(proportion_sum)
        k = math.e ** log_proportion_sum.mean(axis=-1)
        sigma = np.std(log_proportion_sum, axis=-1, ddof=1)
        self.k = k
        self.sigma = sigma
        return k, sigma

    def fit_transform(self, w, h, inplace=False) -> tuple[ArrayLike, ArrayLike] | None:
        """
        Conduct the transformation.
        """
        self.fit(w)
        if inplace:
            w /= self.k
            h *= self.k
            return
        w = w / self.k
        h = h * self.k
        return w, h


class NMFKmeansPermuter:
    # TODO: This class has not been tested.
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
        self.labels_ = None

    def check_labels(self):
        """
        Check if the labels of permuted endmembers are right.

        The labels of each submatrix H should be 0, 1, 2, ..., n_endmember - 1,
        which means that every NMF result (matrix H) has a complete set of
        endmembers.
        """
        return np.array_equiv(np.arange(self.n_endmember), np.sort(self.labels_))

    def fit(self, x) -> None:
        """
        Cluster the data and get the permutation labels.

        The first dimension of x should be the number of NMF runs,
        every slice ``x[i,:,:]`` is a chemistry matrix H in the i-th run.
        """
        x = x.reshape(-1, x.shape[-1])
        self.kmeans.fit(x)
        self.labels_ = self.kmeans.labels_.reshape((-1, self.n_endmember))
        labels_right = self.check_labels()
        if not labels_right:
            warnings.warn("labels not right!")

    def transform(self, x, inplace=False):
        """
        Transform the data. ``fit`` required before calling this method.
        """
        y = x if inplace else x.copy()
        for i in range(y.shape[0]):
            y[i] = x[i, self.labels_[i]]

        return y

    def fit_transform(self, x, inplace=False):
        """
        Cluster the data and permute each chemistry matrix H in the same order.

        The first dimension of x should be the number of NMF runs,
        every slice ``x[i,:,:]`` is a chemistry matrix H in the i-th run.

        """
        self.fit(x)
        return self.transform(x, inplace=inplace)


class NMFStdSelector:
    # TODO: This class has not been tested.
    # TODO: Add docstring.
    def __init__(self, n_select, threshold) -> None:
        self.n_runs = None
        self.n_select = n_select
        self.sigmas = None
        self.threshold = threshold
        self.resizer = NMFTrivialResizer()

    def fit(self, ws):
        n_runs = ws.shape[0]
        resizer = self.resizer
        sigmas = np.empty(n_runs)
        for i in range(n_runs):
            w = ws[i]
            resizer.fit(w)
            sigmas[i] = resizer.sigma
        self.n_runs = n_runs
        self.sigmas = sigmas
        return sigmas

    def select(self, sigmas):
        if self.threshold is not None:
            selected_idx = np.where(sigmas < self.threshold)
        elif self.n_select is not None:
            n_sel = self.n_select
            selected_idx = np.argpartition(sigmas, n_sel)[:n_sel]
        else:
            raise ValueError("Either threshold or n_select should be specified.")
        return selected_idx

def sum_log_std(W):
    """
    Calculate the standard deviation of ``log(sum(W, axis=-1))``.
    """
    log_row_sums = np.log(W.sum(axis=-1))
    return np.std(log_row_sums, axis=-1, ddof=1)