import math
import warnings
import random
import numpy as np
from numpy.typing import ArrayLike
from typing import Iterable
from sklearn.decomposition import NMF
from .postprocessing import sum_log_std


class LstsqNMF(NMF):
    # TODO: Check negative values in W.
    """
    Every row of the raw mixing proportion matrix W is rescaled respectively
    to a new matrix W', where each row sums to 1.

    The transformation matrix is given by solving the least square problem:

    W * T = W'

    The new decomposition becomes:

    X = W * H = (W * T) * (T^-1 * H) = W' * H'
    """

    def __init__(self, n_components="warn", **kwargs):
        super().__init__(n_components, **kwargs)
        self.tranformation_ = None  # Transformation matrix
        self.is_valid_ = None

    def fit_transform(self, X, *args, **kwargs):
        """
        Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        *args :
            Other optional positional arguments passed to ``NMF.fit_transform``.

        **kwargs :
            Other optional keyword arguments passed to ``NMF.fit_transform``.
        """
        W = super().fit_transform(X, *args, **kwargs)
        H = self.components_

        # Obtain the target proportion matrix W_target, where each row sums to 1
        row_sums = W.sum(axis=1)
        row_sums = row_sums[:, np.newaxis]
        W_target = W / row_sums

        T, _, _, _ = np.linalg.lstsq(W, W_target, rcond=None)

        W @= T
        H = np.linalg.solve(T, H)

        self.components_ = H
        self.tranformation_ = T
        self.is_valid_ = is_valid_nmf(W, H)

        return W


def is_valid_nmf(W, H):
    """
    Check if W and H are valid (non-negative) NMF results.

    Parameters
    ----------
    W : array-like of shape (n_samples, n_components)
        The mixing proportion matrix.
    H : array-like of shape (n_components, n_features)
        The endmember matrix.

    Returns
    -------
    bool
        True if W and H are valid NMF results, False otherwise.
    """

    if np.any(W < 0):
        return False
    if np.any(H < 0):
        return False
    return True


class TrivialRescaledNMF(NMF):
    """
    The raw mixing proportion matrix ``W``and endmember matrix ``H`` are rescaled
    by a single factor ``k``.

    The transformation is:

    W' = W / k

    H' = H * k

    The transformation makes the sum of each row of ``W'`` as close to 1 as
    possible.  We assume the sum of each row of ``W`` follows a log-normal
    distribution. By resizing, we make the peak of distribution 1.
    """

    def __init__(self, n_components="warn", **kwargs):
        super().__init__(n_components, **kwargs)
        self.k = None  # Resizing factor

    def fit_transform(self, X, *args, **kwargs):
        """
        Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        *args :
            Other optional positional arguments passed to ``NMF.fit_transform``.

        **kwargs :
            Other optional keyword arguments passed to ``NMF.fit_transform``.
        """
        W = super().fit_transform(X, *args, **kwargs)
        H = self.components_

        row_sums = W.sum(axis=-1)
        log_row_sums = np.log(row_sums)
        k = math.e ** log_row_sums.mean(axis=-1)

        self.k = k
        W /= k
        H *= k

        self.components_ = H

        return W


class MultiNMF:
    # TODO: This class has not been tested.
    """
    Perform multiple NMF runs.

    Parameters
    ----------

    n_runs : int
        The number of NMF runs.
    n_components : int
        The number of endmembers.
        Passed to ``sklearn.decomposition.NMF``.
    random_state : int, RandomState instance or None, default None
        Random state paased to ``sklearn.decomposition.NMF``.
        If it is an integer, it is used as the random seed for
        ``np.random.RandomState``, otherwise it is passed to
        ``sklearn.utils.check_random_state``.
    **nmf_kwargs
        Other keyword arguments passed to ``sklearn.decomposition.NMF``.

    Attributes
    ----------

    n_runs : int
        The number of NMF runs.
    nmf : sklearn.decomposition.NMF
        NMF instance.
    n_components_ : int
        The number of endmembers (NMF components).
    components_ : np.ndarray
        The components of NMF results, stored in a 3d array.
        ``components_[i]`` is the i-th run's components.
    """

    def __init__(
        self,
        n_components: int,
        nmf_type: type = NMF,
        n_runs: int | None = None,
        random_state: int | ArrayLike | Iterable | None = None,
        **nmf_kwargs,
    ) -> None:
        if isinstance(random_state, (int, type(None))):
            random_state = random.Random(random_state)
        else:
            random_state = iter(random_state)
        self.n_runs = n_runs
        self.random_state = random_state
        self.count = 0
        self.n_components_ = n_components

        self.nmf = nmf_type(n_components=n_components, **nmf_kwargs)

    def _random(self):
        if isinstance(self.random_state, random.Random):
            randint = self.random_state.randint(0, 2**32 - 1)
        else:
            randint = next(self.random_state)
        return randint

    def one_run(self, X):
        """
        Perform one NMF run.

        Returns NMF results ``W`` and ``H``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.
        """
        randint = self._random()
        self.nmf.random_state = randint
        self.count += 1

        W = self.nmf.fit_transform(X)
        H = self.nmf.components_
        return W, H

    def fit_transform(self, X, n_runs: int | None = None):
        """
        Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        n_runs : int or None, default None
            The number of NMF runs.
            If None, use ``self.n_runs``.
        """
        if n_runs is None:
            n_runs = self.n_runs
        else:
            self.n_runs = n_runs
        assert n_runs is not None
        n_samples, n_features = X.shape
        # Preallocate memory for NMF results
        Hs = np.empty((n_runs, self.n_components_, n_features))

        for i in range(n_runs):
            _, Hs[i] = self.one_run(X)

        return Hs


class MinStdPickedNMF(MultiNMF):
    def __init__(
        self,
        n_selected: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.n_selected = n_selected
        self.raw_random_state = self.random_state
        self.selected_random_states = None
        self.prefitted = False

    def prefit(self, X):
        sigmas = []
        random_series = []
        for _ in range(self.n_runs):
            W, _ = self.one_run(X)
            sigmas.append(sum_log_std(W))
            random_series.append(self.nmf.random_state)

        sigmas = np.array(sigmas)
        random_series = np.array(random_series)
        n_selected = self.n_selected
        selected_indices = np.argpartition(sigmas, n_selected)[:n_selected]
        selected_random = random_series[selected_indices]

        self.selected_random_states = selected_random
        self.count = 0
        self.prefitted = True

    def fit_transform(self, X):
        if not self.prefitted:
            self.prefit(X)

        self.random_state = iter(self.selected_random_states)

        Hs = super().fit_transform(X, n_runs=self.n_selected)
        return Hs

    
