import matplotlib.pyplot as plt
import seaborn as sns
from numpy.typing import ArrayLike


class ChemistryHeatmap:
    """
    Plot a heatmap of endmember chemistry.
    """

    def __init__(self, ax=None) -> None:
        if ax is None:
            ax = plt.gca()
        self.ax = ax

    def plot(self, data: ArrayLike, xlabels: ArrayLike) -> None:
        """
        Plot the heatmap.
        """
        sns.heatmap(data, annot=True, cmap="YlGnBu", xticklabels=xlabels)
