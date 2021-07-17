"""Protocols and abstract base classes to help in the development in performance_curves."""
from typing import Tuple
from abc import ABC
import matplotlib.pyplot as plt


# @todo Consider using Protocols once support for Python 3.6 is dropped.


class Plottable(ABC):
    """A protocol that specifies extenders can be used to generate a plot."""

    def plot(self) -> Tuple[plt.Figure, plt.Axes]:
        """Create a new plot from the plottable object."""
        ...


class PlotElement(ABC):
    """A protocol that specifies extenders can be added to an existing plot."""

    def draw_to(self, fig: plt.Figure, ax: plt.Axes) -> Tuple[plt.Figure, plt.Axes]:
        """Draw the plot element on the given plot."""
        ...
