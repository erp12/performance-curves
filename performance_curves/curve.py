"""Utilities for representing and manipulating curves.

In the context of the `performance_curves` library, a "curve" is a collection of x-y data points.

"""
from operator import le, ge
from dataclasses import dataclass
from typing import Tuple, Optional, Iterable, Any, Mapping, Dict, Set

import matplotlib.pyplot as plt
import numpy as np

from performance_curves._protocols import Plottable, PlotElement
from performance_curves._utils import SynchronizedArrays


@dataclass
class CurveMeta(PlotElement):
    # noinspection PyUnresolvedReferences
    """Metadata about a curve.

    Attributes:
        name: The logical identifier of the curve.
        kind: The kind of curve.
        x_dim: The name of the x-dimension for the points on the curve.
        y_dim: The name of the y-dimension for the points on the curve.
    """
    name: Optional[Any] = None
    kind: Optional[Any] = None
    x_dim: Optional[str] = None
    y_dim: Optional[str] = None

    def draw_to(self, fig: plt.Figure, ax: plt.Axes) -> Tuple[plt.Figure, plt.Axes]:
        """Stylize a plot by adding the curve metadata.

        Args:
            fig: Plot figure.
            ax: Plot axes.

        Returns:
            The updated figure and axes of the plot.
        """
        if self.x_dim is not None:
            ax.set_xlabel(self.x_dim)
        if self.y_dim is not None:
            ax.set_ylabel(self.y_dim)
        return fig, ax


class Curve(Plottable, PlotElement):
    """The data that makes up a plottable curve."""

    def __init__(self, x: np.ndarray, y: np.ndarray, meta: Optional[CurveMeta] = None):
        """Instantiate a `Curve`.

        Args:
            x: Array of values for the x-dimension.
            y: Array of values for the y-dimension. Should correspond to `x` position-wise.
            meta: Optional curve metadata.
        """
        self.x = x
        self.y = y
        self.meta = meta

    @property
    def arrays(self) -> SynchronizedArrays:
        """The underlying arrays that make up the points of the curve."""
        return SynchronizedArrays(arrays={"x": self.x, "y": self.y})

    def draw_line(self, fig: plt.Figure, ax: plt.Axes, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Draw the curve as a line on the given plot.

        @todo Add links to matplotlib documentation for the style overrides.
        @todo Should this method be public or private?

        Args:
            fig: The figure of the plot.
            ax: The axes of the plot.
            **kwargs: Additional plotting options to pass to `.plot` of matplotlib.

        Returns:
            The updated figure and axes of the plot.
        """
        if self.meta is None:
            ax.plot(self.x, self.y, **kwargs)
        else:
            ax.plot(self.x, self.y, label=self.meta.name, **kwargs)
        return fig, ax

    def draw_to(self, fig: plt.Figure, ax: plt.Axes) -> Tuple[plt.Figure, plt.Axes]:
        """Draws the curve on the given plot."""
        return self.draw_line(fig, ax)

    def plot(self) -> Tuple[plt.Figure, plt.Axes]:
        """Create a new plot with the curve on it."""
        fig, ax = plt.subplots()
        if self.meta is not None:
            fig, ax = self.meta.draw_to(fig, ax)
        return self.draw_to(fig, ax)

    def with_meta(self, meta: CurveMeta) -> "Curve":
        """A new `Curve` with the same data and new metadata."""
        return Curve(self.x, self.y, meta)

    def without_meta(self) -> "Curve":
        """A new `Curve` with the same data and no metadata."""
        return Curve(
            self.x,
            self.y,
        )

    def x_cutoff(self, y_limit: float, is_upper_limit: bool = False, find_min_x: bool = False) -> float:
        """Finds the x-dimension value of the closest data point to the given y-limit."""
        # @todo Test on non-monotonic curves.
        # @todo Better docstring.
        # @todo Consider adding support for configurable interpolation between data points.
        pred = le if is_upper_limit else ge
        selector = np.min if find_min_x else np.max
        return selector(self.arrays.filter("y", lambda a: pred(a, y_limit))["x"])


class Curves(Plottable):
    """A collection of `Curve` objects that can be plotted together."""

    def __init__(
        self,
        curves: Iterable[Curve],
        *,
        kind_overrides: Optional[Mapping[Any, Mapping[str, Any]]] = None,
        curve_overrides: Optional[Mapping[Any, Mapping[str, Any]]] = None
    ):
        """Instantiate a `Curves` object from a collection of `Curve` objects and some optional style overrides.

        @todo Add links to matplotlib documentation for the style overrides.

        Args:
            curves: The `Curve` objects.
            kind_overrides: Plot style overrides to apply to curves based their `kind`.
            curve_overrides: Plot style overrides to apply to curves based their `name`.
                Takes priority over `kind_overrides`.
        """
        # @todo Validate that all curves have unique names.
        self.curves = curves
        self.kind_overrides = {} if kind_overrides is None else kind_overrides
        self.curve_overrides = {} if curve_overrides is None else curve_overrides

    def plot(self) -> Tuple[plt.Figure, plt.Axes]:
        """Create a new plot that shows all curves."""
        # Save the distinct set of axis labels to derive final labels.
        x_labs: Set[str] = set()
        y_labs: Set[str] = set()

        fig, ax = plt.subplots()
        for curve in self.curves:
            overrides: Dict[str, Any] = {}
            meta = curve.meta
            if meta is not None:
                overrides.update(self.kind_overrides.get(meta.kind, {}))
                overrides.update(self.curve_overrides.get(meta.name, {}))
                if meta.x_dim is not None:
                    x_labs.add(meta.x_dim)
                if meta.y_dim is not None:
                    y_labs.add(meta.y_dim)
            fig, ax = curve.without_meta().draw_line(fig, ax, **overrides)

        ax.set_xlabel(", ".join(x_labs))
        ax.set_ylabel(", ".join(y_labs))

        return fig, ax
