from pytest import fixture
import numpy as np
import matplotlib.pyplot as plt  # noqa: F401

from performance_curves.curve import CurveMeta, Curve, Curves


@fixture()
def slope1() -> Curve:
    return Curve(x=np.arange(10),
                 y=np.arange(10),
                 meta=CurveMeta(name="slope-1", x_dim="X", y_dim="Y"))


@fixture()
def random_curve() -> Curve:
    return Curve(x=np.arange(10),
                 y=np.random.randint(0, 10, 10),
                 meta=CurveMeta(kind="rand", x_dim="inches"))


@fixture()
def two_curves(slope1: Curve, random_curve: Curve) -> Curves:
    return Curves(
        curves=[slope1, random_curve],
        kind_overrides={"rand": {"linestyle": "dashed"}},
        curve_overrides={"slope-1": {"color": "r"}}
    )


class TestCurve:
    def test_arrays(self, slope1: Curve):
        assert np.array_equiv(slope1.arrays["x"], np.arange(10))
        assert np.array_equiv(slope1.arrays["y"], np.arange(10))

    # # Uncomment during development to see plots.
    # def test_plot_and_draw_to(self, slope1: Curve, random_curve: Curve):
    #     fig, ax = slope1.plot()
    #     fig, ax = random_curve.draw_to(fig, ax)
    #     plt.show()

    def test_with_meta(self, random_curve: Curve):
        meta = CurveMeta(x_dim="x")
        assert random_curve.with_meta(meta).meta is meta

    def test_without_meta(self, slope1: Curve):
        assert slope1.without_meta().meta is None

    def test_x_cutoff(self, slope1: Curve):
        assert slope1.x_cutoff(y_limit=5.5, is_upper_limit=True) == 5.0
        assert slope1.x_cutoff(y_limit=5.5, is_upper_limit=True, find_min_x=True) == 0.0
        assert slope1.x_cutoff(y_limit=5.5, is_upper_limit=False) == 9.0
        assert slope1.x_cutoff(y_limit=5.5, is_upper_limit=False, find_min_x=True) == 6.0


class TestCurves:

    def test_curves(self, two_curves: Curves):
        assert len(list(two_curves.curves)) == 2

    # # Uncomment during development to see plots.
    # def test_plot(self, two_curves: Curves):
    #     two_curves.plot()
    #     plt.show()
