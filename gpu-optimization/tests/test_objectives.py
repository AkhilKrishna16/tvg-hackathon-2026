"""Tests for src/optimizer/objectives.py"""

import numpy as np
import pytest

from src.optimizer.objectives import (
    load_relief_score,
    loss_reduction_score,
    redundancy_score,
    sustainability_score,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CITY_BOUNDS = {
    "south": 30.098,
    "north": 30.516,
    "west": -97.928,
    "east": -97.522,
}

RNG = np.random.default_rng(seed=0)


def _demand(shape=(500, 500)):
    d = RNG.random(shape, dtype=np.float32)
    d /= d.max()
    return d


def _substations_none():
    return {"type": "FeatureCollection", "features": []}


def _substations_few():
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-97.743, 30.267]},
                "properties": {},
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-97.720, 30.350]},
                "properties": {},
            },
        ],
    }


# ---------------------------------------------------------------------------
# Shape and dtype
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_load_relief_shape(self):
        s = load_relief_score(_demand(), _substations_few(), CITY_BOUNDS)
        arr = np.asarray(s)
        assert arr.shape == (500, 500)
        assert arr.dtype == np.float32

    def test_loss_reduction_shape(self):
        s = loss_reduction_score(_demand(), _substations_few(), CITY_BOUNDS)
        arr = np.asarray(s)
        assert arr.shape == (500, 500)
        assert arr.dtype == np.float32

    def test_sustainability_shape(self):
        s = sustainability_score(CITY_BOUNDS)
        arr = np.asarray(s)
        assert arr.shape == (500, 500)
        assert arr.dtype == np.float32

    def test_redundancy_shape(self):
        s = redundancy_score(_substations_few(), CITY_BOUNDS)
        arr = np.asarray(s)
        assert arr.shape == (500, 500)
        assert arr.dtype == np.float32


# ---------------------------------------------------------------------------
# Normalization to [0, 1]
# ---------------------------------------------------------------------------

class TestNormalization:
    @pytest.mark.parametrize("fn,args", [
        (load_relief_score,    (_demand(), _substations_few(), CITY_BOUNDS)),
        (loss_reduction_score, (_demand(), _substations_few(), CITY_BOUNDS)),
        (sustainability_score, (CITY_BOUNDS,)),
        (redundancy_score,     (_substations_few(), CITY_BOUNDS)),
    ])
    def test_range(self, fn, args):
        arr = np.asarray(fn(*args))
        assert arr.min() >= -1e-5, f"{fn.__name__} has values < 0"
        assert arr.max() <= 1.0 + 1e-5, f"{fn.__name__} has values > 1"

    def test_load_relief_max_is_one(self):
        arr = np.asarray(load_relief_score(_demand(), _substations_few(), CITY_BOUNDS))
        assert abs(arr.max() - 1.0) < 1e-4

    def test_redundancy_max_is_one(self):
        arr = np.asarray(redundancy_score(_substations_few(), CITY_BOUNDS))
        assert abs(arr.max() - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# Zero-substation robustness
# ---------------------------------------------------------------------------

class TestNoSubstations:
    def test_load_relief_no_crash(self):
        arr = np.asarray(load_relief_score(_demand(), _substations_none(), CITY_BOUNDS))
        assert arr.shape == (500, 500)
        assert np.isfinite(arr).all()

    def test_loss_reduction_no_crash(self):
        arr = np.asarray(loss_reduction_score(_demand(), _substations_none(), CITY_BOUNDS))
        assert arr.shape == (500, 500)
        assert np.isfinite(arr).all()

    def test_redundancy_no_crash(self):
        arr = np.asarray(redundancy_score(_substations_none(), CITY_BOUNDS))
        assert arr.shape == (500, 500)
        assert np.isfinite(arr).all()

    def test_redundancy_uniform_when_no_substations(self):
        arr = np.asarray(redundancy_score(_substations_none(), CITY_BOUNDS))
        assert arr.std() < 1e-5, "Expected uniform score when no substations exist"


# ---------------------------------------------------------------------------
# Monotonicity checks
# ---------------------------------------------------------------------------

class TestMonotonicity:
    def test_redundancy_peak_away_from_substations(self):
        """Redundancy score should be highest roughly 4 km from the substation."""
        # Single substation at city centre
        sub = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-97.725, 30.307]},
                "properties": {},
            }],
        }
        arr = np.asarray(redundancy_score(sub, CITY_BOUNDS))

        # Find the row/col of the single substation in grid coords
        rows, cols = 500, 500
        north, south = CITY_BOUNDS["north"], CITY_BOUNDS["south"]
        west, east   = CITY_BOUNDS["west"],  CITY_BOUNDS["east"]
        sub_row = int((north - 30.307) / (north - south) * (rows - 1))
        sub_col = int((-97.725 - west) / (east - west) * (cols - 1))

        # Score at substation location should NOT be the maximum
        score_at_sub = arr[sub_row, sub_col]
        assert score_at_sub < arr.max(), (
            "Score at substation location should be below the global maximum"
        )

    def test_sustainability_south_higher_than_north(self):
        """Southern rows should have higher sustainability (more solar)."""
        arr = np.asarray(sustainability_score(CITY_BOUNDS))
        south_mean = arr[400:, :].mean()
        north_mean = arr[:100, :].mean()
        assert south_mean > north_mean, "Southern cells should score higher on sustainability"

    def test_sustainability_west_higher_than_east(self):
        arr = np.asarray(sustainability_score(CITY_BOUNDS))
        west_mean = arr[:, :100].mean()
        east_mean = arr[:, 400:].mean()
        assert west_mean > east_mean, "Western cells should score higher on sustainability"


# ---------------------------------------------------------------------------
# NaN / Inf safety
# ---------------------------------------------------------------------------

class TestNumericalSafety:
    @pytest.mark.parametrize("fn,args", [
        (load_relief_score,    (_demand(), _substations_few(), CITY_BOUNDS)),
        (loss_reduction_score, (_demand(), _substations_few(), CITY_BOUNDS)),
        (sustainability_score, (CITY_BOUNDS,)),
        (redundancy_score,     (_substations_few(), CITY_BOUNDS)),
        (load_relief_score,    (_demand(), _substations_none(), CITY_BOUNDS)),
        (redundancy_score,     (_substations_none(), CITY_BOUNDS)),
    ])
    def test_no_nan_or_inf(self, fn, args):
        arr = np.asarray(fn(*args))
        assert np.isfinite(arr).all(), f"{fn.__name__} produced NaN or Inf"

    def test_zero_demand_no_crash(self):
        zero_demand = np.zeros((500, 500), dtype=np.float32)
        arr = np.asarray(load_relief_score(zero_demand, _substations_few(), CITY_BOUNDS))
        assert np.isfinite(arr).all()
        arr2 = np.asarray(loss_reduction_score(zero_demand, _substations_few(), CITY_BOUNDS))
        assert np.isfinite(arr2).all()
