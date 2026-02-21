"""Tests for src/optimizer/score.py"""

import numpy as np
import pytest

from src.optimizer.score import composite_score, DEFAULT_WEIGHTS

CITY_BOUNDS = {
    "south": 30.098,
    "north": 30.516,
    "west": -97.928,
    "east": -97.522,
}

RNG = np.random.default_rng(seed=1)


def _demand():
    d = RNG.random((500, 500), dtype=np.float32)
    d /= d.max()
    return d


def _mask():
    m = np.ones((500, 500), dtype=np.float32)
    m[50:80, 50:80] = 0.0  # Block a region
    return m


def _substations():
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-97.743, 30.267]},
                "properties": {},
            }
        ],
    }


class TestCompositeScore:
    def test_returns_tuple(self):
        result = composite_score(_demand(), _substations(), CITY_BOUNDS)
        assert isinstance(result, tuple) and len(result) == 3

    def test_composite_shape(self):
        composite, _, _ = composite_score(_demand(), _substations(), CITY_BOUNDS)
        assert composite.shape == (500, 500)
        assert composite.dtype == np.float32

    def test_individual_keys(self):
        _, individual, _ = composite_score(_demand(), _substations(), CITY_BOUNDS)
        assert set(individual.keys()) == {"load_relief", "loss_reduction", "sustainability", "redundancy"}

    def test_individual_shapes(self):
        _, individual, _ = composite_score(_demand(), _substations(), CITY_BOUNDS)
        for name, arr in individual.items():
            assert arr.shape == (500, 500), f"{name} has wrong shape"

    def test_timing_keys(self):
        _, _, timings = composite_score(_demand(), _substations(), CITY_BOUNDS)
        expected = {"load_relief", "loss_reduction", "sustainability", "redundancy", "aggregation", "total"}
        assert expected.issubset(set(timings.keys()))

    def test_timing_total_positive(self):
        _, _, timings = composite_score(_demand(), _substations(), CITY_BOUNDS)
        assert timings["total"] > 0

    def test_composite_no_nan(self):
        composite, _, _ = composite_score(_demand(), _substations(), CITY_BOUNDS)
        assert np.isfinite(composite).all()

    def test_composite_range(self):
        composite, _, _ = composite_score(_demand(), _substations(), CITY_BOUNDS)
        assert composite.min() >= -1e-5
        assert composite.max() <= 1.0 + 1e-5

    def test_forbidden_cells_zeroed(self):
        mask = _mask()
        composite, _, _ = composite_score(_demand(), _substations(), CITY_BOUNDS, forbidden_mask=mask)
        # The blocked region should have zero composite score
        blocked_region = composite[50:80, 50:80]
        assert blocked_region.max() < 1e-6, "Forbidden cells must be zeroed in composite"

    def test_no_mask_all_cells_scored(self):
        composite_no_mask, _, _ = composite_score(_demand(), _substations(), CITY_BOUNDS, forbidden_mask=None)
        assert composite_no_mask.max() > 0

    def test_weight_override_positive(self):
        """Custom weights that are valid should still produce a valid result."""
        w = {"load_relief": 0.5, "loss_reduction": 0.3, "sustainability": 0.1, "redundancy": 0.1}
        composite, _, _ = composite_score(_demand(), _substations(), CITY_BOUNDS, weights=w)
        assert np.isfinite(composite).all()

    def test_weight_renormalization(self):
        """Weights that don't sum to 1 should be silently normalized."""
        # weights sum to 2
        w = {"load_relief": 0.7, "loss_reduction": 0.7, "sustainability": 0.3, "redundancy": 0.3}
        composite, _, _ = composite_score(_demand(), _substations(), CITY_BOUNDS, weights=w)
        assert np.isfinite(composite).all()

    def test_invalid_weights_raise(self):
        # All objectives set to zero → total weight = 0 → ValueError
        all_zero = {"load_relief": 0, "loss_reduction": 0, "sustainability": 0, "redundancy": 0}
        with pytest.raises(ValueError):
            composite_score(_demand(), _substations(), CITY_BOUNDS, weights=all_zero)


class TestRunEndToEnd:
    """Light integration test: run the full CLI pipeline in-process."""

    def test_run_produces_json(self, tmp_path):
        import json
        from src.optimizer.run import main

        out = tmp_path / "candidates.json"
        main([
            "--output", str(out),
        ])
        assert out.exists()
        with out.open() as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert 1 <= len(data) <= 10
        required_keys = {"rank", "lat", "lon", "composite_score",
                         "load_relief_score", "loss_reduction_score",
                         "sustainability_score", "redundancy_score"}
        for item in data:
            assert required_keys.issubset(set(item.keys()))

    def test_candidates_unique_coordinates(self, tmp_path):
        import json
        from src.optimizer.run import main

        out = tmp_path / "candidates.json"
        main(["--output", str(out)])
        with out.open() as f:
            data = json.load(f)
        coords = [(c["lat"], c["lon"]) for c in data]
        assert len(coords) == len(set(coords)), "All candidates must have unique coordinates"

    def test_candidates_ranked_descending(self, tmp_path):
        import json
        from src.optimizer.run import main

        out = tmp_path / "candidates.json"
        main(["--output", str(out)])
        with out.open() as f:
            data = json.load(f)
        scores = [c["composite_score"] for c in data]
        assert scores == sorted(scores, reverse=True), "Candidates must be in descending score order"
