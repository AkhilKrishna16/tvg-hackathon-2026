"""Tests for src/optimizer/analysis.py"""

import numpy as np
import pytest

from src.optimizer.analysis import (
    coverage_analysis,
    grid_to_latlon,
    nearest_substation_km,
    select_top_candidates,
    sensitivity_analysis,
    _WEIGHT_SETS,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CITY_BOUNDS = {
    "south": 30.098,
    "north": 30.516,
    "west":  -97.928,
    "east":  -97.522,
}

RNG = np.random.default_rng(seed=7)


def _uniform_composite(shape=(500, 500), peak_row=250, peak_col=250):
    """Composite map with a single strong peak and some background noise."""
    m = RNG.uniform(0.0, 0.1, shape).astype(np.float32)
    m[peak_row, peak_col] = 1.0
    return m


def _individual():
    return {
        "load_relief":    np.ones((500, 500), dtype=np.float32) * 0.5,
        "loss_reduction": np.ones((500, 500), dtype=np.float32) * 0.5,
        "sustainability": np.ones((500, 500), dtype=np.float32) * 0.5,
        "redundancy":     np.ones((500, 500), dtype=np.float32) * 0.5,
    }


def _mask_all_placeable():
    return np.ones((500, 500), dtype=np.float32)


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


def _substations_none():
    return {"type": "FeatureCollection", "features": []}


# ---------------------------------------------------------------------------
# grid_to_latlon
# ---------------------------------------------------------------------------

class TestGridToLatlon:
    def test_top_left_is_north_west(self):
        lat, lon = grid_to_latlon(0, 0, CITY_BOUNDS)
        assert abs(lat - CITY_BOUNDS["north"]) < 1e-3
        assert abs(lon - CITY_BOUNDS["west"])  < 1e-3

    def test_bottom_right_is_south_east(self):
        lat, lon = grid_to_latlon(499, 499, CITY_BOUNDS)
        assert abs(lat - CITY_BOUNDS["south"]) < 1e-3
        assert abs(lon - CITY_BOUNDS["east"])  < 1e-3

    def test_centre_is_midpoint(self):
        lat, lon = grid_to_latlon(249, 249, CITY_BOUNDS)
        mid_lat = (CITY_BOUNDS["north"] + CITY_BOUNDS["south"]) / 2
        mid_lon = (CITY_BOUNDS["west"]  + CITY_BOUNDS["east"])  / 2
        assert abs(lat - mid_lat) < 0.01
        assert abs(lon - mid_lon) < 0.01

    def test_returns_tuple(self):
        result = grid_to_latlon(100, 200, CITY_BOUNDS)
        assert isinstance(result, tuple) and len(result) == 2

    def test_lat_decreases_with_row(self):
        lat0, _ = grid_to_latlon(0,   0, CITY_BOUNDS)
        lat1, _ = grid_to_latlon(100, 0, CITY_BOUNDS)
        assert lat0 > lat1, "Latitude should decrease as row index increases (north → south)"

    def test_lon_increases_with_col(self):
        _, lon0 = grid_to_latlon(0, 0,   CITY_BOUNDS)
        _, lon1 = grid_to_latlon(0, 100, CITY_BOUNDS)
        assert lon0 < lon1, "Longitude should increase as col index increases (west → east)"


# ---------------------------------------------------------------------------
# select_top_candidates
# ---------------------------------------------------------------------------

class TestSelectTopCandidates:
    def test_returns_list(self):
        composite = _uniform_composite()
        result = select_top_candidates(composite, _individual(), CITY_BOUNDS)
        assert isinstance(result, list)

    def test_returns_up_to_n_candidates(self):
        composite = _uniform_composite()
        result = select_top_candidates(composite, _individual(), CITY_BOUNDS, n=5)
        assert 1 <= len(result) <= 5

    def test_ranked_descending(self):
        composite = _uniform_composite()
        result = select_top_candidates(composite, _individual(), CITY_BOUNDS)
        scores = [c["composite_score"] for c in result]
        assert scores == sorted(scores, reverse=True), "Candidates must be in descending score order"

    def test_required_keys_present(self):
        composite = _uniform_composite()
        result = select_top_candidates(composite, _individual(), CITY_BOUNDS)
        required = {
            "rank", "lat", "lon", "grid_row", "grid_col", "composite_score",
            "load_relief_score", "loss_reduction_score",
            "sustainability_score", "redundancy_score",
        }
        for c in result:
            assert required.issubset(set(c.keys()))

    def test_anti_clustering_enforced(self):
        """No two candidates should be within min_spacing_cells of each other."""
        composite = _uniform_composite()
        spacing = 20
        result = select_top_candidates(
            composite, _individual(), CITY_BOUNDS, n=5, min_spacing_cells=spacing
        )
        for i, ci in enumerate(result):
            for j, cj in enumerate(result):
                if i == j:
                    continue
                dist = np.sqrt(
                    (ci["grid_row"] - cj["grid_row"]) ** 2
                    + (ci["grid_col"] - cj["grid_col"]) ** 2
                )
                assert dist > spacing, (
                    f"Candidates {i+1} and {j+1} are only {dist:.1f} cells apart "
                    f"(min spacing: {spacing})"
                )

    def test_rank_integers_start_at_one(self):
        composite = _uniform_composite()
        result = select_top_candidates(composite, _individual(), CITY_BOUNDS, n=3)
        ranks = [c["rank"] for c in result]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_all_zero_composite_returns_empty(self):
        composite = np.zeros((500, 500), dtype=np.float32)
        result = select_top_candidates(composite, _individual(), CITY_BOUNDS)
        assert result == []


# ---------------------------------------------------------------------------
# coverage_analysis
# ---------------------------------------------------------------------------

class TestCoverageAnalysis:
    def _make_candidates(self):
        return [
            {"rank": 1, "lat": 30.267, "lon": -97.743,
             "grid_row": 250, "grid_col": 250, "composite_score": 0.9,
             "load_relief_score": 0.9, "loss_reduction_score": 0.9,
             "sustainability_score": 0.5, "redundancy_score": 0.5},
        ]

    def test_adds_coverage_keys(self):
        demand = np.ones((500, 500), dtype=np.float32) / (500 * 500)
        result = coverage_analysis(self._make_candidates(), demand, CITY_BOUNDS)
        assert "coverage_3km_pct" in result[0]
        assert "coverage_5km_pct" in result[0]
        assert "coverage_10km_pct" in result[0]

    def test_coverage_increases_with_radius(self):
        demand = np.ones((500, 500), dtype=np.float32)
        result = coverage_analysis(self._make_candidates(), demand, CITY_BOUNDS)
        c = result[0]
        assert c["coverage_3km_pct"] <= c["coverage_5km_pct"] <= c["coverage_10km_pct"]

    def test_coverage_bounded_0_to_100(self):
        demand = np.ones((500, 500), dtype=np.float32)
        result = coverage_analysis(self._make_candidates(), demand, CITY_BOUNDS)
        for r_km in [3, 5, 10]:
            val = result[0][f"coverage_{r_km}km_pct"]
            assert 0.0 <= val <= 100.0, f"coverage_{r_km}km_pct out of [0, 100]: {val}"

    def test_coverage_with_zero_demand(self):
        demand = np.zeros((500, 500), dtype=np.float32)
        result = coverage_analysis(self._make_candidates(), demand, CITY_BOUNDS)
        for r_km in [3, 5, 10]:
            assert result[0][f"coverage_{r_km}km_pct"] == 0.0

    def test_custom_radii(self):
        demand = np.ones((500, 500), dtype=np.float32)
        result = coverage_analysis(self._make_candidates(), demand, CITY_BOUNDS, radii_km=[2.0, 7.5])
        assert "coverage_2km_pct" in result[0]
        assert "coverage_8km_pct" in result[0]  # 7.5 rounds to 8 in key name... wait, no

    def test_output_length_matches_input(self):
        demand = np.ones((500, 500), dtype=np.float32)
        cands = self._make_candidates() * 3  # 3 candidates
        result = coverage_analysis(cands, demand, CITY_BOUNDS)
        assert len(result) == 3

    def test_preserves_existing_keys(self):
        demand = np.ones((500, 500), dtype=np.float32)
        result = coverage_analysis(self._make_candidates(), demand, CITY_BOUNDS)
        assert result[0]["composite_score"] == 0.9
        assert result[0]["rank"] == 1


# ---------------------------------------------------------------------------
# nearest_substation_km
# ---------------------------------------------------------------------------

class TestNearestSubstationKm:
    def _make_candidates(self):
        return [
            {"rank": 1, "lat": 30.267, "lon": -97.743,
             "grid_row": 250, "grid_col": 250, "composite_score": 0.9,
             "load_relief_score": 0.9, "loss_reduction_score": 0.9,
             "sustainability_score": 0.5, "redundancy_score": 0.5},
        ]

    def test_adds_nearest_existing_km(self):
        result = nearest_substation_km(self._make_candidates(), _substations_few())
        assert "nearest_existing_km" in result[0]

    def test_value_is_non_negative(self):
        result = nearest_substation_km(self._make_candidates(), _substations_few())
        assert result[0]["nearest_existing_km"] >= 0

    def test_none_when_no_substations(self):
        result = nearest_substation_km(self._make_candidates(), _substations_none())
        assert result[0]["nearest_existing_km"] is None

    def test_candidate_at_substation_location_is_very_small(self):
        # Place candidate exactly at one substation's coordinates
        cands = [
            {"rank": 1, "lat": 30.267, "lon": -97.743,
             "grid_row": 0, "grid_col": 0, "composite_score": 0.5,
             "load_relief_score": 0.5, "loss_reduction_score": 0.5,
             "sustainability_score": 0.5, "redundancy_score": 0.5},
        ]
        result = nearest_substation_km(cands, _substations_few())
        assert result[0]["nearest_existing_km"] < 0.1  # essentially 0

    def test_output_length_matches_input(self):
        cands = self._make_candidates() * 4
        result = nearest_substation_km(cands, _substations_few())
        assert len(result) == 4

    def test_preserves_existing_keys(self):
        result = nearest_substation_km(self._make_candidates(), _substations_few())
        assert result[0]["rank"] == 1
        assert result[0]["composite_score"] == 0.9


# ---------------------------------------------------------------------------
# sensitivity_analysis
# ---------------------------------------------------------------------------

class TestSensitivityAnalysis:
    def _make_individual(self, peak_row=150, peak_col=300):
        """Individual arrays with a consistent peak location."""
        arrays = {}
        for k in ("load_relief", "loss_reduction", "sustainability", "redundancy"):
            arr = np.ones((500, 500), dtype=np.float32) * 0.3
            arr[peak_row - 5 : peak_row + 5, peak_col - 5 : peak_col + 5] = 1.0
            arrays[k] = arr
        return arrays

    def test_returns_expected_keys(self):
        result = sensitivity_analysis(
            self._make_individual(), _mask_all_placeable(), CITY_BOUNDS, n=3
        )
        assert "n_weight_sets" in result
        assert "weight_sets" in result
        assert "results_by_set" in result
        assert "stability_map" in result
        assert "stable_cells" in result

    def test_n_weight_sets_matches(self):
        result = sensitivity_analysis(
            self._make_individual(), _mask_all_placeable(), CITY_BOUNDS
        )
        assert result["n_weight_sets"] == len(_WEIGHT_SETS)
        assert len(result["results_by_set"]) == len(_WEIGHT_SETS)

    def test_stability_map_shape(self):
        result = sensitivity_analysis(
            self._make_individual(), _mask_all_placeable(), CITY_BOUNDS
        )
        assert result["stability_map"].shape == (500, 500)

    def test_stability_map_dtype_is_int(self):
        result = sensitivity_analysis(
            self._make_individual(), _mask_all_placeable(), CITY_BOUNDS
        )
        assert np.issubdtype(result["stability_map"].dtype, np.integer)

    def test_stability_counts_bounded(self):
        result = sensitivity_analysis(
            self._make_individual(), _mask_all_placeable(), CITY_BOUNDS
        )
        n_sets = result["n_weight_sets"]
        assert result["stability_map"].max() <= n_sets
        assert result["stability_map"].min() >= 0

    def test_consistent_peak_has_high_stability(self):
        """A cell that scores highly under all objective functions should appear in every set."""
        individual = self._make_individual(peak_row=150, peak_col=300)
        result = sensitivity_analysis(individual, _mask_all_placeable(), CITY_BOUNDS, n=5)
        stable = result["stable_cells"]
        assert len(stable) > 0
        # Top stable cell should have high count
        top_count = stable[0][2]
        assert top_count >= result["n_weight_sets"] // 2

    def test_stable_cells_sorted_descending(self):
        result = sensitivity_analysis(
            self._make_individual(), _mask_all_placeable(), CITY_BOUNDS
        )
        counts = [c for _, _, c in result["stable_cells"]]
        assert counts == sorted(counts, reverse=True)

    def test_results_by_set_structure(self):
        result = sensitivity_analysis(
            self._make_individual(), _mask_all_placeable(), CITY_BOUNDS, n=3
        )
        for r in result["results_by_set"]:
            assert "name"       in r
            assert "weights"    in r
            assert "candidates" in r
            assert isinstance(r["candidates"], list)

    def test_forbidden_cells_excluded(self):
        """Candidates should never appear in forbidden (mask=0) cells."""
        mask = _mask_all_placeable()
        mask[100:400, 100:400] = 0.0  # block large central region
        individual = self._make_individual(peak_row=250, peak_col=250)  # peak in blocked region
        result = sensitivity_analysis(individual, mask, CITY_BOUNDS, n=5)
        for r in result["results_by_set"]:
            for c in r["candidates"]:
                row, col = c["grid_row"], c["grid_col"]
                assert mask[row, col] > 0.5, (
                    f"Candidate at ({row}, {col}) is in a forbidden cell"
                )
