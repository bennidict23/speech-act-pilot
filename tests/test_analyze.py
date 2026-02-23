"""Tests for analyze.py — statistical correctness against known values.

Uses synthetic data with known properties. NO live LLM required.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from analyze import (
    action_type_distribution,
    chi_square_test,
    cramers_v,
    load_results,
    loop_rate_by_style,
    mean_steps_by_style,
    pairwise_fisher_vs_neutral,
    per_task_breakdown,
    plot_recovery_rate,
    recovery_rate_by_style,
    wilson_ci,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_records() -> list[dict]:
    """100 records: 20 per style, with known recovery rates.

    neutral: 15/20 = 75%
    diagnostic: 18/20 = 90%
    directive: 10/20 = 50%
    suggestive: 14/20 = 70%
    accusatory: 8/20 = 40%
    """
    records = []
    style_rates = [
        ("neutral", 15),
        ("diagnostic", 18),
        ("directive", 10),
        ("suggestive", 14),
        ("accusatory", 8),
    ]
    for style, n_recovered in style_rates:
        for i in range(20):
            recovered = i < n_recovered
            records.append({
                "task": "file_write",
                "style": style,
                "run": i,
                "seed": i,
                "recovered": recovered,
                "steps": 3 if recovered else 5,
                "loop_count": 0 if recovered else 2,
                "actions": ["action1"],
                "action_types": (
                    ["correct_recovery"] if recovered
                    else ["retry", "retry"]
                ),
                "trajectory": None,
                "error": None,
            })
    return records


# ---------------------------------------------------------------------------
# wilson_ci
# ---------------------------------------------------------------------------


class TestWilsonCI:

    def test_known_value(self):
        lo, hi = wilson_ci(15, 20)
        assert 0.50 < lo < 0.75
        assert 0.75 < hi < 1.00

    def test_zero_successes(self):
        lo, hi = wilson_ci(0, 20)
        assert lo >= 0.0
        assert hi > 0.0  # Wilson CI is non-zero even at 0/n

    def test_all_successes(self):
        lo, hi = wilson_ci(20, 20)
        assert lo < 1.0  # Wilson CI is below 1 even at n/n
        assert hi <= 1.0

    def test_zero_total(self):
        assert wilson_ci(0, 0) == (0.0, 0.0)

    def test_bounds(self):
        lo, hi = wilson_ci(10, 20)
        assert 0.0 <= lo <= hi <= 1.0


# ---------------------------------------------------------------------------
# recovery_rate_by_style
# ---------------------------------------------------------------------------


class TestRecoveryRate:

    def test_rates(self, synthetic_records):
        rr = recovery_rate_by_style(synthetic_records)
        assert abs(rr["neutral"].rate - 0.75) < 0.01
        assert abs(rr["diagnostic"].rate - 0.90) < 0.01
        assert abs(rr["directive"].rate - 0.50) < 0.01
        assert abs(rr["suggestive"].rate - 0.70) < 0.01
        assert abs(rr["accusatory"].rate - 0.40) < 0.01

    def test_ci_within_bounds(self, synthetic_records):
        rr = recovery_rate_by_style(synthetic_records)
        for style, stats in rr.items():
            assert 0.0 <= stats.ci_lower <= stats.rate <= stats.ci_upper <= 1.0

    def test_counts(self, synthetic_records):
        rr = recovery_rate_by_style(synthetic_records)
        assert rr["neutral"].total == 20
        assert rr["neutral"].recovered == 15


# ---------------------------------------------------------------------------
# chi_square_test
# ---------------------------------------------------------------------------


class TestChiSquare:

    def test_significant_with_synthetic(self, synthetic_records):
        """With rates from 40% to 90%, chi-square should be significant."""
        result = chi_square_test(synthetic_records)
        assert result.chi2 > 0
        assert result.p_value < 0.05
        assert result.dof == 4  # (5-1) * (2-1)

    def test_contingency_shape(self, synthetic_records):
        result = chi_square_test(synthetic_records)
        assert len(result.contingency_table) == 5
        assert all(len(row) == 2 for row in result.contingency_table)


# ---------------------------------------------------------------------------
# pairwise_fisher
# ---------------------------------------------------------------------------


class TestFisherPairwise:

    def test_four_comparisons(self, synthetic_records):
        results = pairwise_fisher_vs_neutral(synthetic_records)
        assert len(results) == 4

    def test_bonferroni_applied(self, synthetic_records):
        results = pairwise_fisher_vs_neutral(synthetic_records)
        for fr in results:
            assert fr.p_value_corrected >= fr.p_value_raw
            assert fr.p_value_corrected <= 1.0

    def test_styles_covered(self, synthetic_records):
        results = pairwise_fisher_vs_neutral(synthetic_records)
        styles = {fr.style for fr in results}
        assert styles == {"diagnostic", "directive", "suggestive", "accusatory"}


# ---------------------------------------------------------------------------
# mean_steps
# ---------------------------------------------------------------------------


class TestMeanSteps:

    def test_recovered_only(self, synthetic_records):
        ms = mean_steps_by_style(synthetic_records)
        # All recovered runs have steps=3 in the fixture
        for style, stats in ms.items():
            assert abs(stats.mean_steps - 3.0) < 0.01

    def test_n_matches_recovered(self, synthetic_records):
        ms = mean_steps_by_style(synthetic_records)
        assert ms["neutral"].n == 15
        assert ms["diagnostic"].n == 18


# ---------------------------------------------------------------------------
# loop_rate
# ---------------------------------------------------------------------------


class TestLoopRate:

    def test_rates(self, synthetic_records):
        lr = loop_rate_by_style(synthetic_records)
        # Non-recovered runs have loop_count=2, recovered have 0
        # neutral: 5/20 not recovered => loop_rate = 0.25
        assert abs(lr["neutral"] - 0.25) < 0.01
        # accusatory: 12/20 not recovered => loop_rate = 0.60
        assert abs(lr["accusatory"] - 0.60) < 0.01


# ---------------------------------------------------------------------------
# cramers_v
# ---------------------------------------------------------------------------


class TestCramersV:

    def test_zero(self):
        assert cramers_v(0.0, 100, 2) == 0.0

    def test_known_value(self):
        # chi2=10, n=100, k=2 => V = sqrt(10/100) = sqrt(0.1) ≈ 0.316
        v = cramers_v(10.0, 100, 2)
        assert abs(v - math.sqrt(0.1)) < 0.001

    def test_zero_n(self):
        assert cramers_v(10.0, 0, 2) == 0.0


# ---------------------------------------------------------------------------
# per_task_breakdown
# ---------------------------------------------------------------------------


class TestPerTaskBreakdown:

    def test_single_task(self, synthetic_records):
        ptb = per_task_breakdown(synthetic_records)
        assert "file_write" in ptb
        assert abs(ptb["file_write"]["neutral"] - 0.75) < 0.01

    def test_multi_task(self):
        records = [
            {"task": "a", "style": "neutral", "recovered": True},
            {"task": "a", "style": "neutral", "recovered": False},
            {"task": "b", "style": "neutral", "recovered": True},
        ]
        ptb = per_task_breakdown(records)
        assert abs(ptb["a"]["neutral"] - 0.50) < 0.01
        assert abs(ptb["b"]["neutral"] - 1.00) < 0.01


# ---------------------------------------------------------------------------
# load_results
# ---------------------------------------------------------------------------


class TestLoadResults:

    def test_loads_valid(self, tmp_path):
        f = tmp_path / "data.jsonl"
        records = [
            {"task": "a", "style": "neutral", "recovered": True, "error": None},
            {"task": "b", "style": "neutral", "recovered": False, "error": None},
        ]
        f.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        loaded = load_results(f)
        assert len(loaded) == 2

    def test_skips_error_rows(self, tmp_path):
        f = tmp_path / "data.jsonl"
        records = [
            {"task": "a", "style": "neutral", "recovered": True, "error": None},
            {"task": "b", "style": "neutral", "recovered": False, "error": "boom"},
        ]
        f.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        loaded = load_results(f)
        assert len(loaded) == 1


# ---------------------------------------------------------------------------
# plot (smoke test: just verify it creates a file)
# ---------------------------------------------------------------------------


class TestPlot:

    def test_creates_png(self, synthetic_records, tmp_path):
        out = tmp_path / "test_plot.png"
        plot_recovery_rate(synthetic_records, out)
        assert out.exists()
        assert out.stat().st_size > 0
