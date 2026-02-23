"""Statistical analysis and visualization for B1 experiment results.

Reads results/b1_results.jsonl, computes metrics, prints tables,
generates plots. All statistical functions are pure (take data, return
results) for testability.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Style display order: neutral first, then alphabetical
STYLE_ORDER = ["neutral", "diagnostic", "directive", "suggestive", "accusatory"]
STYLE_COLORS = {
    "neutral": "#888888",
    "diagnostic": "#4477AA",
    "directive": "#CC3311",
    "suggestive": "#228833",
    "accusatory": "#EE7733",
}


# ---------------------------------------------------------------------------
# Data models (immutable results)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RecoveryStats:
    style: str
    recovered: int
    total: int
    rate: float
    ci_lower: float
    ci_upper: float


@dataclass(frozen=True)
class ChiSquareResult:
    chi2: float
    p_value: float
    dof: int
    contingency_table: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class FisherResult:
    style: str
    odds_ratio: float
    p_value_raw: float
    p_value_corrected: float
    significant: bool


@dataclass(frozen=True)
class StepsStats:
    style: str
    mean_steps: float
    se: float
    n: int


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(path: Path) -> list[dict]:
    """Load JSONL results, skipping error rows."""
    records: list[dict] = []
    skipped = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("error") is not None:
                skipped += 1
                continue
            records.append(record)
    if skipped:
        print(f"Warning: skipped {skipped} error rows", file=sys.stderr)
    return records


# ---------------------------------------------------------------------------
# Pure statistical functions
# ---------------------------------------------------------------------------

def wilson_ci(
    successes: int,
    total: int,
    z: float = 1.96,
) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = (
        z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def recovery_rate_by_style(records: list[dict]) -> dict[str, RecoveryStats]:
    """Recovery rate with Wilson 95% CI per style."""
    by_style: dict[str, list[bool]] = {}
    for r in records:
        by_style.setdefault(r["style"], []).append(r["recovered"])
    result = {}
    for style, outcomes in by_style.items():
        total = len(outcomes)
        recovered = sum(outcomes)
        rate = recovered / total if total else 0.0
        ci_lo, ci_hi = wilson_ci(recovered, total)
        result[style] = RecoveryStats(
            style=style,
            recovered=recovered,
            total=total,
            rate=rate,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
        )
    return result


def chi_square_test(records: list[dict]) -> ChiSquareResult:
    """Chi-square test on 5x2 contingency table (styles x outcome)."""
    by_style: dict[str, list[bool]] = {}
    for r in records:
        by_style.setdefault(r["style"], []).append(r["recovered"])

    # Build table in consistent order
    table = []
    for style in STYLE_ORDER:
        if style not in by_style:
            continue
        outcomes = by_style[style]
        recovered = sum(outcomes)
        not_recovered = len(outcomes) - recovered
        table.append([recovered, not_recovered])

    observed = np.array(table)
    chi2, p, dof, _ = stats.chi2_contingency(observed)
    contingency = tuple(tuple(int(x) for x in row) for row in observed)
    return ChiSquareResult(
        chi2=float(chi2),
        p_value=float(p),
        dof=int(dof),
        contingency_table=contingency,
    )


def pairwise_fisher_vs_neutral(records: list[dict]) -> list[FisherResult]:
    """Fisher exact test for each non-neutral style vs neutral.

    Bonferroni correction: 4 comparisons.
    """
    by_style: dict[str, list[bool]] = {}
    for r in records:
        by_style.setdefault(r["style"], []).append(r["recovered"])

    neutral = by_style.get("neutral", [])
    n_rec_neutral = sum(neutral)
    n_not_neutral = len(neutral) - n_rec_neutral

    results = []
    n_comparisons = len(STYLE_ORDER) - 1  # 4
    for style in STYLE_ORDER:
        if style == "neutral":
            continue
        outcomes = by_style.get(style, [])
        n_rec = sum(outcomes)
        n_not = len(outcomes) - n_rec

        table_2x2 = [[n_rec, n_not], [n_rec_neutral, n_not_neutral]]
        odds_ratio, p_raw = stats.fisher_exact(table_2x2)
        p_corrected = min(1.0, p_raw * n_comparisons)
        results.append(FisherResult(
            style=style,
            odds_ratio=float(odds_ratio),
            p_value_raw=float(p_raw),
            p_value_corrected=float(p_corrected),
            significant=p_corrected < 0.05,
        ))
    return results


def mean_steps_by_style(records: list[dict]) -> dict[str, StepsStats]:
    """Mean steps to recovery (recovered runs only) per style."""
    by_style: dict[str, list[int]] = {}
    for r in records:
        if r["recovered"]:
            by_style.setdefault(r["style"], []).append(r["steps"])
    result = {}
    for style, step_counts in by_style.items():
        n = len(step_counts)
        mean = sum(step_counts) / n if n else 0.0
        se = (
            (sum((s - mean) ** 2 for s in step_counts) / (n - 1)) ** 0.5 / n**0.5
            if n > 1 else 0.0
        )
        result[style] = StepsStats(style=style, mean_steps=mean, se=se, n=n)
    return result


def loop_rate_by_style(records: list[dict]) -> dict[str, float]:
    """Proportion of runs with loop_count > 0 per style."""
    counts: dict[str, list[bool]] = {}
    for r in records:
        counts.setdefault(r["style"], []).append(r["loop_count"] > 0)
    return {
        style: sum(flags) / len(flags) if flags else 0.0
        for style, flags in counts.items()
    }


def action_type_distribution(records: list[dict]) -> dict[str, dict[str, float]]:
    """Normalized action type frequency per style."""
    by_style: dict[str, Counter] = {}
    for r in records:
        counter = by_style.setdefault(r["style"], Counter())
        for at in r["action_types"]:
            counter[at] += 1
    result = {}
    for style, counter in by_style.items():
        total = sum(counter.values())
        result[style] = {
            at: count / total if total else 0.0
            for at, count in counter.items()
        }
    return result


def cramers_v(chi2: float, n: int, k: int) -> float:
    """Cramer's V effect size. k = min(rows, cols) of contingency table."""
    if n == 0 or k <= 1:
        return 0.0
    return math.sqrt(chi2 / (n * (k - 1)))


def per_task_breakdown(records: list[dict]) -> dict[str, dict[str, float]]:
    """Recovery rate per task x style."""
    by_task_style: dict[str, dict[str, list[bool]]] = {}
    for r in records:
        task_dict = by_task_style.setdefault(r["task"], {})
        task_dict.setdefault(r["style"], []).append(r["recovered"])
    result = {}
    for task, style_dict in by_task_style.items():
        result[task] = {
            style: sum(outcomes) / len(outcomes) if outcomes else 0.0
            for style, outcomes in style_dict.items()
        }
    return result


# ---------------------------------------------------------------------------
# Output: printing
# ---------------------------------------------------------------------------

def print_summary(records: list[dict]) -> None:
    """Print full statistical summary to stdout."""
    print(f"\nTotal records: {len(records)}\n")

    # Recovery rate
    print("=" * 60)
    print("Recovery Rate by Style")
    print("=" * 60)
    rr = recovery_rate_by_style(records)
    for style in STYLE_ORDER:
        if style not in rr:
            continue
        s = rr[style]
        print(
            f"  {style:<12s}  {s.rate:5.1%}  "
            f"[{s.ci_lower:.3f}, {s.ci_upper:.3f}]  "
            f"({s.recovered}/{s.total})"
        )

    # Chi-square
    print("\n" + "=" * 60)
    print("Chi-Square Test (5 styles)")
    print("=" * 60)
    cs = chi_square_test(records)
    n_total = sum(sum(row) for row in cs.contingency_table)
    v = cramers_v(cs.chi2, n_total, min(len(cs.contingency_table), 2))
    print(f"  chi2 = {cs.chi2:.4f},  p = {cs.p_value:.6f},  dof = {cs.dof}")
    print(f"  Cramer's V = {v:.4f}")
    if cs.p_value < 0.05:
        print("  ** Significant at p < 0.05 **")
    elif cs.p_value < 0.10:
        print("  * Marginal at p < 0.10 *")
    else:
        print("  Not significant (p > 0.10)")

    # Pairwise Fisher
    print("\n" + "=" * 60)
    print("Pairwise Fisher Exact (vs Neutral, Bonferroni corrected)")
    print("=" * 60)
    for fr in pairwise_fisher_vs_neutral(records):
        sig_marker = " *" if fr.significant else ""
        print(
            f"  {fr.style:<12s}  OR={fr.odds_ratio:6.3f}  "
            f"p_raw={fr.p_value_raw:.4f}  "
            f"p_corrected={fr.p_value_corrected:.4f}{sig_marker}"
        )

    # Mean steps
    print("\n" + "=" * 60)
    print("Mean Steps to Recovery (recovered runs only)")
    print("=" * 60)
    ms = mean_steps_by_style(records)
    for style in STYLE_ORDER:
        if style not in ms:
            continue
        s = ms[style]
        print(f"  {style:<12s}  mean={s.mean_steps:.2f}  SE={s.se:.2f}  n={s.n}")

    # Loop rate
    print("\n" + "=" * 60)
    print("Loop Rate by Style (runs with consecutive retries)")
    print("=" * 60)
    lr = loop_rate_by_style(records)
    for style in STYLE_ORDER:
        if style in lr:
            print(f"  {style:<12s}  {lr[style]:5.1%}")

    # Action type distribution
    print("\n" + "=" * 60)
    print("Action Type Distribution")
    print("=" * 60)
    atd = action_type_distribution(records)
    all_types = sorted({at for dist in atd.values() for at in dist})
    header = f"  {'style':<12s}" + "".join(f"  {at:<16s}" for at in all_types)
    print(header)
    for style in STYLE_ORDER:
        if style not in atd:
            continue
        row = f"  {style:<12s}"
        for at in all_types:
            row += f"  {atd[style].get(at, 0.0):14.1%}  "
        print(row)

    # Per-task breakdown
    print("\n" + "=" * 60)
    print("Per-Task Recovery Rate Breakdown")
    print("=" * 60)
    ptb = per_task_breakdown(records)
    header = f"  {'task':<15s}" + "".join(f"  {s:<12s}" for s in STYLE_ORDER)
    print(header)
    for task in sorted(ptb.keys()):
        row = f"  {task:<15s}"
        for style in STYLE_ORDER:
            rate = ptb[task].get(style, 0.0)
            row += f"  {rate:<12.1%}"
        print(row)


# ---------------------------------------------------------------------------
# Output: plotting
# ---------------------------------------------------------------------------

def plot_recovery_rate(
    records: list[dict],
    output_path: Path,
) -> None:
    """Bar chart of recovery rate by style with error bars."""
    rr = recovery_rate_by_style(records)

    styles = [s for s in STYLE_ORDER if s in rr]
    rates = [rr[s].rate for s in styles]
    ci_lower = [rr[s].rate - rr[s].ci_lower for s in styles]
    ci_upper = [rr[s].ci_upper - rr[s].rate for s in styles]
    colors = [STYLE_COLORS.get(s, "#000000") for s in styles]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(styles))
    ax.bar(x, rates, yerr=[ci_lower, ci_upper], color=colors,
           capsize=5, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(styles, fontsize=11)
    ax.set_ylabel("Recovery Rate", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("B1 Recovery Rate by Speech-Act Style", fontsize=13)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze B1 experiment results",
    )
    parser.add_argument(
        "--input", default="results/b1_results.jsonl",
        help="Input JSONL path",
    )
    parser.add_argument(
        "--plot-dir", default="results/",
        help="Directory for output plots",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip plot generation",
    )
    args = parser.parse_args()

    records = load_results(Path(args.input))
    if not records:
        print("No records found.", file=sys.stderr)
        sys.exit(1)

    print_summary(records)

    if not args.no_plot:
        plot_recovery_rate(
            records,
            Path(args.plot_dir) / "b1_recovery_rate.png",
        )


if __name__ == "__main__":
    main()
