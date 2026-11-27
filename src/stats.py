"""Statistics of the sweeps: confidence intervals over graphs, paired tests, effect sizes, equivalence tests,
multiple-comparison correction."""
import numpy as np
from scipy import stats


def bootstrap_ci(values, level=0.95, resamples=2000, seed=0):
    """Mean and percentile bootstrap confidence interval of the mean: (mean, low, high)."""
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (resamples, len(x)))].mean(axis=1)
    alpha = (1 - level) / 2
    return float(x.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def paired_test(a, b):
    """Two-sided Wilcoxon signed-rank test of a vs b, paired by graph. Returns (mean of a - b, p-value);
    p = 1 when every pair is equal (nothing to test)."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if np.all(d == 0):
        return 0.0, 1.0
    return float(d.mean()), float(stats.wilcoxon(d, zero_method="wilcox").pvalue)


def paired_difference_ci(a, b, level=0.95, resamples=2000, seed=0):
    """Mean of a - b, paired by graph, and its percentile bootstrap confidence interval: (mean, low, high).
    The size of the difference in the unit of the measure, to read next to the p-value."""
    return bootstrap_ci(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64), level, resamples, seed)


def rank_biserial(a, b):
    """Matched-pairs rank-biserial correlation of a vs b (Kerby 2014): (R+ - R-) / (R+ + R-), where R+ and R-
    are the sums of the ranks of |a - b| over the pairs where a > b and a < b; equal pairs are dropped, as in
    the Wilcoxon test. In [-1, 1]: -1 when a is lower on every graph, 0 when neither side dominates. It does
    not grow with the number of graphs, unlike the p-value. Usual reading: 0.1 small, 0.3 medium, 0.5 large."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())


def equivalence_test(a, b, margin):
    """Two one-sided Wilcoxon signed-rank tests (TOST) that a - b lies within (-margin, +margin), paired by
    graph. Returns the p-value, the larger of the two one-sided ones: small means the two methods are
    equivalent up to the margin. "No significant difference" is not evidence of equivalence; this is."""
    if margin <= 0:
        raise ValueError("margin must be positive")
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    above = stats.wilcoxon(d + margin, alternative="greater").pvalue  # H0: a - b <= -margin
    below = stats.wilcoxon(d - margin, alternative="less").pvalue     # H0: a - b >= +margin
    return float(max(above, below))


def holm(p_values):
    """Holm-Bonferroni adjusted p-values (same order as given): control of the family-wise error rate when
    many settings are tested at once."""
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted.tolist()
