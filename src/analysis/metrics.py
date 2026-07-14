"""Tie-neutral ranking metrics for virtual-screening validation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy import stats

from . import config





#########  Input Gate  #########

def _arrays(y_true, scores) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.ndim != 1 or s.ndim != 1 or y.size != s.size:
        raise ValueError("labels and scores must be one-dimensional arrays of equal length")
    if y.size == 0:
        raise ValueError("labels and scores must not be empty")
    if not np.isin(y, (0, 1)).all():                                                    # Only 2 Classes, 0 and 1
        raise ValueError("labels must contain only 0 (decoy) and 1 (active)")
    if not np.isfinite(s).all():                                                        # No Binding Score is infinite
        raise ValueError("scores must all be finite before metric calculation")         # and if they are hard stop.
    if not (np.any(y == 1) and np.any(y == 0)):                                         # Both Classes Present
        raise ValueError("both active and decoy labels are required")
    return y, s



######  Sign Flip Normalizer  ######

def decision_values(scores, score_direction: str = config.SCORE_DIRECTION) -> np.ndarray:
    """Map raw scores to a scale on which larger always means better-ranked."""

    s = np.asarray(scores, dtype=float)
    if score_direction == "lower_is_better":
        return -s
    if score_direction == "higher_is_better":
        return s.copy()
    raise ValueError(f"Unknown score direction: {score_direction}")



######  The Tie Handler  #######

def _ranked_blocks(y: np.ndarray, scores: np.ndarray, score_direction: str):        # Sort best to worst and group compounds that are tied
    dv = decision_values(scores, score_direction)                                   #
    order = np.argsort(-dv, kind="mergesort")                                       # Merge/stable sort for reproducibility
    ranked_y = y[order]                                                             # Ties keep og input order
    ranked_dv = dv[order]                                                           #
    starts = np.r_[0, np.flatnonzero(np.diff(ranked_dv)) + 1]                       # Gap between neighbors adn gaps between blocks
    ends = np.r_[starts[1:], y.size]                                                #
    return ranked_y, ranked_dv, starts, ends                                        # starts, ends are index boundries between tied blocks
















###########    ROC      ###########

def roc_curve(y_true, scores, score_direction: str = config.SCORE_DIRECTION):
    """Return tie-collapsed false-positive rate, true-positive rate, and AUC."""

    y, s = _arrays(y_true, scores)
    ranked_y, _, _, ends = _ranked_blocks(y, s, score_direction)
    positives = int(y.sum())
    negatives = int(y.size - positives)
    true_positives = np.cumsum(ranked_y == 1)[ends - 1]                          # Computed at Block Boundries only
    false_positives = np.cumsum(ranked_y == 0)[ends - 1]                         # last index of each tied block via cumulative sum
    tpr = np.r_[0.0, true_positives / positives]    # Origings                   # Each tie block contributes one point ot the curve
    fpr = np.r_[0.0, false_positives / negatives]   # Origins                    # TIES CONTRIBUTE TO THE SAME POINT
    auc = float(np.trapz(tpr, fpr))                 # trapezoidal integration
    return fpr, tpr, auc




















###############       EF Curve           ###############

def enrichment_curve(y_true, scores, score_direction: str = config.SCORE_DIRECTION):        # same logic as ROC
    """Return a tie-collapsed cumulative recovery curve including the origin."""            # with ties and everything

    y, s = _arrays(y_true, scores)
    ranked_y, _, _, ends = _ranked_blocks(y, s, score_direction)
    screened = ends / y.size
    recovered = np.cumsum(ranked_y)[ends - 1] / y.sum()
    return np.r_[0.0, screened], np.r_[0.0, recovered]




########    Tie Blocks in EF    ########

def _expected_actives_in_top_k(
    ranked_y: np.ndarray, starts: np.ndarray, ends: np.ndarray, k: int              # walks tied blocks from best to worst,
) -> float:                                                                         # filling your budget of k slots
    expected = 0.0                                                                  # if budget fills out mid sort,
    remaining = k                                                                   # TAKE A PROPORTIONAL FRACTION
    for start, end in zip(starts, ends):                                            # EF% might be a fraction, not an integer
        if remaining <= 0:
            break
        block_size = int(end - start)
        take = min(remaining, block_size)                                           # Prevent over shoot
        block_actives = int(ranked_y[start:end].sum())
        expected += block_actives * (take / block_size)
        remaining -= take
    return expected



###############       EF        ###############

def enrichment_factor(
    y_true,
    scores,
    fraction: float,
    score_direction: str = config.SCORE_DIRECTION,
) -> float:
    """Fold enrichment in the top fraction, averaged over tied-score orderings."""

    if not 0 < fraction <= 1:
        raise ValueError("enrichment fraction must be in (0, 1]")
    y, s = _arrays(y_true, scores)
    ranked_y, _, starts, ends = _ranked_blocks(y, s, score_direction)
    k = max(1, int(np.ceil(fraction * y.size)))                                         # Cut off fraction is rounded up
    expected_actives = _expected_actives_in_top_k(ranked_y, starts, ends, k)            # Plus some guard rails so atleast one compound is there
    return float((expected_actives / k) / (y.sum() / y.size))                           # Prevents division by 0 on small samples/test runs




#####  EF with some boilerplate for the report  #####

def enrichment_operating_point(
    y_true,
    scores,                                                                                 # EF is also recomputed
    fraction: float,                                                                        # Mild DRY violation
    score_direction: str = config.SCORE_DIRECTION,
) -> tuple[float, float, float]:
    """Return actual screened fraction, expected active recovery, and EF."""

    if not 0 < fraction <= 1:
        raise ValueError("enrichment fraction must be in (0, 1]")
    y, s = _arrays(y_true, scores)
    ranked_y, _, starts, ends = _ranked_blocks(y, s, score_direction)                         # real screened fraction
    k = max(1, int(np.ceil(fraction * y.size)))                                               # for reporting
    expected_actives = _expected_actives_in_top_k(ranked_y, starts, ends, k)
    screened = k / y.size
    recovered = expected_actives / y.sum()
    ef = (expected_actives / k) / (y.sum() / y.size)
    return float(screened), float(recovered), float(ef)














########################               BEDROC            ########################



def bedroc(
    y_true,
    scores,
    alpha: float = config.BEDROC_ALPHA,
    score_direction: str = config.SCORE_DIRECTION,
) -> float:
    """BEDROC with tied compounds averaged across their possible rank positions."""

    if alpha <= 0:
        raise ValueError("BEDROC alpha must be positive dumbass")                       # alpha value for the steepness of the expo
    y, s = _arrays(y_true, scores)
    ranked_y, _, starts, ends = _ranked_blocks(y, s, score_direction)
    total = y.size
    n_actives = int(y.sum())

    rank_weights = np.exp(-alpha * np.arange(1, total + 1) / total)                       # rank_weights is a decaying exponential over rank position
    weighted_active_sum = 0.0
    for start, end in zip(starts, ends):
        block_actives = int(ranked_y[start:end].sum())
        weighted_active_sum += block_actives * float(rank_weights[start:end].mean())      # actives in tieblock get the blocks avg weight
                                                                                          # Not its sorted position weight.
    random_mean_weight = (                                                                # for neutrality
        (1.0 / total)
        * (1.0 - np.exp(-alpha))
        / np.expm1(alpha / total)
    )                                                                                     # This is the published closed-form
    rie = weighted_active_sum / (n_actives * random_mean_weight)                          # BEDROC from Truchon & Bayly (2007), J. Chem. Inf. Model.
    active_fraction = n_actives / total
    factor1 = (                                                                           # Normalization so BEDROC values 
        active_fraction                                                                   # do not depend on the active/decoy ratio
        * np.sinh(alpha / 2.0)                                                            
        / (
            np.cosh(alpha / 2.0)
            - np.cosh(alpha / 2.0 - alpha * active_fraction)
        )
    )
    factor2 = 1.0 / (1.0 - np.exp(alpha * (1.0 - active_fraction)))
    return float(rie * factor1 + factor2)












########## just bookkeeping ##########


def _describe(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"n": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else None,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }




###########      Mann-Whitney and the direction correction     #############

def score_statistics(
    scores_actives,                                                                 # pre split actives and decoys
    scores_decoys,                                                                  # diff input than the ranked metrics
    *,                                                                              # cause ts is just about a 2 sample test abt 2 groups
    alternative: str = config.MANNWHITNEY_ALTERNATIVE,
    score_direction: str = config.SCORE_DIRECTION,
) -> dict:
    """Descriptive statistics, Mann-Whitney test, and directional effect size."""

    active = np.asarray(scores_actives, dtype=float)
    decoy = np.asarray(scores_decoys, dtype=float)
    if not np.isfinite(active).all() or not np.isfinite(decoy).all():
        raise ValueError("score statistics require finite observed scores")
    result = {"actives": _describe(active), "decoys": _describe(decoy)}
    if active.size == 0 or decoy.size == 0:
        result["mannwhitney"] = {
            "u": None,
            "p_value": None,
            "alternative": alternative,
            "probability_active_better": None,
            "rank_biserial": None,
        }
        return result

    test = stats.mannwhitneyu(active, decoy, alternative=alternative, method="auto")      # the test itself
    raw_greater_probability = float(test.statistic / (active.size * decoy.size))          # better over the t test cause ranks dont skew 
    if score_direction == "lower_is_better":                                              # a t test assumes normality (= bad)
        probability_active_better = 1.0 - raw_greater_probability
    elif score_direction == "higher_is_better":                                           # p is computed on raw scores before the sign flip
        probability_active_better = raw_greater_probability                               # 1-raw flips it back
    else:
        raise ValueError(f"Unknown score direction: {score_direction}")
    result["mannwhitney"] = {
        "u": float(test.statistic),
        "p_value": float(test.pvalue),
        "alternative": alternative,
        "probability_active_better": probability_active_better,
        "rank_biserial": 2.0 * probability_active_better - 1.0,
    }
    return result

















#######   Bundles AUC + BEDROC + EF-at-each-fraction into one dict   ####

def summary_metrics(
    y_true,
    scores,
    *,
    ef_fractions: Iterable[float] = config.EF_FRACTIONS,
    alpha: float = config.BEDROC_ALPHA,
    score_direction: str = config.SCORE_DIRECTION,
) -> dict:
    """Calculate the scalar ranking metrics used in the report."""

    _, _, auc = roc_curve(y_true, scores, score_direction)
    result = {
        "auc": auc,
        "bedroc": bedroc(y_true, scores, alpha, score_direction),
    }
    for fraction in ef_fractions:
        key = f"ef_{int(round(float(fraction) * 100))}pct"
        if key in result:
            raise ValueError(f"EF fractions create duplicate output key {key!r}")
        result[key] = enrichment_factor(y_true, scores, float(fraction), score_direction)
    return result















###########      BOOTSTRAP      ###########

def bootstrap_confidence_intervals(
    y_true,
    scores,
    *,
    n_resamples: int = config.BOOTSTRAP_REPLICATES,
    confidence_level: float = config.CONFIDENCE_LEVEL,
    seed: int = config.BOOTSTRAP_SEED,
    ef_fractions: Iterable[float] = config.EF_FRACTIONS,
    alpha: float = config.BEDROC_ALPHA,
    score_direction: str = config.SCORE_DIRECTION,
) -> dict:
    """Class-stratified percentile bootstrap intervals for ranking metrics."""

    if n_resamples < 0:
        raise ValueError("bootstrap replicate count cannot be negative")
    if n_resamples == 0:
        return {}
    if not 0 < confidence_level < 1:
        raise ValueError("confidence level must be in (0, 1)")
                                                                                    # makes sure the sample i have is solid
    y, s = _arrays(y_true, scores)                                                  # by resampling and calcing metrics 1000 times
    active_scores = s[y == 1]                                                       # with duplicates
    decoy_scores = s[y == 0]
    fractions = tuple(float(f) for f in ef_fractions)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {}

    for _ in range(n_resamples):
        active_sample = rng.choice(active_scores, size=active_scores.size, replace=True)
        decoy_sample = rng.choice(decoy_scores, size=decoy_scores.size, replace=True)
        sample_scores = np.r_[active_sample, decoy_sample]
        sample_labels = np.r_[np.ones(active_sample.size, dtype=int), np.zeros(decoy_sample.size, dtype=int)]
        values = summary_metrics(
            sample_labels,
            sample_scores,                                                  # from the resamples
            ef_fractions=fractions,                                         # recompute all metrics
            alpha=alpha,                                                    # one confidence interval per metric
            score_direction=score_direction,
        )
        for key, value in values.items():
            samples.setdefault(key, []).append(value)

    tail = (1.0 - confidence_level) / 2.0
    return {
        key: {
            "low": float(np.quantile(values, tail)),
            "high": float(np.quantile(values, 1.0 - tail)),
        }
        for key, values in samples.items()
    }
