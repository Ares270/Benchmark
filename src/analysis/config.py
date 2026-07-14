"""Choices for the DYRK1A ranking analysis.

This module contains policy.  Every value used by the
analysis is passed explicitly into the functions that depend on it.
"""

from __future__ import annotations

# Input contract emitted by ``src.harness.dock``.
ID_COLUMN = "molecule_id"
SCORE_COLUMN = "score_kcal_mol"
REFERENCE_ID_COLUMN = "molecule_chembl_id"

# Smina affinity is reported in kcal/mol.
SCORE_DIRECTION = "lower_is_better"
VALID_SCORE_DIRECTIONS = ("lower_is_better", "higher_is_better")

# ``error`` does not tolerate missing values
MISSING_SCORE_POLICY = "error"
VALID_MISSING_SCORE_POLICIES = ("error", "exclude", "rank_last")

# Early-recognition operating points and BEDROC weighting.
EF_FRACTIONS = (0.01, 0.05, 0.10)
BEDROC_ALPHA = 20.0

# The two-sided test detects separation in either direction.  Direction and
# effect size are reported independently so a small p-value cannot disguise an
# inverted ranking.
MANNWHITNEY_ALTERNATIVE = "two-sided"

# Compound-level, class-stratified bootstrap.  These intervals quantify
# sampling variability conditional on this benchmark; they do not capture
# docking stochasticity or chemical-series dependence.
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20260711
CONFIDENCE_LEVEL = 0.95

# Shared figure/output configuration.
COLOR_ACTIVE = "#2563eb"
COLOR_DECOY = "#dc2626"
COLOR_NEUTRAL = "#6b7280"
COLOR_ACCENT = "#059669"
METHOD_PALETTE = [
    "#2563eb", "#dc2626", "#059669", "#d97706",
    "#7c3aed", "#0891b2", "#be185d", "#4b5563",
]

MPL_STYLE = "seaborn-v0_8-whitegrid"
FIG_DPI = 300
FIG_WIDTH_IN = 3.5
FIG_HEIGHT_IN = 2.8
FONT_SIZE = 8
PLOTLY_TEMPLATE = "plotly_white"

FIG_SUBDIR = "figures"
INTERACTIVE_SUBDIR = "interactive"
REPORT_NAME = "report.html"
METRICS_NAME = "metrics.json"
RUN_LOG_NAME = "run_log.json"
RUN_HISTORY_NAME = "run_history.csv"
OUTPUT_SCHEMA_VERSION = 2
