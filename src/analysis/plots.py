"""Static and interactive figures for one validated ranking analysis.

Data comes in already computed from metrics.py


Basically, 

Inputs  - Results
Outputs - Picture of Results

"""

from __future__ import annotations

import contextlib

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

from . import config, metrics

_FONT_STACK = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"


@contextlib.contextmanager
def _mpl_style():
    rc = {
        "font.size": config.FONT_SIZE,
        "axes.titlesize": config.FONT_SIZE + 1,
        "axes.labelsize": config.FONT_SIZE,
        "xtick.labelsize": config.FONT_SIZE - 1,
        "ytick.labelsize": config.FONT_SIZE - 1,
        "legend.fontsize": config.FONT_SIZE - 1,
        "figure.figsize": (config.FIG_WIDTH_IN, config.FIG_HEIGHT_IN),
        "figure.dpi": config.FIG_DPI,
        "savefig.dpi": config.FIG_DPI,
        "savefig.bbox": "tight",
    }
    with plt.style.context(config.MPL_STYLE), mpl.rc_context(rc):
        yield


def _style_plotly(fig: go.Figure, title: str, xaxis: str, yaxis: str) -> go.Figure:
    fig.update_layout(
        template=config.PLOTLY_TEMPLATE,
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis_title=xaxis,
        yaxis_title=yaxis,
        font=dict(family=_FONT_STACK, size=13),
        margin=dict(l=60, r=30, t=55, b=55),
        legend=dict(bgcolor="rgba(255,255,255,0.6)"),
    )
    return fig


def _score_label(score_direction: str) -> str:
    arrow = "← better" if score_direction == "lower_is_better" else "better →"
    return f"Docking score (kcal/mol)  {arrow}"


def _groups(actives, decoys):
    return (
        (np.asarray(decoys, dtype=float), config.COLOR_DECOY, "decoys"),
        (np.asarray(actives, dtype=float), config.COLOR_ACTIVE, "actives"),
    )


def _plot_range(actives: np.ndarray, decoys: np.ndarray) -> tuple[float, float]:
    nonempty = [x for x in (actives, decoys) if x.size]
    if not nonempty:
        return -1.0, 1.0
    lo = min(float(x.min()) for x in nonempty)
    hi = max(float(x.max()) for x in nonempty)
    if lo == hi:
        padding = max(abs(lo) * 0.02, 0.1)
        return lo - padding, hi + padding
    return lo, hi


def _kde_line(values: np.ndarray, grid: np.ndarray):
    if values.size < 2 or np.ptp(values) == 0:
        return None
    try:
        return gaussian_kde(values)(grid)
    except (ValueError, np.linalg.LinAlgError):
        return None


def roc_static(fpr, tpr, auc: float) -> plt.Figure:
    with _mpl_style():
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color=config.COLOR_ACTIVE, lw=1.8, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], color=config.COLOR_NEUTRAL, ls="--", lw=1, label="random")
        ax.set(
            xlabel="False-positive rate",
            ylabel="True-positive rate",
            xlim=(0, 1),
            ylim=(0, 1),
            title="ROC curve",
        )
        ax.legend(loc="lower right", frameon=False)
        fig.tight_layout()
    return fig


def roc_interactive(fpr, tpr, auc: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=fpr, y=tpr, mode="lines", name=f"AUC = {auc:.3f}",
        line=dict(color=config.COLOR_ACTIVE, width=2.5),
        hovertemplate="FPR %{x:.3f}<br>TPR %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="random", hoverinfo="skip",
        line=dict(color=config.COLOR_NEUTRAL, dash="dash", width=1),
    ))
    return _style_plotly(fig, "ROC curve", "False-positive rate", "True-positive rate")


def score_distribution_static(
    actives: np.ndarray, decoys: np.ndarray, score_direction: str
) -> plt.Figure:
    actives = np.asarray(actives, dtype=float)
    decoys = np.asarray(decoys, dtype=float)
    lo, hi = _plot_range(actives, decoys)
    grid = np.linspace(lo, hi, 200)
    bins = np.linspace(lo, hi, 40)
    with _mpl_style():
        fig, ax = plt.subplots()
        for values, color, name in _groups(actives, decoys):
            if not values.size:
                continue
            ax.hist(values, bins=bins, density=True, color=color, alpha=0.4, edgecolor="none")
            kde = _kde_line(values, grid)
            if kde is not None:
                ax.plot(grid, kde, color=color, lw=1.6, label=name)
            else:
                ax.plot([], [], color=color, lw=1.6, label=name)
            ax.axvline(np.mean(values), color=color, ls="--", lw=1)
        ax.set(xlabel=_score_label(score_direction), ylabel="Density", title="Observed score distribution")
        ax.legend(frameon=False)
        fig.tight_layout()
    return fig


def score_distribution_interactive(
    actives: np.ndarray, decoys: np.ndarray, score_direction: str
) -> go.Figure:
    actives = np.asarray(actives, dtype=float)
    decoys = np.asarray(decoys, dtype=float)
    lo, hi = _plot_range(actives, decoys)
    grid = np.linspace(lo, hi, 200)
    fig = go.Figure()
    for values, color, name in _groups(actives, decoys):
        if not values.size:
            continue
        fig.add_trace(go.Histogram(
            x=values, histnorm="probability density", marker_color=color,
            opacity=0.45, name=name, nbinsx=40,
        ))
        kde = _kde_line(values, grid)
        if kde is not None:
            fig.add_trace(go.Scatter(
                x=grid, y=kde, mode="lines", name=f"{name} KDE",
                line=dict(color=color, width=2),
            ))
        fig.add_vline(x=float(np.mean(values)), line=dict(color=color, dash="dash", width=1))
    fig.update_layout(barmode="overlay")
    return _style_plotly(fig, "Observed score distribution", _score_label(score_direction), "Density")


def enrichment_static(frac_screened, frac_found, ef_points) -> plt.Figure:
    """``ef_points`` entries are (requested, actual, recovered, EF)."""

    with _mpl_style():
        fig, ax = plt.subplots()
        ax.plot(frac_screened, frac_found, color=config.COLOR_ACTIVE, lw=1.8, label="observed")
        ax.plot([0, 1], [0, 1], color=config.COLOR_NEUTRAL, ls="--", lw=1, label="random")
        for requested, actual, recovered, ef in ef_points:
            ax.plot([actual], [recovered], "o", color=config.COLOR_ACCENT, ms=4)
            ax.annotate(
                f"EF@{requested:.0%}={ef:.1f}", (actual, recovered),
                textcoords="offset points", xytext=(6, -2),
                fontsize=config.FONT_SIZE - 1, color=config.COLOR_ACCENT,
            )
        ax.set(
            xlabel="Fraction of library screened",
            ylabel="Fraction of actives recovered",
            xlim=(0, 1), ylim=(0, 1), title="Enrichment",
        )
        ax.legend(loc="lower right", frameon=False)
        fig.tight_layout()
    return fig


def enrichment_interactive(frac_screened, frac_found, ef_points) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=frac_screened, y=frac_found, mode="lines", name="observed",
        line=dict(color=config.COLOR_ACTIVE, width=2.5),
        hovertemplate="screened %{x:.1%}<br>actives recovered %{y:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="random", hoverinfo="skip",
        line=dict(color=config.COLOR_NEUTRAL, dash="dash", width=1),
    ))
    if ef_points:
        fig.add_trace(go.Scatter(
            x=[p[1] for p in ef_points], y=[p[2] for p in ef_points], mode="markers",
            marker=dict(color=config.COLOR_ACCENT, size=9), name="EF cutoffs",
            text=[f"EF@{p[0]:.0%} = {p[3]:.2f}<br>actual slice = {p[1]:.2%}" for p in ef_points],
            hovertemplate="%{text}<extra></extra>",
        ))
    return _style_plotly(
        fig, "Enrichment", "Fraction of library screened", "Fraction of actives recovered"
    )


def _ranked(frame: pd.DataFrame, score_direction: str) -> pd.DataFrame:
    result = frame.copy()
    result["decision_value"] = metrics.decision_values(result["score"], score_direction)
    result = result.sort_values(
        ["decision_value", "molecule_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result


def rank_static(frame: pd.DataFrame, score_direction: str) -> plt.Figure:
    ranked = _ranked(frame, score_direction)
    with _mpl_style():
        fig, ax = plt.subplots()
        for label, color, name in ((0, config.COLOR_DECOY, "decoys"), (1, config.COLOR_ACTIVE, "actives")):
            subset = ranked[ranked["label"] == label]
            ax.scatter(
                subset["rank"], subset["score"], s=3, color=color, alpha=0.5,
                edgecolors="none", rasterized=True, label=name,
            )
        imputed_mask = ranked.get("score_imputed", pd.Series(False, index=ranked.index)).astype(bool)
        imputed = ranked[imputed_mask]
        if not imputed.empty:
            ax.scatter(imputed["rank"], imputed["score"], s=12, marker="x", color="black", label="failed; ranked last")
        ax.set(xlabel="Rank (best → worst)", ylabel="Docking score (kcal/mol)", title="Rank-ordered scores")
        ax.legend(frameon=False, markerscale=3)
        fig.tight_layout()
    return fig


def rank_interactive(frame: pd.DataFrame, score_direction: str) -> go.Figure:
    ranked = _ranked(frame, score_direction)
    fig = go.Figure()
    for label, color, name in ((0, config.COLOR_DECOY, "decoys"), (1, config.COLOR_ACTIVE, "actives")):
        subset = ranked[ranked["label"] == label]
        imputed = subset.get("score_imputed", pd.Series(False, index=subset.index)).astype(bool)
        fig.add_trace(go.Scattergl(
            x=subset["rank"], y=subset["score"], mode="markers", name=name,
            marker=dict(
                color=color, size=5, opacity=0.6,
                symbol=np.where(imputed, "x", "circle"),
            ),
            customdata=np.c_[subset["molecule_id"].astype(str), imputed.map({True: "failed; ranked last", False: "observed"})],
            hovertemplate="%{customdata[0]}<br>rank %{x}<br>score %{y:.4f} kcal/mol<br>%{customdata[1]}<extra></extra>",
        ))
    return _style_plotly(fig, "Rank-ordered scores", "Rank (best → worst)", "Docking score (kcal/mol)")


def violin_static(actives: np.ndarray, decoys: np.ndarray) -> plt.Figure:
    groups = [(x, color, name) for x, color, name in _groups(actives, decoys) if x.size]
    with _mpl_style():
        fig, ax = plt.subplots()
        if groups:
            parts = ax.violinplot([g[0] for g in groups], showmeans=False, showmedians=True, showextrema=False)
            for body, (_, color, _) in zip(parts["bodies"], groups):
                body.set_facecolor(color)
                body.set_alpha(0.45)
            parts["cmedians"].set_color(config.COLOR_NEUTRAL)
            ax.set_xticks(range(1, len(groups) + 1))
            ax.set_xticklabels([g[2] for g in groups])
        ax.set(ylabel="Docking score (kcal/mol)", title="Observed score by class")
        fig.tight_layout()
    return fig


def violin_interactive(actives: np.ndarray, decoys: np.ndarray) -> go.Figure:
    fig = go.Figure()
    for values, color, name in _groups(actives, decoys):
        if values.size:
            fig.add_trace(go.Violin(
                y=values, name=name, line_color=color, box_visible=True,
                meanline_visible=True, opacity=0.7,
            ))
    return _style_plotly(fig, "Observed score by class", "", "Docking score (kcal/mol)")


def comparison_bar_static(table: pd.DataFrame, metrics_to_plot=("auc", "bedroc", "ef_1pct", "ef_5pct")) -> plt.Figure:
    columns = [m for m in metrics_to_plot if m in table.columns]
    if not columns:
        raise ValueError("comparison table has none of the requested metrics")
    with _mpl_style():
        fig, axes = plt.subplots(1, len(columns), figsize=(config.FIG_WIDTH_IN * len(columns), config.FIG_HEIGHT_IN))
        axes = [axes] if len(columns) == 1 else axes
        for ax, metric_name in zip(axes, columns):
            colors = [config.METHOD_PALETTE[i % len(config.METHOD_PALETTE)] for i in range(len(table))]
            ax.bar(range(len(table)), table[metric_name].to_numpy(float), color=colors)
            ax.set_xticks(range(len(table)))
            ax.set_xticklabels(table.index, rotation=30, ha="right")
            ax.set_title(metric_name)
        fig.tight_layout()
    return fig


def comparison_bar_interactive(table: pd.DataFrame, metrics_to_plot=("auc", "bedroc", "ef_1pct", "ef_5pct")) -> go.Figure:
    columns = [m for m in metrics_to_plot if m in table.columns]
    if not columns:
        raise ValueError("comparison table has none of the requested metrics")
    fig = make_subplots(rows=1, cols=len(columns), subplot_titles=columns)
    colors = [config.METHOD_PALETTE[i % len(config.METHOD_PALETTE)] for i in range(len(table))]
    for col_index, metric_name in enumerate(columns, 1):
        fig.add_trace(go.Bar(
            x=[str(x) for x in table.index], y=table[metric_name].astype(float),
            marker_color=colors, showlegend=False,
            hovertemplate="%{x}<br>%{y:.4g}<extra></extra>",
        ), row=1, col=col_index)
        fig.update_xaxes(tickangle=-30, row=1, col=col_index)
    fig.update_layout(
        template=config.PLOTLY_TEMPLATE, title=dict(text="Method comparison", x=0.5),
        font=dict(family=_FONT_STACK, size=12), margin=dict(l=50, r=20, t=70, b=90),
    )
    return fig
