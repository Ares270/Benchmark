"""Assemble a self-contained, explicitly qualified HTML analysis report."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from jinja2 import Environment
from markupsafe import Markup
from plotly.offline import get_plotlyjs

from .report_theme import plotly_config, polish_plotly_figure, report_css, report_toolbar


PLOT_LAYOUT = [
    ("roc", "ROC curve", "Global ranking discrimination; ties are handled as score thresholds."),
    ("score_distribution", "Observed score distribution", "Failed dockings are excluded from this distribution even when ranked last."),
    ("enrichment", "Enrichment curve", "Early active recovery; tied blocks are averaged rather than ordered by input position."),
    ("rank", "Rank-ordered scores", "Every analyzed row; crosses denote missing scores ranked last when that policy is selected."),
    ("violin", "Observed score by class", "Distribution summary for finite docking scores only."),
]

CHEMISTRY_PLOT_LAYOUT = [
    ("chemical_distributions", "Parent-property distributions", "Each property keeps its own units and axis; QED and SA are descriptive, not filters."),
    ("chemical_landscape", "All accepted parents in chemical space", "Every accepted parent appears once; hover over a point for TPSA, QED, SA, H-bond, and flexibility details."),
    ("score_property_correlations", "Score–property correlations", "Spearman correlations use observed dockings only. Correlation is diagnostic of scoring bias, not evidence of causation."),
]

COHORTS = ("actives", "decoys")

# Columns of the harness-configuration table, in display order.
HARNESS_CONFIG_DISPLAY = (
    "receptor_path",
    "receptor_sha256",
    "box_center",
    "box_size",
    "exhaustiveness",
    "seed",
    "num_modes",
    "cpu_per_job",
    "num_workers",
    "smina_version",
    "scoring_function",
)

# Scoring actives and decoys under different values of any of these makes the
# two cohorts incomparable, which invalidates every ranking metric derived
# from them. Differences here are fatal, not cosmetic.
HARNESS_CONFIG_COMPARED = (
    "receptor_sha256",
    "box_center",
    "box_size",
    "exhaustiveness",
    "scoring_function",
)

SIZE_CORRELATION_ROWS = (
    ("pooled", "All analyzed molecules"),
    ("actives", "actives"),
    ("decoys", "decoys"),
)

_ABSENT = object()


class HarnessConfigurationMismatch(RuntimeError):
    """Raised when actives and decoys were not docked under one configuration."""



_TEMPLATE = Environment(autoescape=True).from_string(r"""
<div class="report">
{{ toolbar }}

<h1>DYRK1A docking benchmark — {{ meta.name }}</h1>
<p class="sub">{{ meta.timestamp_iso }} · {{ audit.n_analyzed_actives }} actives · {{ audit.n_analyzed_decoys }} benchmark decoys · missing policy: <strong>{{ audit.missing_policy }}</strong></p>
<div class="warning"><strong>Scope:</strong> these results measure ranking of supplied benchmark actives against supplied property-matched decoys. Decoys are presumed negatives, not experimentally confirmed non-binders. Bootstrap intervals are conditional on compound-level resampling and do not include docking stochasticity or chemical-series dependence.</div>

<div class="tiles">
{% for row in metric_rows %}
  <div class="tile"><div class="label">{{ row.label }}</div><div class="value">{{ row.value }}</div>{% if row.interval %}<div class="ci">{{ confidence_label }} CI {{ row.interval }}</div>{% endif %}</div>
{% endfor %}
</div>

<h2>Dataset audit</h2>
<table class="kv"><tr><th>Cohort</th><th class="num">Input</th><th class="num">Scored</th><th class="num">Missing</th><th class="num">Coverage</th></tr>
{% for key in ['actives','decoys'] %}{% set group=audit[key] %}<tr><td>{{ key }}</td><td class="num">{{ group.n_input }}</td><td class="num">{{ group.n_scored }}</td><td class="num">{{ group.n_missing_score }}</td><td class="num">{{ '%.2f%%'|format(100*group.coverage) }}</td></tr>{% endfor %}</table>
{% if audit.reference %}<p class="note">Active-reference check: {{ audit.reference.n_reference_ids }} IDs; {{ audit.reference.n_reference_ids_not_in_scored_actives }} reference actives were not present in this scored-active input. This may reflect the DUDE-Z-compatible active subset.</p>{% endif %}

{% if chemistry %}                      ##### This section appears only when chemistry inputs were supplied
<h2>Chemical-property profile</h2>      ##### It explains the exact parent policy, no tautomer/stereo enumeration, and no QED/SA filtering.
<p class="note"><strong>Evaluated structure:</strong> the raw submitted SMILES remains in the intake audit, but descriptors and docking preparation use only the largest disconnected component. No tautomers, stereoisomers, racemates, or protonation states are enumerated; the submitted parent form is used as-is. QED and SA are descriptive outputs, never hard filters or an “overall score.”</p>
<table class="kv"><tr><th>Cohort</th><th class="num">Submitted</th><th class="num">RDKit-valid</th><th class="num">Unique parents</th><th class="num">Accepted</th><th class="num">Parent extracted</th><th class="num">Validity</th><th class="num">Parent uniqueness</th><th class="num">Intake wall (s)</th></tr>
{% for key in ['actives','decoys'] %}{% set group=chemistry.intake_audit[key] %}<tr><td>{{ key }}</td><td class="num">{{ group.submitted_rows }}</td><td class="num">{{ group.valid_structures }}</td><td class="num">{{ group.unique_valid_parents }}</td><td class="num">{{ group.accepted_for_preparation }}</td><td class="num">{{ group.parent_extractions }}</td><td class="num">{{ '%.2f%%'|format(100*group.validity) }}</td><td class="num">{{ '%.2f%%'|format(100*group.parent_uniqueness) }}</td><td class="num">{{ '%.2f'|format(group.wall_seconds) }}</td></tr>{% endfor %}</table>
<p class="note">Property distributions cover every accepted parent, including any molecule that later failed 3D preparation or docking. Score–property correlations use observed docking scores only.</p>

<table class="kv"><tr><th>Property</th><th class="num">Active mean</th><th class="num">Active median</th><th class="num">Decoy mean</th><th class="num">Decoy median</th><th class="num">Mean difference<br>active − decoy</th></tr>
{% for row in chemistry_rows %}<tr><td>{{ row.label }}</td><td class="num">{{ 'n/a' if row.active_mean is none else '%.3f'|format(row.active_mean) }}</td><td class="num">{{ 'n/a' if row.active_median is none else '%.3f'|format(row.active_median) }}</td><td class="num">{{ 'n/a' if row.decoy_mean is none else '%.3f'|format(row.decoy_mean) }}</td><td class="num">{{ 'n/a' if row.decoy_median is none else '%.3f'|format(row.decoy_median) }}</td><td class="num">{{ 'n/a' if row.mean_difference is none else '%+.3f'|format(row.mean_difference) }}</td></tr>{% endfor %}</table>

<h2>Chemical-property figures</h2>
{% for key,title,caption in chemistry_plot_layout %}{% if key in plot_divs %}<div class="plotcard"><div class="cap"><strong>{{ title }}</strong> — {{ caption }}</div>{{ plot_divs[key] }}</div>{% endif %}{% endfor %}
{% endif %}


<h2>Observed score statistics</h2>
<table class="kv"><tr><th>Group</th><th class="num">n</th><th class="num">Mean</th><th class="num">Median</th><th class="num">SD</th><th class="num">Min</th><th class="num">Max</th></tr>
{% for key in ['actives','decoys'] %}{% set group=stats[key] %}<tr><td>{{ key }}</td><td class="num">{{ group.n }}</td>{% for field in ['mean','median','std','min','max'] %}<td class="num">{{ 'n/a' if group[field] is none else '%.3f'|format(group[field]) }}</td>{% endfor %}</tr>{% endfor %}</table>
<p class="note">Mann–Whitney U={{ mw.u if mw.u is not none else 'n/a' }}, {{ mw.alternative }} p={{ 'n/a' if mw.p_value is none else '%.3g'|format(mw.p_value) }}. Probability that a randomly selected active ranks ahead of a randomly selected decoy={{ 'n/a' if mw.probability_active_better is none else '%.4f'|format(mw.probability_active_better) }}. The p-value is evidence about distributional separation, not its practical magnitude or direction.</p>

{% if size %}
<h2>Size dependence</h2>
<p class="note">Heavy atom count is recomputed with RDKit from the same <code>.smi</code> inputs that were prepared and docked. It tests whether the docking score is tracking molecular size rather than binding chemistry.</p>
{% if 'size_dependence' in plot_divs %}<div class="plotcard"><div class="cap"><strong>Docking score vs heavy atom count</strong> — one point per molecule with an observed docking score.</div>{{ plot_divs['size_dependence'] }}</div>{% endif %}
<p class="note">{{ size.n_correlated }} of {{ size.n_analyzed }} analyzed molecules are plotted and correlated; {{ size.n_excluded_missing_score }} molecule(s) with a missing or failed score are excluded.</p>

<table class="kv"><tr><th>Set</th><th class="num">n</th><th class="num">Spearman ρ</th><th class="num">p (two-sided)</th><th class="num">Pearson r</th><th class="num">p (two-sided)</th></tr>
{% for row in size_correlation_rows %}<tr><td>{{ row.label }}</td><td class="num">{{ row.n }}</td><td class="num">{{ row.spearman_rho }}</td><td class="num">{{ row.spearman_p_value }}</td><td class="num">{{ row.pearson_r }}</td><td class="num">{{ row.pearson_p_value }}</td></tr>{% endfor %}</table>
<p class="note">Spearman is the primary statistic; it needs only a monotone relationship. Pearson is reported second and assumes linearity.</p>

<table class="kv"><tr><th>Cohort</th><th class="num">n</th><th class="num">Mean heavy atoms</th><th class="num">Median</th><th class="num">SD</th></tr>
{% for key in ['actives','decoys'] %}{% set group=size.heavy_atom_counts[key] %}<tr><td>{{ key }}</td><td class="num">{{ group.n }}</td><td class="num">{{ 'n/a' if group.mean is none else '%.2f'|format(group.mean) }}</td><td class="num">{{ 'n/a' if group.median is none else '%.2f'|format(group.median) }}</td><td class="num">{{ 'n/a' if group.std is none else '%.2f'|format(group.std) }}</td></tr>{% endfor %}
<tr><td colspan="4">Standardized mean difference (actives − decoys) / pooled SD</td><td class="num">{{ 'n/a' if size.standardized_mean_difference_actives_minus_decoys is none else '%+.3f'|format(size.standardized_mean_difference_actives_minus_decoys) }}</td></tr></table>
<p class="note">Heavy atom counts cover every analyzed molecule, including any whose docking failed; the correlations above cover only observed scores.</p>
<p class="note"><strong>Scope:</strong> a correlation between docking score and heavy atom count is a property of the scoring function and of these particular compound sets, and does not by itself establish that any individual ranking is wrong.</p>
{% endif %}

<h2>Harness configuration</h2>
<p class="note">Recorded by the docking harness when each cohort was docked and read back from the <code>_dock_summary.json</code> files whose SHA-256 hashes match the analyzed score tables. Nothing in this table is read from the harness configuration as it stands at analysis time.</p>
<table class="kv"><tr><th>Cohort</th><th>Receptor</th><th>Receptor SHA-256</th><th>Box centre (Å)</th><th>Box size (Å)</th><th class="num">Exhaustiveness</th><th class="num">Seed</th><th class="num">Modes</th><th class="num">CPU/job</th><th class="num">Workers</th><th>Smina</th><th>Scoring</th></tr>
{% for row in harness_config_rows %}{% if row.recorded %}<tr><td>{{ row.cohort }}</td><td class="mono">{{ row.fields.receptor_path }}</td><td class="mono">{{ row.fields.receptor_sha256 }}</td><td>{{ row.fields.box_center }}</td><td>{{ row.fields.box_size }}</td><td class="num">{{ row.fields.exhaustiveness }}</td><td class="num">{{ row.fields.seed }}</td><td class="num">{{ row.fields.num_modes }}</td><td class="num">{{ row.fields.cpu_per_job }}</td><td class="num">{{ row.fields.num_workers }}</td><td>{{ row.fields.smina_version }}</td><td>{{ row.fields.scoring_function }}</td></tr>{% else %}<tr><td>{{ row.cohort }}</td><td colspan="11">not recorded</td></tr>{% endif %}{% endfor %}</table>
{% if not harness_config_recorded %}<p class="note">“not recorded” means the docking summary for that cohort predates configuration capture. No value is inferred or substituted for it.</p>{% endif %}

{% if computational_cost %}             #### brand new section for computational cost, only appears when the harness supplies it
<h2>Recorded docking cost</h2>
<p class="note">Wall time and throughput describe the recorded invocation. “Fresh” excludes cached poses. Requested CPU-slot hours are an explicit wall-time estimate from concurrent jobs × Smina CPU/job; they are not measured operating-system CPU consumption and should not be treated as billing data.</p>
<table class="kv"><tr><th>Cohort</th><th class="num">Total</th><th class="num">Fresh</th><th class="num">Cached</th><th class="num">Failed</th><th class="num">Workers</th><th class="num">CPU/job</th><th class="num">Wall (h)</th><th class="num">Fresh/h</th><th class="num">Requested slot-h</th></tr>
{% for key in ['actives','decoys'] %}{% set group=computational_cost[key] %}{% if group %}<tr><td>{{ key }}</td><td class="num">{{ group.counts.total }}</td><td class="num">{{ group.counts.ok }}</td><td class="num">{{ group.counts.cached }}</td><td class="num">{{ group.counts.failed }}</td><td class="num">{{ group.timing.workers_requested }}</td><td class="num">{{ group.timing.cpu_slots_per_task }}</td><td class="num">{{ '%.3f'|format(group.timing.wall_seconds/3600) }}</td><td class="num">{{ 'n/a' if group.timing.fresh_successes_per_wall_second is none else '%.1f'|format(3600*group.timing.fresh_successes_per_wall_second) }}</td><td class="num">{{ '%.3f'|format(group.timing.estimated_requested_cpu_slot_hours) }}</td></tr>{% endif %}{% endfor %}</table>
{% endif %}


<h2>Docking-performance figures</h2>
{% for key,title,caption in plot_layout %}{% if key in plot_divs %}<div class="plotcard"><div class="cap"><strong>{{ title }}</strong> — {{ caption }}</div>{{ plot_divs[key] }}</div>{% endif %}{% endfor %}

<div class="foot"><h2>Provenance</h2>
<p>Git commit <code>{{ provenance.git.commit }}</code> · dirty worktree: <strong>{{ provenance.git.dirty }}</strong> · wall time {{ '%.1f'|format(meta.wall_seconds) }} s</p>
{% for name,record in provenance.inputs.items() %}<p>{{ name }}: <code>{{ record.path }}</code><br>SHA-256 <code>{{ record.sha256 }}</code></p>{% endfor %}
<p>Software: {% for key,value in versions.items() %}{{ key }} <code>{{ value }}</code>{% if not loop.last %} · {% endif %}{% endfor %}</p>
{% if harness_config_recorded %}<p><strong>Harness configuration source:</strong> the receptor, box, seed, and search settings shown above are read from the dock-time summary files whose SHA-256 hashes match the analyzed score files. They are a record of the configuration in effect when these scores were produced, not an analysis-time snapshot.</p>
{% else %}<p><strong>Harness warning:</strong> at least one cohort has no dock-time configuration record, so receptor, box, seed, and search settings for it are unknown here. Any such settings in the run log are an analysis-time snapshot only. The score-file hashes identify the analyzed inputs; this script cannot prove which harness configuration generated them.</p>
{% endif %}
</div></div>
""")


def _format_metric(key: str, value: float, intervals: dict) -> dict[str, Any]:
    label = {"auc": "ROC AUC", "bedroc": "BEDROC"}.get(key, key.replace("_", " ").upper())
    digits = 3 if key in {"auc", "bedroc"} else 2
    interval = intervals.get(key)
    formatted_interval = None
    if interval:
        formatted_interval = f"[{interval['low']:.{digits}f}, {interval['high']:.{digits}f}]"
    return {"label": label, "value": f"{value:.{digits}f}", "interval": formatted_interval}


def _format_config_value(value: Any) -> str:
    """Render one recorded configuration value; absent means absent."""

    if value is _ABSENT or value is None:
        return "n/a"
    if isinstance(value, (list, tuple)):
        return ", ".join(
            f"{float(item):g}" if isinstance(item, (int, float)) else str(item)
            for item in value
        )
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _recorded_configurations(computational_cost: dict | None) -> dict[str, dict]:
    """Dock-time configuration blocks, keyed by cohort; absent ones omitted."""

    records = {}
    for cohort in COHORTS:
        summary = (computational_cost or {}).get(cohort) or {}
        recorded = summary.get("harness_config")
        if isinstance(recorded, dict) and recorded:
            records[cohort] = recorded
    return records


def check_harness_configuration(computational_cost: dict | None) -> None:
    """Public entry point so a caller can fail before doing any work."""

    _require_one_configuration(_recorded_configurations(computational_cost))


def _require_one_configuration(records: dict[str, dict]) -> None:
    """Abort when the two cohorts were not docked under the same settings."""

    if set(records) != set(COHORTS):
        return
    mismatched = [
        (
            field,
            records["actives"].get(field, _ABSENT),
            records["decoys"].get(field, _ABSENT),
        )
        for field in HARNESS_CONFIG_COMPARED
        if records["actives"].get(field, _ABSENT)
        != records["decoys"].get(field, _ABSENT)
    ]
    if not mismatched:
        return
    detail = "; ".join(
        f"{field}: actives={_format_config_value(active)!r} "
        f"decoys={_format_config_value(decoy)!r}"
        for field, active, decoy in mismatched
    )
    raise HarnessConfigurationMismatch(
        "Actives and decoys were docked under different harness configurations, "
        "so their scores cannot be ranked against each other. Mismatched "
        f"field(s): {detail}"
    )


def _harness_config_rows(computational_cost: dict | None) -> tuple[list[dict], bool]:
    """One row per cohort from the dock-time records; never from config.py."""

    records = _recorded_configurations(computational_cost)
    rows: list[dict[str, Any]] = []
    for cohort in COHORTS:
        recorded = records.get(cohort)
        if recorded is None:
            rows.append({"cohort": cohort, "recorded": False})
            continue
        rows.append(
            {
                "cohort": cohort,
                "recorded": True,
                # Not "values": Jinja resolves row.values to dict.values first.
                "fields": {
                    field: _format_config_value(recorded.get(field, _ABSENT))
                    for field in HARNESS_CONFIG_DISPLAY
                },
            }
        )
    _require_one_configuration(records)
    return rows, len(records) == len(COHORTS)


def _size_correlation_rows(size: dict | None) -> list[dict[str, Any]]:
    if size is None:
        return []
    rows = []
    for key, label in SIZE_CORRELATION_ROWS:
        record = size["correlations"][key]
        rows.append(
            {
                "label": label,
                "n": record["n"],
                **{
                    field: (
                        "n/a" if record[field] is None else f"{record[field]:.4g}"
                    )
                    for field in (
                        "spearman_rho", "spearman_p_value",
                        "pearson_r", "pearson_p_value",
                    )
                },
            }
        )
    return rows


def _fig_to_div(key: str, figure: go.Figure) -> Markup:
    polish_plotly_figure(figure)
    return Markup(figure.to_html(
        full_html=False, include_plotlyjs=False, div_id=f"plot-{key}",
        config=plotly_config(),
    ))


def render_report(
    meta: dict,
    metric_values: dict,
    intervals: dict,
    stats: dict,
    figures: dict[str, go.Figure],
    versions: dict,
    audit: dict,
    provenance: dict,
    chemistry: dict | None = None,
    computational_cost: dict | None = None,
    size: dict | None = None,
) -> str:
    metric_rows = [_format_metric(key, value, intervals) for key, value in metric_values.items()]
    confidence_label = f"{100 * meta['confidence_level']:.0f}% bootstrap"
    chemistry_rows = []
    if chemistry is not None:
        for definition in chemistry["property_definitions"]:
            key = definition["key"]
            active = chemistry["cohorts"]["actives"]["properties"][key]
            decoy = chemistry["cohorts"]["decoys"]["properties"][key]
            chemistry_rows.append(
                {
                    "label": definition["label"],
                    "active_mean": active["mean"],
                    "active_median": active["median"],
                    "decoy_mean": decoy["mean"],
                    "decoy_median": decoy["median"],
                    "mean_difference": chemistry[
                        "mean_difference_actives_minus_decoys"
                    ][key],
                }
            )
    harness_config_rows, harness_config_recorded = _harness_config_rows(
        computational_cost
    )
    body = _TEMPLATE.render(
        meta=meta,
        toolbar=Markup(report_toolbar("Labelled validation report")),
        metric_rows=metric_rows,
        confidence_label=confidence_label,
        stats=stats,
        mw=stats["mannwhitney"],
        plot_divs={key: _fig_to_div(key, fig) for key, fig in figures.items()},
        plot_layout=PLOT_LAYOUT,
        chemistry=chemistry,
        chemistry_rows=chemistry_rows,
        computational_cost=computational_cost,
        harness_config_rows=harness_config_rows,
        harness_config_recorded=harness_config_recorded,
        size=size,
        size_correlation_rows=_size_correlation_rows(size),
        chemistry_plot_layout=CHEMISTRY_PLOT_LAYOUT,
        versions=versions,
        audit=audit,
        provenance=provenance,
    )
    title = str(meta["name"]).replace("<", "").replace(">", "")
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>DYRK1A benchmark — {title}</title><script>{get_plotlyjs()}</script>"
        f"<style>{report_css()}</style></head><body>{body}</body></html>"
    )
