"""Assemble a self-contained, explicitly qualified HTML analysis report."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from jinja2 import Environment
from markupsafe import Markup
from plotly.offline import get_plotlyjs


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



_TEMPLATE = Environment(autoescape=True).from_string(r"""
<div class="report">
<style>
  .report{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;color:#1f2937;max-width:1100px;margin:0 auto;padding:24px 20px 60px;line-height:1.5}
  .report h1{font-size:1.7rem;margin:0 0 4px}.report h2{font-size:1.15rem;margin:34px 0 12px;padding-bottom:6px;border-bottom:1px solid #e5e7eb}
  .sub,.note{color:#6b7280;font-size:.88rem}.tiles{display:flex;flex-wrap:wrap;gap:12px;margin:18px 0 6px}
  .tile{flex:1 1 145px;background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:14px 16px;box-shadow:0 1px 2px rgba(0,0,0,.04)}
  .label{font-size:.72rem;text-transform:uppercase;letter-spacing:.04em;color:#6b7280}.value{font-size:1.45rem;font-weight:650;color:#111827;margin-top:2px}.ci{font-size:.75rem;color:#6b7280}
  table.kv{border-collapse:collapse;width:100%;font-size:.88rem;background:#fff;border:1px solid #e5e7eb}table.kv th,table.kv td{text-align:left;padding:9px 12px;border-bottom:1px solid #f0f1f3}table.kv th{background:#f9fafb}.num{text-align:right!important;font-variant-numeric:tabular-nums}
  .plotcard{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:10px 12px 4px;margin:16px 0;box-shadow:0 1px 3px rgba(0,0,0,.05)}
  .cap{color:#6b7280;font-size:.83rem;margin:2px 4px 8px}.warning{background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:10px 12px;font-size:.86rem}.foot{margin-top:40px;font-size:.8rem;color:#6b7280}.foot code{word-break:break-all;background:#f3f4f6;padding:1px 5px;border-radius:4px}
</style>

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
<p><strong>Harness warning:</strong> receptor, box, seed, and search settings in the run log are an analysis-time snapshot only. The score-file hashes identify the analyzed inputs; this script cannot prove which harness configuration generated them.</p>
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


def _fig_to_div(key: str, figure: go.Figure) -> Markup:
    return Markup(figure.to_html(
        full_html=False, include_plotlyjs=False, div_id=f"plot-{key}",
        config={"displaylogo": False, "responsive": True},
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
    body = _TEMPLATE.render(
        meta=meta,
        metric_rows=metric_rows,
        confidence_label=confidence_label,
        stats=stats,
        mw=stats["mannwhitney"],
        plot_divs={key: _fig_to_div(key, fig) for key, fig in figures.items()},
        plot_layout=PLOT_LAYOUT,
        chemistry=chemistry,
        chemistry_rows=chemistry_rows,
        computational_cost=computational_cost,
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
        f"</head><body style='margin:0;background:#f8fafc'>{body}</body></html>"
    )
