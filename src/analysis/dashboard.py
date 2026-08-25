"""Generate an offline home page for the DYRK1A benchmark result tree."""

from __future__ import annotations

import argparse
import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from src.analysis.interpretation import candidate_interpretation
from src.analysis.report_theme import report_css, report_toolbar
from src.harness import config


ARM_CATALOG = (
    {
        "key": "baseline",
        "label": "Naive baseline",
        "generation": "ChEMBL 37 property-matched control",
        "conditioning": "No target information",
        "boundary": "Primary chemical-space control; not a learned generator",
        "accent": "amber",
    },
    {
        "key": "gen1",
        "label": "Gen1",
        "generation": "GuacaMol SMILES-LSTM",
        "conditioning": "Target-unaware",
        "boundary": "Frozen character-level chemical prior",
        "accent": "blue",
    },
    {
        "key": "gen2",
        "label": "Gen2",
        "generation": "WarmMolGenOne",
        "conditioning": "DYRK1A kinase-domain sequence",
        "boundary": "Frozen protein-sequence-conditioned generator",
        "accent": "violet",
    },
    {
        "key": "gen3",
        "label": "Gen3",
        "generation": "Molexar Omni",
        "conditioning": "Locked 7O7K protein pocket",
        "boundary": "Frozen pocket-coordinate-conditioned generator",
        "accent": "teal",
    },
)


class DashboardError(ValueError):
    """Raised when a result tree cannot be summarized safely."""


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DashboardError(f"Cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise DashboardError(f"{path} must contain one JSON object")
    return value


def _record_path(value: object) -> Path | None:
    if isinstance(value, dict):
        value = value.get("path")
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path.resolve() if path.is_file() else None


def _candidate_from_campaign(summary: dict) -> dict | None:
    path = _record_path(summary.get("outputs", {}).get("candidate_metrics"))
    return _read_json(path) if path else None


def _candidate_from_pipeline(summary: dict) -> dict | None:
    report = _record_path(summary.get("outputs", {}).get("candidate_report"))
    metrics = report.parent / "metrics.json" if report else None
    return _read_json(metrics) if metrics and metrics.is_file() else None


def _arm_key(name: str, role: str) -> str:
    lowered = name.lower()
    if role == "naive_baseline" or "naive" in lowered or "baseline" in lowered:
        return "baseline"
    for key in ("gen1", "gen2", "gen3"):
        if key in lowered:
            return key
    if "guacamol" in lowered:
        return "gen1"
    if "warmmol" in lowered:
        return "gen2"
    if "molexar" in lowered:
        return "gen3"
    return "other"


def _report_from_summary(summary: dict, root: Path) -> Path | None:
    outputs = summary.get("outputs", {})
    for key in ("combined_report", "candidate_report", "comparison_report"):
        path = _record_path(outputs.get(key) if isinstance(outputs, dict) else None)
        if path:
            return path
    direct = root / "report.html"
    return direct.resolve() if direct.is_file() else None


def _run_record(summary_path: Path, kind: str) -> dict:
    summary = _read_json(summary_path)
    root = summary_path.parent
    if kind == "campaign":
        candidate = _candidate_from_campaign(summary)
        role = str(candidate.get("role", "model") if candidate else "model")
        registered = bool(summary.get("design", {}).get("registered_campaign"))
    else:
        candidate = _candidate_from_pipeline(summary)
        role = str(summary.get("role", candidate.get("role", "unassigned") if candidate else "unassigned"))
        registered = role in {"model", "naive_baseline"}
    name = str(summary.get("name") or root.name)
    funnel = summary.get("funnel", {})
    neutral = (
        candidate.get("interpretation", {}).get("neutral_summary")
        if candidate
        else None
    )
    if candidate and not neutral:
        neutral = candidate_interpretation(candidate)
    return {
        "kind": kind,
        "name": name,
        "root": root.resolve(),
        "summary_path": summary_path.resolve(),
        "arm": _arm_key(name, role),
        "role": role,
        "registered": registered,
        "funnel": funnel,
        "candidate": candidate,
        "report": _report_from_summary(summary, root),
        "modified": summary_path.stat().st_mtime,
        "neutral": neutral or "The run has durable stage records but no candidate analysis summary yet.",
    }

def _summary_files(results_dir: Path) -> dict[str, list[Path]]:
    """Find durable summaries while pruning molecule-heavy artifact folders."""

    found = {
        "campaign_summary.json": [],
        "pipeline_summary.json": [],
        "all_arms_summary.json": [],
    }
    pruned_names = {
        "candidate_analysis",
        "candidates",
        "docking",
        "prepared",
        "intake",
        "gate",
        "full_cohort_analysis",
        "full_cohort_intake",
        "ligands",
        "report_visual_qa_20260810",
    }
    for current, directories, filenames in os.walk(results_dir):
        current_path = Path(current)
        for filename in found:
            if filename in filenames:
                found[filename].append(current_path / filename)
        if "campaign_summary.json" in filenames or "pipeline_summary.json" in filenames:
            directories[:] = []
            continue
        directories[:] = [
            name
            for name in directories
            if name not in pruned_names and not name.startswith("samples_")
        ]
    for paths in found.values():
        paths.sort()
    return found


def discover_runs(results_dir: Path) -> tuple[list[dict], list[dict]]:
    """Return completed run records and full-comparison records."""

    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise DashboardError(f"Results directory does not exist: {results_dir}")
    summaries = _summary_files(results_dir)
    campaign_paths = summaries["campaign_summary.json"]
    campaign_roots = {path.parent.resolve() for path in campaign_paths}
    records = [_run_record(path, "campaign") for path in campaign_paths]
    for path in summaries["pipeline_summary.json"]:
        if any(root in path.resolve().parents for root in campaign_roots):
            continue
        records.append(_run_record(path, "pipeline"))
    comparisons = []
    for path in summaries["all_arms_summary.json"]:
        summary = _read_json(path)
        report = _record_path(summary.get("comparison_report"))
        comparisons.append(
            {
                "name": path.parent.name,
                "root": path.parent.resolve(),
                "summary_path": path.resolve(),
                "submitted": summary.get("submitted_per_arm"),
                "report": report,
                "modified": path.stat().st_mtime,
            }
        )
    records.sort(key=lambda item: item["modified"], reverse=True)
    comparisons.sort(key=lambda item: item["modified"], reverse=True)
    return records, comparisons


def _relative_link(target: Path | None, output: Path) -> str | None:
    if target is None:
        return None
    relative = os.path.relpath(target, output.parent).replace(os.sep, "/")
    return quote(relative, safe="/._-")


def _number(value: object, digits: int = 0) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:,.{digits}f}"


def _funnel_html(funnel: dict) -> str:
    stages = (
        ("submitted", "Submitted"),
        ("accepted_at_intake", "Intake"),
        ("passed_predock_gate", "Gate"),
        ("prepared_pdbqt_available", "Prepared"),
        ("successfully_scored", "Scored"),
    )
    try:
        total = max(1, int(funnel.get("submitted", 0)))
    except (TypeError, ValueError):
        total = 1
    rows = []
    for key, label in stages:
        if key not in funnel:
            continue
        value = int(funnel[key])
        width = max(2.0, min(100.0, 100 * value / total))
        rows.append(
            '<div class="funnel-row"><span>'
            f"{label}</span><div class=\"funnel-track\"><i style=\"width:{width:.2f}%\"></i></div>"
            f"<strong>{value:,}</strong></div>"
        )
    return "".join(rows) or '<p class="note">Funnel not available yet.</p>'


def _compact_metrics(record: dict) -> str:
    candidate = record.get("candidate") or {}
    intake = candidate.get("intake", {})
    docking = candidate.get("docking", {})
    scores = docking.get("score_distribution_kcal_mol", {})
    values = (
        ("Validity", _number(100 * float(intake.get("validity", 0)), 1) + "%" if intake else "n/a"),
        ("Uniqueness", _number(100 * float(intake.get("parent_uniqueness", 0)), 1) + "%" if intake else "n/a"),
        ("Median Smina", _number(scores.get("median"), 3)),
        ("CPU-slot h", _number(candidate.get("computational_cost", {}).get("timing", {}).get("estimated_requested_cpu_slot_hours"), 2)),
    )
    return "".join(
        f'<div><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'
        for label, value in values
    )


def _run_card(record: dict, output: Path) -> str:
    report_link = _relative_link(record.get("report"), output)
    summary_link = _relative_link(record.get("summary_path"), output)
    badge = "Registered" if record["registered"] else "Pilot"
    actions = []
    if report_link:
        actions.append(f'<a class="primary-link" href="{report_link}">Open report</a>')
    if summary_link:
        actions.append(f'<a href="{summary_link}">Summary JSON</a>')
    return f"""
<article class="run-card arm-{html.escape(record['arm'])}">
  <div class="run-card-head"><div><span class="status-dot"></span><span class="eyebrow">{html.escape(record['arm'].upper())}</span>
  <h3>{html.escape(record['name'])}</h3></div><span class="run-badge">{badge}</span></div>
  <p class="run-path">{html.escape(str(record['root']))}</p>
  <div class="compact-metrics">{_compact_metrics(record)}</div>
  <div class="funnel-mini">{_funnel_html(record.get('funnel', {}))}</div>
  <details><summary>Neutral readout</summary><p>{html.escape(str(record['neutral']))}</p></details>
  <div class="run-actions">{''.join(actions)}</div>
</article>"""


def _arm_card(arm: dict, latest: dict | None, output: Path) -> str:
    status = "No completed run indexed"
    link = ""
    if latest:
        status = f"Latest: {latest['name']}"
        href = _relative_link(latest.get("report"), output)
        if href:
            link = f'<a href="{href}">Open latest report →</a>'
    return f"""
<article class="design-card accent-{arm['accent']}">
  <span class="design-step">{html.escape(arm['label'])}</span>
  <h3>{html.escape(arm['generation'])}</h3>
  <p class="conditioning">{html.escape(arm['conditioning'])}</p>
  <p>{html.escape(arm['boundary'])}</p>
  <div class="design-status">{html.escape(status)} {link}</div>
</article>"""


def dashboard_css() -> str:
    return """
.dashboard-intro{display:grid;grid-template-columns:1.35fr .65fr;gap:18px;align-items:stretch;margin:18px 0}
.dashboard-intro .card{margin:0}.command-card{color:#d9fff8;background:#102a3a;border-color:#274b58}.command-card code{display:block;margin-top:10px;padding:10px 12px;color:#d9fff8;background:#071d29}
.design-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin:18px 0 30px}.design-card{position:relative;overflow:hidden;padding:18px;background:#fff;border:1px solid var(--line);border-top:5px solid var(--teal);border-radius:16px;box-shadow:var(--shadow-sm)}
.design-card.accent-amber{border-top-color:#d97706}.design-card.accent-blue{border-top-color:#2563eb}.design-card.accent-violet{border-top-color:#7c3aed}.design-card h3{margin:10px 0 5px;color:var(--ink);font-size:1.03rem}.design-step,.eyebrow{color:var(--muted);font-size:.67rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase}.conditioning{color:#173b48;font-weight:720}.design-card p{margin:6px 0;font-size:.82rem}.design-status{margin-top:14px;padding-top:11px;border-top:1px solid var(--line);color:var(--muted);font-size:.73rem}.design-status a{display:block;margin-top:4px;font-weight:720}
.run-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.run-card{min-width:0;padding:18px 19px;background:rgba(255,255,255,.97);border:1px solid var(--line);border-left:5px solid var(--teal);border-radius:17px;box-shadow:var(--shadow-sm)}.run-card.arm-baseline{border-left-color:#d97706}.run-card.arm-gen1{border-left-color:#2563eb}.run-card.arm-gen2{border-left-color:#7c3aed}.run-card-head{display:flex;justify-content:space-between;gap:12px}.run-card h3{margin:5px 0 0;color:var(--ink);font-size:1.05rem}.status-dot{display:inline-block;width:8px;height:8px;margin-right:7px;background:#0f9f76;border-radius:50%;box-shadow:0 0 0 4px #e6f4f1}.run-badge{align-self:flex-start;padding:4px 8px;color:#0b625b;background:var(--teal-soft);border-radius:99px;font-size:.65rem;font-weight:800;text-transform:uppercase}.run-path{overflow:hidden;color:var(--muted);font-family:monospace;font-size:.68rem;text-overflow:ellipsis;white-space:nowrap}
.compact-metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;margin:14px 0;overflow:hidden;background:var(--line);border:1px solid var(--line);border-radius:11px}.compact-metrics div{padding:9px;background:#fafcfd}.compact-metrics span{display:block;color:var(--muted);font-size:.6rem;font-weight:750;text-transform:uppercase}.compact-metrics strong{font-size:.9rem;font-variant-numeric:tabular-nums}.funnel-mini{display:grid;gap:6px;margin:13px 0}.funnel-row{display:grid;grid-template-columns:64px 1fr 42px;gap:8px;align-items:center;color:var(--muted);font-size:.68rem}.funnel-row strong{text-align:right;color:var(--ink);font-size:.72rem}.funnel-track{height:7px;overflow:hidden;background:#e8eef2;border-radius:99px}.funnel-track i{display:block;height:100%;background:linear-gradient(90deg,#0f766e,#2ea89d);border-radius:inherit}.run-card details{margin-top:12px;padding-top:10px;border-top:1px solid var(--line);font-size:.78rem}.run-card summary{cursor:pointer;color:#31505c;font-weight:720}.run-actions{display:flex;gap:8px;margin-top:14px}.run-actions a{padding:7px 10px;background:#edf3f5;border-radius:8px;font-size:.72rem;font-weight:720;text-decoration:none}.run-actions .primary-link{color:#fff;background:var(--teal)}
.comparison-strip{display:flex;align-items:center;justify-content:space-between;gap:20px;padding:16px 18px;background:#fff;border:1px solid var(--line);border-radius:14px}.comparison-strip strong{display:block}.comparison-strip span{color:var(--muted);font-size:.8rem}.comparison-strip a{flex:0 0 auto;padding:8px 12px;color:#fff;background:var(--teal);border-radius:9px;font-size:.76rem;font-weight:720;text-decoration:none}
@media(max-width:900px){.design-grid{grid-template-columns:repeat(2,1fr)}.run-grid{grid-template-columns:1fr}.dashboard-intro{grid-template-columns:1fr}}
@media(max-width:520px){.design-grid{grid-template-columns:1fr}.compact-metrics{grid-template-columns:repeat(2,1fr)}}
"""


def render_dashboard(results_dir: Path, output: Path) -> str:
    records, comparisons = discover_runs(results_dir)
    latest = {
        key: next((record for record in records if record["arm"] == key), None)
        for key in ("baseline", "gen1", "gen2", "gen3")
    }
    registered = sum(1 for record in records if record["registered"])
    pilots = len(records) - registered
    design_cards = "".join(
        _arm_card(arm, latest.get(arm["key"]), output) for arm in ARM_CATALOG
    )
    run_cards = "".join(_run_card(record, output) for record in records)
    comparison_rows = "".join(
        '<div class="comparison-strip"><div><strong>'
        f"{html.escape(item['name'])}</strong><span>{_number(item.get('submitted'))} submitted per arm</span></div>"
        + (
            f'<a href="{_relative_link(item.get("report"), output)}">Open four-arm report</a>'
            if item.get("report")
            else "<span>Comparison report unavailable</span>"
        )
        + "</div>"
        for item in comparisons
    ) or '<div class="card"><p>No completed registered four-arm comparison has been indexed yet.</p></div>'
    generated = datetime.now(timezone.utc).isoformat()
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DYRK1A benchmark home</title><style>{report_css()}{dashboard_css()}</style></head>
<body>{report_toolbar("Benchmark home")}<h1>DYRK1A generative benchmark</h1>
<div class="dashboard-intro"><div class="card"><strong>One study, four locked arms</strong><p>This page indexes immutable results without recomputing or re-ranking any molecule. Open a run for its complete provenance, chemistry, failure accounting, and candidate gallery.</p></div><div class="card command-card"><strong>Refresh this page</strong><code>python -m src.benchmark dashboard</code></div></div>
<div class="metrics"><div class="metric">Completed runs<strong>{len(records):,}</strong></div><div class="metric">Registered arms<strong>{registered:,}</strong></div><div class="metric">Pilots<strong>{pilots:,}</strong></div><div class="metric">Four-arm comparisons<strong>{len(comparisons):,}</strong></div></div>
<h2>Study design</h2><p class="note">The progression changes available target information. It is not an architecture-only ablation.</p><div class="design-grid">{design_cards}</div>
<h2>Four-arm comparisons</h2><div class="comparison-list">{comparison_rows}</div>
<h2>Completed run index</h2><p class="note">Newest durable summaries first. Expand the neutral readout only when needed; all headline statistics remain separate.</p><div class="run-grid">{run_cards or '<div class="card"><p>No completed run summaries found.</p></div>'}</div>
<div class="foot">Generated {html.escape(generated)} from <code>{html.escape(str(Path(results_dir).resolve()))}</code>. This mutable index links to immutable scientific artifacts; refreshing it does not alter a run.</div>
</body></html>"""


def write_dashboard(results_dir: Path, output: Path) -> Path:
    """Atomically refresh the non-scientific result index."""

    results_dir = Path(results_dir)
    output = Path(output)
    rendered = render_dashboard(results_dir, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the offline benchmark home page")
    parser.add_argument("--results-dir", type=Path, default=config.REPO_ROOT / "results")
    parser.add_argument("--output", type=Path, default=config.REPO_ROOT / "results/benchmark_home.html")
    args = parser.parse_args()
    try:
        output = write_dashboard(args.results_dir, args.output)
    except DashboardError as error:
        parser.error(str(error))
    print(f"Benchmark home: {output}")


if __name__ == "__main__":
    main()
