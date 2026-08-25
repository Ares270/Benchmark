"""Compact RDKit structure galleries for candidate reports.

The galleries are presentation-only. They select already ranked observed rows
from both ends of the docking-score distribution and never feed structures or
properties back into filtering, ranking, or comparison logic.
"""

from __future__ import annotations

import html
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D


GALLERY_FIELDS = (
    ("molecular_weight", "MW", ".1f"),
    ("clogp", "cLogP", ".2f"),
    ("qed", "QED", ".3f"),
    ("sa_score", "SA", ".2f"),
)


def molecule_svg(smiles: str, *, width: int = 270, height: int = 175) -> str:
    """Return an inline SVG depiction, or a labelled placeholder."""

    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return '<div class="molecule-missing">Structure unavailable</div>'
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.padding = 0.08
    options.bondLineWidth = 1.6
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    start = svg.find("<svg")
    return svg[start:] if start >= 0 else svg


def _formatted(value: Any, spec: str) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return format(number, spec) if np.isfinite(number) else "n/a"


def _card(row: pd.Series, *, total: int, edge: str) -> str:
    molecule_id = html.escape(str(row.get("molecule_id", "unknown")))
    smiles = html.escape(str(row.get("parent_smiles", "")))
    overall_rank = int(row["_overall_rank"])
    score = _formatted(row.get("score"), ".3f")
    properties = "".join(
        '<span class="molecule-stat"><small>'
        f"{html.escape(label)}</small>{_formatted(row.get(column), spec)}</span>"
        for column, label, spec in GALLERY_FIELDS
    )
    return (
        '<article class="molecule-card">'
        '<div class="molecule-card-head">'
        f'<span class="molecule-edge {edge}">{html.escape(edge.title())}</span>'
        f'<span class="molecule-rank">Rank {overall_rank:,} / {total:,}</span>'
        "</div>"
        f'<div class="molecule-drawing">{molecule_svg(str(row.get("parent_smiles", "")))}</div>'
        f'<div class="molecule-id" title="{molecule_id}">{molecule_id}</div>'
        f'<div class="molecule-score"><small>Smina</small>{score} <em>kcal/mol</em></div>'
        f'<div class="molecule-stats">{properties}</div>'
        f'<details class="smiles-details"><summary>Parent SMILES</summary><code>{smiles}</code></details>'
        "</article>"
    )


def render_top_bottom_galleries(
    candidates: pd.DataFrame,
    *,
    n: int = 10,
    gallery_id: str = "candidate-gallery",
) -> str:
    """Render top-N and bottom-N observed candidates as switchable galleries."""

    required = {"molecule_id", "parent_smiles", "score"}
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError(f"Molecule gallery is missing columns: {missing}")
    observed = candidates.loc[candidates["score"].notna()].copy()
    observed = observed.sort_values(["score", "molecule_id"], kind="stable")
    if observed.empty:
        return '<div class="card"><p>No observed docking scores are available for a molecular gallery.</p></div>'
    observed["_overall_rank"] = np.arange(1, len(observed) + 1)
    count = min(int(n), len(observed))
    top = observed.head(count)
    bottom = observed.tail(count).sort_values(
        ["score", "molecule_id"], ascending=[False, True], kind="stable"
    )
    overlap = len(set(top["molecule_id"]).intersection(bottom["molecule_id"]))
    overlap_note = (
        f" The two views overlap by {overlap} molecule{'s' if overlap != 1 else ''} "
        "because this scored pilot contains fewer than 20 distinct rows."
        if overlap
        else ""
    )
    safe_id = "".join(character for character in gallery_id if character.isalnum() or character in "-_")
    top_cards = "".join(
        _card(row, total=len(observed), edge="best") for _, row in top.iterrows()
    )
    bottom_cards = "".join(
        _card(row, total=len(observed), edge="worst") for _, row in bottom.iterrows()
    )
    return f"""
<section class="molecule-gallery-shell" id="{safe_id}">
  <div class="section-heading-row"><div><h2>Candidate structure gallery</h2>
  <p>Rapid chemistry reference for the two ends of the observed Smina ranking. Lower scores are better.{html.escape(overlap_note)}</p></div>
  <div class="gallery-tabs" role="tablist" aria-label="Candidate gallery view">
    <button type="button" class="active" data-gallery-target="top" role="tab">Top {count}</button>
    <button type="button" data-gallery-target="bottom" role="tab">Bottom {count}</button>
  </div></div>
  <div class="gallery-panel active" data-gallery-panel="top"><div class="molecule-grid">{top_cards}</div></div>
  <div class="gallery-panel" data-gallery-panel="bottom"><div class="molecule-grid">{bottom_cards}</div></div>
</section>
<script>(function(){{
  const root=document.getElementById({safe_id!r}); if(!root) return;
  root.querySelectorAll('[data-gallery-target]').forEach((button)=>{{
    button.addEventListener('click',()=>{{
      const target=button.dataset.galleryTarget;
      root.querySelectorAll('[data-gallery-target]').forEach((item)=>item.classList.toggle('active',item===button));
      root.querySelectorAll('[data-gallery-panel]').forEach((panel)=>panel.classList.toggle('active',panel.dataset.galleryPanel===target));
    }});
  }});
}})();</script>"""
