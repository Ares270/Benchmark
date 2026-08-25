"""Deterministic, neutral prose for benchmark reports.

These helpers summarize recorded values without ranking methods, inferring
biochemical activity, or adding statistics that were not already computed.
"""

from __future__ import annotations

from typing import Mapping


def _integer(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _percent(value: object) -> str:
    try:
        return f"{100 * float(value):.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def candidate_interpretation(profile: Mapping) -> str:
    """Summarize one unlabelled candidate cohort without claiming efficacy."""

    intake = profile.get("intake", {})
    docking = profile.get("docking", {})
    distribution = docking.get("score_distribution_kcal_mol", {})
    submitted = _integer(intake.get("submitted_rows"))
    accepted = _integer(intake.get("accepted_for_preparation"))
    scored = _integer(docking.get("n_with_observed_score"))
    median = distribution.get("median")
    median_text = "n/a"
    try:
        median_text = f"{float(median):.3f} kcal/mol"
    except (TypeError, ValueError):
        pass
    return (
        f"This unlabelled cohort submitted {submitted:,} raw rows; "
        f"{_percent(intake.get('validity'))} were structurally valid, "
        f"{accepted:,} unique parents were accepted at intake, and {scored:,} "
        f"received observed Smina scores. The median observed score was "
        f"{median_text}. These values describe generator output and computational "
        "screening behavior only; they do not demonstrate biochemical activity "
        "or superiority to another arm."
    )


def campaign_interpretation(campaign: Mapping, candidate: Mapping | None) -> str:
    """Summarize campaign completion and its interpretation boundary."""

    design = campaign.get("design", {})
    funnel = campaign.get("funnel", {})
    kind = "registered campaign" if design.get("registered_campaign") else "pilot"
    conditioning = str(design.get("conditioning") or "the arm's locked inputs")
    submitted = _integer(funnel.get("submitted"))
    scored = _integer(funnel.get("successfully_scored"))
    score_clause = ""
    if candidate:
        median = candidate.get("docking", {}).get(
            "score_distribution_kcal_mol", {}
        ).get("median")
        try:
            score_clause = f" Median observed Smina score was {float(median):.3f} kcal/mol."
        except (TypeError, ValueError):
            pass
    return (
        f"This {kind} used {conditioning}. Of {submitted:,} raw screening "
        f"submissions, {scored:,} were successfully scored.{score_clause} "
        "The result documents pipeline behavior under the locked protocol; it "
        "does not by itself establish target activity or a model ranking."
    )


def comparison_interpretation(summary_rows: list[Mapping]) -> str:
    """Describe a comparison's scope while deliberately avoiding a winner."""

    if not summary_rows:
        return "No comparable candidate runs were available."
    submitted = {_integer(row.get("submitted")) for row in summary_rows}
    budget = submitted.pop() if len(submitted) == 1 else None
    budget_text = (
        f"an equal raw budget of {budget:,} submissions per arm"
        if budget is not None
        else "the recorded per-arm submission budgets"
    )
    return (
        f"The four arms were evaluated with {budget_text} and one authenticated "
        "docking protocol. Validity, uniqueness, funnel survival, molecular "
        "properties, docking distributions, and compute cost remain separate. "
        "No composite score or automatic winner is produced; any scientific "
        "conclusion must consider those trade-offs and each arm's conditioning."
    )
