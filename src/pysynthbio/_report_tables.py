"""Readable tables from a saved VERDICT trial report (no network, no plotting).

The numbers match the VERDICT results page and PDF: probabilities are model
estimates, unavailable values stay missing rather than being filled in.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ._tissue_names import TISSUE_NAMES


def _trial(report: dict[str, Any]) -> dict[str, Any]:
    return report["request"]["trial"]


def _requested(report: dict[str, Any], kind: str) -> bool:
    return kind in report["request"].get("requested_analyses", [])


def _outputs(report: dict[str, Any], component: str) -> list[dict[str, Any]]:
    for item in report.get("components", []):
        if item["component"] == component:
            return item.get("outputs") or []
    return []


def _arm_labels(report: dict[str, Any]) -> dict[str, str]:
    return {
        arm["arm_id"]: arm.get("label") or arm["arm_id"]
        for arm in _trial(report)["arms"]
    }


def _role(endpoint: dict[str, Any]) -> str:
    analysis = endpoint.get("outcome_analysis")
    if not analysis:
        return "not stated"
    return "secondary" if analysis.get("endpoint_role") == "secondary" else "primary"


def comparison_label(report: dict[str, Any], comparison: dict[str, Any]) -> str:
    arms = _arm_labels(report)
    experimental = arms.get(comparison["experimental_arm_id"])
    comparator = arms.get(comparison["comparator_arm_id"])
    return f"{experimental} vs {comparator}"


def efficacy_table(report: dict[str, Any]) -> pd.DataFrame:
    """One row per comparison and endpoint: probability of significance, direction
    and magnitude, or the predicted hazard ratio for survival endpoints."""
    columns = [
        "comparison",
        "endpoint",
        "role",
        "p_significant",
        "p_favors_experimental",
        "expected_abs_log_effect",
        "hazard_ratio",
        "status",
    ]
    if not _requested(report, "efficacy"):
        return pd.DataFrame(columns=columns)
    trial = _trial(report)
    endpoints = {e["endpoint_id"]: e for e in trial["endpoints"]}
    outputs = _outputs(report, "outcomes")
    rows = []
    for comparison in trial.get("comparisons", []):
        label = comparison_label(report, comparison)
        for endpoint_id in comparison["endpoint_ids"]:
            endpoint = endpoints.get(endpoint_id)
            if endpoint is None:
                continue
            match = [
                o
                for o in outputs
                if o.get("comparison_id") == comparison["comparison_id"]
                and o.get("endpoint_id") == endpoint_id
            ]
            outcome = next((o for o in match if o["kind"] == "analysis_outcome"), None)
            survival = next((o for o in match if o["kind"] == "survival_effect"), None)
            prediction = (outcome or {}).get("prediction") or {}
            rows.append(
                {
                    "comparison": label,
                    "endpoint": endpoint.get("name"),
                    "role": _role(endpoint),
                    "p_significant": prediction.get("probability_significant"),
                    "p_favors_experimental": prediction.get(
                        "probability_first_arm_favored"
                    ),
                    "expected_abs_log_effect": prediction.get(
                        "expected_absolute_log_effect"
                    ),
                    "hazard_ratio": (survival or {}).get("hazard_ratio"),
                    "status": prediction.get("status")
                    or ("predicted" if survival else "no prediction"),
                }
            )
    table = pd.DataFrame(rows, columns=columns)
    # Comparisons keep the trial's order; within each, primary endpoints come first.
    comparison_order = {
        label: i for i, label in enumerate(dict.fromkeys(table["comparison"]))
    }
    role_order = {"primary": 0, "secondary": 1, "not stated": 2}
    return table.sort_values(
        by=["comparison", "role"],
        key=lambda s: s.map(role_order if s.name == "role" else comparison_order),
        kind="stable",
    ).reset_index(drop=True)


def _overall_query(report: dict[str, Any]) -> str | None:
    for query in report["request"].get("safety_queries", []):
        if (
            query.get("seriousness") == "serious"
            and query["term"].strip().lower() == "any serious adverse event"
        ):
            return query["query_id"]
    return None


def safety_table(
    report: dict[str, Any], *, include_overall: bool = True
) -> pd.DataFrame:
    """Predicted share of patients with each serious adverse event, by arm.

    Rows are events (overall first, then highest rate); columns are arms. Missing
    values mean the model made no prediction for that event and arm.
    """
    if not _requested(report, "toxicity"):
        return pd.DataFrame()
    arms = _arm_labels(report)
    incidence = [
        o
        for o in _outputs(report, "safety")
        if o.get("kind") == "safety_incidence" and o.get("status") == "predicted"
    ]
    arm_ids = [a for a in arms if any(o["arm_id"] == a for o in incidence)]
    if not arm_ids:
        return pd.DataFrame()
    rate = {(o["arm_id"], o["query_id"]): o.get("incidence") for o in incidence}
    overall = _overall_query(report)
    rows = {}
    for query in report["request"].get("safety_queries", []):
        if query["query_id"] == overall and not include_overall:
            continue
        values = [rate.get((a, query["query_id"])) for a in arm_ids]
        if all(v is None for v in values):
            continue
        term = (
            "Any serious adverse event"
            if query["query_id"] == overall
            else query["term"][:1].upper() + query["term"][1:]
        )
        rows[term] = values
    table = pd.DataFrame.from_dict(
        rows, orient="index", columns=[arms[a] for a in arm_ids]
    )
    table.index.name = "event"
    if table.empty:
        return table
    first = (
        ["Any serious adverse event"]
        if "Any serious adverse event" in table.index
        else []
    )
    rest = (
        table.drop(index=first)
        .assign(_max=lambda t: t.max(axis=1))
        .sort_values("_max", ascending=False)
        .drop(columns="_max")
    )
    return pd.concat([table.loc[first], rest])


def biology_table(report: dict[str, Any]) -> pd.DataFrame:
    """Mean absolute log2 CPM change between arms, by tissue (rows) and comparison."""
    if not _requested(report, "biology"):
        return pd.DataFrame()
    trial = _trial(report)
    comparisons = {
        c["comparison_id"]: comparison_label(report, c)
        for c in trial.get("comparisons", [])
    }
    rows: dict[str, dict[str, float]] = {}
    for o in _outputs(report, "gem2"):
        contrast = o.get("contrast") or {}
        if (
            o.get("kind") == "expression_effect"
            and o.get("feature_kind") == "aggregate"
            and o.get("metric") == "mean_absolute_log2_cpm_change"
            and contrast.get("kind") == "trial_comparison"
        ):
            label = comparisons.get(
                contrast["comparison_id"], contrast["comparison_id"]
            )
            tissue = TISSUE_NAMES.get(o["tissue_id"], o["tissue_id"])
            rows.setdefault(tissue, {})[label] = o.get("value")
    table = pd.DataFrame.from_dict(rows, orient="index")
    if not table.empty:
        table = table.sort_values(table.columns[0], ascending=False)
        table.index.name = "tissue"
    return table


def headlines(report: dict[str, Any]) -> dict[str, Any]:
    """The few numbers to read first: overall status, P(Success) against historical
    odds, the main comparison's primary result and the overall serious-event rate."""
    out: dict[str, Any] = {"status": report.get("status")}
    for analysis in report.get("analyses", []):
        if analysis["analysis"] == "trial_characteristics":
            estimate = next(
                (
                    e
                    for e in analysis.get("estimates", [])
                    if e.get("kind") == "phase_passage"
                ),
                None,
            )
            if estimate:
                out["p_success"] = estimate.get("probability")
                out["historical_odds"] = estimate.get("reference_probability")
    efficacy = efficacy_table(report)
    if not efficacy.empty:
        main = efficacy.iloc[0]
        out["main_comparison"] = main["comparison"]
        out["main_endpoint"] = main["endpoint"]
        out["p_significant"] = main["p_significant"]
        out["p_favors_experimental"] = main["p_favors_experimental"]
        out["hazard_ratio"] = main["hazard_ratio"]
    safety = safety_table(report)
    if "Any serious adverse event" in getattr(safety, "index", []):
        for arm, value in safety.loc["Any serious adverse event"].items():
            out[f"serious_ae: {arm}"] = value
    return {
        k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in out.items()
    }


def percent(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    pct = value * 100
    return f"{pct:.1f}%" if 0 < pct < 10 else f"{pct:.0f}%"
