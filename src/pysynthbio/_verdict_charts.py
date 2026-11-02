"""Static notebook charts using the same saved quantities as VERDICT's report UI."""

from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

PURPLE, TEAL = "#6A55A0", "#25857D"
TISSUES = dict(
    zip(
        [
            "0000014",
            "0002046",
            "0002048",
            "0002084",
            "0000310",
            "0002107",
            "0000945",
            "0001264",
            "0002113",
            "0000178",
            "0001911",
            "0002097",
        ],
        [
            "Skin",
            "Thyroid",
            "Lung",
            "Heart · left ventricle",
            "Mammary gland",
            "Liver",
            "Stomach",
            "Pancreas",
            "Kidney",
            "Blood",
            "Mammary gland",
            "Skin",
        ],
    )
)
TRAINING = {
    "in_training": "Used to train this model",
    "selection_only": "Used for model selection",
    "not_in_training": "Not used for training or model selection",
    "unknown": "Training membership unavailable",
}


def component_outputs(report, name):
    component = next((c for c in report["components"] if c["component"] == name), None)
    if not component:
        print(f"{name}: no component supplied")
        return [], "unavailable"
    evidence = f"{component['mode']} · {component['status']}"
    print(f"{name}: {evidence}")
    return component.get("outputs", []), evidence


def style(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)


def comparison_label(trial, comparison_id):
    arms = {a["arm_id"]: a["label"] for a in trial["arms"]}
    pair = next(c for c in trial["comparisons"] if c["comparison_id"] == comparison_id)
    return f"{arms[pair['experimental_arm_id']]} vs {arms[pair['comparator_arm_id']]}"


def plot_efficacy(report):
    outputs, evidence = component_outputs(report, "outcomes")
    rows = [o for o in outputs if o["kind"] == "analysis_outcome"]
    if not rows:
        print("No endpoint predictions supplied.")
        return
    trial = report["request"]["trial"]
    endpoints = {e["endpoint_id"]: e["name"] for e in trial["endpoints"]}
    fig, axes = plt.subplots(
        len(rows), 1, figsize=(11, 3.5 * len(rows)), squeeze=False, layout="constrained"
    )
    for ax, row in zip(axes[:, 0], rows):
        prediction = row["prediction"]
        for y, field, status, color in [
            (1, "probability_significant", "status", PURPLE),
            (0, "probability_first_arm_favored", "direction_status", TEAL),
        ]:
            value = prediction.get(field)
            if value is not None and prediction.get(status) == "predicted":
                ax.barh(y, value, height=0.42, color=color)
                ax.text(
                    min(value + 0.02, 0.99),
                    y,
                    f"{value:.1%}",
                    va="center",
                    ha="right" if value > 0.9 else "left",
                )
            else:
                ax.text(
                    0.02,
                    y,
                    (
                        "Not applicable"
                        if prediction.get(status) == "not_applicable"
                        else "Unsupported input"
                    ),
                    va="center",
                )
        magnitude = prediction.get("expected_absolute_log_effect")
        magnitude_text = (
            f"{magnitude:.3f}"
            if magnitude is not None
            and prediction.get("magnitude_status") == "predicted"
            else (
                "Not applicable"
                if prediction.get("magnitude_status") == "not_applicable"
                else "Unsupported input"
            )
        )
        ax.set(
            yticks=[1, 0],
            yticklabels=[
                "Probability significant",
                "Probability experimental arm favored,\ngiven significant superiority",
            ],
            xlim=(0, 1),
            ylim=(-0.65, 1.65),
        )
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(
            fill(endpoints.get(row["endpoint_id"], row["endpoint_id"]), 75)
            + "\n"
            + comparison_label(trial, row["comparison_id"]),
            loc="left",
            fontsize=11,
        )
        ax.set_xlabel(
            f"Expected absolute log-ratio magnitude: {magnitude_text}\n"
            f"{TRAINING.get(prediction.get('training_status'), TRAINING['unknown'])}"
        )
        style(ax)
    fig.suptitle(f"Efficacy · {evidence}", fontsize=17)
    plt.show()
    print(
        "Magnitude is unsigned |ln(ratio)|, not a hazard ratio or uncertainty "
        "interval. Significance is not trial-success probability."
    )


def plot_safety(report, comparison_id=None):
    outputs, evidence = component_outputs(report, "safety")
    trial = report["request"]["trial"]
    pairs = trial.get("comparisons", [])
    if not outputs or not trial.get("arms"):
        print("No safety arm results supplied.")
        return
    if pairs or comparison_id is not None:
        pair = next(
            (
                c
                for c in pairs
                if c["comparison_id"] == (comparison_id or pairs[0]["comparison_id"])
            ),
            None,
        )
        if pair is None:
            raise ValueError("No comparison with this ID")
        arms = [pair["experimental_arm_id"], pair["comparator_arm_id"]]
    else:
        arms = [arm["arm_id"] for arm in trial["arms"][:2]]
    labels = {a["arm_id"]: a["label"] for a in trial["arms"]}
    lookup = {
        (o["query_id"], o["arm_id"]): o
        for o in outputs
        if o["kind"] == "safety_incidence"
    }

    def value(query, arm):
        row = lookup.get((query["query_id"], arm), {})
        return row.get("incidence") if row.get("status") == "predicted" else None

    queries = report["request"].get("safety_queries", [])
    for seriousness in dict.fromkeys(q["seriousness"] for q in queries):
        group = [q for q in queries if q["seriousness"] == seriousness]
        group.sort(
            key=lambda q: -(value(q, arms[0]) if value(q, arms[0]) is not None else -1)
        )
        fig, ax = plt.subplots(
            figsize=(11, max(4, len(group) * 0.45 + 2)), layout="constrained"
        )
        maximum = 0.0
        for y, query in enumerate(group):
            values = [value(query, arm) for arm in arms]
            if len(values) == 2 and all(v is not None for v in values):
                ax.plot(values, [y, y], color="#C7C1CD", linewidth=3)
            for arm_index, (arm, v, color) in enumerate(
                zip(arms, values, [PURPLE, TEAL])
            ):
                if v is None:
                    ax.annotate(
                        f"{labels[arm]}: unavailable",
                        (0, y),
                        xytext=(3, 9 - 18 * arm_index),
                        textcoords="offset points",
                        fontsize=7,
                    )
                    continue
                maximum = max(maximum, v)
                interval = lookup[(query["query_id"], arm)].get("interval")
                if interval:
                    ax.plot(
                        [interval["lower"], interval["upper"]],
                        [y, y],
                        color=color,
                        alpha=0.3,
                        linewidth=6,
                    )
                    maximum = max(maximum, interval["upper"])
                ax.scatter(v, y, color=color, s=50, zorder=3)
        candidates = [
            (0.02, 4),
            (0.05, 5),
            (0.1, 5),
            (0.15, 3),
            (0.2, 4),
            (0.3, 3),
            (0.4, 4),
            (0.5, 5),
            (0.6, 3),
            (0.8, 4),
            (1, 5),
        ]
        limit, divisions = next(
            ((v, n) for v, n in candidates if v >= maximum * 1.04), (1, 5)
        )
        ax.set(
            xlim=(0, limit),
            xticks=[limit * i / divisions for i in range(divisions + 1)],
            yticks=range(len(group)),
            yticklabels=[fill(q["term"].capitalize(), 32) for q in group],
            xlabel="Predicted share of patients with the event",
        )
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        for arm, color in zip(arms, [PURPLE, TEAL]):
            ax.scatter([], [], color=color, label=labels[arm])
        ax.legend(loc="lower right", frameon=False)
        ax.set_title(
            f"{'Serious' if seriousness == 'serious' else 'Other'} "
            f"adverse events · {evidence}",
            loc="left",
            fontsize=16,
        )
        style(ax)
        plt.show()
    print("Lines connect the two arms; bands appear only when intervals are supplied.")
    print(
        "; ".join(
            sorted(
                {
                    TRAINING.get(o.get("training_status"), TRAINING["unknown"])
                    for o in outputs
                }
            )
        )
    )


def plot_biology(report, contrast=None, tissue_id=None):
    outputs, evidence = component_outputs(report, "gem2")
    aggregates = [
        o
        for o in outputs
        if o["kind"] == "expression_effect"
        and o["feature_kind"] == "aggregate"
        and o["metric"] == "mean_absolute_log2_cpm_change"
    ]
    if not aggregates:
        print("No tissue summaries supplied.")
        return
    contrast = contrast or aggregates[0]["contrast"]
    rows = [o for o in aggregates if o["contrast"] == contrast]
    if not rows:
        raise ValueError("No aggregates for this contrast")
    has_median = any(o.get("median_donor_shift") is not None for o in rows)
    statistic = "median_donor_shift" if has_median else "value"
    rows.sort(
        key=lambda o: (
            -(o[statistic] if o.get(statistic) is not None else -1),
            o["tissue_id"],
        )
    )

    def tissue_name(key):
        return TISSUES.get(key.removeprefix("UBERON:"), key)

    trial = report["request"]["trial"]
    arms = {a["arm_id"]: a["label"] for a in trial["arms"]}
    label = (
        comparison_label(trial, contrast["comparison_id"])
        if contrast["kind"] == "trial_comparison"
        else f"{arms[contrast['arm_id']]} vs untreated"
    )
    fig, ax = plt.subplots(figsize=(11, 6.5), layout="constrained")
    for y, row in enumerate(rows):
        mean = row.get("value")
        interval = row.get("interval")
        if interval and mean is not None:
            ax.plot(
                [interval["lower"], interval["upper"]],
                [y, y],
                color=TEAL,
                alpha=0.4,
                linewidth=3,
            )
        if mean is not None:
            ax.scatter(mean, y, facecolors="white", edgecolors=TEAL, s=55, zorder=3)
        if row.get(statistic) is not None:
            ax.scatter(row[statistic], y, color=PURPLE, s=55, zorder=4)
        else:
            ax.annotate("Median not supplied", (0, y), fontsize=8)
    ax.scatter(
        [], [], color=PURPLE, label="Median donor" if has_median else "Mean reference"
    )
    ax.scatter(
        [],
        [],
        facecolors="white",
        edgecolors=TEAL,
        label="Mean reference (bands: supplied mean intervals)",
    )
    ax.set(
        yticks=range(len(rows)),
        yticklabels=[
            f"{tissue_name(o['tissue_id'])} (n={o['reference_count']})" for o in rows
        ],
        xlabel="Typical change per gene, up or down · log2(CPM+1)",
        xlim=(0, None),
    )
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.set_title(f"Biology · {evidence}\n{label}", loc="left", fontsize=15)
    style(ax)
    plt.show()
    selected = tissue_id or rows[0]["tissue_id"]
    if selected not in {o["tissue_id"] for o in rows}:
        raise ValueError("No aggregate for this tissue")
    for kind, count in [("gene", 3), ("pathway", 4)]:
        features = [
            o
            for o in outputs
            if o.get("contrast") == contrast
            and o.get("tissue_id") == selected
            and o.get("feature_kind") == kind
            and o.get("metric") == "mean_signed_log2_cpm_change"
            and o.get("value") is not None
        ]
        features.sort(key=lambda o: -abs(o["value"]))
        features = features[:count]
        if not features:
            print(f"No {kind} summaries supplied.")
            continue
        fig, ax = plt.subplots(figsize=(11, 3.5), layout="constrained")
        ax.barh(
            range(len(features)),
            [o["value"] for o in features],
            color=[TEAL if o["value"] >= 0 else PURPLE for o in features],
        )
        ax.set(
            yticks=range(len(features)),
            yticklabels=[
                fill(o["feature_id"].replace("_", "_ "), 36).replace("_ ", "_")
                for o in features
            ],
            xlabel="Down ← signed mean change · log2(CPM+1) → Up",
        )
        maximum = max(0.001, *(abs(o["value"]) for o in features)) * 1.2
        ax.set_xlim(-maximum, maximum)
        ax.invert_yaxis()
        ax.axvline(0, color="#777777", linewidth=1)
        ax.set_title(
            f"{tissue_name(selected)} · largest {kind} changes · {evidence}",
            loc="left",
            fontsize=14,
        )
        style(ax)
        plt.show()
    print(
        "Tissue changes measure expression, not efficacy or toxicity. "
        "Intervals describe means, not medians."
    )
    print(
        "Pathways: equal-symbol signed mean; not enrichment or causal pathway activity."
    )
    if report["request"]["trial"].get("registry_id") == "NCT02993523":
        print(
            "VIALE-A uses normal reference tissues without bone marrow; "
            "this does not model AML marrow response."
        )
