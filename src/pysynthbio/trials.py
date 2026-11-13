"""Trial input files and saved, displayable VERDICT reports.

Loading files, inspecting result properties, and drawing charts never execute
models. Only VerdictClient.run_trial submits a trial; its retry key is explicit.
"""

import html
import importlib.metadata
import json
from pathlib import Path


def load_trial(path):
    """Read a workflow input JSON file. No API calls or inference."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or "trial" not in value
        or "requested_analyses" not in value
    ):
        raise ValueError("Expected a VERDICT trial workflow input JSON file")
    return value


def load_trial_result(path, *, api_base_url="https://verdict.synthesize.bio"):
    """Read a saved report offline. Does not submit, poll, or contact the API."""
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    return TrialResult(report, api_base_url.rstrip("/"), "Saved API report · offline")


def _comparable_request(value):
    normalized = json.loads(json.dumps(value))
    for endpoint in normalized["trial"]["endpoints"]:
        analysis = endpoint.get("outcome_analysis")
        if analysis is not None:
            analysis.setdefault("endpoint_role", "primary")
    return normalized


def _validate_report(report, run_id=None, trial_input=None):
    if not isinstance(report, dict) or not all(
        k in report for k in ("run_id", "request", "components", "analyses", "status")
    ):
        raise ValueError(
            "Expected a VERDICT trial result, not an individual model result"
        )
    if run_id is not None and report["run_id"] != run_id:
        raise ValueError("Report does not match the requested run")
    if trial_input is not None and _comparable_request(
        report["request"]
    ) != _comparable_request(trial_input):
        raise ValueError("Report does not match the submitted trial input")
    if any(
        c.get("mode") != "live"
        for c in report["components"]
        if c.get("status") != "not_run"
    ):
        raise ValueError("Report contains a non-live component")


def _escape(value):
    return html.escape(str(value), quote=True)


def _provenance():
    try:
        package = importlib.metadata.distribution("pysynthbio")
        direct = json.loads(package.read_text("direct_url.json") or "{}")
        return {
            "package": "pysynthbio",
            "version": package.version,
            "git_commit": direct.get("vcs_info", {}).get("commit_id", "unknown"),
        }
    except importlib.metadata.PackageNotFoundError:
        return {"package": "pysynthbio", "version": "not installed"}


class TrialResult:
    """Saved scientific report with notebook-friendly views and raw data access."""

    def __init__(self, report, api_base_url, source, execution=None):
        _validate_report(report)
        self.raw = report
        self.run_id = report["run_id"]
        self.ui_url = f"{api_base_url}/app/virtual-trials/runs/{self.run_id}"
        self.source = source
        # Local client provenance is separate from the report's model provenance.
        self.provenance = {"local_client": _provenance(), "execution": execution}

    def headlines(self):
        """The numbers to read first (P(Success), main efficacy, serious AEs)."""
        from ._report_tables import headlines

        return headlines(self.raw)

    def __repr__(self):
        from ._report_tables import percent

        h = self.headlines()
        trial = self.raw["request"]["trial"]
        lines = [f"{trial['name']} — {h['status']}", f"  {self.ui_url}"]
        if h.get("p_success") is not None:
            lines.append(
                f"  P(Success) {percent(h['p_success'])}"
                f" (historical odds {percent(h.get('historical_odds'))})"
            )
        if h.get("main_endpoint"):
            effect = (
                f"HR {h['hazard_ratio']:.2f}"
                if h.get("hazard_ratio") is not None
                else f"P(significant) {percent(h.get('p_significant'))}"
            )
            lines.append(
                f"  Efficacy · {h['main_comparison']} · {h['main_endpoint']}: {effect}"
            )
        for key, value in h.items():
            if key.startswith("serious_ae: "):
                lines.append(
                    f"  Any serious AE · {key.removeprefix('serious_ae: ')}: "
                    f"{percent(value)}"
                )
        lines += [f"  {a['analysis']}: {a['status']}" for a in self.raw["analyses"]]
        lines.append(
            "  Tables: .efficacy.table  .safety.table  .biology.table  (.raw for JSON)"
        )
        return "\n".join(lines)

    def _repr_html_(self):
        from ._report_tables import percent

        h = self.headlines()
        cards = []
        if h.get("p_success") is not None:
            cards.append(
                (
                    "P(Success)",
                    percent(h["p_success"]),
                    f"historical odds {percent(h.get('historical_odds'))}",
                )
            )
        if h.get("main_endpoint"):
            value = (
                f"HR {h['hazard_ratio']:.2f}"
                if h.get("hazard_ratio") is not None
                else percent(h.get("p_significant"))
            )
            cards.append(
                ("Efficacy", value, f"{h['main_comparison']} · {h['main_endpoint']}")
            )
        for key, value in h.items():
            if key.startswith("serious_ae: "):
                cards.append(
                    ("Any serious AE", percent(value), key.removeprefix("serious_ae: "))
                )
        card_html = "".join(
            "<div style='border:1px solid #ddd;border-radius:8px;"
            "padding:10px 14px;min-width:150px'>"
            f"<div style='font-size:12px;color:#666'>{_escape(t)}</div>"
            f"<div style='font-size:24px;font-weight:600'>{_escape(v)}</div>"
            f"<div style='font-size:12px;color:#666'>{_escape(c)}</div></div>"
            for t, v, c in cards
        )
        sections = "".join(
            f"<details><summary><strong>{_escape(a.title)}</strong> · "
            f"{_escape(a.status)}</summary>"
            f"{a._table_html()}</details>"
            for a in (self.efficacy, self.safety, self.biology)
        )
        return (
            self.summary._repr_html_()
            + "<div style='display:flex;gap:10px;flex-wrap:wrap;margin:10px 0'>"
            + f"{card_html}</div>"
            + sections
        )

    def save_json(self, path):
        """Write the complete report to a local JSON file; never call the API."""
        Path(path).write_text(json.dumps(self.raw, indent=2), encoding="utf-8")

    @property
    def summary(self):
        return _Summary(self)

    @property
    def biology(self):
        return _Analysis(self, "biology", "Biology", "plot_biology")

    @property
    def efficacy(self):
        return _Analysis(self, "efficacy", "Efficacy", "plot_efficacy")

    @property
    def safety(self):
        return _Analysis(self, "toxicity", "Safety", "plot_safety")


class _Summary:
    def __init__(self, prediction):
        self.prediction = prediction

    @property
    def data(self):
        """Analysis statuses and issues, without making network requests."""
        return self.prediction.raw["analyses"]

    def __repr__(self):
        return "\n".join(
            [repr(self.prediction), self.prediction.ui_url]
            + [f"{a['analysis']}: {a['status']}" for a in self.data]
        )

    def _repr_html_(self):
        p = self.prediction
        report = p.raw
        trial = report["request"]["trial"]
        rows = "".join(
            "<tr>"
            + "".join(
                "<td style='padding:8px;border-bottom:1px solid #ddd'>"
                f"{_escape(v)}</td>"
                for v in (
                    a["analysis"].replace("_", " ").title(),
                    a["status"],
                    "; ".join(i["message"] for i in a.get("issues", [])),
                )
            )
            + "</tr>"
            for a in report["analyses"]
        )
        provenance = {
            **p.provenance,
            "components": [
                {
                    k: c.get(k)
                    for k in ("component", "mode", "status", "release", "issues")
                }
                for c in report["components"]
            ],
        }
        return (
            f"<h2>{_escape(trial['name'])}</h2>"
            f"<p>{_escape(p.source)} · Overall status: "
            f"<strong>{_escape(report['status'])}</strong></p>"
            f"<p>{_escape(trial.get('registry_id', 'Custom trial'))} "
            f"· Run: {_escape(p.run_id)}</p>"
            f"<p><a href='{_escape(p.ui_url)}'>Open this same report in VERDICT</a></p>"
            "<table><thead><tr><th>Analysis</th><th>Status</th><th>Notes</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
            "<details><summary>Model and client provenance</summary>"
            f"<pre>{_escape(json.dumps(provenance, indent=2))}</pre></details>"
        )


class _Analysis:
    def __init__(self, prediction, kind, title, plot_name):
        self.prediction, self.kind, self.title, self.plot_name = (
            prediction,
            kind,
            title,
            plot_name,
        )

    @property
    def data(self):
        """This analysis's metadata and model outputs, without network requests."""
        analysis = next(
            (a for a in self.prediction.raw["analyses"] if a["analysis"] == self.kind),
            None,
        )
        components = (
            []
            if analysis is None
            else [
                c
                for c in self.prediction.raw["components"]
                if c["component"] in analysis.get("components", [])
            ]
        )
        return {"analysis": analysis, "components": components}

    @property
    def status(self):
        analysis = self.data["analysis"]
        return "not requested" if analysis is None else analysis["status"]

    @property
    def table(self):
        """This analysis as a readable pandas DataFrame (no network requests).

        Efficacy: one row per comparison and endpoint. Safety: events by arm.
        Biology: mean absolute log2 CPM change by tissue and comparison.
        """
        from . import _report_tables as tables

        return {
            "efficacy": tables.efficacy_table,
            "toxicity": tables.safety_table,
            "biology": tables.biology_table,
        }[self.kind](self.prediction.raw)

    def _table_html(self):
        table = self.table
        if table.empty:
            return "<p>No predictions.</p>"
        percent_columns = [c for c in table.columns if str(c).startswith("p_")]
        if self.kind == "toxicity":
            percent_columns = list(table.columns)
        from ._report_tables import percent

        formatted = table.copy()
        for column in percent_columns:
            formatted[column] = formatted[column].map(percent)
        return formatted.to_html(na_rep="—", index=self.kind != "efficacy")

    def __repr__(self):
        table = self.table
        body = "" if table.empty else "\n" + table.head(15).to_string()
        hint = "(.table for the DataFrame, .data for records, .plot() for charts)"
        return f"{self.title}: {self.status}{body}\n{hint}"

    def plot(self, **selection):
        """Plot saved model outputs; no inference. Biology accepts contrast/tissue_id;
        safety accepts comparison_id. Requires the pysynthbio[verdict] extra.
        """
        try:
            from . import _verdict_charts
        except ImportError as exc:
            raise ImportError(
                "Install pysynthbio[verdict] to display trial charts"
            ) from exc
        return getattr(_verdict_charts, self.plot_name)(
            self.prediction.raw, **selection
        )

    def _ipython_display_(self):
        from IPython.display import HTML, display

        report = self.prediction.raw
        analysis = next(
            (a for a in report["analyses"] if a["analysis"] == self.kind), None
        )
        if analysis is None:
            display(HTML(f"<h2>{self.title}</h2><p>Not requested in this run.</p>"))
            return
        notes = "".join(
            f"<p>{_escape(i['message'])}</p>" for i in analysis.get("issues", [])
        )
        display(
            HTML(
                f"<h2>{self.title}</h2><p>Status: "
                f"<strong>{_escape(analysis['status'])}</strong></p>{notes}"
                + self._table_html()
            )
        )
        try:
            self.plot()
        except ImportError:
            pass  # charts need the pysynthbio[verdict] extra; the table is shown
