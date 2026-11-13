import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pysynthbio import Trial, TrialDesignError, TrialReviewError, trial_vocabulary
from pysynthbio.trial_design import _variant_key, choose
from pysynthbio.verdict import VerdictClient


def hcc_trial():
    trial = Trial("HCC phase 3")
    trial.population.set_disease("hepatocellular carcinoma")
    trial.edit(
        phase="phase 3", allocation="randomized", blinding="open label", min_age=18
    )
    trial.arms.add("Durvalumab", type="experimental", treatments=["durvalumab"])
    trial.arms.add(
        "Sorafenib",
        type="active comparator",
        treatments=[{"name": "sorafenib", "dose": "400 mg twice daily"}],
    )
    trial.endpoints.add(
        "Overall survival",
        family="overall survival",
        effect_measure="hazard ratio",
        hypothesis="superiority",
    )
    return trial


def test_forgiving_vocabulary_and_helpful_errors():
    assert choose("phase", "Phase 3") == "PHASE3"
    assert choose("phase", "early phase 1") == "EARLY_PHASE1"
    assert choose("blinding", "open label") == "NONE"
    assert choose("arm_type", "Active comparator") == "active_comparator"
    assert choose("effect_measure", "Hazard Ratio") == "hazard_ratio"
    with pytest.raises(TrialDesignError, match="DOUBLE"):
        choose("blinding", "sorta blind")
    assert "QUADRUPLE" in trial_vocabulary().describe("blinding")
    fields = Trial.describe()
    assert {"details.phases", "arms[].treatments[].chembl", "comparisons[]"} <= set(
        fields["field"]
    )


def test_builds_the_clinical_design_contract_and_round_trips(tmp_path):
    trial = hcc_trial()
    trial.details.edit(phase="Phase 1/2", start_date="2024-03")
    design = trial.to_dict()
    assert design["schemaVersion"] == "verdict.clinical-design/v1"
    assert design["population"]["phases"] == ["PHASE1", "PHASE2"]
    assert design["metadata"]["blinding"] == "NONE"
    assert design["arms"][1]["components"] == [
        {"name": "sorafenib", "doseNotes": "400 mg twice daily", "doseStated": True}
    ]
    assert design["endpoints"][0]["effectMeasure"] == "hazard_ratio"
    # Two arms and no explicit comparison: the server builds the default one.
    assert "comparisons" not in design
    assert design["requestedAnalyses"] == ["efficacy", "toxicity", "biology"]
    path = trial.save(tmp_path / "design.json")
    assert Trial.load(path).to_dict() == design


def test_arms_endpoints_and_comparisons_stay_consistent():
    trial = hcc_trial()
    trial.arms.add(
        "Placebo", type="placebo", treatments=[{"name": "Placebo", "role": "placebo"}]
    )
    trial.comparisons.all_pairs("Sorafenib")
    assert [c.label for c in trial.comparisons] == [
        "Durvalumab vs Sorafenib",
        "Placebo vs Sorafenib",
    ]
    trial.endpoints.add(
        "Progression-free survival",
        effect_measure="hazard ratio",
        hypothesis="superiority",
    )
    assert all(
        len(c.endpoints) == 2 for c in trial.comparisons
    )  # new endpoints join comparisons
    trial.comparisons.add("Sorafenib", "Durvalumab")  # an existing pair reverses
    assert (
        trial.comparisons.get("Sorafenib vs Durvalumab").experimental.label
        == "Sorafenib"
    )
    trial.comparisons.edit(
        "Sorafenib", "Durvalumab", remove_endpoints="Overall survival"
    )
    trial.arms.remove("Placebo")
    assert [c.label for c in trial.comparisons] == ["Sorafenib vs Durvalumab"]
    trial.endpoints.remove("Progression-free survival")
    assert len(trial.comparisons) == 0  # left without endpoints
    design = trial.to_dict()
    assert "comparisons" not in design or design["comparisons"] == []


def test_edits_are_validated_atomically():
    trial = hcc_trial()
    with pytest.raises(TrialDesignError):
        trial.edit(sex="male", min_age=80, max_age=40)
    assert trial.population.sex is None and trial.population.min_age == 18
    with pytest.raises(TrialDesignError, match="already exists"):
        trial.arms.rename("Durvalumab", "sorafenib")
    trial.arms.rename("Durvalumab", "Durvalumab")  # renaming to itself is fine
    with pytest.raises(TrialDesignError, match="at most one"):
        trial.arms["Durvalumab"].treatments.edit(
            "durvalumab", chembl="CHEMBL1", smiles="C"
        )
    with pytest.raises(TrialDesignError, match="Unknown"):
        trial.edit(sample_size=100)


def test_variants_are_independent_and_keys_are_valid():
    trial = hcc_trial()
    variants = trial.vary(sex=["female", "male"], min_age=[18, 65])
    assert len(variants) == 4
    assert variants["sex=male, min_age=65"].population.sex == "male"
    assert trial.population.sex is None
    key = _variant_key("hcc-sweep", "sex=male, min_age=65")
    assert VerdictClient._DESIGN_KEY.match(key)


def test_imports_the_form_draft():
    proposal = json.loads(
        (Path(__file__).parent / "data" / "import_proposal.json").read_text()
    )
    trial = Trial.from_import(proposal)
    assert trial.population.disease == "Breast cancer"
    assert trial.details.phases == ["PHASE3"]
    assert trial.population.healthy_volunteers is False
    assert (
        trial.arms["Ribociclib + letrozole"].treatments["ribociclib"].chembl
        == "CHEMBL3545110"
    )
    assert [c.label for c in trial.comparisons] == [
        "Ribociclib + letrozole vs Placebo + letrozole"
    ]
    assert trial.to_dict()["endpoints"][0]["title"] == "Progression-free survival"


def fake_client(availability):
    client = MagicMock()

    def prepare(design, key):
        outcome = {
            a: availability.get(a, "accepted") for a in design["requestedAnalyses"]
        }
        return {
            "issues": [],
            "entities": [],
            "assumptions": [],
            "availability": [
                {
                    "analysis": a,
                    "outcome": o,
                    "reasons": ["why"] if o == "unavailable" else [],
                }
                for a, o in outcome.items()
            ],
            "reviewFingerprint": "f" * 64,
        }

    client.prepare_trial_design.side_effect = prepare
    client.run_trial_design.return_value = {"id": "run-1", "status": "queued"}
    client.wait_for_trial.return_value = "result"
    return client


def test_run_skips_unavailable_analyses_and_reuses_the_key():
    client = fake_client({"biology": "unavailable"})
    assert hcc_trial().run(client, idempotency_key="hcc-1") == "result"
    design, fingerprint = client.run_trial_design.call_args.args
    assert design["requestedAnalyses"] == ["efficacy", "toxicity"]
    assert client.run_trial_design.call_args.kwargs["idempotency_key"] == "hcc-1"
    assert all(
        call.args[1] == "hcc-1" for call in client.prepare_trial_design.call_args_list
    )


def test_run_refuses_when_nothing_can_run():
    client = fake_client(
        {a: "unavailable" for a in ("efficacy", "toxicity", "biology")}
    )
    with pytest.raises(TrialReviewError):
        hcc_trial().run(client, idempotency_key="hcc-2")
    client.run_trial_design.assert_not_called()


def test_trial_set_submits_everything_before_waiting():
    client = fake_client({})
    calls = []
    client.run_trial_design.side_effect = lambda *a, **k: calls.append("submit") or {
        "id": k["idempotency_key"]
    }
    client.wait_for_trial.side_effect = lambda run_id, **k: calls.append(
        "wait"
    ) or MagicMock(headlines=lambda: {"status": "complete", "run": run_id})
    results = (
        hcc_trial()
        .vary(sex=["female", "male"])
        .run(client, idempotency_prefix="by-sex")
    )
    assert calls == ["submit", "submit", "wait", "wait"]
    assert list(results.table["run"]) == ["by-sex:sex-female", "by-sex:sex-male"]


def test_client_design_endpoints():
    client = VerdictClient("key")
    client._request = MagicMock(return_value={"ok": True})
    client.prepare_trial_design(hcc_trial(), "k1")
    method, path, body = client._request.call_args.args
    assert (method, path) == ("POST", "trial-designs/prepare")
    assert body["idempotencyKey"] == "k1" and body["design"]["name"] == "HCC phase 3"
    with pytest.raises(ValueError, match="letters"):
        client.run_trial_design({}, "f" * 64, idempotency_key="has spaces")


def test_results_read_as_tables_not_json():
    from pysynthbio.trials import TrialResult

    report = json.loads(
        (Path(__file__).parent / "data" / "trial_report.json").read_text()
    )
    result = TrialResult(report, "https://verdict.synthesize.bio", "test")
    efficacy = result.efficacy.table
    # The trial's first comparison leads, primary endpoint first, like the web report.
    assert efficacy.iloc[0]["hazard_ratio"] == 0.75
    assert list(efficacy["role"][:2]) == ["primary", "secondary"]
    safety = result.safety.table
    assert safety.index[0] == "Any serious adverse event"
    assert safety.iloc[1:].max(axis=1).is_monotonic_decreasing
    assert result.biology.table.index[0] == "Lung"
    text = repr(result)
    assert "P(Success) 60%" in text and "Any serious AE" in text
    assert "<details>" in result._repr_html_()
    assert result.headlines()["p_success"] == pytest.approx(0.6043, abs=1e-3)


def test_run_drops_toxicity_when_the_release_needs_a_query():
    client = fake_client({})
    first = client.prepare_trial_design.side_effect

    def prepare(design, key):
        reply = first(design, key)
        if "toxicity" in design["requestedAnalyses"]:
            reply["issues"] = [{"field": "query.0.term", "message": "needs a query"}]
            reply["reviewFingerprint"] = None
        return reply

    client.prepare_trial_design.side_effect = prepare
    hcc_trial().run(client, idempotency_key="hcc-3")
    design, _ = client.run_trial_design.call_args.args
    assert design["requestedAnalyses"] == ["efficacy", "biology"]
