"""File and display contracts only; these tests do not simulate model execution."""

import json

import pytest

from pysynthbio import VerdictClient, load_trial, load_trial_result


def test_saved_result_roundtrip_and_escaped_summary(tmp_path):
    report = {
        "run_id": "saved-report",
        "status": "unavailable",
        "request": {"trial": {"name": "<script>example</script>", "endpoints": []}},
        "components": [],
        "analyses": [{"analysis": "biology", "status": "unavailable", "issues": []}],
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    result = load_trial_result(path)
    assert result.raw == report
    assert result.biology.data["analysis"]["status"] == "unavailable"
    assert result.efficacy.data["analysis"] is None
    assert "&lt;script&gt;" in result.summary._repr_html_()
    assert "offline" in result.summary._repr_html_()
    result.save_json(path)
    assert load_trial_result(path).raw == report
    with pytest.raises(ValueError, match="workflow input"):
        load_trial(path)


def test_trial_file_keeps_nulls_arrays_and_numbers(tmp_path):
    trial = {
        "trial": {"endpoints": [], "age": None, "value": 0.123456789012345},
        "requested_analyses": ["biology"],
    }
    path = tmp_path / "trial.json"
    path.write_text(json.dumps(trial))
    assert load_trial(path) == trial
    with pytest.raises(ValueError, match="trial result"):
        load_trial_result(path)


def test_invalid_retry_key_and_polling_limits_fail_before_network():
    client = VerdictClient(api_key="unused-for-local-validation")
    for key in [None, "", "  "]:
        with pytest.raises(ValueError, match="idempotency_key"):
            client.run_trial({}, idempotency_key=key)
    for value in [0, -1, float("nan"), float("inf")]:
        with pytest.raises(ValueError, match="finite and positive"):
            client.run_trial({}, idempotency_key="unused", timeout=value)
