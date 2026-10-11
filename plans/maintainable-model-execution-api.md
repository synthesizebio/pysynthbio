---
linear_project_url: "https://linear.app/synthesize-bio/project/maintainable-model-execution-api-13ac060c775d"
repository: "synthesizebio/pysynthbio"
delivery_mode: "incremental"
plan_branch: "main"
last_reconciled: "2026-10-09"
template_source: "https://github.com/synthesizebio/docs/blob/0deb68948fec1d906d12f223fb744e551610a33a/project-management/plan-template.md"
---

# Maintainable model execution API

<!-- sbio-delivery-modes:v1 -->

## Contribution and boundaries

[Linear project](https://linear.app/synthesize-bio/project/maintainable-model-execution-api-13ac060c775d).
This increment makes Platform validation errors readable in Python and preserves structured
responses for programmatic use. The larger SDK model/run interface remains future work.
No scientific, vocabulary, transport URL, polling, successful result or self-hosted change.

## Current state

- Overall acceptance: unassigned for the whole project. APP-2804 owns milestone 1 only.
- Delivery mode: incremental; this plan becomes canonical on main after its initial PR merges.
- Completed: none for this SDK increment; Platform #2698 is merged and its production error was verified.
- In progress: [APP-2820](https://linear.app/synthesize-bio/issue/APP-2820), one coherent SDK PR.
- Current executor: Codex in isolated codex/app-2820-validation-errors; accountable lead Stein.
- Next: test the recorded production response, review before pushing, complete CI and review.
- Needs: normal human review and package publication remain; dates unknown.

## Approach and sequence

1. Render validated path/message arrays from the server; keep full raw and parsed responses.
2. Preserve fallback text for older, malformed and nonstructured errors.
3. After this increment, propose model()/run() separately; do not introduce client-side schema copies.

## Deliverables and steps — APP-2820

- [x] Implement readable validation errors and structured response access.
- [x] Test the five-field production fixture and malformed/legacy compatibility through SDK transport.
- [x] Document catching the exception/condition and accessing its fields.
- [x] Run repository checks, package build/installed-artifact smoke test and dependency scan.

## Decisions and discoveries

- 2026-10-09 — Stein authorized this small SDK follow-up. Platform owns validation; SDKs only
  display its corrections. HTTP 400/422 retain status_code, response_body and response_json.
  Incomplete errors arrays fall back to original text rather than hiding entries.
- 2026-10-09 — The fixture records the exact October 8 production 422 body for the HCT116 request;
  the original attempt created zero jobs. Regression replay uses mocked HTTP, not new inference.

## Pre-merge validation

Local verification: 66 tests passed, 2 live integration tests skipped without credentials/base URL;
ruff and black pass. Package build and installed-wheel transport replay pass. Dependency scan is clean
after minimum fixed cryptography 50.0.0 and urllib3 2.8.0 lockfile updates; Linux CI covers SecretStorage.
Required checks: pytest, ruff, black, package build and installed wheel smoke test; sbio scan deps.

## Merge gates

Normal CI and human review must pass on the proposed revision. No automatic merge or publication.

## Interfaces and dependencies

Platform #2698 supplies structured errors. Older responses remain supported.
The companion R APP-2821 change is independently reviewable and releasable.

## Release and overall acceptance

Stein owns this deliverable's acceptance in APP-2820. Record package publication and effective
installed behavior there after release; neither a PR nor its merge establishes publication.
Project-level and milestone 1 acceptance remain distinct.

The generated local Claude skills symlink is ignored because the signed API commit path cannot
preserve symlink modes; the canonical generated skills are committed in .agents/skills.
