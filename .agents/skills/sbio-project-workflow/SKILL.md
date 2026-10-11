---
name: sbio-project-workflow
description: Bootstrap, execute, hand off, or report on a Linear project, including Linear-only non-code work and repository work using sbio. Use for project adoption, execution, status drafts, and closeout; ordinary standalone changes need only the commit/review skill.
---

# Project execution and reporting

Before applying this skill, read the repository's `.sbio.toml` and `AGENTS.md`.
If `[workflow] optional = true`, use this skill only when the user explicitly
requests the sbio workflow or the relevant sbio command. Otherwise follow the
repository's ordinary workflow; do not require sbio installation, authentication,
scanning, or App commits merely because its configuration or skills exist.
The user's selection persists through the task. Repository checks still apply.

The [project protocol](https://github.com/synthesizebio/docs/blob/main/project-management/protocol.md)
owns policy. Start with the Linear brief and its named execution records. This
skill operationalizes that policy; it grants no posting, merging, deployment,
workspace-setting, permission-expansion, or commitment-changing authority.

## Choose the execution path

Read the outcome, accountable lead, done criteria, initiative requirements, and
optional handoff brief. Reuse existing records. Proposals need no branches or plans.
Resolve execution-critical gaps within the user's scope; do not invent ownership,
deadlines, evidence, or resource commitments.

- **Linear-only:** use milestones, owned issues, linked documents and result
  artifacts. Skip repository setup, Git, and `sbio project inspect`. Scientific
  review, a signed agreement, or a customer handoff can be acceptance evidence.
- **Repository:** read repo instructions and each maintained `plans/<slug>.md`.
  Use [repository procedures](https://github.com/synthesizebio/tools/blob/main/docs/project-workflows.md)
  for setup, metadata, linking, inspection, reconciliation, and closeout.
- **Mixed:** inspect both paths and their dependencies; Linear owns coordination
  and combined acceptance. A repo's successful checks do not prove the whole outcome.

Check intended end artifacts and reviews required by the applicable initiative,
following the linked shared protocol and the initiative brief.
Draft artifacts describe intended results; they do not establish achieved evidence.
Research may complete with a negative result if its agreed stopping criteria permit
that; distinguish cycle completion from reaching the metric target.

## Start, execute, or hand off

Before taking over, inspect existing work and identify the current executor. Agree
who continues, then record the executor, work in progress, branch/PR or artifact,
outstanding validation, and next action in the existing issue or execution record.
Keep the accountable owner distinct. An assignee or comment is not a concurrency
lock; use isolated checkouts for concurrent repository work.

For repo setup, read `sbio project init --help`, fetch origin, and preview with
`--dry-run` before applying. New repo work defaults to incremental delivery; prefer
one coherent PR when sufficient. Existing plans without delivery_mode retain
project-branch behavior. Never migrate implicitly. Read the recorded canonical
branch and reconcile mode/identity conflicts before relying on a plan. Preserve
repo-specific checks and authorization; change generated skills at their tools
source, not in consumers.

Use `sbio gh pr --project <slug> --ticket <deliverable-ticket> --link-mode related`
with prose in a scratch file. Verify the published plan and explicit membership,
not just a title or branch name. Reconcile concurrent plan edits and validate the
affected combined behavior before merge. Follow the commit/review and dependency
scan skills. Detailed command and branch procedures belong in the tools reference.

## Surface decisions and blockers

Track an unresolved decision in an owned issue when it blocks committed work,
threatens a milestone, changes an agreed commitment, or requires cross-project
coordination. Reuse an existing issue that owns the action. Routine questions stay
with the work; another person's involvement alone does not justify a new issue.

Within authorized scope, record the specific action/decision, owner, needed-by date
or unknown, affected milestone, and consequence. Use native issue blocking relations
for specific dependencies, including across projects. Keep blocked deliverables
assigned to their delivery owners; assign unblocking issues to whoever must act.
Project dependencies are finish-to-start sequencing; do not mark an entire project
blocked when only one deliverable awaits input. Do not add redundant blocked labels
or statuses. If no write tool or authorization is available, provide a precise draft.

A resolved decision retains its rationale and evidence in the issue. Reconcile any
authorized commitment changes in the brief. A decision's resolution does not prove
the blocked deliverable is complete. Surface capacity conflicts for the accountable
owner; agents do not allocate people or invent commitments.

## Inspect and report

Scan every active project's summary, latest update, next milestone, dependencies,
and freshness before choosing deeper reads. Investigate changed work, upcoming gates,
uncertainty, conflicting claims, or stale reporting. Check sufficient current evidence
for every material progress, health, and acceptance claim, including green projects.
An update older than the weekly cadence is a reporting gap, not proof of a blocker;
an old plan date alone says nothing about execution. Never refresh dates without
checking the state they describe.

For repo contributions, run `sbio project inspect <slug> --json` and read the
published plan at its returned revision. Inspect collects PR/check evidence, not
Linear state, review discussions, external artifacts, or acceptance decisions;
read those separately. Report missing/retargeted membership, pagination limits,
identity conflicts, inaccessible sources, and alternate-identity reads as coverage
gaps. Local/open-PR changes are proposals; do not union pending checklists or
double-count final and child PRs. Distinguish implemented, merged, released, and
validated results when material. Read the reference for mode-specific baselines.

Draft Progress (1–3 sentences), Needs / blockers / dependencies, Next milestone,
and evidence links, usually 75–125 words. Name action/decision, owner, timing and
consequence, or explicit unknowns/None. Missing evidence cannot support “None.”
Keep detailed checked date, identity, coverage, and reconciliation in the existing
record or update; no separate audit document is mandatory. Preserve prior detailed
evidence before shortening an existing update or publishing a superseding summary.

Initiative summaries lead with cross-project needs and outcome evidence, not ticket
counts or averaged health. Meeting preparation highlights intervention needs plus
progress and upcoming milestones; do not ignore unflagged projects. Publish and
change statuses only within authorization. Human owners decide commitment changes.

## Finish

Verify all agreed criteria, reviews, required handoffs, and cross-project acceptance.
For repo work, record deliverable release evidence and overall acceptance in their
owned tickets; follow the reference for plan maintenance and link repair. For
Linear-only work, link reviewed results or other required artifacts in the named
acceptance record. A merge, issued report, or resolved blocker alone is insufficient.
Preserve accepted gaps and their authorizing decisions, distinguish later recovery,
and retain discoverable history for canceled work. Publish a concise final update
when authorized. Leave unperformed checks and missing human feedback explicit.
