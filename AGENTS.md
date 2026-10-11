# Repository checks

Run `uv sync --frozen --all-extras`, `uv run ruff check .`, `uv run black --check .`,
`uv run pytest`, and `uv run python -m build`. Live integration tests require credentials;
never claim skipped tests ran. Verify changed behavior against the installed wheel.

<!-- sbio: start -->
## sbio workflow

Use [.agents/skills/sbio-commit-and-review/SKILL.md](.agents/skills/sbio-commit-and-review/SKILL.md)
for formatting, commits, pull requests, and the review loop.
Run `sbio scan deps` alongside this repository's checks; follow
[.agents/skills/sbio-scan-deps/SKILL.md](.agents/skills/sbio-scan-deps/SKILL.md)
when handling dependency findings.
For Linear project adoption, execution, reporting, or closeout, use
[.agents/skills/sbio-project-workflow/SKILL.md](.agents/skills/sbio-project-workflow/SKILL.md).

The `sbio-*` skills installed by `sbio skills install` are committed generated
artifacts owned by `synthesizebio/tools`. Change them there; use `sbio skills diff`
to review updates and `sbio skills check` to detect drift.
<!-- sbio: end -->

<!-- sbio-project: start -->
## Linear project execution

Read the [project protocol](https://github.com/synthesizebio/docs/blob/main/project-management/protocol.md), the Linear brief, and its named execution records.
Repository work uses `plans/<project-slug>.md`; Linear-only work needs no repo commands.
Use the [project skill](.agents/skills/sbio-project-workflow/SKILL.md) for execution and reporting.
Linear owns commitments and acceptance; execution records own approach, current work, and evidence.
Follow the recorded delivery mode; never migrate existing plans implicitly.
Preserve this repo's checks, review, security, merge, and deployment requirements and existing authorization.
<!-- sbio-project: end -->
