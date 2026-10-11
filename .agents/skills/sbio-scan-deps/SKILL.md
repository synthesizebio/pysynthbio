---
name: sbio-scan-deps
description: Scan and remediate dependency CVEs with `sbio scan deps` (Trivy). Use during self-verification after making changes, when Trivy reports MEDIUM/HIGH/CRITICAL findings, when adding or upgrading packages, or when the user asks to scan or fix vulnerabilities. Covers uv and bun, direct vs transitive upgrades, lockfile sync, `.trivyignore`, and the `deps` commit scope.
---

# Scanning and fixing dependency CVEs with `sbio`

Before applying this skill, read the repository's `.sbio.toml` and `AGENTS.md`.
If `[workflow] optional = true`, use this skill only when the user explicitly
requests the sbio workflow or the relevant sbio command. Otherwise follow the
repository's ordinary workflow; do not require sbio installation, authentication,
scanning, or App commits merely because its configuration or skills exist.
The user's selection persists through the task. Repository checks still apply.

`sbio scan deps` wraps Trivy so every repo scans the same way: tracked lock
files, `MEDIUM,HIGH,CRITICAL`, `--ignore-unfixed`, the repo's `.trivyignore`.
It exits 1 on findings, so it is a gate. New CVEs appear against packages
nobody touched, so run it on every verification cycle, not only after a
dependency change.

```bash
sbio scan deps
```

`sbio scan deps --help` carries the flags. Prefer it over anything restated
here. `--path` (repeatable) narrows to a directory or lock file after you
have fixed one target.

A repo with no tracked lock files is a clean no-op. Do not install Trivy just
to be told there is nothing to scan.

## If Trivy is missing

Skip the scan and tell the user:

```
Trivy is not installed — run `sbio scan deps` for install instructions.
```

Do not block the session.

## If the scan is clean

Done. Do not upgrade anything "while you are here."

## If there are findings

Fix every fixable MEDIUM/HIGH/CRITICAL, including packages this change did
not touch. Use the **minimum fixed version** Trivy prints, not the latest
available. In `pyproject.toml`, hard-pin with `==`.

Work in the directory that owns the lock file. After any manifest edit, sync
the lock — never hand-edit `uv.lock` or `bun.lock`. Then re-run the scan
(narrow with `--path` if only one tree changed) and the checks that tree
already has (lint, typecheck, tests). Retry the scan-and-check loop **once**.
A second failure: stop and explain.

### `uv.lock`

Direct: bump the pin in that project's `pyproject.toml`, then `uv lock && uv sync`.

Transitive: `uv tree --invert --package <pkg>` to find the parent. Upgrade the
parent first. If that does not clear it, add an override in
`[tool.uv] override-dependencies` (for example `"cryptography==46.0.7"`), then
`uv lock && uv sync`.

### `bun.lock`

Direct: `bun update <package>` in the directory that owns the lock file.

Transitive: find the parent with `grep <package> bun.lock` (text lockfile;
`bun.lockb` is the older binary — do not grep it). Try `bun update <parent>`.
If that does not clear it, add an `overrides` entry in that `package.json`
and run `bun install`.

### Other lock files

Same idea: upgrade the owning manifest, regenerate the lock with that
ecosystem's tool, re-scan.

### Cannot fix

No upstream patch, or the upgrade breaks compatibility: add the id to
`.trivyignore` with a comment that says why it is safe to ignore, matching
the existing pattern in that file. Re-run `sbio scan deps` to confirm clean.

## Committing the fix

Scope is `deps` unless a Linear ticket is already in context. Follow
`sbio-commit-and-review` for how to land it.
