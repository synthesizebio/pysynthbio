---
name: sbio-commit-and-review
description: Commit, open a pull request, and work a review to completion in a Synthesize Bio repository using the sbio CLI. Use when committing changes, opening or updating a PR, or addressing review comments from Copilot or a human reviewer, in any repository containing a .sbio.toml file. Covers formatting before commit, why git commit and git push are not used here, how to pass prose safely, the comments/reply/resolve loop, and waiting for CI.
---

# Committing and handling review with `sbio`

Before applying this skill, read the repository's `.sbio.toml` and `AGENTS.md`.
If `[workflow] optional = true`, use this skill only when the user explicitly
requests the sbio workflow or the relevant sbio command. Otherwise follow the
repository's ordinary workflow; do not require sbio installation, authentication,
scanning, or App commits merely because its configuration or skills exist.
The user's selection persists through the task. Repository checks still apply.

This repository attributes agent work to the `synthesize-sandbox-agent[bot]`
GitHub App, so commits land `Verified` and are distinguishable from human ones
in the history. `sbio` is the shared CLI that does it. It infers owner/repo from
the git remote and the PR from the current branch, so the same commands work
unmodified in every repo.

If this repository has no `.sbio.toml`, this skill does not apply — stop here and
use ordinary git.

## Formatting

```bash
sbio fmt
```

Run **before every** `sbio gh commit`. The App commit path uses GraphQL
`createCommitOnBranch`, which never runs `git commit`, so lefthook and other
pre-commit hooks do not fire. `sbio fmt` formats the changed files instead,
inferring ruff (Python), biome (TypeScript/JSON), or tofu/terraform (`.tf`) from
the paths and the nearest config in the repo tree. No per-repo setup is required.

If `sbio fmt` fails, fix the reported issues and re-run before committing.

## Verifying

Run the repository's own checks — its `AGENTS.md` names them — and run them
**after your last edit, not merely before the commit**. A check is only evidence
about the tree it ran against, so anything you touch afterwards is unverified,
including the file you added last. Adding a test after typechecking and then
committing means CI is the first thing to typecheck that test, which is the
slowest possible place to learn it does not compile.

## Committing

```bash
sbio gh scratch commit.md    # -> <repo>/.sbio/commit.md, which git ignores
sbio gh commit "feat(SEC-123): headline" --body-file <the path it printed>
```

Do not use `git commit` or `git push`. They land as the user, unverified, and
lose the attribution the App exists to create.

**Write the body to a file and pass `--body-file`.** Keep `--body` for genuine
one-liners. A heredoc puts `$`, parentheses and newlines on the command line, and
command substitution runs before `sbio` does, so no approval rule can safely wave
it through — an agent using one stops for manual approval on every commit.

**Ask `sbio gh scratch` where to put it rather than choosing somewhere.** This
command commits everything that differs from the remote, untracked files included
— that sweep is what lets it run without staging — so a body file written next to
the code is part of the change it describes.

`scratch` hands back a path in `.sbio/`, a directory in this checkout that
`sbio` keeps ignored, so the sweep never sees it. Being part of the checkout, it
is also distinct between worktrees and between clones without any name to
collide.

The path is derived from where you are standing, not generated, so it is the same
every time — a later step can run `sbio gh scratch commit.md` again rather than
remember it. Do not substitute `$(sbio gh scratch ...)` into the command: command
substitution is the thing `--body-file` exists to avoid. Run it, read the path,
use it literally.

Two agents sharing one checkout is not solved by any of this, and cannot be: a
working tree is what `commit` snapshots, so concurrent commits from one checkout
sweep each other's half-finished work whatever the scratch files are called. Give
concurrent agents their own worktrees.

Write a body that explains *why*, the trade-offs, and what a reviewer would
otherwise have to reconstruct. Not a restatement of the diff, and not just a
ticket footer.

Headlines are Conventional Commits: `type(scope): subject`, imperative, no
trailing period. The scope is the Linear ticket (`SEC-123`, `APP-456`) or `deps`.
If no ticket is in context, ask rather than guessing.

## Opening the PR

```bash
sbio gh scratch pr.md    # prints a path; write the description there
sbio gh pr --body-file <the path it printed>
```

The title defaults to the first commit's headline, and reviewers come from
`.sbio.toml`. Without a body or explicit linking flags, the legacy default closes
the ticket inferred from the branch. For project work, use `--ticket APP-123
--link-mode related` and include project/plan links in the body; closeout may
remain after merge. Use `--base <project-branch>` for child PRs.
A child's first commit also needs `sbio gh commit --base <project-branch>`;
its local HEAD must match that remote base. See the project skill for lifecycle
and reporting; ordinary standalone changes retain the existing default. `--as-me` opens it
under your own account instead of the App's.

Through `scratch` again, and here it matters more than for a commit: `pr` cannot
drop the file the way `commit` drops its own, because the file outlives the
command. It is the *next* commit that sweeps it up, by which point nothing knows
it was ever a PR body.

## The review loop

**A PR is not done when it is opened.** Copilot posts a first pass within a couple
of minutes, so stopping at `sbio gh pr` leaves review comments arriving with
nobody reading them. Wait for it, then work the loop until comments is empty
and CI is green.

```bash
sbio gh scratch review.jsonl      # the path the next two lines write and read
sbio gh comments > <that path>    # unresolved threads, one JSON object per line
# add a "reply" to the rows you answered, then post the round in one invocation
sbio gh reply --replies-file <that path>
sbio gh resolve <thread_node_id> <thread_node_id>   # space-separated, any number
sbio gh comments                  # repeat until this prints nothing
gh pr checks --watch              # a red job is another round
```

The review file goes through `scratch` for the same reason the bodies do, and
this loop is where it bites hardest: it writes the file, then tells you to commit
the fixes before replying — so a review file the sweep can see is committed
verbatim beside the change answering it.

That last `comments` line is what makes this a loop rather than a single pass.
Keep going until `comments` prints nothing. Then wait for CI. A red check is
another review round: fix, commit, then `comments` again in case Copilot posted
on the new commit.

The answer goes in `reply`. **`body` is the reviewer's text and must be left
alone** — that separation is the only thing stopping an unedited file from
posting every review comment straight back at its author. Rows without a `reply`
are skipped, so the earlier messages `comments` prints for context need no edit.

Answer a whole round in one invocation rather than one comment at a time: each
invocation mints its own installation token, which is most of what the command
costs.

Commit the fixes *before* posting the replies, so the replies describe code that
is actually pushed.

### Reviewers can be wrong

Reply to every thread, including the ones you decline, and say why. A declined
comment with a reason reads as considered; a declined comment in silence reads as
unread.

Address every comment on its merits. A reviewer, human or bot, can be mistaken.
Where one is, say so in the reply and explain why, rather than changing the code
to silence it.

Check the claim before accepting it: run the test it says will fail, read the
file it says is broken. A comment with a wrong premise often still points at
something real — look for that before dismissing it. Both halves matter, and the
common failure is doing neither: accepting a wrong claim because a bot made it,
or dismissing a whole comment because its stated reason was wrong.

## When something fails

`sbio gh --help` carries the current flags and the whole sequence, and is kept up
to date in the tool. Prefer it over anything restated here.

Two failures look alike and are not:

- A bare `{"message": "Forbidden"}` with no `error` key is API Gateway rejecting
  the request before the token service ran — an IAM or request-signing problem,
  not a scope one.
- A 403 naming the App and a repository means the `synthesize-sandbox-agent` App
  is not installed on that repository. Installing it is the whole fix.

## Escape hatch

Plain `git commit` / `git push` and `gh pr create` still work for what `sbio`
cannot do: files over 10 MB, complex rebases, or when AWS access is unavailable.
Those commits land as the user, unverified.

`sbio gh commit` also refuses to add a brand-new executable file, because the
GitHub API it uses carries contents but not file mode, and the bit would be
silently dropped. Either commit the executable separately with plain git, or keep
it non-executable and invoke it as `bash path/to/script.sh`.
