# pysynthbio docs (Mintlify source)

This directory holds the **Python SDK** section of the [Synthesize Bio docs site](https://synthesizebio.mintlify.app), which is published from [`synthesizebio/mintlify-docs`](https://github.com/synthesizebio/mintlify-docs).

## How publishing works

1. Edit MDX files here in `docs/` on `main` (or in a PR targeting `main`).
2. When the PR merges, the [`Sync docs to mintlify-docs`](../.github/workflows/sync-docs-to-mintlify.yml) workflow runs.
3. That workflow opens / updates a single PR in `synthesizebio/mintlify-docs` named `sync/pysynthbio` that copies `pysynthbio/docs/**` into `mintlify-docs/python-sdk/**`.
4. Merging that PR triggers a Mintlify production deploy.

The mapping is:

| `pysynthbio/docs/...`                       | Live URL                                                                |
| ------------------------------------------- | ----------------------------------------------------------------------- |
| `index.mdx`                                 | `/python-sdk`                                                           |
| `installation.mdx`                          | `/python-sdk/installation`                                              |
| `getting-started.mdx`                       | `/python-sdk/getting-started`                                           |
| `models/baseline.mdx`                       | `/python-sdk/models/baseline`                                           |
| `models/reference-conditioning.mdx`         | `/python-sdk/models/reference-conditioning`                             |
| `models/metadata-prediction.mdx`            | `/python-sdk/models/metadata-prediction`                                |
| `license.mdx`                               | `/python-sdk/license`                                                   |

## Adding a new page

1. Create a new `.mdx` file under `docs/` (or a subfolder).
2. Add a frontmatter block:

   ```mdx
   ---
   title: "My new page"
   description: "Short, public-facing summary."
   ---
   ```

3. **Update [`mintlify-docs/docs.json`](https://github.com/synthesizebio/mintlify-docs/blob/main/docs.json)** in a separate PR to add the page to the `Python SDK` group's navigation. The sync workflow only copies file content — it doesn't edit `docs.json`.

## Local preview

The fastest preview loop is in `mintlify-docs` itself (`mint dev`). To preview a draft of these MDX files locally:

```bash
git clone git@github.com:synthesizebio/mintlify-docs.git
cd mintlify-docs
rm -rf python-sdk && cp -R ../pysynthbio/docs python-sdk
npx mint dev
```

## Required GitHub configuration

The sync workflow needs a repository secret named **`MINTLIFY_DOCS_TOKEN`**: a fine-grained PAT (or GitHub App installation token) scoped to `synthesizebio/mintlify-docs` with `Contents: read+write` and `Pull requests: read+write`.
