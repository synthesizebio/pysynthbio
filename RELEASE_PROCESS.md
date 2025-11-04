# Release Process

## Branching Strategy

We follow a **stable mainline** approach where `main` always reflects the latest published version on PyPI.

### Workflow

1. Create a pre-release branch from `main` (e.g., `release/v3.1.0`)
1. Internal testing by installing from GitHub branch

- `pip install git+https://github.com/synthesizebio/pysynthbio@release/v3.1.0`

1. Merge to `main` when validated
1. Tag with the version number (e.g., `v3.1.0`)
1. Pushing the tag will trigger the `new-release` github action
