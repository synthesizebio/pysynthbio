# pysynthbio <img src="assets/logomark.png" style="width: 80px;" alt="Logomark">

`pysynthbio` is an Python package that provides a convenient interface to the [Synthesize Bio](https://www.synthesize.bio/) API, allowing users to generate realistic gene expression data based on specified biological conditions. This package enables researchers to easily access AI-generated transcriptomic data for various modalities including bulk RNA-seq and single-cell RNA-seq.

To generate datasets without code, use our [web platform](https://app.synthesize.bio/datasets/).

[See the full documentation here](https://synthesizebio.github.io/pysynthbio/)

For questions, suggestions, and support, email us at [support@synthesize.bio](mailto:support@synthesize.bio)

## Installation

To start using pysynthbio, first you need to have an account with synthesize.bio.
[Go here to create one](https://app.synthesize.bio/)

Then on your machine you can install using pip:

```
pip install pysynthbio
```

To ensure it installed you can run `pip show pysynthbio`.

### Installing from a GitHub Release

Alternatively, you can install a specific version directly from its GitHub Release page. This is useful for testing pre-releases or specific tagged versions.

1.  Go to the [Releases page](https://github.com/synthesizebio/pysynthbio/releases) of the repository.
2.  Find the release tag you want to install (e.g., `v2.2.0`).
3.  Expand the "Assets" section for that release.
4.  Download the `.whl` (wheel) file or the `.tar.gz` (source distribution) file. The wheel file is generally preferred if available for your platform.
5.  Install the downloaded file using pip, replacing `<path_to_downloaded_file>` with the actual path to the file:

    ```bash
    # Example using a downloaded wheel file
    pip install /path/to/pysynthbio-2.2.0-py3-none-any.whl

    # Example using a downloaded source distribution
    pip install /path/to/pysynthbio-2.2.0.tar.gz
    ```

## Usage

[See the full documentation here](https://synthesizebio.github.io/pysynthbio/)
