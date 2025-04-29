def get_model_endpoints() -> dict:
    """
    Returns a dictionary of model endpoints.

    Returns
    -------
    dict
        A dictionary containing the model endpoints.
    """
    return MODELS


def get_valid_modalities() -> dict:
    """
    Returns a dictionary of possible modalities per model.

    Returns
    -------
    dict
        A dictionary containing the modalities.
    """
    return MODEL_MODALITIES


def get_valid_query(model) -> dict:
    """
    Generates a sample query for prediction and validation.

    Parameters
    ----------
    model: str
        name of the model (e.g., 'meanv1.0', 'rMetalv0.6')
    Returns
    -------
    dict
        A dictionary representing a valid query with modality and input data.
    """
    # Extract the version number from the model name
    match = re.search(r"v(\d+\.\d+)", model)
    if not match:
        raise ValueError("Model name is not valid. Use `get_model_endpoints()`.")

    version_number = float(match.group(1))

    if version_number < 1.0:
        return {
            "modality": "sra",
            "num_samples": 5,
            "inputs": [
                str(
                    {
                        "cell_line": "A-549",
                        "perturbation": "ABL1",
                        "perturbation_type": "crispr",
                        "perturbation_time": "96 hours",
                        "sample_type": "cell line",
                    }
                ),
                str(
                    {
                        "disease": "gastrointestinal stromal tumor",
                        "age": "65 years",
                        "sex": "female",
                        "sample_type": "primary tissue",
                        "tissue": "stomach",
                    }
                ),
            ],
        }
    else:
        return {
            "output_modality": "sra",
            "mode": "mean estimation",
            "return_classifier_probs": True,
            "seed": 11,
            "inputs": [
                {
                    "metadata": {
                        "cell_line": "A-549",
                        "perturbation": "ABL1",
                        "perturbation_type": "crispr",
                        "perturbation_time": "96 hours",
                        "sample_type": "cell line",
                    },
                    "num_samples": 5,
                },
                {
                    "metadata": {
                        "disease": "gastrointestinal stromal tumor",
                        "age": "65 years",
                        "sex": "female",
                        "sample_type": "primary tissue",
                        "tissue": "stomach",
                    },
                    "num_samples": 5,
                },
            ],
        }


def predict_query(
    query: dict,
    as_counts: bool = True,
    endpoint: str = ENDPOINT_URL,
) -> dict[str, pd.DataFrame]:
    """
    Sends a query to the Hugging Face API for prediction and retrieves samples.

    Parameters
    ----------
    query : dict
        A dictionary representing the query data to send to the API.
    as_counts : bool, optional
        If True, transforms the predicted expression data into counts (default is True).
    endpoint : str, optional

    Returns
    -------
    dict
        metadata: pd.DataFrame containing metadata for each sample
        expression: pd.DataFrame containing expression data for each sample

    Raises
    ------
    KeyError
        If the HF_TOKEN environment variable is not set
    ValueError
        If API fails

    """

    if "HF_TOKEN" not in os.environ:
        raise KeyError("Please set the HF_TOKEN environment variable")

    validate_query(query)

    response = requests.post(
        url=endpoint,
        headers={
            "Accept": "application/json",
            "Authorization": "Bearer " + os.environ["HF_TOKEN"],
            "Content-Type": "application/json",
        },
        json=query,
    )

    # Check the HTTP status code
    if response.status_code != 200:
        raise ValueError(
            f"API request failed with status {response.status_code}: {response.text}"
        )
    # Parse the response JSON
    try:
        content = response.json()
        if not isinstance(content, dict):  # prev versions returned list
            content = content[0]
    except json.JSONDecodeError:
        raise ValueError(f"Failed to decode JSON from API response: {response.text}")

    # this is hack due to status 200 returning errors
    for key in ("error", "errors"):
        if key in content:
            raise ValueError(f"Error in response from API received: {content[key]}")

    # deprecate this logic once we deprecate pre-v1 models
    if "samples" in content:
        samples = content["samples"]
        metadata = expand_metadata(query)
        expression = process_samples(samples, query["modality"])
    else:
        expression = pd.concat(
            [
                pd.DataFrame(output["expression"], columns=content["gene_order"])
                for output in content["outputs"]
            ],
            ignore_index=True,
        )
        metadata_rows = [
            output["metadata"]
            for output in content["outputs"]
            for _ in range(len(output["expression"]))
        ]
        metadata = pd.DataFrame(metadata_rows)

    # logic for counts
    if as_counts and endpoint in LOG_CPM_ENDPOINTS:
        # convert to counts if user requests and model returns log1p cpm values
        expression = transform_to_counts(expression)
    elif as_counts and endpoint not in LOG_CPM_ENDPOINTS:
        # convert to integer for models that return counts and user wants counts
        expression = expression.astype(int)
    elif not as_counts and endpoint not in LOG_CPM_ENDPOINTS:
        # if user does not want counts and is hitting a counts model, return log1p cpm
        expression = log_cpm(expression)
    else:
        # if not as counts and model returns log1p cpm, return as is
        pass

    return {"metadata": metadata, "expression": expression}


def validate_query(query: dict) -> None:
    """
    Validates the structure and contents of the query.

    Parameters
    ----------
    query : dict
        The query dictionary.

    Raises
    ------
    TypeError
        If the query is not a dictionary.
    ValueError
        If the query is missing required keys.
    """
    # Check that the query is a dictionary
    if not isinstance(query, dict):
        raise TypeError(
            f"Expected `query` to be a dictionary, but got {type(query).__name__}"
        )

    # Validate required keys
    required_keys = {"inputs"}
    missing_keys = required_keys - query.keys()
    if missing_keys:
        raise ValueError(
            f"Missing required keys in query: {missing_keys}. Use `get_valid_query()` to get an example."
        )


def process_samples(samples: list, modality: str) -> pd.DataFrame:
    """
    Processes the samples returned from the API into a DataFrame.

    Parameters
    ----------

    samples : list of lists
        A list of lists containing the expression data for each sample.

    modality : str
        The modality of the data (e.g. "bulk_rna-seq", "lincs", "single_cell_rna-seq", "sra").

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the expression data for all samples with ensembl gene id in columns

    """

    expression = pd.DataFrame(
        data=samples,
        columns=get_gene_order(modality),
    ).clip(lower=0)

    return expression


@lru_cache()
def get_gene_order(modality):
    """
    Reads the gene order for a given modality.

    Parameters
    ----------
    modality : str
        The modality for which to read the gene order.

    Returns
    -------
    list
        A list of gene ids in the order predicted by the model.

    """
    with open("utils/ai_gene_order.json", "r") as file:
        data = json.load(file)
    return data[modality]


def expand_metadata(query: dict) -> pd.DataFrame:
    """
    Replicates metadata for each sample in the query.

    Parameters
    ----------
    query : dict
        A dictionary containing the query data. (Assumes pre-v1 format with strings)

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the metadata for each sample as different rows.
    """

    dicts = [ast.literal_eval(item) for item in query["inputs"]]
    metadata = pd.DataFrame(
        [item for item in dicts for _ in range(query["num_samples"])]
    )
    return metadata


def validate_modality(query: dict, endpoint: str) -> None:
    """
    Validates the modality in the query is within the allowed modalities for model.

    Parameters
    ----------
    query : dict
        A dictionary containing the query data.

    Raises
    ------
    AssertionError
        If the modality is not in the allowed modalities.
    """
    if "modality" in query:
        selected_modality = query["modality"]
    else:
        selected_modality = query["inputs"]["modality"]

    model_name = next(k for k, v in MODELS.items() if v == endpoint)
    assert (
        selected_modality in MODEL_MODALITIES[model_name]
    ), f"Invalid modality: '{selected_modality}' not in {MODEL_MODALITIES[model_name]}"

    return None


def transform_to_counts(expression: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms expression data from log1p cpm into counts data with approximately 30M total counts.

    Parameters
    ----------
    expression : pd.DataFrame
        A DataFrame containing expression data.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing counts data.
    """
    counts = (np.expm1(expression) * 30).astype(int)
    return counts


def log_cpm(expression: pd.DataFrame):
    """
    Transforms expression data into log1p cpm.

    Parameters
    ----------
    expression : pd.DataFrame
        A DataFrame containing expression data.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing log1p cpm data.
    """

    cpm = expression.div(expression.sum(axis=1), axis=0) * 1e6
    log_cpm = np.log1p(cpm)

    return log_cpm
