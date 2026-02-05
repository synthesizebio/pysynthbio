"""Get example queries for models from the Synthesize Bio API."""

from pysynthbio.http_client import API_BASE_URL, api_request


def get_example_query(model_id: str, api_base_url: str = API_BASE_URL):
    """
    Get the example query for a given model.

    Parameters
    ----------
    model_id : str
        The ID of the model to get an example query for.
    api_base_url : str, optional
        Base URL for the API server. Defaults to the production host.

    Returns
    -------
    dict
        Example query dictionary for the model.

    Raises
    ------
    KeyError
        If no API token is configured.
    AuthenticationError
        If the token is invalid.
    NotFoundError
        If the model doesn't exist.
    SynthesizeAPIError
        For other API errors.
    """
    return api_request(
        "GET", f"/api/models/{model_id}/example-query", api_base_url=api_base_url
    )
