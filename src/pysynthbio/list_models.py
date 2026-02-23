"""List available models from the Synthesize Bio API."""

from pysynthbio.http_client import API_BASE_URL, api_request


def list_models(api_base_url: str = API_BASE_URL):
    """
    List all models available in the Synthesize Bio API.

    Parameters
    ----------
    api_base_url : str, optional
        Base URL for the API server. Defaults to the production host.

    Returns
    -------
    list
        List of available models.

    Raises
    ------
    KeyError
        If no API token is configured.
    AuthenticationError
        If the token is invalid.
    SynthesizeAPIError
        For other API errors.
    """
    return api_request("GET", "/api/models", api_base_url=api_base_url)
