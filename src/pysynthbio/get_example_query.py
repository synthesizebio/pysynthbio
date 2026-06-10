"""Get example queries for models from the Synthesize Bio API."""

from typing import Optional

from pysynthbio.http_client import (
    api_request,
    get_self_hosted,
    resolve_base_url,
    self_hosted_enabled,
)


def get_example_query(
    model_id: str,
    api_base_url: Optional[str] = None,
    self_hosted: Optional[bool] = None,
):
    """
    Get the example query for a given model.

    Parameters
    ----------
    model_id : str
        The ID of the model to get an example query for.
    api_base_url : str, optional
        Base URL for the API server. Defaults to the per-model env var
        ``SYNTHESIZE_API_BASE_URL__<MODEL>``, then ``SYNTHESIZE_API_BASE_URL``,
        then the production host.
    self_hosted : bool, optional
        Talk to a self-hosted container (auth optional). Defaults to the
        ``SYNTHESIZE_SELF_HOSTED`` environment variable.

    Returns
    -------
    dict
        Example query dictionary for the model.
    """
    base_url = resolve_base_url(api_base_url, model_id=model_id)
    endpoint = f"/api/models/{model_id}/example-query"
    if self_hosted_enabled(self_hosted):
        return get_self_hosted(endpoint, api_base_url=base_url)
    return api_request("GET", endpoint, api_base_url=base_url)
