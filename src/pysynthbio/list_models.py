"""List available models from the Synthesize Bio API."""

from typing import Optional

from pysynthbio.http_client import (
    api_request,
    get_self_hosted,
    resolve_base_url,
    self_hosted_enabled,
)


def list_models(
    api_base_url: Optional[str] = None,
    self_hosted: Optional[bool] = None,
):
    """
    List all models available in the Synthesize Bio API.

    Parameters
    ----------
    api_base_url : str, optional
        Base URL for the API server. Defaults to ``SYNTHESIZE_API_BASE_URL`` or
        the production host.
    self_hosted : bool, optional
        Talk to a self-hosted container (auth optional). Defaults to the
        ``SYNTHESIZE_SELF_HOSTED`` environment variable.

    Returns
    -------
    list
        List of available models.
    """
    base_url = resolve_base_url(api_base_url)
    if self_hosted_enabled(self_hosted):
        return get_self_hosted("/api/models", api_base_url=base_url)
    return api_request("GET", "/api/models", api_base_url=base_url)
