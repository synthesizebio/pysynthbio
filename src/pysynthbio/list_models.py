import os

import requests

from pysynthbio.key_handlers import has_synthesize_token

API_BASE_URL = "https://app.synthesize.bio"


def list_models(
    api_base_url: str = API_BASE_URL,
):
    """
    List all models available in the Synthesize Bio API.
    """
    if not has_synthesize_token():
        raise KeyError(
            "No API token found. Set the SYNTHESIZE_API_KEY environment variable or "
            + "call set_synthesize_token() before making API requests."
        )

    url = f"{api_base_url}/api/models"
    response = requests.get(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": "Bearer " + os.environ["SYNTHESIZE_API_KEY"],
        },
    )
    models = response.json()
    return models
