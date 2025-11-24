import os

import requests

from pysynthbio.key_handlers import has_synthesize_token

API_BASE_URL = "https://app.synthesize.bio"


def get_example_query(
    model_id: str,
    api_base_url: str = API_BASE_URL,
):
    """
    Get the example query for a given model.
    """
    if not has_synthesize_token():
        raise KeyError(
            "No API token found. Set the SYNTHESIZE_API_KEY environment variable or "
            + "call set_synthesize_token() before making API requests."
        )

    url = f"{api_base_url}/api/models/{model_id}/example-query"
    response = requests.get(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": "Bearer " + os.environ["SYNTHESIZE_API_KEY"],
        },
    )
    example_query = response.json()
    return example_query
