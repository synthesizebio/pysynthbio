# read version from installed package
from importlib.metadata import version

__version__ = version("pysynthbio")


# populate package namespace
from .call_model_api import (
    get_available_models as get_available_models,
    get_valid_modalities as get_valid_modalities,
    get_valid_query as get_valid_query,
    predict_query as predict_query,
)
