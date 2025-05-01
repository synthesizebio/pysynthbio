# read version from installed package
from importlib.metadata import version

__version__ = version("pysynthbio")


# populate package namespace
from .call_model_api import (
    get_available_models as get_available_models,
    get_valid_modalities as get_valid_modalities,
    get_valid_query as get_valid_query,
    predict_query as predict_query,
    validate_query as validate_query,
    process_samples as process_samples,
    get_gene_order as get_gene_order,
    expand_metadata as expand_metadata,
    validate_modality as validate_modality,
    transform_to_counts as transform_to_counts,
    log_cpm as log_cpm,
)
