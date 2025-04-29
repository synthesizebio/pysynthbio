import pandas as pd
import numpy as np
from functools import lru_cache
import os
import requests
import json
import re
import ast

__version__ = version("pysynthbio")


# populate package namespace
from pysynthbio.call_model_api import (
    get_model_endpoints,
    get_valid_modalities,
    get_valid_query,
    predict_query,
    validate_query,
    process_samples,
    get_gene_order,
    expand_metadata,
    validate_modality,
    transform_to_counts,
    log_cpm
)

MODEL_MODALITIES = {
    "rMetalv0.2": {
        "bulk_rna-seq",
        "lincs",
        "sra",
    },
    "rMetalv0.4": {
        "bulk_rna-seq",
        "lincs",
        "sra",
    },
    "rMetalv0.5": {
        "bulk_rna-seq",
        "lincs",
        "sra",
    },
    "rMetalv0.6": {
        "bulk_rna-seq",
        "lincs",
        "sra",
    },
    "DoGMAv0.1": {
        "bulk_rna-seq",
        "lincs",
        "sra",
    },
    "DoGMAv0.2": {
        "bulk_rna-seq",
        "lincs",
        "sra",
    },
    "DoGMAv0.3": {
        "bulk_rna-seq",
        "lincs",
        "sra",
    },
    "combinedv1.0": {
        "bulk_rna-seq",
        "lincs",
        "sra",
        "single_cell_rna-seq",
        "microarray",
        "pseudo_bulk",
    },
}

MODELS = {
    "rMetalv0.2": "https://tp3fsonurm7s6gcl.us-east-1.aws.endpoints.huggingface.cloud",
    "rMetalv0.4": "https://nduvneyj3ynapbv3.us-east-1.aws.endpoints.huggingface.cloud",
    "rMetalv0.5": "https://mut58aklpmk6mr1u.us-east-1.aws.endpoints.huggingface.cloud",
    "rMetalv0.6": "https://k8bcgyqfijgrl2bq.us-east-1.aws.endpoints.huggingface.cloud",
    "DoGMAv0.1": "https://y9e80x48xyreg7vn.us-east-1.aws.endpoints.huggingface.cloud",
    "DoGMAv0.2": "https://w6zrdrulbi04x6sv.us-east-1.aws.endpoints.huggingface.cloud",
    "DoGMAv0.3": "https://gfiu95rp5maj75yg.us-east-1.aws.endpoints.huggingface.cloud",
    "combinedv1.0": "https://uk9hdkibyzwxn0hz.us-east-1.aws.endpoints.huggingface.cloud",
}

DEFAULT_MODEL = "combinedv1.0"
ENDPOINT_URL = MODELS[DEFAULT_MODEL]


LOG_CPM_MODELS = {"rMetalv0.2", "rMetalv0.4", "rMetalv0.5", "DoGMAv0.1", "DoGMAv0.2"}
LOG_CPM_ENDPOINTS = {MODELS[model] for model in LOG_CPM_MODELS}
