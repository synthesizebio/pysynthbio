# read version from installed package
from importlib.metadata import version

__version__ = version("pysynthbio")


# populate package namespace
from .call_model_api import (
    get_valid_modalities as get_valid_modalities,
)
from .call_model_api import (
    get_valid_query as get_valid_query,
)
from .call_model_api import (
    predict_query as predict_query,
)
from .key_handlers import (
    clear_synthesize_token as clear_synthesize_token,
)
from .key_handlers import (
    has_synthesize_token as has_synthesize_token,
)
from .key_handlers import (
    load_synthesize_token_from_keyring as load_synthesize_token_from_keyring,
)
from .key_handlers import (
    set_synthesize_token as set_synthesize_token,
)
