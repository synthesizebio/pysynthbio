import numpy as np
import pandas as pd

from pysynthbio.call_model_api import log_cpm


def test_log_cpm():
    """Tests transforming counts to logCPM."""
    counts_data = pd.DataFrame(
        {"gene1": [1000000, 3000000], "gene2": [2000000, 6000000]}
    )
    expected_log_cpm = pd.DataFrame(
        {
            "gene1": [np.log1p(1e6 / 3), np.log1p(1e6 / 3)],
            "gene2": [np.log1p(2e6 / 3), np.log1p(2e6 / 3)],
        }
    )

    result_log_cpm = log_cpm(counts_data)
    pd.testing.assert_frame_equal(
        result_log_cpm, expected_log_cpm, check_dtype=False, rtol=1e-5
    )


def test_log_cpm_zero_counts():
    """Tests log_cpm handles rows with zero total counts."""
    counts_data = pd.DataFrame({"gene1": [10, 0], "gene2": [20, 0]})
    expected_log_cpm = pd.DataFrame(
        {
            "gene1": [np.log1p(10 / 30 * 1e6), 0.0],
            "gene2": [np.log1p(20 / 30 * 1e6), 0.0],
        }
    )
    result_log_cpm = log_cpm(counts_data)
    pd.testing.assert_frame_equal(
        result_log_cpm, expected_log_cpm, check_dtype=False, rtol=1e-5
    )
