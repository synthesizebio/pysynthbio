"""
Unit tests for the Synthesize Bio API authentication workflow.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from pysynthbio.key_handlers import (
    KEYRING_AVAILABLE,
    has_synthesize_token,
    set_synthesize_token,
)


class TestTokenManagement(unittest.TestCase):
    """Test cases for token management functions."""

    def setUp(self):
        """Set up test environment."""
        # Save existing API key if it exists
        self.original_api_key = os.environ.get("SYNTHESIZE_API_KEY")
        # Clear any existing token for tests
        if "SYNTHESIZE_API_KEY" in os.environ:
            del os.environ["SYNTHESIZE_API_KEY"]

    def tearDown(self):
        """Restore environment after tests."""
        # Restore original API key if it existed
        if self.original_api_key is not None:
            os.environ["SYNTHESIZE_API_KEY"] = self.original_api_key
        else:
            # Clear the API key if it was added during tests
            if "SYNTHESIZE_API_KEY" in os.environ:
                del os.environ["SYNTHESIZE_API_KEY"]

    def test_has_synthesize_token_when_not_set(self):
        """Test has_synthesize_token when no token is set."""
        # Ensure token is not set
        if "SYNTHESIZE_API_KEY" in os.environ:
            del os.environ["SYNTHESIZE_API_KEY"]

        # Test function returns False when token not set
        self.assertFalse(has_synthesize_token())

    def test_has_synthesize_token_when_set(self):
        """Test has_synthesize_token when token is set."""
        # Set a test token
        os.environ["SYNTHESIZE_API_KEY"] = "test-token-value"

        # Test function returns True when token is set
        self.assertTrue(has_synthesize_token())

    def test_has_synthesize_token_when_empty(self):
        """Test has_synthesize_token when token is empty string."""
        # Set an empty token
        os.environ["SYNTHESIZE_API_KEY"] = ""

        # Test function returns False when token is empty
        self.assertFalse(has_synthesize_token())

    @patch("webbrowser.open")
    @patch("getpass.getpass")
    def test_set_synthesize_token_interactive(self, mock_getpass, mock_browser):
        """Test setting token interactively."""
        # Configure mocks
        mock_getpass.return_value = "test-interactive-token"

        # Call function
        set_synthesize_token()

        # Verify browser was opened
        mock_browser.assert_called_once()

        # Verify getpass was called
        mock_getpass.assert_called_once()

        # Verify token was set
        self.assertEqual(os.environ["SYNTHESIZE_API_KEY"], "test-interactive-token")

    def test_set_synthesize_token_direct(self):
        """Test setting token directly."""
        # Call function with direct token
        set_synthesize_token(token="test-direct-token")

        # Verify token was set
        self.assertEqual(os.environ["SYNTHESIZE_API_KEY"], "test-direct-token")

    @patch("keyring.set_password")
    def test_set_synthesize_token_with_keyring(self, mock_set_password):
        """Test setting token with keyring storage."""
        # Skip if keyring is not available
        if not KEYRING_AVAILABLE:
            self.skipTest("Keyring package not available")

        # Call function with keyring storage
        set_synthesize_token(token="test-keyring-token", use_keyring=True)

        # Verify token was set in environment
        self.assertEqual(os.environ["SYNTHESIZE_API_KEY"], "test-keyring-token")

        # Verify token was stored in keyring
        mock_set_password.assert_called_once_with(
            "pysynthbio", "api_token", "test-keyring-token"
        )


class TestApiWithAuthentication(unittest.TestCase):
    """Test cases for API functions that use authentication."""

    def setUp(self):
        """Set up test environment."""
        # Save existing API key if it exists
        self.original_api_key = os.environ.get("SYNTHESIZE_API_KEY")
        # Clear any existing token for tests
        if "SYNTHESIZE_API_KEY" in os.environ:
            del os.environ["SYNTHESIZE_API_KEY"]

    def tearDown(self):
        """Restore environment after tests."""
        # Restore original API key if it existed
        if self.original_api_key is not None:
            os.environ["SYNTHESIZE_API_KEY"] = self.original_api_key
        else:
            # Clear the API key if it was added during tests
            if "SYNTHESIZE_API_KEY" in os.environ:
                del os.environ["SYNTHESIZE_API_KEY"]

    @patch("pysynthbio.call_model_api.set_synthesize_token")
    @patch("pysynthbio.call_model_api.requests.get")
    @patch("pysynthbio.call_model_api.requests.post")
    def test_predict_query_auto_authenticate(self, mock_post, mock_get, mock_set_token):
        """Test auto authentication in predict_query."""
        # Import here to avoid circular imports in tests
        from pysynthbio.call_model_api import predict_query

        # Make the mock set_synthesize_token function
        # actually set the environment variable
        def mock_set_token_implementation(use_keyring=False):
            os.environ["SYNTHESIZE_API_KEY"] = "mock-token-for-test"
            return True

        mock_set_token.side_effect = mock_set_token_implementation

        # POST /predict returns modelQueryId
        post_resp = MagicMock()
        post_resp.status_code = 200
        post_resp.json.return_value = {"modelQueryId": "abc123"}
        mock_post.return_value = post_resp

        # Then status ready and download with final JSON
        get_status_ready = MagicMock()
        get_status_ready.status_code = 200
        get_status_ready.json.return_value = {
            "status": "ready",
            "downloadUrl": "https://example.com/final.json",
        }
        get_download = MagicMock()
        get_download.status_code = 200
        get_download.json.return_value = {
            "outputs": [
                {
                    "counts": [1, 2, 3],
                    "metadata": {
                        "sample_id": "test1",
                        "age_years": "25",
                        "sex": "female",
                        "cell_line_ontology_id": "CVCL_0023",
                        "perturbation_ontology_id": "ENSG00000156127",
                        "perturbation_type": "crispr",
                        "sample_type": "cell line",
                    },
                }
            ],
            "gene_order": ["gene1", "gene2", "gene3"],
            "model_version": 2,
        }
        mock_get.side_effect = [get_status_ready, get_download]

        # Call function with auto-authentication
        # Query content doesn't matter since HTTP responses are mocked
        query = {"samples": [{"test": "data"}]}
        results = predict_query(query, model_id="gem-1-bulk", auto_authenticate=True)

        # Verify set_token was called
        mock_set_token.assert_called_once_with(use_keyring=True)

        # Verify API was called
        mock_post.assert_called_once()

        # Verify results structure
        self.assertIn("metadata", results)
        self.assertIn("expression", results)

        # Verify data dimensions match new structure
        self.assertEqual(len(results["metadata"]), 1)  # One sample
        self.assertEqual(len(results["expression"]), 1)  # One row
        self.assertEqual(len(results["expression"].columns), 3)  # Three genes

    @patch("pysynthbio.call_model_api.requests.post")
    def test_predict_query_without_auto_authenticate(self, mock_post):
        """Test predict_query without auto authentication."""
        # Import here to avoid circular imports in tests
        from pysynthbio.call_model_api import predict_query

        # Call function without auto-authentication and no token
        # Query content doesn't matter since we're testing auth failure
        query = {"samples": [{"test": "data"}]}
        with self.assertRaises(KeyError):
            predict_query(query, model_id="gem-1-bulk", auto_authenticate=False)

        # Verify API was not called
        mock_post.assert_not_called()

    @patch("pysynthbio.call_model_api.requests.get")
    @patch("pysynthbio.call_model_api.requests.post")
    def test_predict_query_with_token(self, mock_post, mock_get):
        """Test predict_query with token already set."""
        # Import here to avoid circular imports in tests
        from pysynthbio.call_model_api import predict_query

        # Set a token
        os.environ["SYNTHESIZE_API_KEY"] = "test-api-token"

        # POST /predict -> modelQueryId
        post_resp = MagicMock()
        post_resp.status_code = 200
        post_resp.json.return_value = {"modelQueryId": "bulk-xyz"}
        mock_post.return_value = post_resp

        # Then ready + download
        get_status_ready = MagicMock()
        get_status_ready.status_code = 200
        get_status_ready.json.return_value = {
            "status": "ready",
            "downloadUrl": "https://example.com/bulk.json",
        }
        get_download = MagicMock()
        get_download.status_code = 200
        get_download.json.return_value = {
            "outputs": [
                {"counts": [100, 200, 300], "metadata": {"sample_id": "test1"}},
                {"counts": [150, 250, 350], "metadata": {"sample_id": "test2"}},
            ],
            "gene_order": ["gene1", "gene2", "gene3"],
            "model_version": 2,
        }
        mock_get.side_effect = [get_status_ready, get_download]

        # Call function without auto-authentication but with token set
        # Query content doesn't matter since HTTP responses are mocked
        query = {"samples": [{"test": "data"}]}
        results = predict_query(query, model_id="gem-1-bulk", auto_authenticate=False)

        # Verify API was called with correct token
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-api-token")

        # Verify results structure
        self.assertIn("metadata", results)
        self.assertIn("expression", results)

        # Verify data dimensions for two samples
        self.assertEqual(len(results["metadata"]), 2)  # Two samples
        self.assertEqual(len(results["expression"]), 2)  # Two rows
        self.assertEqual(len(results["expression"].columns), 3)  # Three genes

        # Verify actual data values match the mock
        self.assertEqual(list(results["expression"].iloc[0]), [100, 200, 300])
        self.assertEqual(list(results["expression"].iloc[1]), [150, 250, 350])

    @patch("pysynthbio.call_model_api.requests.get")
    @patch("pysynthbio.call_model_api.requests.post")
    def test_predict_query_single_vs_multiple_samples(self, mock_post, mock_get):
        """Test that the code correctly handles both single and multiple samples."""
        # Import here to avoid circular imports in tests
        from pysynthbio.call_model_api import predict_query

        # Set a token
        os.environ["SYNTHESIZE_API_KEY"] = "test-api-token"

        # Test with single sample
        post_resp_single = MagicMock()
        post_resp_single.status_code = 200
        post_resp_single.json.return_value = {"modelQueryId": "bulk-1"}
        mock_post.return_value = post_resp_single

        get_status_ready_1 = MagicMock()
        get_status_ready_1.status_code = 200
        get_status_ready_1.json.return_value = {
            "status": "ready",
            "downloadUrl": "https://example.com/bulk1.json",
        }
        get_download_1 = MagicMock()
        get_download_1.status_code = 200
        get_download_1.json.return_value = {
            "outputs": [
                {"counts": [10, 20, 30, 40], "metadata": {"sample_id": "single_test"}},
            ],
            "gene_order": ["gene1", "gene2", "gene3", "gene4"],
            "model_version": 2,
        }
        mock_get.side_effect = [get_status_ready_1, get_download_1]

        # Query content doesn't matter since HTTP responses are mocked
        query = {"samples": [{"test": "data"}]}
        results_single = predict_query(
            query, model_id="gem-1-bulk", auto_authenticate=False
        )

        # Verify single sample results
        self.assertEqual(len(results_single["metadata"]), 1)
        self.assertEqual(len(results_single["expression"]), 1)
        self.assertEqual(len(results_single["expression"].columns), 4)

        # Reset mocks for multiple samples test
        mock_post.reset_mock()
        mock_get.reset_mock()

        # Test with multiple samples
        post_resp_multi = MagicMock()
        post_resp_multi.status_code = 200
        post_resp_multi.json.return_value = {"modelQueryId": "bulk-2"}
        mock_post.return_value = post_resp_multi

        get_status_ready_2 = MagicMock()
        get_status_ready_2.status_code = 200
        get_status_ready_2.json.return_value = {
            "status": "ready",
            "downloadUrl": "https://example.com/bulk2.json",
        }
        get_download_2 = MagicMock()
        get_download_2.status_code = 200
        get_download_2.json.return_value = {
            "outputs": [
                {"counts": [10, 20], "metadata": {"sample_id": "multi_test_1"}},
                {"counts": [30, 40], "metadata": {"sample_id": "multi_test_2"}},
                {"counts": [50, 60], "metadata": {"sample_id": "multi_test_3"}},
            ],
            "gene_order": ["gene1", "gene2"],
            "model_version": 2,
        }
        mock_get.side_effect = [get_status_ready_2, get_download_2]

        results_multiple = predict_query(
            query, model_id="gem-1-bulk", auto_authenticate=False
        )

        # Verify multiple sample results
        self.assertEqual(len(results_multiple["metadata"]), 3)
        self.assertEqual(len(results_multiple["expression"]), 3)
        self.assertEqual(len(results_multiple["expression"].columns), 2)

        # Verify data content
        self.assertEqual(list(results_multiple["expression"].iloc[0]), [10, 20])
        self.assertEqual(list(results_multiple["expression"].iloc[1]), [30, 40])
        self.assertEqual(list(results_multiple["expression"].iloc[2]), [50, 60])


if __name__ == "__main__":
    unittest.main()
