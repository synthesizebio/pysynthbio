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
    @patch("pysynthbio.call_model_api.requests.post")
    def test_predict_query_auto_authenticate(self, mock_post, mock_set_token):
        """Test auto authentication in predict_query."""
        # Import here to avoid circular imports in tests
        from pysynthbio.call_model_api import get_valid_query, predict_query

        # Make the mock set_synthesize_token function
        # actually set the environment variable
        def mock_set_token_implementation(use_keyring=False):
            os.environ["SYNTHESIZE_API_KEY"] = "mock-token-for-test"
            return True

        mock_set_token.side_effect = mock_set_token_implementation

        # Create mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [
                {
                    "metadata": {"sample_id": "test1"},
                    "expression": [[1, 2, 3], [4, 5, 6]],
                }
            ],
            "gene_order": ["gene1", "gene2", "gene3"],
        }
        mock_post.return_value = mock_response

        # Call function with auto-authentication
        query = get_valid_query()
        results = predict_query(query, auto_authenticate=True)

        # Verify set_token was called
        mock_set_token.assert_called_once_with(use_keyring=True)

        # Verify API was called
        mock_post.assert_called_once()

        # Verify results structure
        self.assertIn("metadata", results)
        self.assertIn("expression", results)

    @patch("pysynthbio.call_model_api.requests.post")
    def test_predict_query_without_auto_authenticate(self, mock_post):
        """Test predict_query without auto authentication."""
        # Import here to avoid circular imports in tests
        from pysynthbio.call_model_api import get_valid_query, predict_query

        # Call function without auto-authentication and no token
        query = get_valid_query()
        with self.assertRaises(KeyError):
            predict_query(query, auto_authenticate=False)

        # Verify API was not called
        mock_post.assert_not_called()

    @patch("pysynthbio.call_model_api.requests.post")
    def test_predict_query_with_token(self, mock_post):
        """Test predict_query with token already set."""
        # Import here to avoid circular imports in tests
        from pysynthbio.call_model_api import get_valid_query, predict_query

        # Set a token
        os.environ["SYNTHESIZE_API_KEY"] = "test-api-token"

        # Create mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [
                {
                    "metadata": {"sample_id": "test1"},
                    "expression": [[1, 2, 3], [4, 5, 6]],
                }
            ],
            "gene_order": ["gene1", "gene2", "gene3"],
        }
        mock_post.return_value = mock_response

        # Call function without auto-authentication but with token set
        query = get_valid_query()
        results = predict_query(query, auto_authenticate=False)

        # Verify API was called with correct token
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-api-token")

        # Verify results structure
        self.assertIn("metadata", results)
        self.assertIn("expression", results)


if __name__ == "__main__":
    unittest.main()
