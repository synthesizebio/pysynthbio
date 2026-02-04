"""
Unit tests for the HTTP client and error handling.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

import requests

from pysynthbio.call_model_api import _clean_error_message
from pysynthbio.http_client import (
    AuthenticationError,
    NotFoundError,
    SynthesizeAPIError,
    ValidationError,
    api_request,
    get_json,
)


class TestCleanErrorMessage(unittest.TestCase):
    """Test cases for the _clean_error_message helper."""

    def test_clean_message_with_traceback(self):
        """Test that server-side tracebacks are stripped."""
        message = (
            "Metadata validation failed: [\"Query 1 has bad values.\"]\n"
            "Traceback (most recent call last):\n"
            '  File "/opt/ml/model/code/inference.py", line 100\n'
            "ValueError: something went wrong"
        )
        cleaned = _clean_error_message(message)
        self.assertEqual(
            cleaned, 'Metadata validation failed: ["Query 1 has bad values."]'
        )

    def test_clean_message_without_traceback(self):
        """Test that messages without tracebacks are unchanged."""
        message = "Simple error message"
        cleaned = _clean_error_message(message)
        self.assertEqual(cleaned, "Simple error message")

    def test_clean_message_with_multiple_tracebacks(self):
        """Test that only content before first traceback is kept."""
        message = (
            "Error occurred\n"
            "Traceback (most recent call last):\n"
            "  first traceback\n"
            "Traceback (most recent call last):\n"
            "  second traceback"
        )
        cleaned = _clean_error_message(message)
        self.assertEqual(cleaned, "Error occurred")


class TestApiRequestErrors(unittest.TestCase):
    """Test cases for API request error handling."""

    def setUp(self):
        """Set up test environment."""
        self.original_api_key = os.environ.get("SYNTHESIZE_API_KEY")
        os.environ["SYNTHESIZE_API_KEY"] = "test-token"

    def tearDown(self):
        """Restore environment after tests."""
        if self.original_api_key is not None:
            os.environ["SYNTHESIZE_API_KEY"] = self.original_api_key
        elif "SYNTHESIZE_API_KEY" in os.environ:
            del os.environ["SYNTHESIZE_API_KEY"]

    @patch("pysynthbio.http_client.requests.request")
    def test_authentication_error_401(self, mock_request):
        """Test that 401 raises AuthenticationError."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response
        )
        mock_request.return_value = mock_response

        with self.assertRaises(AuthenticationError) as context:
            api_request("GET", "/api/models")

        self.assertEqual(context.exception.status_code, 401)
        self.assertIn("Authentication failed", str(context.exception))

    @patch("pysynthbio.http_client.requests.request")
    def test_authentication_error_403(self, mock_request):
        """Test that 403 raises AuthenticationError."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response
        )
        mock_request.return_value = mock_response

        with self.assertRaises(AuthenticationError) as context:
            api_request("GET", "/api/models")

        self.assertEqual(context.exception.status_code, 403)

    @patch("pysynthbio.http_client.requests.request")
    def test_not_found_error_404(self, mock_request):
        """Test that 404 raises NotFoundError."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response
        )
        mock_request.return_value = mock_response

        with self.assertRaises(NotFoundError) as context:
            api_request("GET", "/api/models/nonexistent")

        self.assertEqual(context.exception.status_code, 404)
        self.assertIn("/api/models/nonexistent", str(context.exception))

    @patch("pysynthbio.http_client.requests.request")
    def test_validation_error_400(self, mock_request):
        """Test that 400 raises ValidationError."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request: invalid query format"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response
        )
        mock_request.return_value = mock_response

        with self.assertRaises(ValidationError) as context:
            api_request("POST", "/api/models/gem-1-bulk/predict", json={"bad": "data"})

        self.assertEqual(context.exception.status_code, 400)

    @patch("pysynthbio.http_client.requests.request")
    def test_validation_error_422(self, mock_request):
        """Test that 422 raises ValidationError."""
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.text = "Unprocessable Entity"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response
        )
        mock_request.return_value = mock_response

        with self.assertRaises(ValidationError) as context:
            api_request("POST", "/api/models/gem-1-bulk/predict", json={})

        self.assertEqual(context.exception.status_code, 422)

    @patch("pysynthbio.http_client.requests.request")
    def test_generic_api_error_500(self, mock_request):
        """Test that 500 raises SynthesizeAPIError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response
        )
        mock_request.return_value = mock_response

        with self.assertRaises(SynthesizeAPIError) as context:
            api_request("GET", "/api/models")

        self.assertEqual(context.exception.status_code, 500)

    @patch("pysynthbio.http_client.requests.request")
    def test_network_error(self, mock_request):
        """Test that network errors raise SynthesizeAPIError."""
        mock_request.side_effect = requests.exceptions.ConnectionError(
            "Failed to connect"
        )

        with self.assertRaises(SynthesizeAPIError) as context:
            api_request("GET", "/api/models")

        self.assertIn("Network error", str(context.exception))
        self.assertIsNone(context.exception.status_code)

    def test_missing_token_raises_key_error(self):
        """Test that missing token raises KeyError."""
        del os.environ["SYNTHESIZE_API_KEY"]

        with self.assertRaises(KeyError):
            api_request("GET", "/api/models")

    @patch("pysynthbio.http_client.requests.request")
    def test_successful_request(self, mock_request):
        """Test successful API request returns JSON."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": ["gem-1-bulk"]}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        result = api_request("GET", "/api/models")

        self.assertEqual(result, {"models": ["gem-1-bulk"]})


class TestGetJson(unittest.TestCase):
    """Test cases for the get_json helper."""

    @patch("pysynthbio.http_client.requests.get")
    def test_successful_download(self, mock_get):
        """Test successful JSON download."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "test"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = get_json("https://example.com/data.json")

        self.assertEqual(result, {"data": "test"})

    @patch("pysynthbio.http_client.requests.get")
    def test_download_http_error(self, mock_get):
        """Test HTTP error during download."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = "Access Denied"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response
        )
        mock_get.return_value = mock_response

        with self.assertRaises(SynthesizeAPIError) as context:
            get_json("https://example.com/data.json")

        self.assertEqual(context.exception.status_code, 403)


class TestExceptionHierarchy(unittest.TestCase):
    """Test that exception hierarchy is correct."""

    def test_authentication_error_is_synthesize_api_error(self):
        """Test AuthenticationError inherits from SynthesizeAPIError."""
        err = AuthenticationError("test", status_code=401)
        self.assertIsInstance(err, SynthesizeAPIError)

    def test_not_found_error_is_synthesize_api_error(self):
        """Test NotFoundError inherits from SynthesizeAPIError."""
        err = NotFoundError("test", status_code=404)
        self.assertIsInstance(err, SynthesizeAPIError)

    def test_validation_error_is_synthesize_api_error(self):
        """Test ValidationError inherits from SynthesizeAPIError."""
        err = ValidationError("test", status_code=400)
        self.assertIsInstance(err, SynthesizeAPIError)

    def test_can_catch_all_with_base_exception(self):
        """Test that all errors can be caught with SynthesizeAPIError."""
        errors = [
            AuthenticationError("auth", status_code=401),
            NotFoundError("not found", status_code=404),
            ValidationError("validation", status_code=400),
            SynthesizeAPIError("generic", status_code=500),
        ]

        for err in errors:
            try:
                raise err
            except SynthesizeAPIError as caught:
                self.assertIsNotNone(caught.status_code)


if __name__ == "__main__":
    unittest.main()
