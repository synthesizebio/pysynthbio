Python API Wrapper Style Guide
This style guide outlines conventions and best practices for developing a Python package that serves as an API wrapper. Adhering to these guidelines will ensure a consistent, robust, and user-friendly package that interacts gracefully with external APIs.

I. General Principles
Clarity and Readability: Code should be easy to understand by others and your future self. Prioritize clarity over cleverness.

Idiomatic Python: Write Python code that feels natural and familiar to Python users. Embrace PEP 8 and Pythonic constructs.

Robustness: Anticipate and handle common API issues like network errors, rate limits, and unexpected response structures.

User-Friendliness: Provide clear functions, helpful error messages, and comprehensive documentation.

II. Package Structure and Organization
Follow standard Python package directory structure (e.g., setuptools or Poetry conventions).

myapipackage/: The root directory for the Python package source code.

__init__.py: Marks the directory as a Python package.

auth.py: Authentication functions.

client.py: Core API client for making requests.

resources/: (Optional) Sub-package for API resources (e.g., users.py, products.py) if the API is large.

exceptions.py: Custom exceptions for API-specific errors.

utils.py: General utility functions.

tests/: Contains unit and integration tests. Organize with subdirectories reflecting your source structure (e.g., tests/unit/test_client.py, tests/integration/test_live_api.py).

docs/: (Recommended) Documentation source files (e.g., Sphinx, MkDocs).

examples/: (Recommended) Example scripts demonstrating package usage.

pyproject.toml or setup.py: Project metadata and build configuration (for Poetry or setuptools).

README.md: Project overview.

.env: (Local, not committed) For environment variables like API keys.

.gitignore: Files to ignore in version control.

Example Structure:

myapipackage_repo/
├── myapipackage/
│   ├── __init__.py
│   ├── auth.py             # Authentication functions
│   ├── client.py           # Core API client
│   ├── resources/
│   │   ├── __init__.py
│   │   ├── users.py
│   │   └── products.py
│   ├── exceptions.py       # Custom exceptions
│   └── utils.py            # General utilities
├── tests/
│   ├── unit/
│   │   ├── test_auth.py
│   │   └── test_client.py
│   └── integration/
│       └── test_live_api.py
├── docs/
├── examples/
├── pyproject.toml          # or setup.py, requirements.txt
├── README.md
└── .env

III. Naming Conventions (PEP 8 Adherence)
Modules and Packages: snake_case (e.g., auth.py, myapipackage).

Classes: PascalCase (e.g., ApiClient, AuthHandler).

Functions and Methods: snake_case (e.g., get_user_data, authenticate_client).

Variables: snake_case (e.g., api_key, response_data).

Constants: UPPER_SNAKE_CASE (e.g., API_BASE_URL, TIMEOUT_SECONDS).

Private/Internal Members: Prefix with a single underscore (e.g., _build_headers).

IV. Code Layout and Readability (PEP 8 Adherence)
Indentation: Use 4 spaces for indentation.

Line Length: Limit lines to 79 characters. Break lines gracefully for long function calls, arguments, or string literals using parentheses or explicit line continuations (\).

Blank Lines:

Two blank lines between top-level function and class definitions.

One blank line between method definitions within a class.

Use blank lines judiciously to separate logical blocks of code.

Imports:

Imports should usually be on separate lines.

Imports should be grouped in the following order:

Standard library imports.

Third-party imports.

Local application/library specific imports.

Each group should be separated by a blank line.

Use isort and black for automatic formatting and import sorting.

Docstrings: Use triple double quotes ("""Docstring content""") for docstrings. Refer to Section V for more details.

Example (Code Layout):

import os
import requests

from myapipackage.exceptions import ApiError
from myapipackage.utils import parse_json_response


class ApiClient:
    """
    A client for interacting with the MyAPI service.
    """
    API_BASE_URL = "https://api.example.com/v1"
    DEFAULT_TIMEOUT = 30

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._session = requests.Session()

    def _build_headers(self) -> dict:
        """Constructs headers for API requests."""
        return {"Authorization": f"Bearer {self.api_key}"}

    def get_user_data(self, user_id: str,
                      include_details: bool = False) -> dict:
        """
        Fetches data for a specific user from the API.

        Args:
            user_id: The unique identifier of the user.
            include_details: Whether to include additional user details.

        Returns:
            A dictionary containing user data.

        Raises:
            ApiError: If the API request fails.
        """
        url = f"{self.API_BASE_URL}/users/{user_id}"
        params = {"details": "true"} if include_details else {}

        try:
            response = self._session.get(
                url,
                headers=self._build_headers(),
                params=params,
                timeout=self.DEFAULT_TIMEOUT
            )
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            return parse_json_response(response)
        except requests.exceptions.RequestException as e:
            raise ApiError(f"API request failed: {e}") from e


V. Commenting and Docstrings (PEP 257 Adherence)
Docstrings: Use triple double quotes ("""Docstring content""") for docstrings.

Module Docstrings: Describe the module's purpose and contents.

Class Docstrings: Describe the class's purpose and any important attributes.

Function/Method Docstrings: Describe what the function does, its arguments (Args:), what it returns (Returns:), and any exceptions it might raise (Raises:). Use Sphinx, NumPy, or Google style for parameters and return types. Google style is often preferred for its readability.

Inline Comments: Use # for inline comments to explain complex logic, non-obvious code, or tricky algorithms. Keep them concise and relevant.

Example Google-style Docstring:

def fetch_data_with_pagination(endpoint: str,
                               params: dict = None,
                               page_limit: int = None) -> list:
    """
    Fetches data from the API endpoint, handling pagination.

    Args:
        endpoint: The API endpoint (e.g., "items").
        params: Optional dictionary of query parameters.
        page_limit: Maximum number of pages to fetch. If None, fetches all pages.

    Returns:
        A list of dictionaries, where each dictionary represents a data record.

    Raises:
        ApiError: If an API request fails or encounters an unexpected response.
        RateLimitExceeded: If the API's rate limit is hit.
    """
    # Function implementation

VI. API Wrapper Specific Guidelines
6.1. API Keys and Authentication
Environment Variables: Never hardcode API keys or sensitive credentials. Require users to set API keys as environment variables. Provide clear instructions on how to do this (e.g., using a .env file and python-dotenv).

Function/Class Arguments: Allow API keys to be passed as function arguments or class constructor arguments for explicit usage or testing, but prioritize environment variables if the argument is None.

Secure Handling: Use os.getenv() or dotenv.load_dotenv() + os.getenv() to retrieve API keys.

Authentication Flow: Clearly define and implement the authentication process (e.g., API keys in headers, OAuth 2.0 flows, Bearer tokens).

Example (API Key Handling):

import os
from dotenv import load_dotenv

# In your main script or package __init__.py
load_dotenv() # Load .env file at application start

def _get_api_key(key_name: str = "MYAPI_KEY", api_key_arg: str = None) -> str:
    """
    Retrieves the API key from arguments or environment variables.

    Args:
        key_name: The name of the environment variable to check.
        api_key_arg: An API key passed directly as an argument.

    Returns:
        The retrieved API key string.

    Raises:
        ValueError: If the API key is not found.
    """
    if api_key_arg:
        return api_key_arg
    
    api_key_env = os.getenv(key_name)
    if not api_key_env:
        raise ValueError(
            f"API key '{key_name}' not found. Please set it as an "
            "environment variable or pass it directly."
        )
    return api_key_env

6.2. HTTP Requests and Error Handling
requests Package: Use the requests library for making HTTP requests. It is the de facto standard for Python HTTP clients due to its simplicity and robustness.

Status Codes: Explicitly check HTTP status codes.

Use response.raise_for_status() to automatically raise requests.exceptions.HTTPError for 4xx or 5xx responses.

Provide informative custom exceptions or messages for specific API error codes.

Rate Limiting:

Implement client-side rate limiting (e.g., using ratelimit library or custom decorator/logic with time.sleep).

Parse Retry-After headers if available in API responses to back off appropriately.

Network Errors: Handle common requests.exceptions (e.g., ConnectionError, Timeout, RequestException) gracefully using try-except blocks.

Timeout: Always set explicit timeout values for requests calls to prevent hanging indefinitely.

Example (Error Handling with requests):

import requests
from myapipackage.exceptions import RateLimitExceeded, ApiError

def make_api_call(url: str, headers: dict, params: dict = None, timeout: int = 30) -> dict:
    try:
        response = requests.get(url, headers=headers, params=params, timeout=timeout)
        response.raise_for_status() # Raises HTTPError for 4xx/5xx responses
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429: # Too Many Requests
            retry_after = e.response.headers.get("Retry-After")
            raise RateLimitExceeded(f"Rate limit hit. Retry after {retry_after} seconds.") from e
        raise ApiError(f"API returned error {e.response.status_code}: {e.response.text}") from e
    except requests.exceptions.Timeout as e:
        raise ApiError(f"API request timed out after {timeout} seconds.") from e
    except requests.exceptions.ConnectionError as e:
        raise ApiError(f"Network connection error: {e}") from e
    except requests.exceptions.RequestException as e:
        raise ApiError(f"An unexpected API request error occurred: {e}") from e

6.3. Request and Response Parsing
Consistent Data Structures: Aim to return consistent Python data structures (e.g., dictionaries, lists of dictionaries, Pandas DataFrames) regardless of API response variations.

JSON Handling: Use response.json() from requests. Handle potential json.JSONDecodeError if the response is not valid JSON.

Data Validation: Validate incoming API response data using tools like Pydantic or custom schema validation functions to ensure expected structure.

Flattening/Normalization: Provide helper functions to flatten nested JSON structures into usable flat dictionaries or Pandas DataFrames where appropriate.

Key Naming: Standardize dictionary keys or DataFrame column names to snake_case in your package's output, even if the API uses camelCase or PascalCase.

6.4. Pagination
If the API supports pagination, implement automatic fetching of all pages up to a user-defined limit or until no more data is available.

Clearly expose pagination parameters to the user (e.g., page_size, limit, offset).

VII. Logging
Use Python's built-in logging module for all internal messages.

Configure log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL) appropriately.

Do not print directly to console (print()) for non-debug messages in library code.

Example:

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Set default level for the module

# In a function
logger.debug(f"Fetching data from {url} with params {params}")
# ...
logger.error(f"Failed to parse response: {e}")

VIII. Testing
Comprehensive testing is critical for API wrappers due to external dependencies.

Unit Tests (unittest or pytest):

Test individual functions and methods in isolation.

Crucially, mock API calls using unittest.mock (or pytest-mock for pytest) to ensure tests are fast, reliable, and don't depend on network connectivity or live API availability. This avoids hitting rate limits during development.

Integration Tests:

Run tests against a live API environment (e.g., a staging environment or a dedicated test account).

These should be run less frequently than unit tests, perhaps only in CI/CD pipelines.

Require a valid API key (e.g., from environment variables) to run.

CI/CD Integration: Automate test execution (unit and integration) using GitHub Actions, GitLab CI, or similar platforms.

Example (tests/unit/test_client.py with mocking using pytest):

# tests/unit/test_client.py
import pytest
from unittest.mock import Mock, patch

from myapipackage.client import ApiClient
from myapipackage.exceptions import ApiError, RateLimitExceeded


@pytest.fixture
def mock_client():
    return ApiClient(api_key="mock_api_key")

def test_get_user_data_success(mock_client):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock() # Ensure this method exists
    mock_response.json.return_value = {"id": 1, "name": "Alice"}

    with patch("requests.Session.get", return_value=mock_response) as mock_get:
        user_data = mock_client.get_user_data(user_id="123")
        assert user_data == {"id": 1, "name": "Alice"}
        mock_get.assert_called_once_with(
            "https://api.example.com/v1/users/123",
            headers={"Authorization": "Bearer mock_api_key"},
            params={},
            timeout=30
        )

def test_get_user_data_api_error(mock_client):
    mock_response = Mock()
    mock_response.status_code = 400
    mock_response.raise_for_status = Mock(side_effect=requests.exceptions.HTTPError(response=mock_response))
    mock_response.text = "Bad Request" # Add text for the error message

    with patch("requests.Session.get", return_value=mock_response):
        with pytest.raises(ApiError, match="API returned error 400"):
            mock_client.get_user_data(user_id="invalid")

def test_get_user_data_rate_limit(mock_client):
    mock_response = Mock()
    mock_response.status_code = 429
    mock_response.headers = {"Retry-After": "60"}
    mock_response.raise_for_status = Mock(side_effect=requests.exceptions.HTTPError(response=mock_response))
    mock_response.text = "Too Many Requests"

    with patch("requests.Session.get", return_value=mock_response):
        with pytest.raises(RateLimitExceeded, match="Rate limit hit. Retry after 60 seconds."):
            mock_client.get_user_data(user_id="some_id")

IX. Dependency Management
pyproject.toml (Poetry/Hatch) or setup.py (setuptools): Define all package dependencies explicitly.

requirements.txt: Can be generated from pyproject.toml for deployment environments if not using Poetry directly.

Virtual Environments: Encourage the use of virtual environments (venv, conda, uv venv) to isolate project dependencies.

X. Contribution Guidelines
New Branches: Create a new branch for each feature or bug fix.

Pull Requests: Submit pull requests for review.

Ensure all new code adheres to this style guide.

Include comprehensive docstrings for new functions/classes.

Add or update unit and integration tests as appropriate.

Run linters (flake8, pylint), formatters (black), and type checkers (mypy) before submitting.

Issue Tracking: Link pull requests to relevant issues.

By consistently applying these guidelines, you will build a high-quality Python API wrapper that is easy to develop, maintain, and use.
