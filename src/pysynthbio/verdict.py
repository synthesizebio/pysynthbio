"""VERDICT's versioned JSON API, independent of legacy Platform defaults."""

import os
import time
from urllib.parse import quote, urlparse
from uuid import uuid4

import requests


class VerdictAPIError(RuntimeError):
    """HTTP failure with the server's structured validation details preserved."""

    def __init__(self, status_code, details):
        self.status_code = status_code
        self.details = details
        super().__init__(f"VERDICT HTTP {status_code}: {details}")


class VerdictClient:
    """Run VERDICT models and trial workflows; return scientific JSON unchanged.

    Create a key at https://verdict.synthesize.bio/app/api-keys. Supply it here or
    through VERDICT_API_KEY. Existing Platform functions keep their defaults.
    """

    def __init__(self, api_key=None, api_base_url=None, timeout=60):
        self.api_key = api_key or os.environ.get("VERDICT_API_KEY")
        if not self.api_key:
            raise ValueError("Set VERDICT_API_KEY or supply a VERDICT API key")
        self.api_base_url = (
            api_base_url
            or os.environ.get("VERDICT_API_BASE_URL")
            or "https://verdict.synthesize.bio"
        ).rstrip("/")
        url = urlparse(self.api_base_url)
        if (
            url.scheme != "https"
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("api_base_url must be an HTTPS origin without credentials")
        self.timeout = timeout

    def _request(self, method, path, body=None, idempotency_key=None):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key
        response = requests.request(
            method,
            f"{self.api_base_url}/api/v1/{path}",
            headers=headers,
            json=body,
            timeout=self.timeout,
            allow_redirects=False,
        )
        try:
            result = response.json()
        except ValueError:
            raise VerdictAPIError(
                response.status_code,
                {"error": "non_json_response", "body": response.text},
            ) from None
        if not 200 <= response.status_code < 300:
            raise VerdictAPIError(response.status_code, result)
        return result

    @staticmethod
    def _submission_key(value):
        if value is None:
            return str(uuid4())
        if not isinstance(value, str) or not value.strip() or len(value) > 180:
            raise ValueError(
                "idempotency_key must be a nonblank string of at most 180 characters"
            )
        return value

    def models(self):
        """Discover models, operations, input schemas, releases and limits."""
        return self._request("GET", "models")

    def example(self, model, operation=None):
        path = f"models/{quote(model, safe='')}/example-query"
        if operation:
            path += f"?operation={quote(operation, safe='')}"
        return self._request("GET", path)

    def predict(self, model, operation, input, *, release=None, idempotency_key=None):
        """Submit a model call; reuse the key after uncertain transport errors."""
        body = {"operation": operation, "input": input}
        if release is not None:
            body["release"] = release
        return self._request(
            "POST",
            f"models/{quote(model, safe='')}/predict",
            body,
            idempotency_key=self._submission_key(idempotency_key),
        )

    def run(self, run_id):
        return self._request("GET", f"runs/{quote(run_id, safe='')}")

    def result(self, run_id):
        """Retrieve saved scientific results without running inference again."""
        return self._request("GET", f"runs/{quote(run_id, safe='')}/result")

    def workflows(self):
        return self._request("GET", "workflows")

    def submit_workflow(self, input, *, idempotency_key=None):
        return self._request(
            "POST",
            "workflows",
            {
                "input": input,
                "idempotencyKey": self._submission_key(idempotency_key),
            },
        )

    def wait(self, run_id, *, timeout=1800, interval=5):
        """Poll an existing run; timeouts retain its ID for later resumption."""
        if timeout <= 0 or interval <= 0:
            raise ValueError("timeout and interval must be positive")
        deadline = time.monotonic() + timeout
        while True:
            run = self.run(run_id)
            if run["status"] not in {"queued", "running"}:
                return run
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"VERDICT run {run_id} is still running; resume polling"
                )
            time.sleep(min(interval, remaining))
