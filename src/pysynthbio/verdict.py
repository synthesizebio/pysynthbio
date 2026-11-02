"""VERDICT's versioned JSON API, independent of legacy Platform defaults."""

import math
import os
import time
from pathlib import Path
from urllib.parse import quote, urlparse
from uuid import uuid4

import requests


class VerdictAPIError(RuntimeError):
    """HTTP failure with the server's structured validation details preserved."""

    def __init__(self, status_code, details):
        self.status_code = status_code
        self.details = details
        super().__init__(f"VERDICT HTTP {status_code}: {details}")


class TrialTimeoutError(TimeoutError):
    """An existing trial is still running; resume with wait_for_trial(run_id)."""

    def __init__(self, run_id):
        self.run_id = run_id
        super().__init__(
            f"Trial {run_id} is still running; use wait_for_trial({run_id!r})"
        )


class TrialRunError(RuntimeError):
    """A terminal trial has no report. The run ID and server state are retained."""

    def __init__(self, run_id, status):
        self.run_id, self.status = run_id, status
        super().__init__(
            f"Trial {run_id} has no report: {status.get('status')} / "
            f"{status.get('failureCode')}"
        )


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

    def _request(
        self,
        method,
        path,
        body=None,
        idempotency_key=None,
        *,
        form=None,
        files=None,
        timeout=None,
    ):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key
        response = requests.request(
            method,
            f"{self.api_base_url}/api/v1/{path}",
            headers=headers,
            json=body,
            timeout=self.timeout if timeout is None else timeout,
            **(
                {"data": form, "files": files}
                if form is not None or files is not None
                else {}
            ),
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

    def import_trial(self, *, nct_id=None, document=None, timeout=210):
        """Propose a trial draft using Synthia 2.0; never submit an analysis.

        Supply exactly one NCT ID or local PDF/DOCX/TXT path (up to 5 MiB).
        Returns draft and metadata, including evidence and fields needing review.
        Extraction errors, including streamed HTTP-200 errors, raise VerdictAPIError.
        A retry performs a fresh extraction; imports have no saved run/idempotency key.
        """
        if (nct_id is None) == (document is None):
            raise ValueError("Supply exactly one of nct_id or document")
        if (
            not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be a positive finite number")
        if nct_id is not None:
            if not isinstance(nct_id, str) or not nct_id.strip():
                raise ValueError("nct_id must be a nonblank string")
            result = self._request(
                "POST", "synthia/import", form={"nctId": nct_id}, timeout=timeout
            )
        else:
            path = Path(document)
            if path.suffix.lower() not in {".pdf", ".docx", ".txt"}:
                raise ValueError("document must be a PDF, DOCX, or TXT file")
            with path.open("rb") as source:
                content = source.read(5 * 1024 * 1024 + 1)
            if not content or len(content) > 5 * 1024 * 1024:
                raise ValueError("document must contain between 1 byte and 5 MiB")
            result = self._request(
                "POST",
                "synthia/import",
                files={"document": (path.name, content)},
                timeout=timeout,
            )
        if (
            not isinstance(result, dict)
            or "error" in result
            or not isinstance(result.get("result"), dict)
        ):
            raise VerdictAPIError(200, result)
        return result["result"]

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
        while time.monotonic() < deadline:
            run = self.run(run_id)
            if run["status"] not in {"queued", "running"}:
                return run
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(interval, remaining))
        raise TimeoutError(f"VERDICT run {run_id} is still running; resume polling")

    @staticmethod
    def _trial_polling_options(timeout, poll_interval):
        if (
            not math.isfinite(timeout)
            or timeout <= 0
            or not math.isfinite(poll_interval)
            or poll_interval <= 0
        ):
            raise ValueError("timeout and poll_interval must be finite and positive")

    def run_trial(self, trial_input, *, idempotency_key, timeout=3600, poll_interval=5):
        """Submit a live trial and wait for a TrialResult.

        Reuse idempotency_key for retries of the same input, including after a
        lost response or kernel restart. Use a new key for intentional new
        inference. No local cache or automatic filesystem writes are involved.
        A TrialTimeoutError retains run_id for wait_for_trial. HTTP errors remain
        VerdictAPIError; failed runs without reports raise TrialRunError.
        """
        from .trials import _validate_report

        self._trial_polling_options(timeout, poll_interval)
        if idempotency_key is None:
            raise ValueError("run_trial requires an explicit idempotency_key")
        key = self._submission_key(idempotency_key)
        execution = self.workflows()["execution"]
        if execution.get("mode") != "live":
            raise RuntimeError("Deployment is not in live mode; no trial was submitted")
        submitted = self.submit_workflow(trial_input, idempotency_key=key)
        result = self.wait_for_trial(
            submitted["id"], timeout=timeout, poll_interval=poll_interval
        )
        _validate_report(result.raw, submitted["id"], trial_input)
        result.provenance["execution"] = execution
        return result

    def wait_for_trial(self, run_id, *, timeout=3600, poll_interval=5):
        """Wait for an existing trial and return its report; never submit inference."""
        self._trial_polling_options(timeout, poll_interval)
        try:
            status = self.wait(run_id, timeout=timeout, interval=poll_interval)
        except TimeoutError as exc:
            raise TrialTimeoutError(run_id) from exc
        if status.get("result") is None:
            raise TrialRunError(run_id, status)
        return self.get_trial_result(run_id)

    def get_trial_result(self, run_id):
        """Fetch a completed TrialResult once. No submission or polling."""
        from .trials import TrialResult, _validate_report

        report = self.result(run_id)
        _validate_report(report, run_id)
        return TrialResult(report, self.api_base_url, "Retrieved from the VERDICT API")
