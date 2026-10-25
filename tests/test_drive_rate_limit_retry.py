"""Offline tests for Drive HTTP 403/429/5xx classification and bounded retry (contract C1 / A11).

time.sleep is patched in every test; the recorded delays are asserted to be bounded.
"""
from __future__ import annotations

import email.utils
import json
import math
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple
from unittest import mock

from colab_adapter.errors import DriveAuthenticationError, DriveQuotaExceededError, DriveTransferError
from colab_adapter.storage import ArtifactType, DriveCredentials, GoogleDriveArtifactStore
from colab_adapter.storage.google_drive import _MAX_RETRY_AFTER_SEC, _parse_retry_after
from tests.drive_api_mock import StrictDriveMock

SLEEP_TARGET = "colab_adapter.storage.google_drive.time.sleep"


def _forbidden(reason: str) -> bytes:
    return json.dumps({
        "error": {
            "code": 403,
            "message": reason,
            "errors": [{"domain": "usageLimits", "reason": reason, "message": reason}],
        }
    }).encode("utf-8")


class ScriptedHttpClient:
    """Returns scripted (status, headers, body) tuples in order; exceptions are raised."""

    def __init__(self, responses: Sequence[Any]):
        self.responses: List[Any] = list(responses)
        self.calls = 0

    def request(self, method, url, headers=None, data=None, timeout=30.0):
        self.calls += 1
        if not self.responses:
            return 200, {}, b"{}"
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class RateLimitRetryBase(unittest.TestCase):
    MAX_RETRIES = 5

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="rate_limit_")
        self.addCleanup(lambda: shutil.rmtree(self.test_dir, ignore_errors=True))
        self.sleep_patch = mock.patch(SLEEP_TARGET)
        self.sleep = self.sleep_patch.start()
        self.addCleanup(self.sleep_patch.stop)

    def _store(self, client) -> GoogleDriveArtifactStore:
        creds = DriveCredentials(
            client_id="cid",
            client_secret="csecret",
            refresh_token="rtoken",
            access_token="atoken",
            token_expiry=time.time() + 3600,
        )
        return GoogleDriveArtifactStore(
            credentials=creds,
            local_staging_dir=Path(self.test_dir) / "staging",
            http_client=client,
            max_retries=self.MAX_RETRIES,
            initial_backoff_sec=0.5,
        )

    def _delays(self) -> List[float]:
        return [float(c.args[0]) for c in self.sleep.call_args_list]

    def _assert_bounded(self):
        for d in self._delays():
            self.assertGreaterEqual(d, 0.0)
            self.assertLessEqual(d, _MAX_RETRY_AFTER_SEC)
            self.assertFalse(math.isnan(d))


class TestDriveRateLimitRetry(RateLimitRetryBase):
    URL = "https://www.googleapis.com/drive/v3/files?q=x"

    def test_user_rate_limit_403_then_200_succeeds_with_bounded_delay(self):
        client = ScriptedHttpClient([(403, {}, _forbidden("userRateLimitExceeded")), (200, {}, b"ok")])
        status, _, body = self._store(client)._request_with_retry("GET", self.URL)
        self.assertEqual((status, body), (200, b"ok"))
        self.assertEqual(client.calls, 2)
        self.assertEqual(len(self._delays()), 1)
        self._assert_bounded()

    def test_rate_limit_exceeded_403_then_200_succeeds(self):
        client = ScriptedHttpClient([(403, {}, _forbidden("rateLimitExceeded")), (200, {}, b"ok")])
        status, _, _ = self._store(client)._request_with_retry("GET", self.URL)
        self.assertEqual(status, 200)
        self.assertEqual(client.calls, 2)
        self._assert_bounded()

    def test_sharing_rate_limit_403_is_retried(self):
        client = ScriptedHttpClient([(403, {}, _forbidden("sharingRateLimitExceeded")), (200, {}, b"ok")])
        status, _, _ = self._store(client)._request_with_retry("GET", self.URL)
        self.assertEqual(status, 200)
        self.assertEqual(client.calls, 2)

    def test_rate_limit_budget_exhausted_raises_quota_error(self):
        client = ScriptedHttpClient([(403, {}, _forbidden("userRateLimitExceeded"))] * (self.MAX_RETRIES + 1))
        with self.assertRaises(DriveQuotaExceededError):
            self._store(client)._request_with_retry("GET", self.URL)
        self.assertEqual(client.calls, self.MAX_RETRIES + 1)
        self.assertEqual(len(self._delays()), self.MAX_RETRIES)
        self._assert_bounded()

    def test_plain_permission_403_is_authentication_error_and_not_retried(self):
        client = ScriptedHttpClient([(403, {}, _forbidden("insufficientPermissions")), (200, {}, b"ok")])
        with self.assertRaises(DriveAuthenticationError):
            self._store(client)._request_with_retry("GET", self.URL)
        self.assertEqual(client.calls, 1)
        self.assertEqual(self._delays(), [])

    def test_storage_and_daily_quota_403_are_quota_errors_not_retried(self):
        for reason in ("storageQuotaExceeded", "dailyLimitExceeded", "quotaExceeded"):
            with self.subTest(reason=reason):
                self.sleep.reset_mock()
                client = ScriptedHttpClient([(403, {}, _forbidden(reason)), (200, {}, b"ok")])
                with self.assertRaises(DriveQuotaExceededError):
                    self._store(client)._request_with_retry("GET", self.URL)
                self.assertEqual(client.calls, 1)
                self.assertEqual(self._delays(), [])

    def test_429_with_retry_after_honours_value(self):
        client = ScriptedHttpClient([(429, {"retry-after": "2"}, b""), (200, {}, b"ok")])
        status, _, _ = self._store(client)._request_with_retry("GET", self.URL)
        self.assertEqual(status, 200)
        self.assertEqual(self._delays(), [2.0])

    def test_429_without_retry_after_uses_bounded_backoff(self):
        client = ScriptedHttpClient([(429, {}, b""), (429, {}, b""), (200, {}, b"ok")])
        status, _, _ = self._store(client)._request_with_retry("GET", self.URL)
        self.assertEqual(status, 200)
        delays = self._delays()
        self.assertEqual(len(delays), 2)
        self.assertGreater(delays[0], 0.0)
        self.assertLessEqual(delays[0], 0.5 * 1.5 + 1e-9)
        self.assertLessEqual(delays[1], 1.0 * 1.5 + 1e-9)
        self._assert_bounded()

    def test_retry_after_http_date_capped_and_invalid_values_ignored(self):
        future = email.utils.formatdate(time.time() + 3600, usegmt=True)
        past = email.utils.formatdate(time.time() - 3600, usegmt=True)
        parsed_future = _parse_retry_after(future)
        self.assertIsNotNone(parsed_future)
        self.assertGreater(parsed_future, _MAX_RETRY_AFTER_SEC)
        self.assertIsNone(_parse_retry_after(past))
        self.assertIsNone(_parse_retry_after("-5"))
        self.assertIsNone(_parse_retry_after("nan"))
        self.assertIsNone(_parse_retry_after("inf"))
        self.assertIsNone(_parse_retry_after("not a date"))
        self.assertIsNone(_parse_retry_after(""))
        self.assertEqual(_parse_retry_after("0"), 0.0)

        client = ScriptedHttpClient([(429, {"Retry-After": future}, b""), (200, {}, b"ok")])
        self._store(client)._request_with_retry("GET", self.URL)
        self.assertEqual(self._delays(), [_MAX_RETRY_AFTER_SEC])

        self.sleep.reset_mock()
        client = ScriptedHttpClient([(429, {"retry-after": "-7"}, b""), (200, {}, b"ok")])
        self._store(client)._request_with_retry("GET", self.URL)
        self._assert_bounded()
        self.assertNotEqual(self._delays(), [-7.0])

    def test_malformed_json_plain_403_is_never_retried(self):
        client = ScriptedHttpClient([(403, {}, b"<html>Forbidden</html>"), (200, {}, b"ok")])
        with self.assertRaises(DriveAuthenticationError):
            self._store(client)._request_with_retry("GET", self.URL)
        self.assertEqual(client.calls, 1)
        self.assertEqual(self._delays(), [])

    def test_delays_stay_bounded_under_long_retry_after_storm(self):
        storm = [(429, {"retry-after": "3600"}, b"")] * 8 + [(200, {}, b"ok")]
        client = ScriptedHttpClient(storm)
        store = self._store(client)
        store.max_retries = 8
        status, _, _ = store._request_with_retry("GET", self.URL)
        self.assertEqual(status, 200)
        self.assertEqual(len(self._delays()), 8)
        self.assertTrue(all(d == _MAX_RETRY_AFTER_SEC for d in self._delays()))


class TestNoDuplicatePublicationAfterAmbiguousResponses(RateLimitRetryBase):
    """Rate-limit responses on upload calls must not create a second remote file or session."""

    def _publish_with_mock(self, mock_client: StrictDriveMock, data: bytes):
        src = Path(self.test_dir) / "model.pt"
        src.write_bytes(data)
        creds = DriveCredentials(
            client_id="cid",
            client_secret="csecret",
            refresh_token="rtoken",
            access_token="atoken",
            token_expiry=time.time() + 3600,
        )
        store = GoogleDriveArtifactStore(
            credentials=creds,
            local_staging_dir=Path(self.test_dir) / "staging",
            http_client=mock_client,
            initial_backoff_sec=0.01,
        )
        return store.publish(src, "proj_rl", "run_rl", ArtifactType.CHECKPOINT.value)

    def test_rate_limited_session_init_then_success_creates_single_upload(self):
        client = StrictDriveMock()
        client.inject_fault("init", status_code=403, body=_forbidden("userRateLimitExceeded"), count=1)
        art = self._publish_with_mock(client, b"RL_INIT_" + b"x" * 64)
        self.assertEqual(client.upload_sessions_initiated, 1)
        names = [f for f in client.mock_files if f.get("name") == "model.pt"]
        self.assertEqual(len(names), 1)
        self.assertEqual(names[0]["id"], art.remote_file_id)
        self._assert_bounded()

    def test_rate_limited_listing_then_success_creates_single_upload(self):
        client = StrictDriveMock()
        client.inject_fault("list", status_code=403, body=_forbidden("rateLimitExceeded"), count=2)
        art = self._publish_with_mock(client, b"RL_LIST_" + b"y" * 64)
        self.assertTrue(art.is_latest)
        self.assertEqual(client.upload_sessions_initiated, 1)
        names = [f for f in client.mock_files if f.get("name") == "model.pt"]
        self.assertEqual(len(names), 1)
        self._assert_bounded()


if __name__ == "__main__":
    unittest.main()
