"""Offline tests for the live Drive smoke harness and the human-run authorize helper.

Everything here runs against tests/drive_api_mock.py or fake HTTP/browser/receiver objects.
No test opens a socket or contacts Google. The harness is never given the real HTTP client.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import stat
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab_adapter.storage.google_drive import DriveCredentials  # noqa: E402
from scripts.live import drive_authorize as auth  # noqa: E402
from scripts.live import drive_smoke as ds  # noqa: E402
from tests.drive_api_mock import StrictDriveMock  # noqa: E402

ACCESS = "ya29.test-access-token-value-0001"
REFRESH = "1//test-refresh-token-value-0002"
CLIENT_SECRET = "GOCSPX-test-client-secret-0003"
CLIENT_ID = "123-test.apps.googleusercontent.com"
FIXED_NOW = datetime(2026, 10, 25, 12, 0, 0, tzinfo=timezone.utc)


class RecordingClient:
    """Fake HTTP client that records every call and answers 500 (never reached in dry-run)."""

    def __init__(self) -> None:
        self.calls = []

    def request(self, method, url, headers=None, data=None, timeout=30.0):
        self.calls.append((method, url))
        return 500, {}, b"{}"


def _private_dir(parent: Path, name: str) -> Path:
    d = parent / name
    d.mkdir(mode=0o700)
    os.chmod(d, 0o700)
    return d


def _token_file(root: Path) -> Path:
    path = root / "token.json"
    DriveCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, refresh_token=REFRESH,
                     access_token=ACCESS, token_expiry=9999999999.0).save(path)
    return path


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="drive-harness-test-")
        self.root = Path(self._tmp.name)
        os.chmod(self.root, 0o700)

    def tearDown(self) -> None:
        self._tmp.cleanup()


# ---------------------------------------------------------------------------
# drive_authorize: pure parts and refusal paths
# ---------------------------------------------------------------------------

class AuthorizePureTests(_Base):
    def test_pkce_matches_rfc7636_example(self):
        verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
        v, challenge = auth.pkce_pair(verifier)
        self.assertEqual(v, verifier)
        self.assertEqual(challenge, "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM")

    def test_generated_pkce_is_s256_without_padding(self):
        verifier, challenge = auth.pkce_pair()
        self.assertTrue(43 <= len(verifier) <= 128)
        expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        self.assertEqual(challenge, expected)
        self.assertNotIn("=", challenge)

    def test_invalid_verifier_is_refused(self):
        with self.assertRaises(auth.AuthorizeError):
            auth.pkce_pair("short")
        with self.assertRaises(auth.AuthorizeError):
            auth.pkce_pair("a" * 43 + "+/")

    def test_auth_url_requests_only_drive_file_with_pkce_and_state(self):
        url = auth.build_auth_url(CLIENT_ID, "http://127.0.0.1:5555/callback", "STATE123", "CHAL456")
        self.assertTrue(url.startswith(auth.AUTH_URI + "?"))
        query = url.split("?", 1)[1]
        import urllib.parse as up
        params = dict(up.parse_qsl(query))
        self.assertEqual(params["scope"], "https://www.googleapis.com/auth/drive.file")
        self.assertEqual(params["code_challenge_method"], "S256")
        self.assertEqual(params["code_challenge"], "CHAL456")
        self.assertEqual(params["state"], "STATE123")
        self.assertEqual(params["access_type"], "offline")
        self.assertEqual(params["include_granted_scopes"], "false")
        self.assertNotIn(CLIENT_SECRET, url)

    def test_callback_accepts_matching_state(self):
        self.assertEqual(auth.parse_callback("code=abc123&state=S1", "S1"), "abc123")

    def test_callback_refuses_state_mismatch_and_missing_state(self):
        with self.assertRaises(auth.AuthorizeError) as cm:
            auth.parse_callback("code=abc123&state=WRONG", "S1")
        self.assertNotIn("abc123", str(cm.exception))
        with self.assertRaises(auth.AuthorizeError):
            auth.parse_callback("code=abc123", "S1")

    def test_callback_refuses_provider_error_and_duplicate_code(self):
        with self.assertRaises(auth.AuthorizeError) as cm:
            auth.parse_callback("error=access_denied&state=S1", "S1")
        self.assertIn("access_denied", str(cm.exception))
        with self.assertRaises(auth.AuthorizeError):
            auth.parse_callback("code=a&code=b&state=S1", "S1")

    def test_scope_check_refuses_broader_drive_scope(self):
        auth.check_granted_scope("https://www.googleapis.com/auth/drive.file")
        with self.assertRaises(auth.AuthorizeError):
            auth.check_granted_scope("https://www.googleapis.com/auth/drive")
        with self.assertRaises(auth.AuthorizeError):
            auth.check_granted_scope("https://www.googleapis.com/auth/drive.file https://www.googleapis.com/auth/drive.readonly")
        with self.assertRaises(auth.AuthorizeError):
            auth.check_granted_scope("https://www.googleapis.com/auth/drive.readonly")
        with self.assertRaises(auth.AuthorizeError):
            auth.check_granted_scope(None)

    def test_client_config_requires_desktop_client(self):
        ok = {"installed": {"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET}}
        self.assertEqual(auth.extract_client_config(ok), (CLIENT_ID, CLIENT_SECRET))
        with self.assertRaises(auth.AuthorizeError):
            auth.extract_client_config({"web": {"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET}})
        with self.assertRaises(auth.AuthorizeError):
            auth.extract_client_config({"installed": {"client_id": CLIENT_ID}})

    def test_client_secret_file_mode_is_enforced_and_secret_not_in_error(self):
        path = self.root / "client.json"
        path.write_text(json.dumps({"installed": {"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET}}))
        os.chmod(path, 0o644)
        with self.assertRaises(auth.AuthorizeError) as cm:
            auth.load_client_credentials(str(path), {})
        self.assertNotIn(CLIENT_SECRET, str(cm.exception))
        os.chmod(path, 0o600)
        self.assertEqual(auth.load_client_credentials(str(path), {}), (CLIENT_ID, CLIENT_SECRET))

    def test_client_secret_symlink_is_refused(self):
        real = self.root / "real.json"
        real.write_text("{}")
        os.chmod(real, 0o600)
        link = self.root / "link.json"
        os.symlink(real, link)
        with self.assertRaises(auth.AuthorizeError):
            auth.load_client_credentials(str(link), {})

    def test_env_credentials_required_both(self):
        env = {auth.ENV_CLIENT_ID: CLIENT_ID}
        with self.assertRaises(auth.AuthorizeError):
            auth.load_client_credentials(None, env)
        env[auth.ENV_CLIENT_SECRET] = CLIENT_SECRET
        self.assertEqual(auth.load_client_credentials(None, env), (CLIENT_ID, CLIENT_SECRET))

    def test_token_path_inside_repo_or_existing_is_refused(self):
        with self.assertRaises(auth.AuthorizeError):
            auth.check_token_path(str(REPO_ROOT / "token.json"), REPO_ROOT)
        existing = self.root / "exists.json"
        existing.write_text("x")
        with self.assertRaises(auth.AuthorizeError):
            auth.check_token_path(str(existing), REPO_ROOT)
        fresh = auth.check_token_path(str(self.root / "new-token.json"), REPO_ROOT)
        self.assertEqual(fresh.name, "new-token.json")

    def test_non_interactive_is_refused(self):
        class Fake(io.StringIO):
            def __init__(self, tty):
                super().__init__()
                self._tty = tty

            def isatty(self):
                return self._tty

        with self.assertRaises(auth.AuthorizeError):
            auth.ensure_interactive(Fake(False), Fake(True))
        with self.assertRaises(auth.AuthorizeError):
            auth.ensure_interactive(Fake(True), Fake(False))
        auth.ensure_interactive(Fake(True), Fake(True))


class _FakeReceiver:
    def __init__(self, port_redirect="http://127.0.0.1:43210/callback", code="AUTHCODE-XYZ", state_override=None):
        self.redirect_uri = port_redirect
        self.code = code
        self.state_override = state_override
        self.seen_state = None

    def wait_for_code(self, expected_state, timeout=0):
        self.seen_state = expected_state
        query = f"code={self.code}&state={self.state_override or expected_state}"
        return auth.parse_callback(query, expected_state)


class _FakeTokenClient:
    def __init__(self, status=200, body=None):
        self.calls = []
        self.status = status
        self.body = body if body is not None else {
            "access_token": "ya29.fake-issued-access-token-0004",
            "refresh_token": "1//fake-issued-refresh-token-0005",
            "expires_in": 3599,
            "scope": "https://www.googleapis.com/auth/drive.file",
            "token_type": "Bearer",
        }

    def request(self, method, url, headers=None, data=None, timeout=30.0):
        self.calls.append((method, url, data))
        return self.status, {}, json.dumps(self.body).encode("utf-8")


class AuthorizeFlowTests(_Base):
    def test_full_flow_with_fakes_saves_0600_token_and_prints_no_secrets(self):
        token_path = self.root / "drive-token.json"
        http = _FakeTokenClient()
        lines = []
        opened = []
        auth.run_authorization(CLIENT_ID, CLIENT_SECRET, token_path, http_client=http,
                               open_browser=opened.append, receiver=_FakeReceiver(), emit=lines.append)
        self.assertEqual(len(opened), 1)
        self.assertIn("scope=https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive.file", opened[0])
        self.assertEqual(len(http.calls), 1)
        self.assertEqual(http.calls[0][0], "POST")
        self.assertEqual(http.calls[0][1], auth.TOKEN_URI)
        self.assertEqual(stat.S_IMODE(os.stat(token_path).st_mode), 0o600)
        self.assertTrue(any("File mode: 0600" in line for line in lines))
        output = "\n".join(lines)
        for secret in (CLIENT_SECRET, "AUTHCODE-XYZ", "ya29.fake-issued-access-token-0004",
                       "1//fake-issued-refresh-token-0005"):
            self.assertNotIn(secret, output)
        saved = json.loads(token_path.read_text())
        self.assertEqual(saved["refresh_token"], "1//fake-issued-refresh-token-0005")

    def test_state_mismatch_stops_before_token_exchange(self):
        token_path = self.root / "t.json"
        http = _FakeTokenClient()
        with self.assertRaises(auth.AuthorizeError):
            auth.run_authorization(CLIENT_ID, CLIENT_SECRET, token_path, http_client=http,
                                   open_browser=lambda u: None,
                                   receiver=_FakeReceiver(state_override="FORGED"), emit=lambda s: None)
        self.assertEqual(http.calls, [])
        self.assertFalse(token_path.exists())

    def test_broader_scope_in_response_is_not_saved(self):
        token_path = self.root / "t.json"
        body = {"access_token": "a" * 20, "refresh_token": "r" * 20, "expires_in": 3600,
                "scope": "https://www.googleapis.com/auth/drive"}
        with self.assertRaises(auth.AuthorizeError):
            auth.run_authorization(CLIENT_ID, CLIENT_SECRET, token_path,
                                   http_client=_FakeTokenClient(body=body),
                                   open_browser=lambda u: None, receiver=_FakeReceiver(), emit=lambda s: None)
        self.assertFalse(token_path.exists())

    def test_token_exchange_failure_reports_reason_only(self):
        token_path = self.root / "t.json"
        http = _FakeTokenClient(status=400, body={"error": "invalid_grant", "error_description": "AUTHCODE-XYZ"})
        with self.assertRaises(auth.AuthorizeError) as cm:
            auth.run_authorization(CLIENT_ID, CLIENT_SECRET, token_path, http_client=http,
                                   open_browser=lambda u: None, receiver=_FakeReceiver(), emit=lambda s: None)
        self.assertIn("invalid_grant", str(cm.exception))
        self.assertNotIn("AUTHCODE-XYZ", str(cm.exception))

    def test_main_refuses_without_tty_and_makes_no_request(self):
        class NoTty(io.StringIO):
            def isatty(self):
                return False

        http = _FakeTokenClient()
        out = io.StringIO()
        rc = auth.main(["--token-file", str(self.root / "t.json")],
                       environ={auth.ENV_CLIENT_ID: CLIENT_ID, auth.ENV_CLIENT_SECRET: CLIENT_SECRET},
                       stdin=NoTty(), stdout=out, http_client=http,
                       open_browser=lambda u: None, receiver=_FakeReceiver(), repo_root=REPO_ROOT)
        self.assertEqual(rc, 2)
        self.assertEqual(http.calls, [])
        self.assertNotIn(CLIENT_SECRET, out.getvalue())

    def test_main_refuses_missing_client_config_without_request(self):
        class Tty(io.StringIO):
            def isatty(self):
                return True

        http = _FakeTokenClient()
        out = io.StringIO()
        rc = auth.main(["--token-file", str(self.root / "t.json")], environ={},
                       stdin=Tty(), stdout=out, http_client=http,
                       open_browser=lambda u: None, receiver=_FakeReceiver(), repo_root=REPO_ROOT)
        self.assertEqual(rc, 2)
        self.assertEqual(http.calls, [])


# ---------------------------------------------------------------------------
# drive_smoke: dry-run, refusals, secret scanner, guard
# ---------------------------------------------------------------------------

class DryRunAndRefusalTests(_Base):
    def test_dry_run_prints_plan_and_makes_zero_requests(self):
        client = RecordingClient()
        out = io.StringIO()
        rc = ds.run([], http_client=client, stdout=out, now=FIXED_NOW, rand_hex="a1b2c3")
        self.assertEqual(rc, 0)
        self.assertEqual(client.calls, [])
        text = out.getvalue()
        self.assertIn("colab-live-smoke-20261025T120000Z-a1b2c3", text)
        self.assertIn("<owner Google account: not queried in dry-run>", text)
        self.assertIn("NOT EXECUTED", text)
        self.assertIn("DRY-RUN: no network request was made", text)

    def test_dry_run_flag_is_explicit_and_cannot_combine_with_execute(self):
        client = RecordingClient()
        out = io.StringIO()
        rc = ds.run(["--dry-run"], http_client=client, stdout=out, now=FIXED_NOW, rand_hex="a1b2c3")
        self.assertEqual(rc, 0)
        self.assertEqual(client.calls, [])
        out = io.StringIO()
        rc = ds.run(["--dry-run", "--execute", "--approval-phrase", ds.APPROVAL_PHRASE],
                    http_client=client, stdout=out, now=FIXED_NOW, rand_hex="a1b2c3")
        self.assertEqual(rc, 2)
        self.assertEqual(client.calls, [])

    def test_execute_without_exact_phrase_refuses_with_zero_requests(self):
        client = RecordingClient()
        token = _token_file(self.root)
        for phrase in ("", "approve drive smoke test", "APPROVE DRIVE SMOKE TEST ", "yes"):
            out = io.StringIO()
            rc = ds.run(["--execute", "--approval-phrase", phrase, "--token-file", str(token),
                         "--evidence-dir", str(self.root / "ev")],
                        http_client=client, stdout=out, now=FIXED_NOW, rand_hex="a1b2c3")
            self.assertEqual(rc, 2, phrase)
            self.assertIn("No request was made", out.getvalue())
        self.assertEqual(client.calls, [])
        self.assertFalse((self.root / "ev").exists())

    def test_execute_refuses_token_inside_repo_and_loose_mode(self):
        client = RecordingClient()
        inside = REPO_ROOT / "scripts" / "live" / "should-not-exist-token.json"
        out = io.StringIO()
        rc = ds.run(["--execute", "--approval-phrase", ds.APPROVAL_PHRASE, "--token-file", str(inside),
                     "--evidence-dir", str(self.root / "ev")], http_client=client, stdout=out)
        self.assertEqual(rc, 2)
        self.assertEqual(client.calls, [])
        token = _token_file(self.root)
        os.chmod(token, 0o644)
        out = io.StringIO()
        rc = ds.run(["--execute", "--approval-phrase", ds.APPROVAL_PHRASE, "--token-file", str(token),
                     "--evidence-dir", str(self.root / "ev")], http_client=client, stdout=out)
        self.assertEqual(rc, 2)
        self.assertEqual(client.calls, [])

    def test_revisions_limited_to_three_or_five(self):
        import contextlib

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            ds.run(["--revisions", "4"], stdout=io.StringIO())
        with self.assertRaises(ds.HarnessRefusal):
            ds.build_checkpoints(6)

    def test_checkpoints_are_deterministic_and_capped_at_one_mib(self):
        a1 = ds.build_checkpoints(3)
        a2 = ds.build_checkpoints(3)
        self.assertEqual([c.sha256 for c in a1], [c.sha256 for c in a2])
        for c in ds.build_checkpoints(5):
            self.assertLessEqual(c.size, ds.MAX_CHECKPOINT_BYTES)
            self.assertEqual(c.md5, hashlib.md5(c.data).hexdigest())
        with self.assertRaises(ValueError):
            ds.synthetic_bytes("X", ds.MAX_CHECKPOINT_BYTES + 1)

    def test_root_folder_name_format(self):
        name = ds.make_root_name(FIXED_NOW, "0f1e2d")
        self.assertEqual(name, "colab-live-smoke-20261025T120000Z-0f1e2d")


class GuardTests(_Base):
    def test_delete_is_refused_locally_and_never_reaches_inner_client(self):
        inner = RecordingClient()
        guard = ds.GuardedHttpClient(inner)
        with self.assertRaises(ds.DeletionForbidden):
            guard.request("DELETE", "https://www.googleapis.com/drive/v3/files/file-1")
        self.assertEqual(inner.calls, [])
        self.assertTrue(guard.aborted)

    def test_call_cap_aborts_before_the_over_limit_request(self):
        inner = RecordingClient()
        guard = ds.GuardedHttpClient(inner, max_calls=2)
        guard.request("GET", "https://www.googleapis.com/drive/v3/files?q=x")
        guard.request("GET", "https://www.googleapis.com/drive/v3/files/f1")
        with self.assertRaises(ds.CallCapExceeded):
            guard.request("GET", "https://www.googleapis.com/drive/v3/files/f2")
        self.assertEqual(len(inner.calls), 2)
        self.assertEqual(guard.calls, 2)
        self.assertIn("cap", guard.abort_reason)

    def test_request_classification_covers_drive_ops(self):
        c = ds.classify_request
        self.assertEqual(c("POST", "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable"), "upload_init")
        self.assertEqual(c("PUT", "https://x/upload?session_id=s", {"Content-Range": "bytes */10"}), "upload_query")
        self.assertEqual(c("PUT", "https://x/upload?session_id=s", {"content-range": "bytes 0-9/10"}), "upload_chunk")
        self.assertEqual(c("PATCH", "https://www.googleapis.com/drive/v3/files/f1"), "patch")
        self.assertEqual(c("GET", "https://www.googleapis.com/drive/v3/files/f1?alt=media"), "download")
        self.assertEqual(c("GET", "https://www.googleapis.com/drive/v3/files?q=name"), "list")
        self.assertEqual(c("POST", "https://www.googleapis.com/drive/v3/files?fields=id"), "folder_create")


class SecretScannerTests(_Base):
    SECRETS = {"access_token": ACCESS, "refresh_token": REFRESH, "client_secret": CLIENT_SECRET}

    def test_scanner_reports_names_not_values(self):
        leaks = ds.find_secret_leaks(f"log says {ACCESS} here", self.SECRETS)
        self.assertEqual(leaks, ["access_token"])
        self.assertNotIn(ACCESS, ",".join(leaks))
        self.assertEqual(ds.find_secret_leaks("nothing to see", self.SECRETS), [])

    def test_redact_replaces_values(self):
        text = ds.redact(f"x {REFRESH} y {CLIENT_SECRET}", self.SECRETS)
        self.assertNotIn(REFRESH, text)
        self.assertNotIn(CLIENT_SECRET, text)

    def test_emit_suppresses_line_containing_token(self):
        out = io.StringIO()
        ctx = ds.Ctx.__new__(ds.Ctx)
        ctx.secret_values = self.SECRETS
        ctx.log_lines = []
        ctx.leak_count = 0
        ctx.stdout = out
        ctx.emit(f"oops {ACCESS}")
        self.assertNotIn(ACCESS, out.getvalue())
        self.assertEqual(ctx.leak_count, 1)

    def test_injected_token_in_fake_log_is_caught_by_run(self):
        def leaky_d1(ctx, res):
            logging.getLogger("colab_adapter.fake").warning("debug dump %s", ACCESS)
            ds._check(res, "always true", True)

        token = _token_file(self.root)
        ev = _private_dir(self.root, "ev")
        out = io.StringIO()
        rc = ds.run(["--execute", "--approval-phrase", ds.APPROVAL_PHRASE, "--token-file", str(token),
                     "--evidence-dir", str(ev), "--work-dir", str(self.root / "wd"), "--revisions", "3"],
                    http_client=StrictDriveMock(), stdout=out, now=FIXED_NOW, rand_hex="a1b2c3",
                    test_funcs={"D1": leaky_d1, "D2": leaky_d1, "D3": leaky_d1, "D4": leaky_d1, "D5": leaky_d1})
        self.assertEqual(rc, 1)
        files = list(ev.iterdir())
        self.assertEqual(len(files), 1)
        evidence_text = files[0].read_text()
        self.assertNotIn(ACCESS, evidence_text)
        self.assertNotIn(ACCESS, out.getvalue())
        doc = json.loads(evidence_text)
        self.assertGreaterEqual(doc["secret_scan"]["leaks"], 1)
        self.assertEqual(doc["overall"], "FAIL")


# ---------------------------------------------------------------------------
# End-to-end D1-D5 against the offline mock
# ---------------------------------------------------------------------------

class EndToEndMockTests(_Base):
    def _run(self, revisions=3, test_funcs=None, cap=None):
        token = _token_file(self.root)
        ev = _private_dir(self.root, "ev")
        mock_client = StrictDriveMock()
        out = io.StringIO()
        argv = ["--execute", "--approval-phrase", ds.APPROVAL_PHRASE, "--token-file", str(token),
                "--evidence-dir", str(ev), "--work-dir", str(self.root / "wd"),
                "--revisions", str(revisions)]
        patches = []
        if cap is not None:
            patches.append(mock.patch.object(ds, "_CALL_CAP", cap))
        for p in patches:
            p.start()
        try:
            rc = ds.run(argv, http_client=mock_client, stdout=out, now=FIXED_NOW, rand_hex="a1b2c3",
                        test_funcs=test_funcs)
        finally:
            for p in patches:
                p.stop()
        files = list(ev.iterdir())
        doc = json.loads(files[0].read_text()) if files else None
        return rc, doc, mock_client, out.getvalue(), ev

    def test_full_d1_to_d5_passes_against_mock_with_valid_evidence(self):
        rc, doc, mock_client, out, ev = self._run(revisions=3)
        self.assertEqual(rc, 0, out)
        self.assertEqual(doc["overall"], "PASS")
        self.assertEqual(doc["not_run"], [])
        self.assertEqual([t["status"] for t in doc["tests"]], ["PASS"] * 5)
        self.assertEqual(ds.validate_evidence(doc), [])
        self.assertLessEqual(doc["api"]["total"], ds.MAX_CALLS)
        self.assertFalse(doc["api"]["aborted"])
        self.assertEqual(doc["cleanup_plan"]["performed"], False)
        self.assertEqual(len(doc["revisions"]), 3)
        self.assertEqual(doc["secret_scan"]["leaks"], 0)
        self.assertEqual(stat.S_IMODE(os.stat(ev).st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(os.stat(next(ev.iterdir())).st_mode), 0o600)
        # evidence holds no secret values and no account address
        text = json.dumps(doc)
        for secret in (ACCESS, REFRESH, CLIENT_SECRET):
            self.assertNotIn(secret, text)
        # no deletion was ever sent to the remote mock
        self.assertNotIn("DELETE", [r["method"] for r in mock_client.recorded_requests])

    def test_stop_on_first_failure_issues_no_later_writes(self):
        writes_seen = {}

        def failing_d1(ctx, res):
            ds._check(res, "injected failure", False, "forced")

        def should_not_run(ctx, res):  # pragma: no cover - must never execute
            raise AssertionError("later test ran after D1 failure")

        rc, doc, mock_client, out, ev = self._run(
            test_funcs={"D1": failing_d1, "D2": should_not_run, "D3": should_not_run,
                        "D4": should_not_run, "D5": should_not_run})
        self.assertEqual(rc, 1)
        self.assertEqual(doc["tests"][0]["status"], "FAIL")
        self.assertEqual(doc["not_run"], ["D2", "D3", "D4", "D5"])
        # D1 publishes at most revision A before failing; nothing after it may publish.
        self.assertLessEqual(len(doc["revisions"]), 1)

    def test_failure_in_d2_stops_before_d3_and_d4(self):
        calls = []

        def ok(ctx, res):
            calls.append(res["id"])
            ds._check(res, "ok", True)

        def bad(ctx, res):
            calls.append(res["id"])
            ds._check(res, "bad", False)

        rc, doc, _m, _out, _ev = self._run(test_funcs={"D1": ok, "D2": bad, "D3": ok, "D4": ok, "D5": ok})
        self.assertEqual(rc, 1)
        self.assertEqual(calls, ["D1", "D2"])
        self.assertEqual(doc["not_run"], ["D3", "D4", "D5"])

    def test_call_cap_aborts_run_and_reports_it(self):
        rc, doc, mock_client, out, _ev = self._run(revisions=3, cap=10)
        self.assertEqual(rc, 1)
        self.assertTrue(doc["api"]["aborted"])
        self.assertIn("cap", doc["api"]["abort_reason"])
        self.assertLessEqual(doc["api"]["total"], 10)
        self.assertLessEqual(len(mock_client.recorded_requests), 10)
        self.assertEqual(doc["overall"], "FAIL")

    def test_evidence_schema_rejects_incomplete_document(self):
        problems = ds.validate_evidence({"schema": "wrong"})
        self.assertTrue(problems)
        self.assertTrue(any("missing keys" in p for p in problems))

    def test_plan_lists_operations_and_cleanup_without_claiming_deletion(self):
        plan = ds.build_plan(3, "colab-live-smoke-x")
        self.assertEqual(plan["api_call_cap"], ds.MAX_CALLS)
        self.assertIn("NOT EXECUTED", plan["cleanup_plan"])
        self.assertEqual([c["label"] for c in plan["checkpoints"]], ["A", "B", "C"])


if __name__ == "__main__":
    unittest.main()
