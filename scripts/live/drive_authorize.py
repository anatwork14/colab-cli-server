#!/usr/bin/env python3
"""HUMAN-RUN helper: authorize the Google Drive REST client with the narrowest scope.

Implements the installed-app (Desktop client) OAuth 2.0 flow with loopback redirect,
PKCE (S256) and a state check. Requested scope: ``drive.file`` only (files created by
this app). The resulting refresh/access tokens are saved ONLY through
``DriveCredentials.save()`` (atomic, mode 0600).

Safety properties:
- Refuses to run unless stdin and stdout are both TTYs (a human is present).
- Client id/secret come from environment variables or a 0600 client-secret JSON file
  path. They are never accepted as command-line values and are never printed.
- The token file path must be outside this repository and must not already exist.
- Authorization codes and tokens are never printed or logged.
- The token response must grant exactly the drive.file scope family; anything broader
  is refused and nothing is saved.

Usage (in your own terminal, not from an agent or CI):

    export COLAB_DRIVE_OAUTH_CLIENT_ID=...          # or use --client-secret-file
    export COLAB_DRIVE_OAUTH_CLIENT_SECRET=...
    python3 scripts/live/drive_authorize.py --token-file ~/.config/colab-drive/token.json

See docs/live/DRIVE_SETUP.md for the full owner procedure.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import http.server
import json
import os
import secrets
import socket
import stat
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab_adapter.storage.google_drive import (  # noqa: E402
    DriveCredentials,
    StandardDriveHttpClient,
)

SCOPE = "https://www.googleapis.com/auth/drive.file"
DRIVE_SCOPE_PREFIX = "https://www.googleapis.com/auth/drive"
AUTH_URI = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URI = "https://oauth2.googleapis.com/token"
ENV_CLIENT_ID = "COLAB_DRIVE_OAUTH_CLIENT_ID"
ENV_CLIENT_SECRET = "COLAB_DRIVE_OAUTH_CLIENT_SECRET"
CALLBACK_PATH = "/callback"
DEFAULT_TIMEOUT_SEC = 300.0


class AuthorizeError(Exception):
    """Refusal or failure with a message that never contains secrets."""


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested offline)
# ---------------------------------------------------------------------------

def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def pkce_pair(verifier: Optional[str] = None) -> Tuple[str, str]:
    """Return (code_verifier, code_challenge) using the S256 method (RFC 7636)."""
    if verifier is None:
        verifier = _b64url(secrets.token_bytes(64))  # 86 chars, within 43..128
    if not (43 <= len(verifier) <= 128) or not all(
        c.isalnum() or c in "-._~" for c in verifier
    ):
        raise AuthorizeError("PKCE code verifier has an invalid length or alphabet")
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def new_state() -> str:
    return _b64url(secrets.token_bytes(32))


def build_auth_url(client_id: str, redirect_uri: str, state: str, code_challenge: str) -> str:
    if not client_id or not redirect_uri or not state or not code_challenge:
        raise AuthorizeError("client_id, redirect_uri, state and code_challenge are required")
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": SCOPE,
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "false",
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    return f"{AUTH_URI}?{urllib.parse.urlencode(params)}"


def loopback_redirect_uri(port: int) -> str:
    return f"http://127.0.0.1:{port}{CALLBACK_PATH}"


def parse_callback(query: str, expected_state: str) -> str:
    """Return the authorization code from a redirect query string, enforcing state.

    Only the error *code* from the provider is surfaced, sanitized to a short token.
    """
    params = urllib.parse.parse_qs(query, keep_blank_values=True)
    if "error" in params:
        code = "".join(ch for ch in params["error"][0] if ch.isalnum() or ch == "_")[:40]
        raise AuthorizeError(f"authorization was not granted (provider reason: {code or 'unknown'})")
    received_state = params.get("state", [""])[0]
    if not received_state or not secrets.compare_digest(received_state, expected_state):
        raise AuthorizeError("state mismatch in redirect; refusing the response")
    code_values = params.get("code", [])
    if len(code_values) != 1 or not code_values[0]:
        raise AuthorizeError("redirect did not carry exactly one authorization code")
    return code_values[0]


def check_granted_scope(scope_field: Optional[str]) -> None:
    """Require drive.file and refuse any broader Drive scope in the token response."""
    if not scope_field:
        raise AuthorizeError("token response did not state granted scopes; refusing to save")
    granted = scope_field.split()
    if SCOPE not in granted:
        raise AuthorizeError("token response does not grant drive.file; refusing to save")
    for item in granted:
        if item.startswith(DRIVE_SCOPE_PREFIX) and item != SCOPE:
            raise AuthorizeError("token response grants a broader Drive scope; refusing to save")


def extract_client_config(data: Mapping[str, Any]) -> Tuple[str, str]:
    """Accept a Google 'installed' (Desktop) client JSON; reject web/service clients."""
    if "web" in data or ("type" in data and data.get("type") != "installed"):
        raise AuthorizeError("client secret JSON is not a Desktop ('installed') OAuth client")
    inner = data.get("installed", data)
    if not isinstance(inner, Mapping):
        raise AuthorizeError("client secret JSON has an unexpected structure")
    client_id = inner.get("client_id")
    client_secret = inner.get("client_secret")
    if not isinstance(client_id, str) or not client_id or not isinstance(client_secret, str) or not client_secret:
        raise AuthorizeError("client secret JSON lacks client_id or client_secret")
    return client_id, client_secret


def check_secret_file(path: Path) -> None:
    """Client-secret JSON must be a regular, non-symlink file with mode 0600 (or stricter)."""
    try:
        st = os.lstat(path)
    except OSError:
        raise AuthorizeError("client secret file is not readable") from None
    if not stat.S_ISREG(st.st_mode):
        raise AuthorizeError("client secret path must be a regular file, not a symlink or special file")
    if st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise AuthorizeError("client secret file must have mode 0600 (no group/other access)")


def load_client_credentials(
    client_secret_file: Optional[str],
    environ: Mapping[str, str],
) -> Tuple[str, str]:
    if client_secret_file:
        path = Path(client_secret_file).expanduser()
        check_secret_file(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise AuthorizeError("client secret file could not be parsed as JSON") from None
        if not isinstance(data, dict):
            raise AuthorizeError("client secret JSON has an unexpected structure")
        return extract_client_config(data)
    client_id = environ.get(ENV_CLIENT_ID, "")
    client_secret = environ.get(ENV_CLIENT_SECRET, "")
    if not client_id or not client_secret:
        raise AuthorizeError(
            f"provide --client-secret-file or both {ENV_CLIENT_ID} and {ENV_CLIENT_SECRET}"
        )
    return client_id, client_secret


def is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def check_token_path(token_file: str, repo_root: Path = REPO_ROOT) -> Path:
    """Token output must be outside the repository and must not already exist."""
    target = Path(token_file).expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    # Resolve the parent (it may not exist yet); the file itself must not be a symlink.
    resolved_parent = target.parent.resolve()
    final = resolved_parent / target.name
    if is_inside(final, repo_root.resolve()):
        raise AuthorizeError("token file must be stored outside the repository")
    if os.path.lexists(final):
        raise AuthorizeError("token file already exists; move or delete it yourself, then rerun")
    return final


def ensure_interactive(stdin: Any, stdout: Any) -> None:
    """Refuse non-interactive use: a human must be present at the terminal."""
    try:
        ok = bool(stdin.isatty()) and bool(stdout.isatty())
    except Exception:
        ok = False
    if not ok:
        raise AuthorizeError("refusing to run without an interactive terminal (stdin and stdout must be TTYs)")


# ---------------------------------------------------------------------------
# Loopback receiver (real network-facing piece; injected in tests)
# ---------------------------------------------------------------------------

class LoopbackReceiver:
    """One-shot HTTP listener bound to 127.0.0.1 for the redirect."""

    def __init__(self, port: int = 0) -> None:
        self._server = http.server.HTTPServer(("127.0.0.1", port), self._make_handler())
        self._server.timeout = 1.0
        self._query: Optional[str] = None
        self.port = self._server.server_address[1]
        self.redirect_uri = loopback_redirect_uri(self.port)

    def _make_handler(self) -> type:
        receiver = self

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - stdlib naming
                parsed = urllib.parse.urlsplit(self.path)
                if parsed.path != CALLBACK_PATH:
                    self.send_response(404)
                    self.end_headers()
                    return
                receiver._query = parsed.query
                body = b"<html><body>Authorization received. You can close this tab.</body></html>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt: str, *args: Any) -> None:  # silence request logging
                return

        return _Handler

    def wait_for_code(self, expected_state: str, timeout: float = DEFAULT_TIMEOUT_SEC) -> str:
        deadline = time.monotonic() + timeout
        try:
            while self._query is None:
                if time.monotonic() >= deadline:
                    raise AuthorizeError("timed out waiting for the browser redirect")
                self._server.handle_request()
        finally:
            self._server.server_close()
        return parse_callback(self._query or "", expected_state)

    def close(self) -> None:
        try:
            self._server.server_close()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Token exchange
# ---------------------------------------------------------------------------

def exchange_code(
    http_client: Any,
    client_id: str,
    client_secret: str,
    code: str,
    redirect_uri: str,
    code_verifier: str,
) -> Dict[str, Any]:
    payload = urllib.parse.urlencode({
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "code_verifier": code_verifier,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri,
    }).encode("utf-8")
    status, _headers, body = http_client.request(
        "POST",
        TOKEN_URI,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data=payload,
    )
    if status != 200:
        reason = ""
        try:
            reason = str(json.loads(body.decode("utf-8")).get("error", ""))
        except Exception:
            reason = ""
        reason = "".join(ch for ch in reason if ch.isalnum() or ch == "_")[:40]
        raise AuthorizeError(f"token exchange failed (HTTP {status}, reason: {reason or 'unknown'})")
    try:
        data = json.loads(body.decode("utf-8"))
    except ValueError:
        raise AuthorizeError("token response was not valid JSON") from None
    if not isinstance(data, dict) or not data.get("access_token") or not data.get("refresh_token"):
        raise AuthorizeError(
            "token response lacked a refresh token; revoke the app's previous grant in your Google "
            "account (Security > Third-party access) and rerun"
        )
    check_granted_scope(data.get("scope"))
    return data


def build_credentials(client_id: str, client_secret: str, token_data: Mapping[str, Any]) -> DriveCredentials:
    expires_in = float(token_data.get("expires_in", 3600))
    return DriveCredentials(
        client_id=client_id,
        client_secret=client_secret,
        refresh_token=str(token_data["refresh_token"]),
        access_token=str(token_data["access_token"]),
        token_expiry=time.time() + expires_in,
        token_uri=TOKEN_URI,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_authorization(
    client_id: str,
    client_secret: str,
    token_path: Path,
    *,
    http_client: Any,
    open_browser: Callable[[str], Any],
    receiver: Any,
    emit: Callable[[str], None],
) -> None:
    """Run the consent flow and save the token. ``receiver`` provides redirect_uri and wait_for_code."""
    verifier, challenge = pkce_pair()
    state = new_state()
    auth_url = build_auth_url(client_id, receiver.redirect_uri, state, challenge)
    emit("Open this URL in a browser on this machine (or tunnel the loopback port, see DRIVE_SETUP.md):")
    emit(auth_url)
    open_browser(auth_url)
    code = receiver.wait_for_code(state)
    token_data = exchange_code(http_client, client_id, client_secret, code, receiver.redirect_uri, verifier)
    creds = build_credentials(client_id, client_secret, token_data)
    creds.save(token_path)
    mode = stat.S_IMODE(os.stat(token_path).st_mode)
    emit(f"Drive token file saved (scope: {SCOPE}). File mode: {mode:04o}.")


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    environ: Optional[Mapping[str, str]] = None,
    stdin: Any = None,
    stdout: Any = None,
    http_client: Any = None,
    open_browser: Optional[Callable[[str], Any]] = None,
    receiver: Any = None,
    repo_root: Path = REPO_ROOT,
) -> int:
    parser = argparse.ArgumentParser(description="Authorize Drive drive.file scope (human-run).")
    parser.add_argument("--token-file", required=True, help="Output path for the 0600 token file (outside the repo).")
    parser.add_argument("--client-secret-file", help="Path to a 0600 Desktop client JSON. Alternative: env vars.")
    parser.add_argument("--port", type=int, default=0, help="Loopback port (0 = ephemeral). Fix it to tunnel over SSH.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SEC, help="Seconds to wait for consent.")
    args = parser.parse_args(argv)

    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    environ = os.environ if environ is None else environ

    def emit(line: str) -> None:
        stdout.write(line + "\n")
        stdout.flush()

    try:
        ensure_interactive(stdin, stdout)
        token_path = check_token_path(args.token_file, repo_root)
        client_id, client_secret = load_client_credentials(args.client_secret_file, environ)
        if not (0 <= args.port <= 65535):
            raise AuthorizeError("port out of range")
        owns_receiver = receiver is None
        if receiver is None:
            receiver = LoopbackReceiver(args.port)
        if open_browser is None:
            import webbrowser

            open_browser = webbrowser.open
        if http_client is None:
            http_client = StandardDriveHttpClient()
        try:
            run_authorization(
                client_id,
                client_secret,
                token_path,
                http_client=http_client,
                open_browser=open_browser,
                receiver=receiver,
                emit=emit,
            )
        finally:
            if owns_receiver:
                receiver.close()
    except AuthorizeError as exc:
        emit(f"Refused or failed: {exc}")
        return 2
    except Exception as exc:  # never echo exception text: it may carry request context
        emit(f"Failed: {type(exc).__name__} (details suppressed to protect credentials)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
