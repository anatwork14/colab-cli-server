#!/usr/bin/env python3
"""Cross-device validation for the read-only Colab monitor (loopback only).

Runs the checks described in docs/live/CROSS_DEVICE_TEST_PLAN.md against a
monitor reachable at --base-url. Safety properties of this tool:

- The owner token is read from an environment variable (default
  COLAB_MONITOR_TOKEN, the same variable ``colab_adapter.cli monitor`` reads).
  It is never accepted on argv and is never printed, logged or written to the
  evidence file.
- Only loopback base URLs are accepted unless --allow-non-loopback is given.
- It only issues GET requests plus read-only probes (POST/PUT/DELETE/PATCH and
  OPTIONS) that the monitor must refuse. It never contacts Colab or Google.

Modes:
  default                 run checks a-g against --base-url
  --assert-no-new-sessions  compare two --sessions-before/--sessions-after
                          snapshot files (no network access)

Exit codes: 0 all checks PASS, 1 at least one check FAIL, 2 usage or
configuration error. Stdlib only, Python 3.8+.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import secrets
import ssl
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

DEFAULT_TOKEN_ENV = "COLAB_MONITOR_TOKEN"
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
MIN_TOKEN_LEN = 16
MAX_BODY_BYTES = 8 * 1024 * 1024
MAX_LOG_BYTES = 64 * 1024 * 1024
MUTATING_METHODS = ("POST", "PUT", "DELETE", "PATCH")
REFUSED_STATUSES = {404, 405}
SCHEMA_VERSION = 1

# Secret-looking shapes. Each pattern captures the candidate value in group "v"
# so placeholders such as "[REDACTED]" or "****" are not reported as leaks.
SECRET_PATTERNS: List[Tuple[str, Any]] = [
    ("bearer_header", re.compile(r"(?i:bearer)\s+(?P<v>[A-Za-z0-9._~+/=-]{8,})")),
    ("access_token_field", re.compile(r"(?i)access_token[\"']?\s*[:=]\s*[\"']?(?P<v>[A-Za-z0-9._~+/=-]{8,})")),
    ("refresh_token_field", re.compile(r"(?i)refresh_token[\"']?\s*[:=]\s*[\"']?(?P<v>[A-Za-z0-9._~+/=-]{8,})")),
    ("client_secret_field", re.compile(r"(?i)client_secret[\"']?\s*[:=]\s*[\"']?(?P<v>[A-Za-z0-9._~+/=-]{8,})")),
    ("google_access_token_shape", re.compile(r"(?P<v>ya29\.[A-Za-z0-9._-]{10,})")),
    ("google_refresh_token_shape", re.compile(r"(?P<v>1//[0-9A-Za-z_-]{20,})")),
]
_PLACEHOLDER = re.compile(r"^(\*+|<.*>|\[?redacted\]?|x+|none|null)$", re.IGNORECASE)


class ConfigError(Exception):
    """Usage or configuration problem; never includes secret values."""


class TransportError(Exception):
    """Network failure; carries only the exception class name."""


def pattern_hits(text: str) -> List[str]:
    """Return the names of secret-looking patterns found in text (no values)."""
    hits: List[str] = []
    for name, pat in SECRET_PATTERNS:
        for match in pat.finditer(text):
            if not _PLACEHOLDER.match(match.group("v")):
                hits.append(name)
                break
    return hits


class Response:
    def __init__(self, status: int, headers: List[Tuple[str, str]], body: bytes) -> None:
        self.status = status
        self.headers = headers
        self.body = body

    def header(self, name: str) -> Optional[str]:
        lname = name.lower()
        for key, value in self.headers:
            if key.lower() == lname:
                return value
        return None

    def json(self) -> Any:
        try:
            return json.loads(self.body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None


class Target:
    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        parts = urllib.parse.urlsplit(base_url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ConfigError("--base-url must be an http:// or https:// URL with a host")
        self.scheme = parts.scheme
        self.host = parts.hostname
        self.port = parts.port or (443 if parts.scheme == "https" else 80)
        self.timeout = timeout

    def describe(self) -> Dict[str, Any]:
        return {"scheme": self.scheme, "host": self.host, "port": self.port}

    def request(self, method: str, path: str, headers: Optional[Dict[str, str]] = None) -> Response:
        if self.scheme == "https":
            conn: http.client.HTTPConnection = http.client.HTTPSConnection(
                self.host, self.port, timeout=self.timeout, context=ssl.create_default_context()
            )
        else:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        try:
            conn.request(method, path, headers=headers or {})
            resp = conn.getresponse()
            body = resp.read(MAX_BODY_BYTES)
            return Response(resp.status, list(resp.getheaders()), body)
        except (OSError, http.client.HTTPException) as exc:
            raise TransportError(type(exc).__name__)
        finally:
            conn.close()


class Checker:
    def __init__(
        self,
        target: Target,
        token: str,
        expect_project: str,
        expect_run: str,
        expect_latest_sha256: Optional[str],
        expect_revisions: Optional[int],
        log_file: Optional[Path],
    ) -> None:
        self.target = target
        self.token = token
        self.expect_project = expect_project
        self.expect_run = expect_run
        self.expect_latest_sha256 = expect_latest_sha256
        self.expect_revisions = expect_revisions
        self.log_file = log_file
        self.checks: List[Dict[str, Any]] = []
        # (label, text, is_api) for every response body and header block seen.
        self.corpus: List[Tuple[str, str, bool]] = []
        self.runs: Optional[List[Any]] = None
        self.run_detail: Optional[Dict[str, Any]] = None
        self.run_metrics: Optional[List[Any]] = None

    # ---- transport helpers -------------------------------------------------

    def _auth(self) -> Dict[str, str]:
        return {"Authorization": "Bearer " + self.token}

    def _send(self, label: str, method: str, path: str, headers: Optional[Dict[str, str]] = None, api: bool = True) -> Response:
        resp = self.target.request(method, path, headers=headers)
        header_text = "\n".join(f"{k}: {v}" for k, v in resp.headers)
        self.corpus.append((label + " [headers]", header_text, False))
        self.corpus.append((label + " [body]", resp.body.decode("utf-8", errors="replace"), api))
        return resp

    def _get(self, label: str, path: str, auth: bool = True) -> Response:
        return self._send(label, "GET", path, self._auth() if auth else None)

    def _record(self, check_id: str, title: str, passed: bool, details: Dict[str, Any], reason: str = "") -> None:
        entry: Dict[str, Any] = {
            "id": check_id,
            "title": title,
            "status": "PASS" if passed else "FAIL",
            "details": details,
        }
        if not passed:
            entry["reason"] = reason or "expectation not met"
        self.checks.append(entry)

    # ---- checks ------------------------------------------------------------

    def check_a_unauthenticated(self) -> None:
        statuses: Dict[str, int] = {}
        for path in ("/api/v1/runs", "/api/v1/summary"):
            statuses[f"GET {path} (no auth)"] = self._send("a1 " + path, "GET", path).status
        ok = all(v == 401 for v in statuses.values())
        self._record("a1_unauthenticated_401", "Unauthenticated API GET returns 401", ok, statuses,
                     "unauthenticated request was not rejected with 401")

        query_status = self._send("a2 query", "GET", "/api/v1/runs?token=" + urllib.parse.quote(self.token), None).status
        self._record("a2_query_token_rejected", "?token= query auth is rejected",
                     query_status == 401, {"GET /api/v1/runs?token=<redacted>": query_status},
                     "query-string token was accepted")

        wrong = secrets.token_urlsafe(MIN_TOKEN_LEN)
        wrong_status = self._send("a3 wrong", "GET", "/api/v1/runs",
                                  {"Authorization": "Bearer " + wrong}).status
        self._record("a3_wrong_bearer_401", "Wrong bearer token returns 401",
                     wrong_status == 401, {"GET /api/v1/runs (wrong bearer)": wrong_status},
                     "wrong bearer token was accepted")

    def check_b_owner_reads(self) -> None:
        results: Dict[str, Any] = {}
        ok = True

        def expect(name: str, resp: Response, shape_ok: bool) -> None:
            nonlocal ok
            results[name] = resp.status
            if resp.status != 200 or not shape_ok:
                ok = False

        health = self._get("b health", "/api/v1/health")
        hb = health.json()
        expect("GET /api/v1/health", health, isinstance(hb, dict) and "status" in hb)

        summary = self._get("b summary", "/api/v1/summary")
        sb = summary.json()
        expect("GET /api/v1/summary", summary,
               isinstance(sb, dict) and {"total_runs", "managed_running_runs", "managed_counts_scope"} <= set(sb))

        runs = self._get("b runs", "/api/v1/runs")
        rb = runs.json()
        expect("GET /api/v1/runs", runs, isinstance(rb, list))
        self.runs = rb if isinstance(rb, list) else []

        sessions = self._get("b sessions", "/api/v1/sessions")
        expect("GET /api/v1/sessions", sessions, isinstance(sessions.json(), list))

        pq = "?project=" + urllib.parse.quote(self.expect_project)
        base = f"/api/v1/runs/{urllib.parse.quote(self.expect_run)}"
        detail = self._get("b detail", base + pq)
        db = detail.json()
        detail_ok = isinstance(db, dict) and db.get("run_id") == self.expect_run \
            and db.get("project_id") == self.expect_project and isinstance(db.get("artifacts"), list)
        expect(f"GET {base}", detail, detail_ok)
        self.run_detail = db if detail_ok else None

        arts = self._get("b artifacts", base + "/artifacts" + pq)
        ab = arts.json()
        expect(f"GET {base}/artifacts", arts,
               isinstance(ab, dict) and isinstance(ab.get("artifacts"), list))

        metrics = self._get("b metrics", base + "/metrics" + pq)
        mb = metrics.json()
        metrics_ok = isinstance(mb, list) and len(mb) >= 1 and all(
            isinstance(p, dict) and isinstance(p.get("metrics"), dict) for p in mb)
        expect(f"GET {base}/metrics", metrics, metrics_ok)
        self.run_metrics = mb if metrics_ok else None

        self._record("b_owner_reads_200", "Owner GETs return 200 with expected schema", ok, results,
                     "one or more owner GET endpoints returned a non-200 or unexpected schema")

    def check_c_identity_and_revisions(self) -> None:
        runs = self.runs or []
        shape_bad = [i for i, r in enumerate(runs)
                     if not (isinstance(r, dict) and isinstance(r.get("project_id"), str) and r.get("project_id")
                             and isinstance(r.get("run_id"), str) and r.get("run_id"))]
        pair_entries = [r for r in runs if isinstance(r, dict)
                        and r.get("project_id") == self.expect_project and r.get("run_id") == self.expect_run]
        same_id_projects = sorted({r.get("project_id") for r in runs
                                   if isinstance(r, dict) and r.get("run_id") == self.expect_run})
        ok_identity = not shape_bad and len(pair_entries) == 1
        self._record("c1_run_key_project_and_run", "Run list entries carry project_id and run_id",
                     ok_identity,
                     {"entries": len(runs), "entries_missing_key_fields": len(shape_bad),
                      "expected_pair_matches": len(pair_entries),
                      "projects_sharing_expected_run_id": len(same_id_projects)},
                     "run list entries lack project_id/run_id or the expected pair is absent or duplicated")

        arts = (self.run_detail or {}).get("artifacts") or []
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for a in arts:
            if isinstance(a, dict):
                groups.setdefault(str(a.get("logical_path") or ""), []).append(a)
        group_report: Dict[str, Any] = {}
        revisions_ok = bool(groups)
        for path_key, items in groups.items():
            heads = [a for a in items if a.get("is_latest") is True]
            known = all(a.get("latest_state") == "known" for a in items)
            bool_ok = all(a.get("is_latest") in (True, False) for a in items)
            group_ok = len(heads) == 1 and known and bool_ok
            revisions_ok = revisions_ok and group_ok
            group_report[path_key or "<unnamed>"] = {
                "revisions": len(items), "head_count": len(heads), "all_latest_state_known": known,
            }
        head_sha = None
        if groups:
            heads_all = [a for items in groups.values() for a in items if a.get("is_latest") is True]
            head_sha = heads_all[0].get("sha256") if len(heads_all) == 1 else None
        if self.expect_latest_sha256 is not None:
            revisions_ok = revisions_ok and head_sha == self.expect_latest_sha256
        checkpoints = [a for a in arts if isinstance(a, dict)
                       and (a.get("artifact_type") == "checkpoint" or a.get("category") == "checkpoint")]
        if self.expect_revisions is not None:
            revisions_ok = revisions_ok and len(checkpoints) == self.expect_revisions
        self._record("c2_revision_latest_state", "Only the manifest head is is_latest; history is not",
                     revisions_ok,
                     {"groups": group_report, "checkpoint_revisions": len(checkpoints),
                      "head_matches_expected": (head_sha == self.expect_latest_sha256)
                      if self.expect_latest_sha256 is not None else None},
                     "latest/historical revision state is not the expected head or is unknown")

        run_list_ok = True
        run_list_detail: Dict[str, Any] = {}
        if self.expect_latest_sha256 is not None:
            if len(pair_entries) == 1:
                entry = pair_entries[0]
                run_list_detail = {"latest_checkpoint_state": entry.get("latest_checkpoint_state"),
                                   "latest_checkpoint_sha_matches": entry.get("latest_checkpoint_sha256") == self.expect_latest_sha256}
                run_list_ok = run_list_detail["latest_checkpoint_state"] == "known" \
                    and run_list_detail["latest_checkpoint_sha_matches"]
            else:
                run_list_ok = False
        self._record("c3_run_list_latest_matches_head", "Run list latest checkpoint matches manifest head",
                     run_list_ok, run_list_detail,
                     "run list latest checkpoint is unknown or does not match the expected head")

    def check_d_read_only(self) -> None:
        before = self._get("d runs before", "/api/v1/runs")
        statuses: Dict[str, int] = {}
        for method in MUTATING_METHODS:
            for path in ("/api/v1/runs", "/api/v1/summary"):
                statuses[f"{method} {path}"] = self._send(f"d {method} {path}", method, path,
                                                          self._auth(), api=True).status
            statuses[f"{method} /api/v1/runs (no auth)"] = self._send(
                f"d {method} anon", method, "/api/v1/runs", None).status
        ok = all(s in REFUSED_STATUSES for s in statuses.values())
        self._record("d1_mutating_methods_refused", "POST/PUT/DELETE/PATCH are refused (405 or 404)",
                     ok, statuses, "a mutating method returned a status other than 405 or 404")

        options = self._send("d options", "OPTIONS", "/api/v1/runs", None, api=False)
        advertised = " ".join(v for k, v in options.headers
                              if k.lower() in ("allow", "access-control-allow-methods"))
        mutating_adv = [m for m in MUTATING_METHODS if re.search(r"\b" + m + r"\b", advertised)]
        self._record("d2_no_mutating_methods_advertised", "No mutating method is advertised",
                     not mutating_adv, {"advertised_methods_checked": True, "mutating_advertised": mutating_adv},
                     "OPTIONS advertises a mutating method")

        after = self._get("d runs after", "/api/v1/runs")
        same = hashlib.sha256(before.body).hexdigest() == hashlib.sha256(after.body).hexdigest()
        self._record("d3_state_unchanged_by_probes", "Run list unchanged after refused writes",
                     same and before.status == 200 and after.status == 200,
                     {"before_status": before.status, "after_status": after.status, "identical": same},
                     "run list changed after refused write probes")

    def check_f_csp(self) -> None:
        resp = self._send("f root", "GET", "/", None, api=False)
        csp = resp.header("Content-Security-Policy") or ""
        nonce_match = re.search(r"'nonce-([A-Za-z0-9_\-]+)'", csp)
        nonce = nonce_match.group(1) if nonce_match else ""
        html = resp.body.decode("utf-8", errors="replace")
        script_tags = re.findall(r"<script\b[^>]*>", html, flags=re.IGNORECASE)
        style_tags = re.findall(r"<style\b[^>]*>", html, flags=re.IGNORECASE)
        unnonced = [t for t in script_tags + style_tags
                    if "src=" not in t.lower() and f'nonce="{nonce}"' not in t]
        inline_handlers = re.findall(r"<[a-zA-Z][^>]*?\s(on[a-zA-Z]+)\s*=", html)
        js_urls = re.findall(r"(?i)\b(?:href|src|action)\s*=\s*[\"']?\s*javascript:", html)
        details = {
            "status": resp.status,
            "csp_present": bool(csp),
            "nonce_length": len(nonce),
            "script_style_tags": len(script_tags) + len(style_tags),
            "tags_missing_nonce": len(unnonced),
            "inline_handler_attributes": len(inline_handlers),
            "javascript_urls": len(js_urls),
            "unsafe_inline_present": "unsafe-inline" in csp,
        }
        ok = (resp.status == 200 and bool(csp) and "script-src 'self'" in csp and len(nonce) >= 16
              and "'unsafe-inline'" not in csp and "'unsafe-eval'" not in csp
              and not unnonced and not inline_handlers and not js_urls)
        self._record("f_csp_nonce_no_inline", "CSP with per-response nonce; no inline handlers",
                     ok, details, "CSP header/nonce missing, unsafe-inline present, untagged script/style or inline handlers found")

    def check_e_secret_hygiene(self) -> None:
        exact_hits: List[str] = []
        pattern_hits_found: List[str] = []
        for label, text, is_api in self.corpus:
            if self.token in text:
                exact_hits.append(label)
            if is_api:
                for name in pattern_hits(text):
                    pattern_hits_found.append(f"{label}:{name}")
        # Response bodies from the root page are not API payloads; they are still
        # checked for the exact token above.
        scanned = len(self.corpus)
        self._record("e1_no_secret_in_responses", "No token or secret-shaped value in responses",
                     not exact_hits and not pattern_hits_found,
                     {"segments_scanned": scanned, "exact_token_hits": len(exact_hits),
                      "pattern_hits": sorted(set(pattern_hits_found))},
                     "the exact token or a secret-shaped value appeared in a response")

        if self.log_file is None:
            self._record("e2_no_secret_in_log", "No token or secret-shaped value in monitor log",
                         True, {"log_file_checked": False}, "")
            return
        try:
            raw = self.log_file.read_bytes()[:MAX_LOG_BYTES]
        except OSError:
            self._record("e2_no_secret_in_log", "No token or secret-shaped value in monitor log",
                         False, {"log_file_checked": False}, "log file could not be read")
            return
        text = raw.decode("utf-8", errors="replace")
        log_exact = self.token in text
        log_patterns = sorted(set(pattern_hits(text)))
        self._record("e2_no_secret_in_log", "No token or secret-shaped value in monitor log",
                     not log_exact and not log_patterns,
                     {"log_file_checked": True, "bytes_scanned": len(raw),
                      "exact_token_found": log_exact, "pattern_hits": log_patterns},
                     "the exact token or a secret-shaped value appears in the log file")

    def run(self) -> None:
        self.check_a_unauthenticated()
        self.check_b_owner_reads()
        self.check_c_identity_and_revisions()
        self.check_d_read_only()
        self.check_f_csp()
        self.check_e_secret_hygiene()


def load_session_ids(path: Path) -> Set[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"cannot read sessions snapshot {path.name}: {type(exc).__name__}")
    items: Any = data.get("sessions") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ConfigError(f"sessions snapshot {path.name} must be a list or contain a 'sessions' list")
    ids: Set[str] = set()
    for item in items:
        if isinstance(item, str) and item:
            ids.add(item)
        elif isinstance(item, dict) and isinstance(item.get("session_id"), str) and item.get("session_id"):
            ids.add(item["session_id"])
        else:
            raise ConfigError(f"sessions snapshot {path.name} has an entry without a session_id")
    return ids


def assert_no_new_sessions(before_path: Path, after_path: Path) -> Dict[str, Any]:
    before = load_session_ids(before_path)
    after = load_session_ids(after_path)
    new_ids = after - before
    grew = bool(new_ids) or len(after) > len(before)
    check = {
        "id": "s1_managed_session_set_not_grown",
        "title": "Managed session set did not grow between snapshots",
        "status": "FAIL" if grew else "PASS",
        "details": {"before_count": len(before), "after_count": len(after),
                    "new_session_count": len(new_ids), "removed_session_count": len(before - after)},
    }
    if grew:
        check["reason"] = "managed session set grew: a new managed session appeared"
    return {"mode": "assert-no-new-sessions", "checks": [check]}


def save_sessions_snapshot(checker: Checker, out_path: Path) -> None:
    resp = checker._get("snapshot sessions", "/api/v1/sessions")
    body = resp.json()
    if resp.status != 200 or not isinstance(body, list):
        raise ConfigError(f"sessions snapshot request failed with status {resp.status}")
    out_path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_evidence(mode: str, target: Dict[str, Any], token_env: str, checks: List[Dict[str, Any]],
                   token: Optional[str]) -> Dict[str, Any]:
    evidence: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool": "scripts/live/monitor_check.py",
        "mode": mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target": target,
        "token_source": {"env_var": token_env, "value": "not recorded"},
        "checks": checks,
    }
    if token is not None:
        serialized = json.dumps(evidence, sort_keys=True)
        if token in serialized or pattern_hits(serialized):
            evidence["checks"].append({
                "id": "g2_evidence_sanitized", "title": "Evidence file contains no secret-shaped value",
                "status": "FAIL", "details": {}, "reason": "evidence failed the self-scan",
            })
    evidence["overall"] = "PASS" if all(c["status"] == "PASS" for c in evidence["checks"]) else "FAIL"
    return evidence


SECRET_ARG = re.compile(r"^--(?:auth[-_]?token|token|access[-_]?token|secret|password|client[-_]?secret)(?:=|$)",
                        re.IGNORECASE)


def refuse_secrets_on_argv(argv: List[str]) -> None:
    """Refuse token-like flags or secret values on argv without echoing them."""
    secret_values = [v for k, v in os.environ.items()
                     if v and re.search(r"TOKEN|SECRET", k, re.IGNORECASE) and len(v) >= MIN_TOKEN_LEN]
    for arg in argv:
        if SECRET_ARG.match(arg):
            raise ConfigError("secrets must not be passed on argv; set the token environment variable "
                              f"(default {DEFAULT_TOKEN_ENV}) from a 0600 file instead")
        if any(value in arg for value in secret_values):
            raise ConfigError("a secret value was found on argv; remove it and use the environment variable")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Validate the read-only monitor for cross-device use. The token is read from an "
                    "environment variable, never from argv.")
    p.add_argument("--base-url", help="Monitor base URL, e.g. http://127.0.0.1:8080")
    p.add_argument("--token-env", default=DEFAULT_TOKEN_ENV,
                   help=f"Environment variable holding the owner token (default: {DEFAULT_TOKEN_ENV}).")
    p.add_argument("--allow-non-loopback", action="store_true",
                   help="Permit a non-loopback host (only for an operator-approved TLS/identity proxy).")
    p.add_argument("--expect-project", help="Project id that must own the expected run.")
    p.add_argument("--expect-run", help="Run id expected under --expect-project.")
    p.add_argument("--expect-latest-sha256", help="Full 64-hex sha256 that must be the manifest head.")
    p.add_argument("--expect-revisions", type=int, help="Expected number of checkpoint revisions.")
    p.add_argument("--log-file", type=Path, help="Monitor/server log to scan for leaked secrets.")
    p.add_argument("--evidence-file", type=Path, help="Write the sanitized JSON evidence here.")
    p.add_argument("--save-sessions-snapshot", type=Path,
                   help="Save GET /api/v1/sessions (managed sessions) to this JSON file.")
    p.add_argument("--assert-no-new-sessions", action="store_true",
                   help="Compare --sessions-before and --sessions-after instead of probing a monitor.")
    p.add_argument("--sessions-before", type=Path)
    p.add_argument("--sessions-after", type=Path)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        refuse_secrets_on_argv(raw_argv)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    try:
        if args.assert_no_new_sessions:
            if not args.sessions_before or not args.sessions_after:
                raise ConfigError("--assert-no-new-sessions needs --sessions-before and --sessions-after")
            result = assert_no_new_sessions(args.sessions_before, args.sessions_after)
            evidence = build_evidence(result["mode"], {}, "none", result["checks"], None)
            return _finish(evidence, args.evidence_file)

        if not args.base_url or not args.expect_project or not args.expect_run:
            raise ConfigError("--base-url, --expect-project and --expect-run are required")
        token = os.environ.get(args.token_env, "")
        if not token:
            raise ConfigError(f"environment variable {args.token_env} is not set (token is never read from argv)")
        if len(token) < MIN_TOKEN_LEN:
            raise ConfigError(f"token in {args.token_env} is shorter than {MIN_TOKEN_LEN} characters")
        target = Target(args.base_url)
        if target.host not in LOOPBACK_HOSTS and not args.allow_non_loopback:
            raise ConfigError("refusing a non-loopback host; use an SSH tunnel to 127.0.0.1 "
                              "(or --allow-non-loopback behind an approved TLS/identity proxy)")
        if args.expect_latest_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", args.expect_latest_sha256):
            raise ConfigError("--expect-latest-sha256 must be 64 lowercase hex characters")
        checker = Checker(target, token, args.expect_project, args.expect_run,
                          args.expect_latest_sha256, args.expect_revisions, args.log_file)
        checker.run()
        if args.save_sessions_snapshot:
            save_sessions_snapshot(checker, args.save_sessions_snapshot)
        evidence = build_evidence("probe", target.describe(), args.token_env, checker.checks, token)
        return _finish(evidence, args.evidence_file)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except TransportError as exc:
        print(f"error: monitor request failed ({exc}); is the monitor running on that address?", file=sys.stderr)
        return 2


def _finish(evidence: Dict[str, Any], evidence_file: Optional[Path]) -> int:
    for check in evidence["checks"]:
        line = f"{check['status']}  {check['id']}"
        if check["status"] == "FAIL" and check.get("reason"):
            line += f"  - {check['reason']}"
        print(line)
    print(f"overall: {evidence['overall']}")
    if evidence_file is not None:
        evidence_file.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"evidence written: {evidence_file}")
    return 0 if evidence["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
