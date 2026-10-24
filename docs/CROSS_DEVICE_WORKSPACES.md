# Cross-Device Workspaces & Session Attachment (V2)

## Motivation

Researchers frequently need to monitor remote Colab training runs from multiple devices (e.g. MacBook, Windows workstation, mobile browser) without:
1. Re-authenticating Google Colab CLI on every device.
2. Sharing sensitive OAuth tokens or private SSH keys.
3. Accidentally provisioning expensive cloud GPU virtual machines merely by viewing status.
4. Losing run visibility once the temporary Colab VM terminates.

The Ubuntu-hosted **Workspace Broker** provides authenticated, multi-client read-only attachment to Colab runs while preserving strict isolation boundaries.

## Architecture

```text
  [MacBook Laptop]       [Windows PC]       [Phone / Tablet]
         \                    |                    /
          \                   |                   /
     Authenticated HTTPS / Tailscale / WireGuard / SSH Tunnel
                              |
                     [Ubuntu Host Server]
                              |
               +------------------------------+
               |       Workspace Broker       |
               |  (Token Auth & Audit Log)    |
               +------------------------------+
                     /                 \
                    /                   \
        [Managed Session Registry]   [Durable Artifact Store]
                   |                             |
      [Active Colab GPU Session]    [Local SSD + Google Drive]
        (Approved Runtime)             (Historical Runs)
```

## Core Guarantees

### 1. Strictly Read-Only in Initial Release
All cross-device inspection actions (`list_runs`, `attach_run`, `get_artifacts`, `get_logs`) are strictly read-only.
Any attempt to invoke writable execution via the broker or monitoring dashboard raises a `NotImplementedError` or returns `HTTP 405 Method Not Allowed`.

### 2. Zero Unintended Cloud VM Allocations
Viewing an existing run or attaching to an expired session **never** triggers `colab up` or auto-provisions a cloud VM.
If an inspected session's lease has expired or the remote VM was terminated:
- The broker reports `runtime_status: "RUNTIME_UNAVAILABLE"` and `is_runtime_active: false`.
- All historical logs, metrics, code snapshots, and Google Drive checkpoints remain fully viewable and downloadable.

### 3. Rejection of Unmanaged Sessions
The broker strictly refuses requests targeting unmanaged sessions (sessions created outside the adapter instance):
- Raises `UnmanagedSessionError` with zero outbound commands dispatched.
- Prevents rogue or unmanaged instances from interfering with researcher accounts.

### 4. Multi-Client Concurrency & Audit History
Multiple independent devices can attach to the same running or historical experiment simultaneously:
- Each client connection is verified via constant-time bearer token validation.
- All attachment events are recorded in an in-memory and persistent audit log with `client_id`, `run_id`, timestamp, and access mode.

## API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/v1/runs` | `GET` | List active and historical runs with checkpoint status and Drive links |
| `/api/v1/runs/<run_id>` | `GET` | Detailed telemetry, event count, and runtime status for a run |
| `/api/v1/runs/<run_id>/artifacts` | `GET` | List verified local and Drive-published artifacts with SHA256 digests |
| `/api/v1/runs/<run_id>/drive` | `GET` | Summary of Google Drive publication state, remote file IDs, and errors |
| `/api/v1/runs/<run_id>/manifest` | `GET` | Full reproducibility manifest (git commit, environment, file hashes) |
| `/api/v1/runs/<run_id>/metrics` | `GET` | Time-series loss and accuracy telemetry points |
| `/api/v1/runs/<run_id>/logs` | `GET` | Filtered standard output and training logs |
| `/api/v1/events/stream` | `GET` | Server-Sent Events (SSE) stream supporting cursor resume (`?from_id=...`) |

All API endpoints strictly require authentication via `Authorization: Bearer <token>`, `X-API-Token: <token>`, or a trusted reverse-proxy identity header (`X-Remote-User`). Query parameter authentication (`?token=...`) is strictly rejected to prevent secret leakage in URLs, browser history, reverse-proxy logs, and referrer headers.

## Remote Access & Deployment Security

1. **Loopback Default Binding**: The monitoring server binds to `127.0.0.1` by default.
2. **Real TLS Socket Encryption & Verified Trusted Proxy**: Binding to an external network interface (`allow_external=True`) strictly requires either:
   - Real TLS socket encryption (`require_tls=True`) with valid `tls_certfile` and `tls_keyfile` wrapped via `ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)` (`TLSv1.2`+), or
   - A verified reverse-proxy identity gateway (`trusted_proxy=True`) configured with a non-empty `trusted_proxy_secret`, peer IP validation against `trusted_proxy_ips` (default `127.0.0.1`, `::1`), `X-Remote-User` format validation (`^[a-zA-Z0-9_.@-]{1,128}$`), and constant-time verification of `X-Proxy-Secret` or HMAC-SHA256 `X-Proxy-Signature`.
   Raw unencrypted HTTP on external interfaces or `trusted_proxy=True` without a shared secret is rejected with `ValueError` / `BrokerAuthenticationError`.
3. **Strict CORS Policy**: Wildcard CORS (`Access-Control-Allow-Origin: *`) is completely eliminated. The server restricts access to same-origin requests or explicitly configured allowed origins, sending `Vary: Origin`.
4. **Persistent Credentials**: Broker tokens can be loaded from and persisted to a protected `tokens_file` with `0600` permissions (parent directory `0700`), ensuring authentication configuration survives service restarts.
