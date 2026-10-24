# M8 — Secure Cross-Device Training & Colab Compute Dashboard

> **Design only — not yet implemented or deployed (2026-10-24).** This is a local Ubuntu web-monitor service, separate from Colab's temporary GPU VM. No GPU allocation, OAuth, public access, or cloud resource changes are authorized by this plan.

## Product objective

From an authenticated browser on any authorized computer or phone, inspect live research jobs, training metrics, account-wide Colab compute-unit (CU) balance/rate, historical snapshots from the date of activation, current sessions, results and alerts. It must remain usable when no Colab VM is running and must never become an unprotected remote-execution interface.

**Core distinction:** the official `colab usage` command reports current **account-wide CU balance**, **aggregate CU/hour consumption rate** and **active assignment count**, not an itemized billing ledger. `colab sessions` reports active assignments. Earlier billing history cannot be reconstructed from today's balance; our own ledger begins when the collector starts. Running projects must emit their own progress/epoch/loss/accuracy/checkpoint events. GPU VRAM/utilization is optional instrumentation on a previously approved running session, not something the account usage command exposes.

## Dashboard UI / UX information architecture

1. **Overview:** CU balance, aggregate CU/h, account active assignments, managed running/queued jobs, data freshness, source health, summary of training runs and warnings. Account balance line graph (24h/7d/30d/custom) with real gaps; running-project cards with status, verified progress, latest metrics, elapsed time and errors.
2. **Training Runs:** searchable/filterable table by project, state, accelerator, start date; real states PLANNED/STARTING/SYNCING/RUNNING/EVALUATING/CHECKPOINTING/COMPLETED/FAILED/INTERRUPTED/CLEANUP_FAILED/UNKNOWN; status sourced from lifecycle events, not fabricated.
3. **Run Detail:** project/revision/session metadata; actual epoch/step and progress where total steps are known; loss, evaluation, learning rate and throughput charts when emitted; time-stamped GPU memory/utilization when instrumentation exists; approximate ETA only when estimated from observed steps; redacted live logs, event timeline, checkpoint hashes/download metadata, errors and cleanup result.
4. **Compute Units:** time series of CU balance, account-wide CU/h rate and assignments, observed net balance change since collection began, collection gaps, increases potentially caused by top-ups/refunds. Distinguish readings from billed spend and keep unattributed account use (including other Colab notebooks) separate from managed jobs.
5. **Sessions:** read-only display of account-wide assignments with managed/unmanaged flags. The monitor cannot stop, SSH into or otherwise control sessions.
6. **History / Artifacts:** completed and failed runs, filters, result metadata, verified checkpoints, authorized CSV/JSON export, no arbitrary filesystem browsing.
7. **Settings / Health:** last account query, stale data, collector error/backoff, data-retention controls, local service status, source version, and explicit monitoring start date. Domain/auth setup is operator-controlled, not public self-service.

**Visual direction:** modern minimalist dark/light responsive dashboard; small typography, accessible contrast, usable phone layout, tabular values, clear error and freshness indicators, keyboard navigation and log virtualization. Never render fake telemetry when source is absent.

## Architecture

~~~text
Authorized laptop/desktop/mobile browser
            |
   VPN OR HTTPS + authenticated access gateway
            |
 Ubuntu host: localhost-bound monitoring web service
       |                |
 React/Vite SPA     Read-only REST API + SSE
       |                |
       +--- private local event/snapshot store (JSONL + index) ---+
                 |               |                      |
        AccountCollector    JobEventIngestor       SessionInventory
        colab usage         manager/trainer        colab sessions
        READ ONLY           event JSONL            READ ONLY
                 |               |
            Account CU      Job progress/metrics/checkpoints
             snapshots         event timeline
~~~

- **Backend:** optional isolated Python >=3.12 web service (e.g. FastAPI/Uvicorn), independent of V1's Python 3.8 standard-library adapter; browser never communicates with Colab API or sees Google credentials.
- **Frontend:** small React/Vite SPA served as static assets; charting and responsive pages; use SSE with safe reconnection/cursors or bounded polling.
- **Storage:** append-only event and account-sample JSONL plus atomic indexed snapshots, rotation, bounded retention and corruption recovery. Private `0700` directory/`0600` files; no mandatory database server, sudo, GPU libraries or Docker.
- **Collector:** run read-only `colab --auth=oauth2 usage` and `colab --auth=oauth2 sessions` on bounded configurable intervals (initial target 60–120 sec with backoff). Poll cadence is independent of browser count and never calls `new`, `run`, SSH, Drive mount or other allocating commands.
- **Training events:** opt-in, time-stamped JSONL events from owned jobs, e.g. `train_step`, `eval_step`, `checkpoint_saved`, `heartbeat`, `job_state`, `cleanup_result`. Support tensorboard-compatible exporter in future, but do not assume every script emits logs that can be parsed reliably. Remote GPU sampling only within previously approved active runtimes, never trigger allocation merely to get GPU telemetry.
- **User-level hosting:** bind 127.0.0.1 by default; run via unprivileged service if available. Operator can set up a private mesh VPN or TLS+identity access gateway (MFA strongly preferred), which allows access from other computers over an encrypted and authenticated path. No direct public HTTP port, wildcard CORS or unauthenticated UI. Some tunnel/VPN setups need extra credentials or installation; obtain authorization separately.

## Data source truth table

| Source | Confirmed information | Limits |
| --- | --- | --- |
| Official CLI `usage` | account CU balance, aggregate CU/hour, active assignments | no full past transactions, no exact per-job charges |
| Official CLI `sessions` | current account assignments | no epoch/loss and not a completed-run history |
| V1 `ExperimentTracker` | runs, state, script, artifacts, cleanup, usage snapshot | cannot infer live loss without instrumentation |
| V2 project runner events | project/run status, epoch/step, model loss/accuracy, checkpoints | only emitted values available |
| Optional session GPU instrumentation | timestamped utilization/VRAM for approved running jobs | unavailable without instrumentation and current session |
| Verified user-provided official export | potentially earlier account records, if supported by source | never extrapolate from incomplete records |

**CU accounting:** track `balance_cu`, `rate_cu_per_hour`, `active_assignments`, `observed_at` and `collected_at`, `source` and `status`. For adjacent observations, label difference as **net balance change**, not verified consumption; positive may be a top-up/refund, negative may include account-wide billing/adjustments. External/unmanaged sessions prevent reliable per-job attribution. Billing may be delayed; show stale snapshots and gaps rather than interpolate false history. Do not display a fabricated lifetime CU total.

## Proposed schemas (illustrative; NOT actual user data)

~~~json
{"schema_version":1,"sample_id":"example-001","observed_at":"2026-10-24T06:00:00Z","source":"colab usage","balance_cu":200.0,"rate_cu_per_hour":0.0,"active_assignments":0,"status":"OK"}
~~~

~~~json
{"schema_version":1,"event_id":"example-event-1","run_id":"sample-run","project_id":"sample-project","emitted_at":"2026-10-24T06:00:02Z","event_type":"train_step","epoch":1,"step":10,"total_steps":100,"metrics":{"loss":1.23,"steps_per_second":2.4},"gpu":{"vram_used_bytes":null,"utilization_pct":null},"source":"trainer_callback"}
~~~

Stable ID + ingestion timestamp + source timestamp are required for deduplication and late events; missing metrics remain null/unavailable.

## Proposed READ-ONLY API (not yet implemented)

| Route | Payload |
| --- | --- |
| `GET /api/v1/health` | collector/backend health; stale source flags |
| `GET /api/v1/summary` | latest account snapshots and managed run totals |
| `GET /api/v1/compute/samples?from=&to=&limit=` | timestamped account usage history |
| `GET /api/v1/sessions` | sanitized owned/unmanaged account assignments |
| `GET /api/v1/runs?project=&status=&cursor=` | managed training records |
| `GET /api/v1/runs/{run_id}` | single managed run detail and provenance |
| `GET /api/v1/runs/{run_id}/metrics` | chart-ready emitted metrics |
| `GET /api/v1/runs/{run_id}/logs` | redacted, bounded, paginated log tail |
| `GET /api/v1/events/stream` | SSE event stream with resume cursor |
| `GET /api/v1/export/compute.csv` | authenticated source-labeled export |

All routes require authentication and authorization, including SSE and exports; restrict filters/pages, protect against cross-origin access, XSS via logs, path traversal, ID enumeration and disclosure of private project information.

**Explicitly out of scope:** `POST /run`, `POST /stop`, shell, Colab SSH, OAuth sign-in and self-approval from browser. Future controls need protected executor and separate review.

## Monitoring implementation slices

- **M8.1 Contracts & design:** APIs, event schema, fixture-based accessible responsive UI, source freshness semantics.
- **M8.2 Account ledger:** bounded read-only official CLI collector, safe parser, atomic durable timestamped history, top-up/delay/gap cases and error backoff.
- **M8.3 Training telemetry:** hook or adapter emitter, event ingestion/dedup, lifecycle projection, charts/checkpoints/logging with sanitization.
- **M8.4 Local server:** read-only authenticated REST, SSE/cursor, two-browser concurrency, bounded queries, retention and export.
- **M8.5 Remote access:** default localhost; independently approved VPN or authenticated HTTPS+MFA gateway, deployment guide, privacy audit. Verify from two separate clients using mock data before any live cloud probes.
- **M8.6 Integration:** Q19–Q31 in expected-results document, CI and release signoff; no GPU allocated by testing.

### Acceptance (offline-first)
- 0 GPU/CPU sessions allocated by dashboard collection or CI.
- No unauthenticated access to any run/compute/sessions endpoint, SSE stream or CSV export.
- 100% seeded secret strings redacted from HTTP/logs/exports.
- No double-counted events across restarts; unknown pre-monitor history labeled UNKNOWN.
- No false per-job CU billing attribution when another account session exists.
- Reliable 404/401/403/429/timeout handling with stale indicators and backoff.
- Training charts show only emitted metrics, with 0 invented VRAM or loss readings.
- Browser access from two authenticated devices verified in separate approved deployment test; localhost-only is the safe default.

## References
- https://github.com/googlecolab/google-colab-cli/blob/main/docs/04_automation_and_utility.md
- https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md
- https://github.com/googlecolab/google-colab-cli/blob/main/AGENTS.md
- https://research.google.com/colaboratory/faq.html
