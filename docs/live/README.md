# Live Test Documentation Index

This directory holds the documents for the live (cloud-backed) validation of colab-cli-server. Offline tests are described elsewhere in the repository.

## Documents

| Document | Purpose |
|---|---|
| [LIVE_TEST_EVIDENCE.md](LIVE_TEST_EVIDENCE.md) | Evidence record: header, rules, per-test results table, pre-flight facts, final verdict, honesty rules. |
| [ENVIRONMENT_PREFLIGHT.md](ENVIRONMENT_PREFLIGHT.md) | Environment checks to run before any live test. |
| [DRIVE_SETUP.md](DRIVE_SETUP.md) | Setup for the Google Drive side of the live tests. |
| [GPU_TEST_PLAN.md](GPU_TEST_PLAN.md) | Plan for the Colab GPU tests (G1 to G6). |
| [CROSS_DEVICE_TEST_PLAN.md](CROSS_DEVICE_TEST_PLAN.md) | Plan for the cross-device test (X1), which needs a second device. |

Some linked documents may not exist yet while sibling work is in progress.

## Reading order

1. ENVIRONMENT_PREFLIGHT.md
2. DRIVE_SETUP.md
3. GPU_TEST_PLAN.md and CROSS_DEVICE_TEST_PLAN.md
4. LIVE_TEST_EVIDENCE.md (record results here)

## Ground rules

- No secrets, tokens, or raw OAuth responses in any document or artifact.
- Account identity is redacted.
- A test is PASS only when its expected assertion is checked against observed output.
- Missing prerequisites mean BLOCKED, not PASS.
