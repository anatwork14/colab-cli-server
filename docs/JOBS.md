# Running jobs, staging inputs and retrieving results

## One-shot GPU smoke test

Verify installation/auth first. Review what will run **without allocating**:

```bash
python3 colab_server.py plan --gpu T4 --script examples/gpu_probe.py
```

Then, as a human at your own terminal, approve a real runtime:

```bash
python3 colab_server.py run --gpu T4 --script examples/gpu_probe.py
```

Type exactly `ALLOCATE T4` when prompted. The wrapper invokes `colab --auth=oauth2 run --gpu T4 --timeout 120 <script>`. Upstream `run` executes a **local .py file remotely**, and attempts immediate VM teardown. It normally propagates Python failures as nonzero exit status. The 120s timeout is a **kernel inactivity timeout, not a charge cap or guaranteed maximum duration**.

## Stateful multi-file training with checkpoints

For large inputs, install requirements and stage files in an explicitly named session:

```bash
# Only after human approval; an allocation may consume compute units
colab --auth=oauth2 new -s demo-trainer --gpu T4

# The local requirements.txt is sent to the remote runtime
colab --auth=oauth2 install -s demo-trainer -r requirements.txt

# Upload any needed input file (do not upload secrets)
colab --auth=oauth2 upload -s demo-trainer ./dataset.csv /content/dataset.csv

# Execute your LOCAL script on the remote Python kernel
colab --auth=oauth2 exec -s demo-trainer --timeout 300 -f train.py

# Confirm remote outputs and download before stopping
colab --auth=oauth2 ls -s demo-trainer /content/output
colab --auth=oauth2 download -s demo-trainer \
  /content/output/model.pt ./model.pt

colab --auth=oauth2 stop -s demo-trainer
colab --auth=oauth2 sessions
```

This example presumes your `train.py` writes `/content/output/model.pt`; **the CLI does not invent that file**. The official `exec` command reads the local script source and sends it to the existing remote kernel, but sibling modules, datasets and requirements are **not automatically copied**.

**Failure handling:** these standalone commands are not automatically chained with a cleanup handler. Even if install/exec/download fails, stop the VM by name. Inspect `colab sessions` after a network interruption. Use `colab log -s demo-trainer` for debugging, and capture required logs/artifacts before terminating.

**Do not trust `colab exec` exit codes alone.** Current upstream `exec` can print a kernel traceback while exiting the shell with success. Verify output files, training metrics and successful completion explicitly, or write a machine-readable `/content/output/status.json` inside your workload.

## Files, notebooks and optional Google Drive

- `colab upload -s NAME LOCAL REMOTE`: send explicit input assets.
- `colab download -s NAME REMOTE LOCAL`: retrieve assets. Validate integrity for critical checkpoints.
- `colab exec -s NAME -f notebook.ipynb`: execute notebook cells; upstream writes a local output notebook.
- `colab drivemount -s NAME`: optional interactive Google Drive mount, requiring human consent. Do not invoke it as an unattended agent step.
- For large artifacts, consider a user-approved storage location accessible to the runtime; never expose public buckets or credential material as a shortcut.

## Scheduling and cost

Check `colab --auth=oauth2 usage` before and after permitted experiments to understand account balance/rate. Stop idle sessions promptly. Do not assume a paid tier grants guaranteed accelerators or unlimited runtime. Do not build background keep-alive mechanisms, automatic unbounded retry, or guaranteed-SLA inference services on top of Colab.
