# v0.2 source-anchor training backend

This sibling backend trains a PPO or FPO specialist on the exact native
environment frozen by a reviewed v0.2 source-anchor manifest. It does not
contain a task/axis/factor table and does not fill any `[REVIEW REQUIRED]`
scientific choice. The legacy `../repro_fpo_ppo/` runner and its artifacts are
read-only inputs; nominal v0/v0.1 provenance is never rewritten here.

The execution order is fixed:

```text
strict anchor/job/runtime digest validation
→ fresh registry environment
→ exact allowlisted MJX model mutation
→ full model-diff and poisoned-nominal audit
→ agent + rollout construction on that same environment object
→ train/evaluate/export with repeated environment digest checks
→ finite actor/obs/golden/metrics validation
→ atomic bundle and hash-bound training-record publication
```

## Safety contracts

- `AnchorManifest` freezes registry config, concrete environment class, runtime
  versions, upstream FPO commit, nominal/bound model digests, operator digest,
  the canonical live model-diff digest, leaf-level before/after digests, and
  exact flattened indices. The package and server use one environment-instance,
  source-anchor, and model-diff projection contract with fixed golden vectors.
- Shifted manifests whose live environment remains nominal are rejected. Model
  mutations outside the reviewed leaf/index allowlist are rejected.
- Protocols embed the complete native trainer config, explicit algorithm,
  budget, evaluation contract, and checkpoint rule. There are no trainer,
  seed, task, factor, or budget defaults. Computed dataclass properties are not
  injected into that exact config: the package/server bridge derives
  `iterations_per_env = num_minibatches*batch_size*unroll_length/num_envs`
  only after proving positive inputs and exact divisibility.
- Actor parameters, observation statistics, rollout transitions, optimizer
  metrics, evaluations, golden observations/actions, and every exported NPZ
  array must be finite. Null/NaN/Inf evidence fails the attempt.
- A queue runs at most one subprocess per listed physical GPU. An attempt is
  published by atomic directory rename, never resumed in place, and retries
  start from outer zero in a new `attempt_NNN` directory.
- Resume skips only an attempt whose queue-result digest, runner status,
  training-record digests, exact export set, bundle manifest, file hashes, and
  numerical bundle contents all validate again.

## Before any real training

Human review must first freeze every RFC `[REVIEW REQUIRED]` literal and create
immutable anchor and protocol JSON files through the package-side v0.2 config
workflow. This backend only derives digests; it must not be used to invent the
six-task axis registry, factors, primary algorithm, seeds, budget, evaluation
episodes, competence thresholds, or formal checkpoint.

Materialize each already-reviewed nominal/shifted specification against the
pinned live registry before planning jobs:

```bash
python -m repro_fpo_ppo_v02.generate_anchor_manifest \
  --spec /absolute/reviewed/anchor-spec.json \
  --fpo-root /absolute/clean/fpo-checkout \
  --output /absolute/immutable/source-anchor-manifest.json
```

The FPO checkout may provide `flow_policy` while the GoRL environment provides
the `mujoco_playground` module. In that layout the materializer parses the
tracked `playground/pyproject.toml` and requires its exact `playground==X` pin
to equal the installed distribution version before importing the registry.

For shifted anchors the reviewed specification must contain the exact model
leaf/flat-index allowlist and a package-side axis-binding digest. The
materializer opens a fresh native environment, mutates only those elements,
checks mass/inertia row coupling where applicable, and independently rebinds
the final manifest. Nominal manifests contain none of those shift fields.

Generate a plan only from those reviewed files (repeat `--anchor-manifest` for
each source anchor):

```bash
python repro_fpo_ppo_v02/generate_manifest.py \
  --anchor-manifest /absolute/reviewed/anchor-a.json \
  --anchor-manifest /absolute/reviewed/anchor-b.json \
  --protocol /absolute/reviewed/training-protocol.json \
  --seeds '<reviewed,sorted,seeds>' \
  --config-digest '<sha256-of-the-exact-v02-config>' \
  --execution-purpose '<audit_smoke|development_discovery|v02_freeze_ready>' \
  --output /absolute/immutable/v02-training-plan.json
```

For `v02_freeze_ready`, also pass `--formal-config` with the absolute canonical
YAML path. The generator and queue independently reload its canonical freeze
manifest and require the exact reviewed 30-anchor × 3-seed grid, task/axis/
operator/factor/binding/leaf semantics, algorithm, native transition budget,
and checkpoint rule. A caller-supplied `config_digest` is never formal
authority. Non-formal plans reject `--formal-config`. Formal freeze also
requires a clean live HEAD and a source-only closure: remove local
`__pycache__`, bytecode/native import artifacts, and symlinks first, then run
with `PYTHONDONTWRITEBYTECODE=1`.

The output path must not already exist. Any semantic change requires a new
reviewed input and therefore a new digest/plan, rather than overwriting an old
plan or run root.

## Dependency-light tests

These tests use the standard library plus NumPy and do not import MuJoCo/JAX or
start GPU training:

```bash
cd /share/songyf/RL_Learnware
python3 -m pytest -q repro_fpo_ppo_v02/tests
```

They cover exact source mutation, source immutability, the required poisoned
shifted/actual-nominal rejection, strict JSON/digest failure, full NPZ finite
checks, deterministic plan generation, atomic attempts, one-job-per-GPU queue
scheduling, and validated resume.

## Launch and monitor

All operational inputs are explicit. `launch.sh` never regenerates a plan and
never supplies scientific values:

```bash
repro_fpo_ppo_v02/launch.sh \
  --plan /absolute/immutable/v02-training-plan.json \
  --execution-purpose '<same-explicit-purpose-frozen-in-plan>' \
  --runs-root /absolute/new/v02-runs \
  --gpus '<physical,gpu,ids>' \
  --max-attempts '<reviewed-operational-limit>' \
  --session '<unique-tmux-name>' \
  --fpo-root /absolute/clean/fpo-checkout \
  --python /home/songyf/miniforge3/envs/GoRL/bin/python \
  --vendor-dir /share/songyf/RL_Learnware/repro_fpo_ppo/_vendor \
  --legacy-policy-io /share/songyf/RL_Learnware/repro_fpo_ppo/policy_io.py
```

`--vendor-dir` is required because the server's GoRL environment does not
provide `wandb`, while the pinned upstream trainer imports it even when online
logging is disabled. The v0.2 queue reuses the read-only dependency tree from
the legacy runner; it never installs or modifies packages. Before allocating a
job, both queue and runner validate the vendored `wandb` distribution and hash
every non-cache regular file in the tree. Symlinks, special files, a missing or
ambiguous `wandb` distribution, and RECORD mismatches fail closed.

The queue prepends the resolved vendor directory to the runner's `PYTHONPATH`,
sets `WANDB_MODE=disabled` and `PYTHONDONTWRITEBYTECODE=1`, and preserves any
inherited paths only after the pinned directory. Absolute vendor path, content
digest, pinned `wandb` version, file count, and byte count are recorded in the
run manifest, queue status, and immutable queue result. Resume revalidates that
the current vendor tree is byte-identical to the successful attempt; changing
it requires a new audited attempt and cannot silently reuse old evidence.

`--legacy-policy-io` is also required; the runner loads that exact file rather
than discovering an exporter by module name or working directory. At attempt
allocation the queue hashes a fixed implementation inventory: the actual
runner, queue master, package bridge, anchor binding, provenance/vendor code,
consumer-side bundle/parity loaders, v0.2 training contract, and the explicit
legacy exporter. The implementation inventory and scientific config/plan are
co-bound by the immutable attempt digest and are repeated in the run manifest,
training record, queue status, and queue result. Formal bridge admission
re-hashes the named runner/exporter plus all fixed modules; any source-byte
drift fails admission even when every caller-provided JSON digest was rehashed.

The runner rejects a non-GPU JAX backend by default. `--allow-non-gpu` exists
only for synthetic/debugging runs and permanently records
`execution_mode=audit_smoke` and `formal_eligible=false`; observing a GPU later
cannot upgrade that attempt. Formal admission requires a single physical CUDA
device, `execution_purpose=v02_freeze_ready`, and a digest chain spanning the
configuration, job/plan, attempt, run, every checkpoint, bundle, training
record, and queue result under the canonical attempt root. A development run
remains non-formal even when it uses a GPU.

Read-only monitoring:

```bash
repro_fpo_ppo_v02/monitor.sh \
  --runs-root /absolute/new/v02-runs \
  --session '<unique-tmux-name>'
```

Use `--follow` for the master log or `--json` for the full queue status. Send
Ctrl-C to the tmux queue master for a graceful stop; it terminates active
process groups, records interrupted attempts, and leaves queued jobs resumable.

## Artifact layout

```text
<runs-root>/
  queue_master.lock
  queue_status.json
  master.log
  jobs/<opaque-job-id>/
    job_manifest.json
    job_state.json
    attempt_001/
      attempt_manifest.json
      stdout.log / stderr.log
      run_manifest.json
      status.json / events.jsonl
      checkpoints/outer_NNNNNN/<legacy-policy-bundle>
      training_record.json
      queue_result.json
```

`job_manifest.json`, attempt manifests, successful training records,
queue-results, checkpoints, and bundles are immutable evidence. Status files
are replaceable progress views. A leftover `.attempt_*.pending-*` is treated as
a torn publication requiring audit, never silently deleted.
