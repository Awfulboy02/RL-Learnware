# Policy Learnware v0.4a — Fixed-Probe Bayesian Reuse

This branch contains the smallest executable v0.4a comparison built on the
v0.31 Raw-Delta RKME operator. It compares reward-free, fixed-probe policy
retrieval within a same-task five-policy candidate set:

- `RAW_DELTA_TASK5`: native Raw-Delta RKME nearest-neighbour retrieval;
- `BPR_FP`: Gaussian-summary BPR fixed-probe adapter;
- `EBPR_FP`: conditional `(s, a, delta-s)` EBPR fixed-probe adapter;
- `EBPR_FP_BPR_U`: low-cost hybrid ablation, not a primary method.

The source BPR/EBPR models are fitted once from source-only evidence. Target
rankings are sealed before the development oracle can be opened. The protocol
keeps `visible_transition_count` and `interaction_cost_steps` as separate
ledgers and uses one nested reward-free probe membership across methods.

## Scientific status

The canonical R4 run is `COMPLETE_DEVELOPMENT`, `formal=false`, and explicitly
not confirmatory. It contains 30 source contexts, 24 frozen development
contexts, seven budgets (`1,2,4,8,16,24,32`), 672 rankings, and 672 post-seal
metric records. It is evidence for method development, not a formal claim that
one method is universally best.

At budget 32, the descriptive Hit@1 values were BPR `0.625`, hybrid `0.500`,
Raw-Delta `0.375`, and EBPR `0.208`. R0–R2 parity/asset attempts remained
recorded NO-GO evidence; R4 completion does not erase them. P4 controls,
full-transition sensitivity, sequential BPR/EBPR, and confirmatory evaluation
remain deferred.

Frozen R4 identity:

- run tree: 439 files, 30,612,855 bytes;
- tree SHA-256: `f8f040ba34a19b69d6eb0d8dc58b7b39e6c197657158e81a918e98d953adff5d`;
- rankings digest: `69d2a4f28ababe5cea3482693b5fbe975c9ea1db72e19c9315aa2c4be630cdf0`;
- ranking-seal file SHA-256: `43fedd846bdb48ba957a7b5e9e6baf6df54082ac600e8e419c13de2b7bf3fa0b`;
- summary file SHA-256: `e2ea65300274b5e13a7b15e11a938a871d035e0196bf485d4c1833c0909f34ef`.

## External artifacts and reports

No experiment payload belongs in Git. Every active branch uses the one external
root `RL_LEARNWARE_ARTIFACTS_ROOT`, resolved in this order:

1. explicit CLI or configuration path;
2. `RL_LEARNWARE_ARTIFACTS_ROOT`;
3. the safe default `<repository-parent>/artifacts`.

The canonical roots are `/Users/jamesmac/Desktop/RL Learnware/artifacts` locally
and `/share/songyf/RL_Learnware/artifacts` on the server. Relative CLI artifact
paths below are resolved against that root. The relevant layout is:

```text
artifacts/
├── relocation_manifest.json
├── v02/exact90/v02-reacher-formal-2r-20260825-r2/
├── v02/formal_inputs/v02-reacher-formal-2r-20260825-r2/
├── v03/runs/
│   ├── v03-main-20260827-r0/source-market/
│   ├── v03-signal-ranking-20260827-r1/{probes,baseline}/
│   └── v031-raw-transition-controls-20260828-r1/
├── shared/runtime/fpo-418c2554/
└── v04a/
    ├── runs/v04a-primary-dev-20260828-r4/
    ├── logs/v04a-primary-dev-20260828-r4/
    └── diagnostics/m2cpu-r1/
```

Shared v02/v03/FPO inputs are referenced, not copied into `v04a/`. Historical
receipts retain their original paths and bytes. The root relocation manifest
binds old and new locations without a symlink compatibility layer.
Experiment plans and reports live in the sibling `reports/` tree; they are not
tracked in this branch. In particular, the unchanged v0.4a coding plan belongs
at `../reports/v04a/Policy_Learnware_v0.4a_Coding_Plan.md`.
The two retained `docs/` files are code-interface notes for the v02 adapter and
v03 anonymous market; they are not experiment reports or coding plans.

## Install and test

Python 3.11 or newer is required.

```bash
python -m venv .venv
.venv/bin/pip install -e '.[research,test]'
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/v04a
```

The downstream v0.5 compatibility check is run from an audited v0.5 worktree:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/v04a tests/v05
```

## Fresh development replay

Run from the repository root. Never use the sealed R4 directory as the output
of a fresh replay.

```bash
export RL_LEARNWARE_ARTIFACTS_ROOT=/absolute/path/to/workspace/artifacts
PY=.venv/bin/python
RUN=v04a/runs/v04a-development-<new-id>

PYTHONPATH=src:. "$PY" -m server.repro_fpo_ppo_v04a.bpr_runner prepare \
  --config configs/v04a_bayesian_reuse.yaml \
  --artifacts-root "$RL_LEARNWARE_ARTIFACTS_ROOT" \
  --run-dir "$RUN" \
  --context-index v03/runs/v03-signal-ranking-20260827-r1/probes/context_index.json \
  --public-policy-market v03/runs/v03-main-20260827-r0/source-market/public_policy_market.json \
  --deployment-private-registry v03/runs/v03-main-20260827-r0/source-market/deployment_private_registry.json \
  --origin-pool-acceptance v02/exact90/v02-reacher-formal-2r-20260825-r2/policy_pool_handoff_a7d10c0/policy_pool_acceptance.json \
  --raw-delta-root v03/runs/v031-raw-transition-controls-20260828-r1 \
  --fpo-root shared/runtime/fpo-418c2554 \
  --source-utility-root v03/runs/v03-signal-ranking-20260827-r1/baseline/oracle

PYTHONPATH=src:. "$PY" -m server.repro_fpo_ppo_v04a.bpr_runner fit-source \
  --artifacts-root "$RL_LEARNWARE_ARTIFACTS_ROOT" --run-dir "$RUN"

PYTHONPATH=src:. "$PY" -m server.repro_fpo_ppo_v04a.bpr_runner smoke-fp \
  --artifacts-root "$RL_LEARNWARE_ARTIFACTS_ROOT" --run-dir "$RUN" \
  --context-id <one-development-context-id> --attempt-id smoke-1

PYTHONPATH=src:. "$PY" -m server.repro_fpo_ppo_v04a.bpr_runner score-fp \
  --artifacts-root "$RL_LEARNWARE_ARTIFACTS_ROOT" --run-dir "$RUN"

PYTHONPATH=src:. "$PY" -m server.repro_fpo_ppo_v04a.bpr_runner seal-rankings \
  --artifacts-root "$RL_LEARNWARE_ARTIFACTS_ROOT" --run-dir "$RUN"

PYTHONPATH=src:. "$PY" -m server.repro_fpo_ppo_v04a.bpr_runner oracle-evaluate \
  --artifacts-root "$RL_LEARNWARE_ARTIFACTS_ROOT" --run-dir "$RUN" \
  --oracle-root v03/runs/v03-signal-ranking-20260827-r1/baseline/oracle

PYTHONPATH=src:. "$PY" -m server.repro_fpo_ppo_v04a.bpr_runner summarize \
  --artifacts-root "$RL_LEARNWARE_ARTIFACTS_ROOT" --run-dir "$RUN"
```

## Verify an R4 relocation before source retirement

The relocation verifier is a pre-retirement operation. It requires three
distinct source directories: the complete R4 run, a directory containing
exactly the seven R4 log files, and a directory containing exactly the three
m2cpu diagnostic files. After copying, it requires exact source/target equality,
validates the source-only and seal-before-oracle status, and can write an
owner-scoped manifest for a future relocation whose source still exists:

```bash
PYTHONPATH=src:. "$PY" -m server.repro_fpo_ppo_v04a.bpr_runner relocation-manifest \
  --artifacts-root "$RL_LEARNWARE_ARTIFACTS_ROOT" \
  --source-run /historical/read-only/v04a-primary-dev-20260828-r4 \
  --source-log-root /historical/read-only/r4-logs-only \
  --source-diagnostic-root /historical/read-only/m2cpu-diagnostics-only
```

The command never copies, edits, or re-labels the R4 evidence. It must not be
run against a reconstructed copy after the historical source has been retired.
For the current server freeze, the old `v04a_runs` source has already been
retired. Authority is therefore the central
`$RL_LEARNWARE_ARTIFACTS_ROOT/relocation_manifest.json` (SHA-256
`81e726c297c78ebc110df017e06e6fb56de73face39371198635299f931bfed9`)
plus `_audit/verification_receipts/server-v04a-source-retirement-precheck.json`
and `server-v04a-source-retirement-complete.json`. No owner manifest is
retroactively fabricated. A reconstructed runtime must be disclosed as
reconstructed and cannot replace original provenance.
