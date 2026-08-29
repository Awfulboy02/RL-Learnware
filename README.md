# Policy Learnware v0.5

This branch implements reward-free, candidate-independent retrieval from a
closed pool of policy learnwares. Every candidate is compared against the same
open-loop probe transitions. Source fitting reads neither target data nor
reward; target scoring calls no candidate policy and reads neither reward nor
the private target label. A complete ranking is persisted and sealed before
private truth is opened.

## Status

- Engineering and protocol: `GO`.
- Scientific result:
  `COMPLETE_DEVELOPMENT / COMPLETE_NO_GO_NO_PARETO_GAIN`.
- Historical development producer:
  `v05@3c06275ebe1bcff2498acfce004af1dcd97fb7b3`.

The experiment used a pool of 30 policies spanning six task families. The
target evidence was an independently collected repeat of a known environment,
not an unseen environment family. The results are descriptive development
evidence: they are not confirmatory, SOTA, formal certification, formal
differential privacy, or off-policy evaluation.

Raw-Delta RKME did not lead the frozen numerical Pareto comparison. Summary
LogReg achieved better retrieval AUC with a smaller and faster representation.
Raw-Delta RKME nevertheless matched the observable Top-1, truth-rank,
confusion, and aggregate metrics of the full Empirical-MMD control while using
a smaller source artifact and faster reuse-time scoring. This does not claim
identical raw scores or complete rankings.

## Frozen method panel

- `RAW_DELTA_RKME`
- `EMPIRICAL_MMD_NN`
- `SUMMARY_LOGREG`
- `KME_KRR`
- `RFF_KME_NN`
- `SWE_NN`

The primary Hit@1 results at one, two, and four short probe trajectories were:

| Method | 1 trajectory | 2 trajectories | 4 trajectories |
|---|---:|---:|---:|
| Raw-Delta RKME | .667 | .667 | .667 |
| Empirical MMD | .667 | .667 | .667 |
| Summary LogReg | .667 | .700 | .667 |
| KME-KRR | .667 | .733 | .733 |
| RFF-KME | .667 | .633 | .667 |
| SWE | .700 | .667 | .700 |

Each short trajectory exposes 64 sampled transitions to the selector. The
associated prefix-equivalent environment interaction cost is 1,000 steps per
trajectory. These were reused historical trajectories, so the run acquired
zero new transitions.

P1 methods, larger markets, full-budget curves, bootstrap confidence intervals,
probability calibration, association-destruction controls, and confirmatory
evaluation remain deferred.

## Scientific and privacy boundary

- Labels come from frozen v0.3 development championization in `OBSERVE` mode.
  Twenty-three of 30 selected champions passed that study's competence check;
  seven did not, and none were filtered after the fact.
- “Certificate” binds frozen label, bundle, championization, and execution-ABI
  provenance. It is not third-party or formal quality certification.
- Source fitting uses 19 training and six validation episodes per source.
  Seven held-repeat episodes are private preparation inputs and are not exposed
  with source identity to the scorer.
- The joint benchmark artifact includes full-support target views required by
  control methods. No process- or storage-level raw isolation is claimed.
  RFF and SWE provide only a method-level fixed-vector scoring interface.
- The endpoint is closed-set environment-to-policy retrieval. It does not
  estimate policy return, regret, or counterfactual value and is not OPE.

## External artifacts

Large trajectories, policy assets, model checkpoints, run/release JSON, and
figures are outside Git. Resolve the shared root in this order:

1. explicit `--artifacts-root`;
2. `RL_LEARNWARE_ARTIFACTS_ROOT`;
3. the `artifacts/` directory beside this repository.

The canonical layout is:

```text
artifacts/
├── v03/runs/v03-main-20260827-r0/
├── v04a/runs/v04a-primary-dev-20260828-r4/
└── v05/
    ├── runs/v05-environment-classification-dev-20260829-r0/
    ├── releases/v05-environment-classification-dev-20260829-r0/
    ├── analysis/
    └── evidence/final-manifest-chain/
```

The runner resolves the v0.3 and v0.4a inputs independently from this root; it
does not depend on their relative directory geometry or a symlink. The legacy
`root_relative_to_r4` config field remains historical provenance only.

Historical receipts, seals, manifests, and results remain byte-identical after
relocation. Their old-to-new paths and tree digests belong in the external
relocation manifest; a relocated or reconstructed run must not claim the
original producer provenance. The frozen run contains private truth-bearing
material and must retain restricted access.

Canonical narrative reports live outside Git under `reports/v05/`.

## Install and test

Python 3.11 or newer is required.

```bash
python -m pip install -e '.[test]'

PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  -q -p no:cacheprovider tests/v05

PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  -q -p no:cacheprovider \
  tests/unit/test_distance.py \
  tests/unit/test_empirical_kme.py \
  tests/unit/test_gaussian_kernel.py \
  tests/unit/test_reducer.py \
  tests/v03 tests/v04a
```

## Run a new development reproduction

Never overwrite the frozen development run. Use a new output directory and
single-threaded BLAS for the small dense linear algebra workload.

```bash
export RL_LEARNWARE_ARTIFACTS_ROOT=/path/to/artifacts

PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -B -m server.repro_fpo_ppo_v05.environment_classifier_runner \
  --config configs/v05_certified_policy_retrieval.yaml \
  --new-run-dir \
    "$RL_LEARNWARE_ARTIFACTS_ROOT/v05/runs/<new-run-id>"
```

If the process is interrupted, pass `--resume` only with exactly the same
inputs, code, config, and output directory. Source checkpoints, score cells,
the unique global ranking seal, and post-seal evaluation are all fail-closed
and no-clobber.

The immutable historical run is reproduced or resumed only at its recorded
producer commit and original runtime contract. The audited successor writes a
new run schema and new provenance; it does not silently adopt the historical
run.
