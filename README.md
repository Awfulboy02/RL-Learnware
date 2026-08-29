# Policy Learnware v0.5 Ablation

This branch is the audited superset of the six-method v0.5 selector. It retains
the reward-free, candidate-independent development runner and adds source/query
few-shot and one-factor-at-a-time compute analyses. The additions are secondary
post-truth exploration, not a second production protocol.

## Status

- Core engineering and protocol: `GO`.
- Core scientific result:
  `COMPLETE_DEVELOPMENT / COMPLETE_NO_GO_NO_PARETO_GAIN`.
- Ablation scope: `SECONDARY_EXPLORATORY_POST_TRUTH`.
- Historical core producer:
  `v05@3c06275ebe1bcff2498acfce004af1dcd97fb7b3`.
- Historical ablation producer:
  `v05-ablation@4e395487ff567ee64d40c70d53d420451d571582`.

The pool contains 30 policies over six task families. Targets are independent
repeats of known environments, not unseen environment families. Neither the
core nor the ablation is confirmatory, SOTA, formal certification, formal DP,
or OPE.

Raw-Delta RKME did not obtain a numerical Pareto lead. Summary LogReg achieved
better retrieval AUC with a smaller and faster representation. Raw-Delta RKME
matched the observable Top-1, truth-rank, confusion, and aggregate metrics of
the full Empirical-MMD control while compressing its source representation and
accelerating reuse scoring; it did not match all raw scores or rankings.

## Methods

The frozen core panel is:

- `RAW_DELTA_RKME`
- `EMPIRICAL_MMD_NN`
- `SUMMARY_LOGREG`
- `KME_KRR`
- `RFF_KME_NN`
- `SWE_NN`

The exploratory additions are:

- `B0_RANDOM`: deterministic chance control.
- `B3A_RAW_MOMENT_NN`: direct mean/std/second-moment control.
- `SUMMARY_NN`: nearest summary prototype.
- `RFF_LOGREG` and `RFF_RIDGE`: supervised heads on fixed RFF features.
- `SWE_1024_NN`: dimension-reduced SWE control.

`B3A_RAW_MOMENT_NN` is an external high-exposure baseline. Direct moments can
leak much more information about uploaded data than a reduced RKME or a fixed
low-dimensional interface; it is not eligible for the low-disclosure
deployment Pareto set.

## Experiment boundary

- Source few-shot varies complete training/validation episodes and,
  separately, rows retained inside each episode.
- Query few-shot varies one, two, or four complete probe trajectories and
  1–64 retained transitions per trajectory.
- Each full short trajectory exposes 64 selector-visible transitions and
  corresponds to 1,000 environment steps. All evidence was reused; no new
  transition was acquired.
- Nested few-shot prefixes are correlated, not independent statistical
  samples. No confidence interval or confirmatory significance claim is made.
- OFAT varies one factor at a time. It reuses the full-source normalizer, so it
  is a source-side compute benchmark rather than an end-to-end upload curve.
- Market sizes above 30 are analytic complexity projections only. No duplicated
  anchors are presented as measured large-market evidence.
- Warm in-memory reuse excludes cold model loading unless reported separately.
- The joint benchmark artifact contains full-support views for controls; no
  process- or storage-level raw isolation is claimed.

The core certificate remains frozen v0.3 development championization in
`OBSERVE` mode. It binds label, bundle, championization, and execution ABI,
not third-party quality. Of 30 champions, 23 passed the historical competence
check and seven did not; none were removed after observing retrieval results.

## External artifacts

No trajectory bank, policy pool, checkpoint, run/release JSON, CSV, or figure
belongs in Git. Resolve the root in this order:

1. explicit `--artifacts-root`;
2. `RL_LEARNWARE_ARTIFACTS_ROOT`;
3. the `artifacts/` directory beside this repository.

```text
artifacts/
├── v03/runs/v03-main-20260827-r0/
├── v04a/runs/v04a-primary-dev-20260828-r4/
└── v05/
    ├── runs/v05-environment-classification-dev-20260829-r0/
    ├── releases/v05-environment-classification-dev-20260829-r0/
    ├── analysis/
    │   ├── v05-ablation-fewshot-20260829-r0/
    │   ├── v05-compute-scale-20260829-r0/
    │   └── v05-ablation-summary-20260829-r0/
    └── evidence/final-manifest-chain/
```

The loader resolves v0.3 and v0.4a independently from this root and verifies
their frozen file and bank digests. It never depends on old relative geometry
or a symlink. Historical receipts, seals, manifests, results, and absolute path
provenance remain byte-identical; relocation is recorded in an external
sidecar.

The original temporary summary/plot sources referenced by the historical
postprocess manifests could not be recovered:
`MISSING_ORIGINAL_SOURCE`. Existing CSV/PNG bytes remain verifiable frozen
evidence, but this branch does not claim exact source-level regeneration of
those derived files. The raw development, few-shot, and OFAT JSON producers are
versioned and reproducible.

Narrative reports live outside Git under `reports/v05/`.

## Install and test

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

## New runs only

Never overwrite a frozen result. New development runs use the core module
documented on `v05`. For the exploratory analyses:

```bash
export RL_LEARNWARE_ARTIFACTS_ROOT=/path/to/artifacts
export PYTHONPATH=src:.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

python -B -m server.repro_fpo_ppo_v05.ablation_analysis \
  --config configs/v05_ablation.yaml \
  --new-analysis-dir \
    "$RL_LEARNWARE_ARTIFACTS_ROOT/v05/analysis/<new-fewshot-id>"

python -B -m server.repro_fpo_ppo_v05.compute_scale_benchmark \
  --plan configs/v05_ablation.yaml \
  --completed-run-dir \
    "$RL_LEARNWARE_ARTIFACTS_ROOT/v05/runs/v05-environment-classification-dev-20260829-r0" \
  --new-output-dir \
    "$RL_LEARNWARE_ARTIFACTS_ROOT/v05/analysis/<new-scale-id>"
```

`--resume` is valid only for the same code, config, inputs, and directory.
Completed outputs reject missing, changed, or extra artifacts before writing.
An audited successor always writes a new result identity; it never adopts the
historical r0 provenance.
