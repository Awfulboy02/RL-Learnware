# Policy Learnware v0.3 / v0.31

This branch is the final v0.3 implementation. It closes the development loop

```text
v0.2 exact-90 handoff
  -> 30-entry source market
  -> candidate-independent transition probes
  -> Raw transition measurement
  -> source Reduced RKME / query Empirical KME
  -> MMD policy ranking
  -> frozen-policy reuse and regret metrics
```

v0.31 supersedes the earlier method **role**, not the immutable experiment
records. The paper candidate is **Raw-Delta RKME Learnware**, with measurement
`(o' - o, a)`. Historical `B3b` remains the Raw-FULL comparator. Learned
EnvironmentSpec (`A-Env`) and competence fusion (`M02/B5`) remain comparison
variants. Encoder-family bake-offs and LOTO are v0.4 work and are not v0.3
completion dependencies.

## Scientific status

All v0.3/v0.31 results are **development evidence** (`formal=false`), not
confirmatory or extrapolation results. Raw-Delta was selected on the same
30-source + 24-development panel on which it was measured, so it must not be
reported as method-selection-blind superiority.

On that panel, Raw-Delta obtained all-context regret `0.0413`, versus `0.1040`
for Raw-FULL, `0.0923` for raw moments, `0.1190` for learned distance-only, and
`0.1709` for learned distance plus global competence. These numbers describe
the frozen development run only. They do not establish differential privacy,
irreversibility, or universal encoder failure.

The full v0.31 report and tables live outside Git under:

```text
<repository-parent>/reports/v03/Policy_Learnware_v0.31_Plan_and_Report.md
<repository-parent>/reports/v03/tables/
```

## External artifacts

Experiment payloads are not stored in this repository. Paths resolve in this
order:

1. explicit CLI/config path;
2. `RL_LEARNWARE_ARTIFACTS_ROOT`;
3. `<repository-parent>/artifacts`.

Canonical roots are `/share/songyf/RL_Learnware/artifacts` on the server and
`/Users/jamesmac/Desktop/RL Learnware/artifacts` locally. The v0.3 closure is:

```text
artifacts/
  relocation_manifest.json                # old -> new; receipts stay immutable
  shared/runtime/fpo-418c2554/             # recovered, attested FPO checkout
  v03/
    runs/
      v03-main-20260827-r0/                # source market + legacy evidence
      v03-signal-ranking-20260827-r1/      # banks, Atlas, oracle, metrics
      v031-raw-transition-controls-20260828-r1/  # eight fixed-Raw views
    restricted/authorities/                # centrally managed nonces
```

The private deployment registry and development oracle are restricted assets.
Historical manifests, receipts, configs, and digests remain byte-identical;
portable code applies a separate relocation manifest when a recorded path has
moved. The exact-90 policies are shared v0.2 assets and are not duplicated
under `artifacts/v03/`.

Fresh policy inference never searches for a workspace-sibling `fpo/` checkout.
It uses the canonical `shared/runtime/fpo-418c2554` path (or an explicit path),
verifies the frozen Git bytes, and requires the explicit
`--allow-reconstructed-runtime` flag. The resulting evidence is always labelled
`RECONSTRUCTED_RUNTIME`; the default remains fail-closed because the original
vendor runtime is missing.

Relocation rows keep the immutable historical `source` as an absolute path and
store `target` relative to the common artifacts root, so the same manifest is
byte-identical across local and server mirrors. Only verified rows whose
`sha256sum-relative-v1` inventory, file count, and regular-file bytes match are
usable.

## Install and focused checks

Python 3.11 or newer is required.

```bash
python -m venv .venv
.venv/bin/pip install -e '.[test,research]'

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  .venv/bin/python -m pytest -q -p no:cacheprovider tests/v03
```

The package CLI exposes the compact numeric checks:

```bash
PYTHONPATH=src python -m policy_learnware_v0.v03 --help
```

## Reproduce the Raw-Delta development calculation

After the external trees have been relocated and verified, the runner can
derive all frozen inputs from the common root. Always use a new output
directory; it will not overwrite the immutable run.

```bash
export RL_LEARNWARE_ARTIFACTS_ROOT=/path/to/RL_Learnware/artifacts

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python -B -m \
  server.repro_fpo_ppo_v03.v031_raw_transition_runner \
  --output-dir /new/empty/v031-recompute \
  --views V_DELTA_ONLY \
  --backend numpy
```

Use explicit `--context-index`, `--public-policy-market`,
`--deployment-private-registry`, or `--oracle-root` to override any derived
input. Omitting them uses, respectively, the context index and development
oracle from `v03-signal-ranking-20260827-r1` and the market from
`v03-main-20260827-r0`.

The full fixed-Raw Table 1 uses the same command without `--views` and evaluates
all eight registered measurements. It reuses stored transition banks and the
development oracle; it does not retrain a policy or encoder and does not read a
confirmatory oracle.

## Other retained entry points

- `server.repro_fpo_ppo_v03.asset_binding`: bind a typed v0.2 intake to source
  evaluation work units. Fresh runtime validation requires
  `--allow-reconstructed-runtime`.
- `server.repro_fpo_ppo_v03.source_market_runner`: select one runnable champion
  per source anchor and publish public/private market views. It performs one
  reconstructed-runtime preflight before creating any run output.
- `server.repro_fpo_ppo_v03.dynamics_probe_collector`: collect the fixed,
  candidate-independent source/development transition banks.
- `server.repro_fpo_ppo_v03.signal_fit_runner` and `signal_bank_runner`: rebuild
  the historical representation/control atlas.
- `server.repro_fpo_ppo_v03.development_baseline_runner`: build representations,
  rank policies, execute the real reuse oracle, and summarize regret.

The last runner also accepts `--artifacts-root` and an optional
`--relocation-manifest`. It resolves a missing historical policy bundle only
through that external mapping; it never edits the frozen private registry.

## Recovery and provenance boundary

The stored banks, source market, Raw-RKME specifications, rankings, oracle, and
metrics are complete and digest-verifiable. The FPO checkout recovered at
`artifacts/shared/runtime/fpo-418c2554` passes both frozen attestation digests.
Two inputs required for an original-provenance *fresh* end-to-end replay are
still unavailable:

- the persisted v0.3 intake/binding triplet must be regenerated from the v0.2
  exact-90 handoff;
- the original `_vendor` dependency tree is missing.

The recovered FPO checkout is verified source evidence, not a candidate. Its
public runtime bridge supports explicitly opted-in, inference-only reconstructed
rollouts and records that provenance, but it does not reconstruct the missing
`_vendor` bytes or original training runtime. A successful reconstructed policy
rollout is therefore not an original-provenance replay. The supported boundary
is stored banks through Raw-RKME ranking and metrics, read-only verification of
the frozen source-market results, and separately labelled reconstructed
inference smoke.

## Privacy boundary

The mathematical interface supports provider-side source reduction and local
query matching, but the development runner is centralized. Reduced supports
remain in interpretable transition coordinates and may expose prototypes.
`B4a/B4b` additionally use target return labels and are privileged upper-bound
controls. No result in this branch proves zero leakage or differential
privacy.
