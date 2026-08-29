# Policy Learnware v0.2

This branch is the frozen, code-only handoff of the v0.2 source-policy pool.
The scientific release is tag `v0.2.0` at
`a7d10c05df069407d1054bf25baa21ac5fa8f961`; audit commits make its paths
portable and its verification surface smaller without rewriting that release
or any receipt.

## Result and boundary

The terminal state is `READY_FOR_V03_MARKET_INTAKE`:

- six tasks, twelve dynamics axes, and 30 source anchors;
- seeds `0/1/2` for every anchor, hence exactly 90 accepted bundles;
- 84 direct terminal records and six audited promotions to the last earlier
  finite checkpoint that passed golden and compiled parity;
- pool digest
  `e478ef1d38b7eea1a38691d4ea2bd25dc0356cd7264f5a5bd6df5e6de5e0d15f`;
- frozen acceptance report digest
  `5a6eba99a795019f036f2597aba0d4001238e353ba4b0f71ed9272f908fb5c00`.

The six promoted cells do not relax parity tolerance or bless their later
failed checkpoint. The selected earlier checkpoint already passed finiteness,
golden parity, and compiled parity at `1e-6` tolerance.

v0.2 does **not** claim championization, a 30-entry market, unseen-dynamics
selection, a development or confirmatory benchmark, full-pool oracle results,
P/O/E tables, or L-min comparisons. Those are v0.3-or-later work. Competence
is downstream `OBSERVE` metadata and is not an acceptance gate for this pool.

The v0.1 result also remains unchanged: engineering checks passed, but its
scientific decision is `NO_GO_CURRENT_POOL_SHIFT` because Gates A and B failed.
An engineering PASS here does not revise that negative result. Hopper was
replaced by `ReacherEasy` during development before this grid was frozen; that
choice is not held-out Paper-I evidence.

## External assets

No policy, checkpoint, run record, runtime checkout, or experiment report is
tracked in Git. Asset-root resolution is strictly:

1. explicit `--artifacts-root` or API argument;
2. `RL_LEARNWARE_ARTIFACTS_ROOT`;
3. `<current-code-repository-parent>/artifacts`.

Explicit and environment values that are empty or whitespace fail closed.
The sibling fallback is enabled only when the supplied code root is the real
Git checkout top-level, has a canonical HEAD and tracked `pyproject.toml`, and
the sibling root already contains a strict root relocation manifest. A
detached audit worktree without that canonical sibling must use an explicit or
environment root. Git discovery uses a root-owned, non-writable executable and
ignores caller `GIT_*` overrides and `PATH` shims.

The only relocation authority is
`$RL_LEARNWARE_ARTIFACTS_ROOT/relocation_manifest.json`, schema
`rl-learnware-relocation/v1`. Its reviewed byte digest is
`81e726c297c78ebc110df017e06e6fb56de73face39371198635299f931bfed9`.
Targets in that manifest are artifacts-root-relative, so the same bytes work
on local and server roots. Unknown, ambiguous, unverified, overlapping,
symlinked, escaping, or incorrectly inventoried mappings fail closed.

Canonical v0.2 assets are:

| Role | Root-relative path | Verified state |
|---|---|---|
| Exact-90 handoff and training evidence | `v02/exact90/v02-reacher-formal-2r-20260825-r2/` | immutable; 13,888 files, 101,180,750 bytes, digest `fbe5b0aa…12b23a2` |
| Formal anchor inputs | `v02/formal_inputs/v02-reacher-formal-2r-20260825-r2/` | immutable; 69 files, 246,292 bytes, digest `3bd5fceb…a3c6dc64` |
| Quiesced coordination sidecar | `v02/runtime_state/v02-reacher-formal-2r-20260825-r2/training_private/coordination/` | operational history only; excluded from immutable exact90 |
| Recovered FPO checkout | `shared/runtime/fpo-418c2554/` | reconstructed source runtime; independent clean-Git attestation required |
| Incomplete legacy recovery | `shared/repro_fpo_ppo/legacy-v02/` | 3,131 files; `policy_io.py` exact bytes verified, original `_vendor` and referenced payloads incomplete |

The FPO relocation row is deliberately not an active `verified` prefix. Code
locates that canonical path directly and requires commit
`418c2554f7cd22d52e14c07d951280929d73bf2f`, 72 tracked files, clean index and
worktree, no untracked or ignored files, head/worktree digest
`7bb5d663d19b1e5099037e56990211c21e43fb5948be55fcdd4f5d983b135783`,
and execution digest
`396f2b4633d1fd0cf1cc753fbe16a458f4e62afabb385cbf2fd3dfb872626083`.
The deleted workspace-level `fpo/` duplicate is neither needed nor consulted.

Historical JSON retains its original absolute path strings and digests. The
resolver overlays location only while opening or semantically comparing a
path; it never edits a receipt or uses basename/string containment shortcuts.

## Three capabilities

| Capability | Status | Meaning |
|---|---|---|
| Handoff verification | **PASS — original immutable evidence** | Re-hash exact90/formal inputs and strictly replay all 90 acceptance cells. |
| Policy-inference assets | **READY — reconstructed provenance only** | FPO and legacy bytes are present and attested; this does not probe JAX, MuJoCo, or host executability. |
| Policy-inference execution | **CONDITIONAL — reconstructed runtime** | Only a successful `load_verified_fpo_upstream` call establishes execution readiness on that host; its receipt remains `RECONSTRUCTED_RUNTIME`. |
| Training replay | **NO-GO** | The original 1,612-file `_vendor` tree (digest `11ea54a9…7f786d4`) is missing. No reconstructed install can satisfy original provenance. |

`capability_status` emits schema
`policy-learnware.v02-capability-status.v1`. Its
`asset_provenance_ready` field is intentionally separate from
`runtime_dependency_ready`, which remains `null` with
`runtime_dependency_check=not_performed`. It never imports JAX or MuJoCo and
must not be treated as an execution smoke.

`load_verified_fpo_upstream(..., allow_reconstructed=True)` always imports
`flow_policy` with an in-memory, inference-only `wandb` shim—even if real
`wandb` is installed. Logging, network, artifact, and write APIs raise; the
caller's `wandb` module namespace is restored after import. Its receipt records
`runtime_status=RECONSTRUCTED_RUNTIME`, shim identity, whether installed
`wandb` was bypassed, and `training_replay_capable=false`. Initialize it once
per worker process and reuse the returned object; a second source load with
cached `flow_policy` modules fails closed. The loader also requires
`sys.dont_write_bytecode is True` and `sys.pycache_prefix is None`; this blocks
both checkout-local bytecode and unverified bytecode caches outside the
attested checkout.

## Verify and reproduce

Run from a source checkout; the retained `server.repro_fpo_ppo_v02` replay
package and script are repository interfaces, not a standalone wheel entry
point. Set the common root explicitly when using a detached Git worktree.

```bash
export RL_LEARNWARE_ARTIFACTS_ROOT=/path/to/sibling/artifacts
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPYCACHEPREFIX

python -m pip install -e '.[test]'
python -m pytest -q tests/test_policy_bundle.py tests/v02

PYTHONPATH=src:. python -B scripts/accept_v02_policy_pool.py verify-assets
PYTHONPATH=src:. python -B scripts/accept_v02_policy_pool.py capabilities
PYTHONPATH=src:. python -B scripts/accept_v02_policy_pool.py replay
```

The replay must report `PASS`, 90 jobs, 30 anchors, seeds `[0,1,2]`, the 84+6
split, the pool digest above, and frozen report digest above. It first verifies
the exact90 and formal-input tree digest/count/bytes from the unique root
manifest; only path-valued receipt fields receive a relocation overlay, while
all other fields and content digests remain exact.

The focused public interfaces are:

```python
from policy_learnware_v0.v02.artifacts import RelocationResolver
from policy_learnware_v0.v02.runtime import (
    load_verified_fpo_upstream,
    original_vendor_status,
    verify_fpo_checkout,
)
from server.repro_fpo_ppo_v02.replay import (
    replay_relocated_policy_pool_acceptance,
)
```

The current test set covers strict JSON and manifest schema, portable roots,
malicious and ambiguous mappings, symlink/containment checks, directory and
file inventories, anchor/attempt/checkpoint path relocation, non-path semantic
equality, exact 84+6 replay, bundle validation, retained provenance, v0.3
intake records, FPO Git attestation, and fail-closed reconstructed imports.

## Consumer and documentation contract

v0.3 and later may consume the pool only through the strict replay and typed
intake boundary. Championization, market publication, quality policy,
benchmark ranking, and oracle release are downstream decisions and must not
mutate or reinterpret v0.2 evidence.

Human-readable audit material lives in the sibling `reports/` root, not this
branch. Immutable evidence remains under `artifacts/`; external reports are
projections and are never acceptance inputs. The final v0.2 projection is
[`reports/v02/FINAL_AUDIT_PROJECTION.md`](../reports/v02/FINAL_AUDIT_PROJECTION.md),
SHA-256
`0b1744034284eedc356270226709d59f9832cba4ce9dc86e1d2d5ffc453996b2`.
