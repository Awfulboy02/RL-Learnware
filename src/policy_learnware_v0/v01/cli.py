"""Least-privilege command line orchestration for Policy Learnware v0.1.

The v0.1 command line is intentionally separate from the v0 pipeline.  The
freeze step is the only command that consumes the complete experiment YAML;
all later commands consume immutable, capability-scoped artifacts.  This
module keeps imports of JAX, MuJoCo and policy implementations behind the
commands that actually need them so config inspection remains CPU-only.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from ..artifacts import ArtifactLayout
from ..hashing import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    sha256_json,
    sha256_ndarrays,
)
from ..io import ArtifactExistsError, read_json
from ..probe.dataset import EpisodeDataset
from .artifacts import V01ArtifactLayout, V01ArtifactLayoutError
from .audit import (
    assert_measurement_isolation,
    assert_measurement_schema_allowlist,
    assert_no_oracle_dependencies,
)
from .base_runtime import BaseRuntimeBindingError, VerifiedBaseRuntime, verify_and_load_base_runtime
from .config import (
    APPROVED_FORMAL_CONFIG_DIGEST,
    V01ConfigError,
    V01ExperimentConfig,
    load_v01_experiment_config,
)
from .plans import build_pair_plan, verify_pair_plan
from .registry import ShiftRegistry, default_shift_registry
from .schemas import (
    MeasurementSchemaView,
    PrivateContextRecord,
    ProtocolIdentifiers,
    ShiftManifest,
    VariantDatasetManifest,
    derive_experiment_protocol_id,
    derive_measurement_protocol_id,
    derive_measurement_run_id,
    derive_oracle_protocol_id,
    derive_variant_id,
)


CLI_SCHEMA = "policy-learnware.v01-cli-result.v0"
RUN_MANIFEST_SCHEMA = "policy-learnware.v01-run-manifest.v0"
COMMANDS = (
    "validate-config",
    "freeze-run",
    "audit-variants",
    "collect-probes",
    "compute-taskspec-matrix",
    "evaluate-oracle",
    "evaluate-gates",
    "audit-recompute",
    "build-report",
)
_V0_REGRESSION_BACKEND_PROBE_SCHEMA = (
    "policy-learnware.v01-regression-backend-probe.v0"
)
_V0_REGRESSION_TEST_RESULT_SCHEMA = (
    "policy-learnware.v01-v0-unittest-result.v0"
)
_V0_REGRESSION_SUBPROCESS_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "JAX_PLATFORMS": "cpu",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
}


class V01CommandFailure(RuntimeError):
    """A v0.1 command cannot safely satisfy its immutable contract."""


class V01IncompleteArtifacts(V01CommandFailure):
    """A downstream command was attempted before all registered work units exist."""


def _v0_regression_backend_probe_passed(
    record: Mapping[str, Any] | None, *, returncode: int
) -> bool:
    """Validate executable evidence that the regression child is CPU-only."""

    return bool(
        returncode == 0
        and record is not None
        and set(record)
        == {"schema", "default_backend", "device_count", "device_platforms"}
        and record.get("schema") == _V0_REGRESSION_BACKEND_PROBE_SCHEMA
        and record.get("default_backend") == "cpu"
        and type(record.get("device_count")) is int
        and int(record["device_count"]) > 0
        and isinstance(record.get("device_platforms"), list)
        and len(record["device_platforms"]) == int(record["device_count"])
        and all(value == "cpu" for value in record["device_platforms"])
    )


def _v0_regression_backend_probe_command() -> list[str]:
    """Return the exact executable probe bound into the regression attestation."""

    return [
        sys.executable,
        "-c",
        (
            "import logging; "
            "logging.getLogger('jax._src.xla_bridge').setLevel(logging.CRITICAL); "
            "import json, jax; "
            "devices = jax.devices(); "
            "print(json.dumps({"
            f"'schema': '{_V0_REGRESSION_BACKEND_PROBE_SCHEMA}', "
            "'default_backend': jax.default_backend(), "
            "'device_count': len(devices), "
            "'device_platforms': [device.platform for device in devices]"
            "}, sort_keys=True))"
        ),
    ]


def _v0_regression_test_command() -> list[str]:
    """Run the legacy unittest suites without an external test-runner package."""

    script = "\n".join(
        [
            "import contextlib, json, sys, unittest",
            "with contextlib.redirect_stdout(sys.stderr):",
            "    unit = unittest.TestLoader().discover('tests/unit', top_level_dir='.')",
            "    integration = unittest.TestLoader().discover('tests/integration', top_level_dir='.')",
            "    unit_discovered = unit.countTestCases()",
            "    integration_discovered = integration.countTestCases()",
            "    suite = unittest.TestSuite([unit, integration])",
            "    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=1).run(suite)",
            "record = {",
            f"    'schema': '{_V0_REGRESSION_TEST_RESULT_SCHEMA}',",
            "    'unit_discovered': unit_discovered,",
            "    'integration_discovered': integration_discovered,",
            "    'tests_run': result.testsRun,",
            "    'failures': len(result.failures),",
            "    'errors': len(result.errors),",
            "    'skipped': len(result.skipped),",
            "    'expected_failures': len(result.expectedFailures),",
            "    'unexpected_successes': len(result.unexpectedSuccesses),",
            "    'successful': result.wasSuccessful(),",
            "}",
            "print(json.dumps(record, sort_keys=True))",
            "sys.exit(0 if result.wasSuccessful() else 1)",
        ]
    )
    return [
        sys.executable,
        "-c",
        script,
    ]


def _v0_regression_test_record_passed(
    record: Mapping[str, Any] | None, *, returncode: int
) -> bool:
    """Validate the structured result emitted by the stdlib unittest runner."""

    return bool(
        returncode == 0
        and record is not None
        and set(record)
        == {
            "schema",
            "unit_discovered",
            "integration_discovered",
            "tests_run",
            "failures",
            "errors",
            "skipped",
            "expected_failures",
            "unexpected_successes",
            "successful",
        }
        and record.get("schema") == _V0_REGRESSION_TEST_RESULT_SCHEMA
        and type(record.get("unit_discovered")) is int
        and int(record["unit_discovered"]) > 0
        and type(record.get("integration_discovered")) is int
        and int(record["integration_discovered"]) > 0
        and type(record.get("tests_run")) is int
        and int(record["tests_run"])
        == int(record["unit_discovered"]) + int(record["integration_discovered"])
        and type(record.get("failures")) is int
        and int(record["failures"]) == 0
        and type(record.get("errors")) is int
        and int(record["errors"]) == 0
        and type(record.get("skipped")) is int
        and 0 <= int(record["skipped"]) < int(record["tests_run"])
        and type(record.get("expected_failures")) is int
        and int(record["expected_failures"]) == 0
        and type(record.get("unexpected_successes")) is int
        and int(record["unexpected_successes"]) == 0
        and record.get("successful") is True
    )


def _v0_regression_resume_json_passed(record: Any) -> bool:
    return bool(
        isinstance(record, Mapping)
        and record.get("status") == "ok"
        and isinstance(record.get("result"), Mapping)
        and record["result"].get("resumed") is True
    )


def _v0_regression_base_resume_digest(
    report: Mapping[str, Any], base_ref: Mapping[str, Any]
) -> str:
    """Rebuild the regression/base-resume binding from persisted fields."""

    binding = _require_sha256(base_ref.get("binding_digest"), "base binding digest")
    return sha256_json(
        {
            "first_binding_digest": binding,
            "second_binding_digest": binding,
            "protocol_manifest_sha256": _require_sha256(
                base_ref.get("protocol_manifest_sha256"),
                "base protocol manifest digest",
            ),
            "pool_manifest_sha256": _require_sha256(
                base_ref.get("pool_manifest_sha256"), "base pool manifest digest"
            ),
            "public_pool_manifest_sha256": _require_sha256(
                base_ref.get("public_pool_manifest_sha256"),
                "base public pool manifest digest",
            ),
            "resume_command": report.get("base_resume_command"),
            "subprocess_environment": report.get("subprocess_environment"),
            "backend_probe_command": report.get("backend_probe_command"),
            "backend_probe_exit_code": report.get("backend_probe_exit_code"),
            "backend_probe_stdout_sha256": report.get(
                "backend_probe_stdout_sha256"
            ),
            "backend_probe_stderr_sha256": report.get(
                "backend_probe_stderr_sha256"
            ),
            "backend_probe_record": report.get("backend_probe_record"),
            "backend_probe_passed": report.get("backend_probe_passed"),
            "test_command": report.get("command"),
            "test_exit_code": report.get("exit_code"),
            "test_stdout_sha256": report.get("test_stdout_sha256"),
            "test_stderr_sha256": report.get("test_stderr_sha256"),
            "test_record": report.get("test_record"),
            "test_runner_passed": report.get("test_runner_passed"),
            "resume_config_sha256": report.get("base_resume_config_sha256"),
            "resume_exit_code": report.get("base_resume_exit_code"),
            "resume_stdout_sha256": report.get("base_resume_stdout_sha256"),
            "resume_stderr_sha256": report.get("base_resume_stderr_sha256"),
            "resume_json_passed": _v0_regression_resume_json_passed(
                report.get("base_resume_json_record")
            ),
        }
    )


def _v0_regression_binding_evidence_valid(
    report: Mapping[str, Any], base_ref: Mapping[str, Any]
) -> bool:
    """Fail closed on forged CPU-probe or base-resume provenance."""

    try:
        probe = report.get("backend_probe_record")
        if not isinstance(probe, Mapping):
            return False
        probe_stdout = json.dumps(dict(probe), sort_keys=True) + "\n"
        probe_stdout_digest = _require_sha256(
            report.get("backend_probe_stdout_sha256"),
            "regression backend probe stdout digest",
        )
        probe_stderr_digest = _require_sha256(
            report.get("backend_probe_stderr_sha256"),
            "regression backend probe stderr digest",
        )
        test_record = report.get("test_record")
        if not isinstance(test_record, Mapping):
            return False
        test_stdout = json.dumps(dict(test_record), sort_keys=True) + "\n"
        test_stdout_digest = _require_sha256(
            report.get("test_stdout_sha256"),
            "regression unittest stdout digest",
        )
        _require_sha256(
            report.get("test_stderr_sha256"),
            "regression unittest stderr digest",
        )
        for name in (
            "base_resume_config_sha256",
            "base_resume_stdout_sha256",
            "base_resume_stderr_sha256",
            "base_resume_digest",
        ):
            _require_sha256(report.get(name), f"regression {name}")
        return bool(
            report.get("subprocess_environment")
            == _V0_REGRESSION_SUBPROCESS_ENVIRONMENT
            and report.get("backend_probe_command")
            == _v0_regression_backend_probe_command()
            and _v0_regression_backend_probe_passed(
                probe,
                returncode=int(report.get("backend_probe_exit_code", -1)),
            )
            and report.get("backend_probe_passed") is True
            and probe_stdout_digest == sha256_bytes(probe_stdout.encode("utf-8"))
            and probe_stderr_digest == sha256_bytes(b"")
            and report.get("command") == _v0_regression_test_command()
            and _v0_regression_test_record_passed(
                test_record,
                returncode=int(report.get("exit_code", -1)),
            )
            and report.get("test_runner_passed") is True
            and test_stdout_digest == sha256_bytes(test_stdout.encode("utf-8"))
            and report.get("passed_test_count")
            == int(test_record["tests_run"]) - int(test_record["skipped"])
            and report.get("failed_test_count") == int(test_record["failures"])
            and report.get("error_test_count") == int(test_record["errors"])
            and report.get("base_resume_digest")
            == _v0_regression_base_resume_digest(report, base_ref)
        )
    except (TypeError, ValueError, V01CommandFailure):
        return False


def _emit(value: Mapping[str, Any], *, stream: Any | None = None) -> None:
    destination = sys.stdout if stream is None else stream
    print(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False),
        file=destination,
        flush=True,
    )


def _object(path: str | Path, where: str) -> Mapping[str, Any]:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise V01IncompleteArtifacts(f"missing {where}: {candidate}")
    try:
        value = read_json(candidate)
    except (OSError, json.JSONDecodeError) as error:
        raise V01CommandFailure(f"cannot read {where} {candidate}: {error}") from error
    if not isinstance(value, Mapping):
        raise V01CommandFailure(f"{where} must be a JSON object: {candidate}")
    return value


def _require_sha256(value: Any, where: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64:
        raise V01CommandFailure(f"{where} is not a SHA-256 digest")
    try:
        int(digest, 16)
    except ValueError as error:
        raise V01CommandFailure(f"{where} is not a SHA-256 digest") from error
    return digest


def _json_bytes_digest(payload: Any) -> str:
    return sha256_bytes(canonical_json_bytes(payload) + b"\n")


def _source_digest(module: Any) -> str:
    path = Path(str(module.__file__)).resolve()
    if not path.is_file():
        raise V01CommandFailure(f"cannot hash source module: {module.__name__}")
    return sha256_file(path)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_state(project_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(project_root), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise V01CommandFailure(f"git provenance inspection failed: {detail}")
        return completed.stdout.strip()

    commit = run("rev-parse", "HEAD").lower()
    if len(commit) not in {40, 64}:
        raise V01CommandFailure("Git commit is not a supported object id")
    try:
        int(commit, 16)
    except ValueError as error:
        raise V01CommandFailure("Git commit is not hexadecimal") from error
    status = run("status", "--porcelain", "--untracked-files=normal")
    return {
        "commit": commit,
        "clean": status == "",
        "porcelain_sha256": sha256_bytes(status.encode("utf-8")),
    }


def _runtime_versions() -> dict[str, str]:
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for distribution in (
        "jax",
        "jaxlib",
        "flax",
        "numpy",
        "scipy",
        "mujoco",
        "playground",
        "mujoco-playground",
    ):
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return result


def _is_formal(config: V01ExperimentConfig) -> bool:
    """Recognise only the human-approved formal contract byte-for-semantics."""

    return config.config_digest == APPROVED_FORMAL_CONFIG_DIGEST


@contextmanager
def _run_lock(path: Path) -> Iterator[None]:
    """Hold one non-blocking parent lock while a mutating command runs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise V01CommandFailure(f"another v0.1 process owns {path}") from error
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):  # pragma: no cover - non-POSIX fallback
            pass
        handle.close()


def _verify_base(config: V01ExperimentConfig, root: Path) -> VerifiedBaseRuntime:
    return verify_and_load_base_runtime(
        root,
        pool_id=config.base.pool_id,
        expected_protocol_id=config.base.expected_protocol_id,
        expected_protocol_draft_hash=config.base.expected_protocol_draft_hash,
    )


def _base_layout(base: VerifiedBaseRuntime) -> ArtifactLayout:
    return ArtifactLayout(base.base_artifacts_root, base.pool_id)


def _source_support_sizes(base: VerifiedBaseRuntime) -> tuple[int, ...]:
    """Project the live public pool into the only workload field it supplies."""

    sizes = tuple(
        sorted(
            int(np.asarray(entry.task_spec.supports).shape[0])
            for entry in base.public_pool.entries
        )
    )
    if not sizes or any(size <= 0 for size in sizes):
        raise V01CommandFailure("verified base contains an invalid TaskSpec support set")
    return sizes


def _verify_analysis_base(
    layout: V01ArtifactLayout, base_artifacts_root: Path
) -> VerifiedBaseRuntime:
    """Load the live base named by a frozen run and verify its binding digest."""

    base_ref = _object(layout.base_protocol_ref, "base protocol reference")
    base = verify_and_load_base_runtime(
        base_artifacts_root,
        pool_id=str(base_ref["pool_id"]),
        expected_protocol_id=str(base_ref["protocol_id"]),
        expected_protocol_draft_hash=str(base_ref["protocol_draft_hash"]),
    )
    if base.binding_digest != base_ref.get("binding_digest"):
        raise V01CommandFailure("analysis base runtime differs from frozen binding")
    return base


def _load_candidates_for_freeze(
    config: V01ExperimentConfig, base: VerifiedBaseRuntime
) -> tuple[Any, ...]:
    from .oracle import load_candidates_from_inventory

    layout = _base_layout(base)
    if not layout.bundle_verification.is_file():
        raise V01IncompleteArtifacts(
            f"base bundle verification is missing: {layout.bundle_verification}"
        )
    return load_candidates_from_inventory(
        layout.policy_inventory,
        tasks=config.tasks.all,
        candidates_per_task=config.base.candidates_per_task,
        checkpoint_outer=config.base.checkpoint_outer,
        expected_environment_steps=config.base.actual_environment_steps,
    )


def _live_source_digests_for_domains(
    domains: Sequence[str],
) -> dict[str, dict[str, str]]:
    """Hash only the requested live v0.1 capability domains.

    The imports are intentionally conditional.  TaskSpec computation requests
    only ``measurement`` and therefore neither imports oracle/analysis code nor
    acquires an artifact capability for those domains.
    """

    cli_digest = sha256_file(Path(__file__))
    requested = frozenset(str(domain) for domain in domains)
    unknown = requested - {"measurement", "oracle", "analysis"}
    if unknown:
        raise V01CommandFailure(f"unknown provenance domains: {sorted(unknown)}")
    result: dict[str, dict[str, str]] = {}
    if "measurement" in requested:
        from . import (
            audit,
            execution_profile,
            live_binding,
            plans,
            probe,
            schemas,
            seeds,
            taskspec,
            variant_env,
        )

        result["measurement"] = {
            "schemas_source": _source_digest(schemas),
            "seeds_source": _source_digest(seeds),
            "probe_source": _source_digest(probe),
            "pair_plan_source": _source_digest(plans),
            "taskspec_source": _source_digest(taskspec),
            "measurement_audit_source": _source_digest(audit),
            "execution_profile_source": _source_digest(execution_profile),
            "variant_operator_source": _source_digest(variant_env),
            "live_binding_source": _source_digest(live_binding),
            "cli_source": cli_digest,
        }
    if "oracle" in requested:
        from . import live_binding, oracle, schemas, seeds, variant_env

        result["oracle"] = {
            "schemas_source": _source_digest(schemas),
            "seeds_source": _source_digest(seeds),
            "oracle_source": _source_digest(oracle),
            "variant_operator_source": _source_digest(variant_env),
            "live_binding_source": _source_digest(live_binding),
            "cli_source": cli_digest,
        }
    if "analysis" in requested:
        from . import analysis as analysis_module
        from . import audit, execution_profile, gates, recompute, report, statistics

        result["analysis"] = {
            "analysis_source": _source_digest(analysis_module),
            "statistics_source": _source_digest(statistics),
            "gates_source": _source_digest(gates),
            "recompute_source": _source_digest(recompute),
            "audit_source": _source_digest(audit),
            "report_source": _source_digest(report),
            "execution_profile_source": _source_digest(execution_profile),
            "cli_source": cli_digest,
        }
    return result


def _component_source_digests() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Hash every domain when constructing the immutable full run."""

    result = _live_source_digests_for_domains(
        ("measurement", "oracle", "analysis")
    )
    return result["measurement"], result["oracle"], result["analysis"]


def _component_digests(base: VerifiedBaseRuntime) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    measurement_sources, oracle_sources, analysis_sources = (
        _component_source_digests()
    )
    measurement = {
        "base_binding": base.binding_digest,
        "base_assets": sha256_json(dict(base.asset_digests)),
        **measurement_sources,
    }
    oracle_components = {
        "base_binding": base.binding_digest,
        **oracle_sources,
    }
    return measurement, oracle_components, analysis_sources


def _validate_live_provenance(
    *,
    formal: Any,
    frozen_git: Any,
    frozen_runtime_versions: Any,
    frozen_source_digests: Any,
    domains: Sequence[str],
    protocol_component_digests: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed when the executing code/runtime differs from a frozen run.

    Formal runs require the exact committed and clean checkout that created the
    manifest.  Smoke runs intentionally permit commit/cleanliness drift so an
    engineer can iterate on tests or documentation; they *still* require exact
    runtime versions and exact source hashes for every requested capability
    domain.  Thus smoke is convenient, but stale scientific code is never
    silently resumed.
    """

    if not isinstance(formal, bool):
        raise V01CommandFailure("provenance formal flag must be boolean")
    if not isinstance(frozen_git, Mapping):
        raise V01CommandFailure("provenance lacks the frozen Git state")
    frozen_commit = str(frozen_git.get("commit", "")).lower()
    if len(frozen_commit) not in {40, 64}:
        raise V01CommandFailure("frozen Git commit is malformed")
    try:
        int(frozen_commit, 16)
    except ValueError as error:
        raise V01CommandFailure("frozen Git commit is malformed") from error
    if not isinstance(frozen_git.get("clean"), bool):
        raise V01CommandFailure("frozen Git cleanliness flag is malformed")
    frozen_porcelain = _require_sha256(
        frozen_git.get("porcelain_sha256"), "frozen Git porcelain digest"
    )
    if frozen_git["clean"] and frozen_porcelain != sha256_bytes(b""):
        raise V01CommandFailure(
            "frozen Git state claims clean with a non-empty porcelain digest"
        )

    live_git = _git_state(_project_root())
    if formal:
        if frozen_git["clean"] is not True:
            raise V01CommandFailure("formal run was not frozen from a clean Git checkout")
        if live_git["clean"] is not True:
            raise V01CommandFailure("formal downstream command requires a clean Git checkout")
        if live_git["commit"] != frozen_commit:
            raise V01CommandFailure(
                "formal downstream command Git commit differs from the frozen run"
            )

    if not isinstance(frozen_runtime_versions, Mapping) or not frozen_runtime_versions:
        raise V01CommandFailure("provenance lacks frozen runtime versions")
    expected_runtime = {
        str(name): str(version)
        for name, version in frozen_runtime_versions.items()
        if isinstance(name, str) and isinstance(version, str) and version
    }
    if len(expected_runtime) != len(frozen_runtime_versions):
        raise V01CommandFailure("frozen runtime versions are malformed")
    live_runtime = _runtime_versions()
    if live_runtime != expected_runtime:
        mismatch = {
            name: {"frozen": expected_runtime.get(name), "live": live_runtime.get(name)}
            for name in sorted(set(expected_runtime) | set(live_runtime))
            if expected_runtime.get(name) != live_runtime.get(name)
        }
        raise V01CommandFailure(f"live runtime differs from frozen provenance: {mismatch}")

    if not isinstance(frozen_source_digests, Mapping):
        raise V01CommandFailure("provenance lacks component source digests")
    requested = tuple(str(domain) for domain in domains)
    if not requested or len(set(requested)) != len(requested):
        raise V01CommandFailure("provenance domains must be unique and non-empty")
    live_by_domain = _live_source_digests_for_domains(requested)
    for domain in requested:
        frozen_components = frozen_source_digests.get(domain)
        if not isinstance(frozen_components, Mapping):
            raise V01CommandFailure(
                f"provenance lacks {domain} component digests"
            )
        if protocol_component_digests is not None and domain in protocol_component_digests:
            protocol_components = protocol_component_digests[domain]
            if not isinstance(protocol_components, Mapping) or dict(protocol_components) != dict(
                frozen_components
            ):
                raise V01CommandFailure(
                    f"{domain} protocol component digests differ from run provenance"
                )
        for name, digest in frozen_components.items():
            if not isinstance(name, str) or not name:
                raise V01CommandFailure(
                    f"{domain} component digest name is malformed"
                )
            _require_sha256(digest, f"{domain} component digest {name}")
        live_sources = live_by_domain[domain]
        frozen_source_keys = {
            str(key) for key in frozen_components if str(key).endswith("_source")
        }
        if frozen_source_keys != set(live_sources):
            raise V01CommandFailure(
                f"{domain} component source coverage differs from live implementation"
            )
        mismatches = {
            name: {"frozen": frozen_components.get(name), "live": digest}
            for name, digest in live_sources.items()
            if frozen_components.get(name) != digest
        }
        if mismatches:
            raise V01CommandFailure(
                f"{domain} component source digest mismatch: {mismatches}"
            )
    return {
        "formal": formal,
        "git_policy": "exact_commit_and_clean" if formal else "smoke_source_scoped",
        "git": live_git,
        "runtime_versions": live_runtime,
        "validated_domains": list(requested),
    }


def _candidate_digests(candidates: Sequence[Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for candidate in candidates:
        identifier = str(candidate.candidate_id)
        digest = _require_sha256(candidate.bundle_digest, f"bundle {identifier}")
        if identifier in result:
            raise V01CommandFailure(f"duplicate candidate id: {identifier}")
        result[identifier] = digest
    return dict(sorted(result.items()))


def _seed_collision_attestation(
    *,
    config: V01ExperimentConfig,
    base: VerifiedBaseRuntime,
    candidates: Sequence[Any],
) -> dict[str, Any]:
    """Enumerate the complete approved seed schedule before freezing a run."""

    from .seeds import (
        V01SeedPlan,
        assert_no_base_seed_pair_collision,
        assert_v01_seed_records_disjoint,
        collect_known_base_seed_pairs,
    )

    plan = V01SeedPlan(config.project_seed)
    probe_records = tuple(
        plan.probe_episode(task, bank, episode)
        for task in config.tasks.all
        for bank in range(config.probe.banks)
        for episode in range(config.probe.max_episodes_per_bank)
    )
    oracle_records = tuple(
        plan.oracle_episode(
            str(candidate.task_private), str(candidate.candidate_id), episode
        )
        for candidate in candidates
        for episode in range(config.oracle.episodes_per_candidate_variant)
    )
    gate0_records = tuple(
        plan.gate0_episode(task, episode)
        for task in config.tasks.all
        for episode in range(config.gates.identity.audit_episodes)
    )
    domains = {
        "v01_probe": probe_records,
        "v01_oracle": oracle_records,
        "v01_gate0": gate0_records,
    }
    assert_v01_seed_records_disjoint(domains)
    base_pairs = collect_known_base_seed_pairs(base.base_run_dir)
    all_v01 = probe_records + oracle_records + gate0_records
    assert_no_base_seed_pair_collision(all_v01, base_pairs)
    return {
        "schema": "policy-learnware.v01-seed-collision-attestation.v0",
        "passed": True,
        "record_counts": {
            name: len(records) for name, records in sorted(domains.items())
        },
        "known_base_pair_count": len(base_pairs),
        "known_base_pairs_digest": sha256_json(
            [list(pair) for pair in sorted(base_pairs)]
        ),
        "v01_pairs_digest": sha256_json(
            [list(record.collision_key) for record in all_v01]
        ),
    }


def _build_contexts(
    *,
    config: V01ExperimentConfig,
    registry: ShiftRegistry,
    measurement_protocol_id: str,
    existing: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return private records plus the internal task/factor plan.

    On resume, the CSPRNG material is loaded and strictly revalidated; no token
    is ever generated a second time for an existing experiment root.
    """

    if existing is None:
        contexts = [
            PrivateContextRecord.new(
                task=task,
                shift_id=config.shift.shift_id,
                factor=factor,
            )
            for task in config.tasks.all
            for factor in config.shift.diagnostic_grid
        ]
    else:
        if set(existing) != {"schema", "experiment_id", "entries"}:
            raise V01CommandFailure("private contexts artifact has unknown or missing fields")
        if existing["schema"] != "policy-learnware.v01-private-context-map.v0":
            raise V01CommandFailure("unsupported private contexts schema")
        if existing["experiment_id"] != config.experiment_id:
            raise V01CommandFailure("private contexts experiment id mismatch")
        raw_entries = existing["entries"]
        if not isinstance(raw_entries, list):
            raise V01CommandFailure("private contexts entries must be a list")
        contexts = []
        for entry in raw_entries:
            if not isinstance(entry, Mapping) or "context" not in entry:
                raise V01CommandFailure("invalid private context entry")
            contexts.append(PrivateContextRecord.from_dict(entry["context"]))

    expected = [
        (task, float(factor))
        for task in config.tasks.all
        for factor in config.shift.diagnostic_grid
    ]
    actual = [(item.task, item.factor) for item in contexts]
    if actual != expected:
        raise V01CommandFailure("private context order/coverage differs from config")

    private_entries: list[dict[str, Any]] = []
    variant_plan: list[dict[str, Any]] = []
    for context in contexts:
        resolved = registry.require(context.shift_id, context.task, context.factor)
        manifest = ShiftManifest.create(
            shift_id=context.shift_id,
            factor=context.factor,
            registry_digest=resolved.registry_digest,
            base_protocol_id=config.base.expected_protocol_id,
            task=context.task,
            private_context_id=context.private_context_id,
        )
        variant_id = derive_variant_id(
            measurement_protocol_id=measurement_protocol_id,
            private_nonce=context.private_nonce,
            shift_manifest_digest=manifest.digest,
        )
        private_entries.append(
            {
                "context": context.to_dict(),
                "shift_manifest": manifest.to_dict(),
                "shift_manifest_digest": manifest.digest,
                "variant_id": variant_id,
            }
        )
        variant_plan.append(
            {
                "task": context.task,
                "factor": context.factor,
                "variant_id": variant_id,
                "context": context,
                "manifest": manifest,
            }
        )

    if existing is not None and list(existing["entries"]) != private_entries:
        raise V01CommandFailure(
            "resume context tokens no longer derive the frozen manifests/variant ids"
        )
    return private_entries, variant_plan


def _audit_subset(
    pair_plan: Mapping[str, Any],
    variant_plan: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze the exact per-task raw audit cells before any result exists."""

    by_task: dict[str, dict[float, str]] = {}
    for item in variant_plan:
        task = str(item["task"])
        factor = float(item["factor"])
        by_task.setdefault(task, {})[factor] = str(item["variant_id"])
    selected_within: list[dict[str, Any]] = []
    selected_between: list[dict[str, Any]] = []
    selected_routing: list[dict[str, Any]] = []
    for task in sorted(by_task):
        factors = by_task[task]
        if 1.0 not in factors or 2.0 not in factors:
            raise V01CommandFailure(
                f"raw audit subset requires nominal and factor-2 variants for {task}"
            )
        nominal_id = factors[1.0]
        factor_two_id = factors[2.0]

        def exactly_one(records: Sequence[Mapping[str, Any]], where: str) -> Mapping[str, Any]:
            if len(records) != 1:
                raise V01CommandFailure(
                    f"raw audit subset expected one {where} record for {task}, got {len(records)}"
                )
            return records[0]

        within = exactly_one(
            [
                record
                for record in pair_plan["within"]
                if record["left_variant_id"] == nominal_id
                and record["right_variant_id"] == nominal_id
                and int(record["left_bank"]) == 0
                and int(record["right_bank"]) == 1
            ],
            "nominal bank0-vs-bank1 within",
        )
        between = exactly_one(
            [
                record
                for record in pair_plan["between"]
                if record["left_variant_id"] == nominal_id
                and record["right_variant_id"] == factor_two_id
                and int(record["left_bank"]) == 0
                and int(record["right_bank"]) == 0
            ],
            "nominal-vs-factor2 bank0 between",
        )
        routing = exactly_one(
            [
                record
                for record in pair_plan["routing"]
                if record["variant_id"] == factor_two_id
                and int(record["bank"]) == 0
            ],
            "factor2 bank0 routing",
        )
        selected_within.append({"task_private": task, **dict(within)})
        selected_between.append({"task_private": task, **dict(between)})
        selected_routing.append({"task_private": task, **dict(routing)})
    return {
        "schema": "policy-learnware.v01-stratified-audit-plan.v0",
        "oracle_reaggregation": "all_episode_rows",
        "taskspec_digest_coverage": "all_datasets_and_semantic_caches",
        "taskspec_aggregation_coverage": "all_frozen_pairs",
        "raw_numeric_subset": {
            "within": selected_within,
            "between": selected_between,
            "routing": selected_routing,
            "selection_time": "before_results",
        },
    }


def _freeze_payloads(
    *,
    config: V01ExperimentConfig,
    base: VerifiedBaseRuntime,
    candidates: Sequence[Any],
    existing_contexts: Mapping[str, Any] | None,
    git_state: Mapping[str, Any],
) -> tuple[V01ArtifactLayout, dict[str, Any], list[dict[str, Any]]]:
    registry = default_shift_registry()
    measurement_components, oracle_components, analysis_components = _component_digests(base)
    runtime_versions = _runtime_versions()
    measurement_components["seed_contract"] = sha256_json(
        {"project_seed": config.project_seed, "namespace": "v01_probe"}
    )
    oracle_components["seed_contract"] = sha256_json(
        {"project_seed": config.project_seed, "namespace": "v01_oracle"}
    )
    base_layout = _base_layout(base)
    candidate_digests = _candidate_digests(candidates)
    seed_collision_attestation = _seed_collision_attestation(
        config=config,
        base=base,
        candidates=candidates,
    )
    oracle_components = {
        **oracle_components,
        "inventory_file": sha256_file(base_layout.policy_inventory),
        "verification_file": sha256_file(base_layout.bundle_verification),
        "candidate_bundles": sha256_json(candidate_digests),
    }
    measurement_protocol_id = derive_measurement_protocol_id(
        config_projection=config.measurement_projection(),
        registry_digest=registry.digest,
        component_digests=measurement_components,
    )
    oracle_protocol_id = derive_oracle_protocol_id(
        config_projection=config.oracle_projection(),
        registry_digest=registry.digest,
        component_digests=oracle_components,
    )
    private_entries, variant_plan = _build_contexts(
        config=config,
        registry=registry,
        measurement_protocol_id=measurement_protocol_id,
        existing=existing_contexts,
    )

    schema_views: dict[str, MeasurementSchemaView] = {}
    variant_schema_digests: dict[str, str] = {}
    for record in variant_plan:
        task = str(record["task"])
        try:
            base_schema = base.env_schemas[task]
        except KeyError as error:
            raise V01CommandFailure(f"base protocol has no approved task {task}") from error
        view = MeasurementSchemaView.from_env_schema(base_schema)
        schema_views[view.schema_view_id] = view
        variant_schema_digests[str(record["variant_id"])] = view.digest

    pair_plan = build_pair_plan(
        variant_plan,
        banks=config.probe.banks,
        gate_prefix=config.probe.gate_b_unreduced_prefix,
        routing_prefix=config.probe.max_episodes_per_bank,
        within_bank_pairs=config.probe.sparse_within_bank_pairs,
        nominal_factor=config.shift.nominal_factor,
    )
    pair_plan_digest = verify_pair_plan(pair_plan)
    measurement_run_id = derive_measurement_run_id(
        measurement_protocol_id=measurement_protocol_id,
        variant_schema_view_digests=variant_schema_digests,
        pair_plan_digest=pair_plan_digest,
    )
    experiment_protocol_id = derive_experiment_protocol_id(
        measurement_run_id=measurement_run_id,
        oracle_protocol_id=oracle_protocol_id,
        analysis_projection=config.analysis_projection(),
        component_digests=analysis_components,
    )
    identifiers = ProtocolIdentifiers(
        measurement_protocol_id=measurement_protocol_id,
        oracle_protocol_id=oracle_protocol_id,
        measurement_run_id=measurement_run_id,
        experiment_protocol_id=experiment_protocol_id,
    )

    base_ref = {
        "schema": "policy-learnware.v01-base-protocol-ref.v0",
        "pool_id": base.pool_id,
        "protocol_id": base.protocol_id,
        "protocol_draft_hash": config.base.expected_protocol_draft_hash,
        "binding_digest": base.binding_digest,
        "asset_digests": dict(base.asset_digests),
        "protocol_manifest_sha256": base.protocol_manifest_sha256,
        "pool_manifest_sha256": base.pool_manifest_sha256,
        "public_pool_manifest_sha256": base.public_pool_manifest_sha256,
    }
    registry_ref = {
        "schema": "policy-learnware.v01-shift-registry-ref.v0",
        "registry": registry.to_dict(),
        "registry_digest": registry.digest,
    }
    measurement_protocol = {
        "schema": "policy-learnware.v01-measurement-protocol.v0",
        "measurement_protocol_id": measurement_protocol_id,
        "config_projection": config.measurement_projection(),
        "registry_digest": registry.digest,
        "component_digests": measurement_components,
    }
    oracle_protocol = {
        "schema": "policy-learnware.v01-oracle-protocol.v0",
        "oracle_protocol_id": oracle_protocol_id,
        "config_projection": config.oracle_projection(),
        "registry_digest": registry.digest,
        "component_digests": oracle_components,
        "candidate_digests": candidate_digests,
    }
    measurement_contract = {
        "schema": "policy-learnware.v01-measurement-contract.v0",
        "measurement_protocol_id": measurement_protocol_id,
        "base_protocol_id": base.protocol_id,
        "probe_banks": config.probe.banks,
        "episodes_per_bank": config.probe.max_episodes_per_bank,
        "prefix_grid": list(config.probe.prefix_grid),
        "gate_prefix": config.probe.gate_b_unreduced_prefix,
        "pair_plan_digest": pair_plan_digest,
        "variant_ids": sorted(variant_schema_digests),
        "schema_view_digests": dict(sorted(variant_schema_digests.items())),
        "visibility": "opaque_variant_only_no_context_policy_or_outcome",
    }
    oracle_contract = {
        "schema": "policy-learnware.v01-oracle-contract.v0",
        "oracle_protocol_id": oracle_protocol_id,
        "base_protocol_id": base.protocol_id,
        "horizon": config.return_contract.horizon,
        "episodes_per_candidate_variant": config.oracle.episodes_per_candidate_variant,
        "paired_across_variants": True,
        "paired_across_candidates": False,
        "project_seed": config.project_seed,
        "candidate_ids": sorted(candidate_digests),
        "candidate_digests": candidate_digests,
    }
    run_ref = {
        "schema": "policy-learnware.v01-measurement-run-ref.v0",
        "measurement_protocol_id": measurement_protocol_id,
        "measurement_run_id": measurement_run_id,
        "measurement_protocol_sha256": _json_bytes_digest(measurement_protocol),
        "base_protocol_ref": {
            "pool_id": base.pool_id,
            "protocol_id": base.protocol_id,
            "protocol_draft_hash": config.base.expected_protocol_draft_hash,
            "binding_digest": base.binding_digest,
        },
        "measurement_contract_digest": _json_bytes_digest(measurement_contract),
        "pair_plan_digest": pair_plan_digest,
        "schema_view_digests": dict(sorted(variant_schema_digests.items())),
        # Public provenance projection: this is intentionally sufficient for
        # TaskSpec computation without granting access to frozen/private roots.
        "formal": _is_formal(config),
        "git": dict(git_state),
        "runtime_versions": runtime_versions,
        "measurement_component_digests": measurement_components,
    }
    contexts_payload = {
        "schema": "policy-learnware.v01-private-context-map.v0",
        "experiment_id": config.experiment_id,
        "entries": private_entries,
    }
    candidate_payload = {
        "schema": "policy-learnware.v01-candidates.v0",
        "oracle_protocol_id": oracle_protocol_id,
        "candidates": [candidate.to_dict() for candidate in candidates],
    }
    source_entries = base.pool_manifest.get("entries")
    if not isinstance(source_entries, Mapping) or not source_entries:
        raise V01CommandFailure("base pool manifest lacks private source-task entries")
    source_task_by_id: dict[str, str] = {}
    for task, raw in source_entries.items():
        if not isinstance(raw, Mapping) or not raw.get("opaque_id"):
            raise V01CommandFailure("base source-task entry lacks an opaque id")
        opaque_id = str(raw["opaque_id"])
        if opaque_id in source_task_by_id:
            raise V01CommandFailure("base source-task opaque ids are not unique")
        source_task_by_id[opaque_id] = str(task)
    source_task_map = {
        "schema": "policy-learnware.v01-private-source-task-map.v0",
        "base_pool_manifest_sha256": base.pool_manifest_sha256,
        "source_task_by_id": dict(sorted(source_task_by_id.items())),
    }
    audit_plan = _audit_subset(pair_plan, variant_plan)
    run_manifest = {
        "schema": RUN_MANIFEST_SCHEMA,
        "experiment_id": config.experiment_id,
        "formal": _is_formal(config),
        "project_seed": config.project_seed,
        "config_digest": config.config_digest,
        "config": config.to_dict(),
        "analysis_contract": config.analysis_projection(),
        "analysis_contract_digest": config.analysis_config_digest,
        "protocol_identifiers": identifiers.to_dict(),
        "base_ref_sha256": _json_bytes_digest(base_ref),
        "registry_ref_sha256": _json_bytes_digest(registry_ref),
        "measurement_protocol_sha256": _json_bytes_digest(measurement_protocol),
        "oracle_protocol_sha256": _json_bytes_digest(oracle_protocol),
        "measurement_contract_sha256": _json_bytes_digest(measurement_contract),
        "oracle_contract_sha256": _json_bytes_digest(oracle_contract),
        "pair_plan_sha256": _json_bytes_digest(pair_plan),
        "measurement_run_ref_sha256": _json_bytes_digest(run_ref),
        "audit_plan_sha256": _json_bytes_digest(audit_plan),
        "contexts_sha256": _json_bytes_digest(contexts_payload),
        "candidate_manifest_sha256": _json_bytes_digest(candidate_payload),
        "source_task_map_sha256": _json_bytes_digest(source_task_map),
        "seed_collision_attestation": seed_collision_attestation,
        "source_digests": {
            "measurement": measurement_components,
            "oracle": oracle_components,
            "analysis": analysis_components,
        },
        "git": dict(git_state),
        "runtime_versions": runtime_versions,
        "frozen_scope": {
            "tasks": list(config.tasks.all),
            "shift_id": config.shift.shift_id,
            "diagnostic_grid": list(config.shift.diagnostic_grid),
            "checkpoint_outer": config.base.checkpoint_outer,
            "environment_steps": config.base.actual_environment_steps,
        },
    }
    payloads = {
        "base_ref": base_ref,
        "registry_ref": registry_ref,
        "measurement_protocol": measurement_protocol,
        "oracle_protocol": oracle_protocol,
        "measurement_contract": measurement_contract,
        "oracle_contract": oracle_contract,
        "pair_plan": pair_plan,
        "audit_plan": audit_plan,
        "run_ref": run_ref,
        "contexts": contexts_payload,
        "candidates": candidate_payload,
        "source_task_map": source_task_map,
        "run_manifest": run_manifest,
        "schema_views": schema_views,
        "identifiers": identifiers,
    }
    return V01ArtifactLayout(Path("."), config.experiment_id), payloads, variant_plan


def _publish_freeze(
    *,
    artifacts_root: Path,
    config: V01ExperimentConfig,
    payloads: Mapping[str, Any],
    variant_plan: Sequence[Mapping[str, Any]],
    resume: bool,
) -> dict[str, Any]:
    layout = V01ArtifactLayout(artifacts_root, config.experiment_id)
    frozen = layout.writer("frozen")
    private = layout.writer("benchmark_private")
    measurement = layout.writer("measurement")
    oracle = layout.writer("oracle_private")

    published: dict[str, str] = {}
    published["contexts"] = private.publish_json(
        layout.contexts, payloads["contexts"], resume=resume
    )
    published["source_task_map"] = private.publish_json(
        layout.source_task_map, payloads["source_task_map"], resume=resume
    )
    for record in variant_plan:
        published[f"shift_manifest:{record['variant_id']}"] = private.publish_json(
            layout.shift_manifest(str(record["task"]), str(record["variant_id"])),
            record["manifest"].to_dict(),
            resume=resume,
        )
    published["base_protocol_ref"] = frozen.publish_json(
        layout.base_protocol_ref, payloads["base_ref"], resume=resume
    )
    published["shift_registry_ref"] = frozen.publish_json(
        layout.shift_registry_ref, payloads["registry_ref"], resume=resume
    )
    published["measurement_protocol"] = frozen.publish_json(
        layout.measurement_protocol, payloads["measurement_protocol"], resume=resume
    )
    published["oracle_protocol"] = frozen.publish_json(
        layout.oracle_protocol, payloads["oracle_protocol"], resume=resume
    )
    published["frozen_measurement_contract"] = frozen.publish_json(
        layout.frozen_measurement_contract,
        payloads["measurement_contract"],
        resume=resume,
    )
    published["oracle_contract"] = frozen.publish_json(
        layout.oracle_contract, payloads["oracle_contract"], resume=resume
    )
    published["audit_plan"] = frozen.publish_json(
        layout.audit_plan, payloads["audit_plan"], resume=resume
    )
    published["pair_plan"] = measurement.publish_json(
        layout.pair_plan, payloads["pair_plan"], resume=resume
    )
    published["measurement_contract"] = measurement.publish_json(
        layout.measurement_contract, payloads["measurement_contract"], resume=resume
    )
    for identifier, view in sorted(payloads["schema_views"].items()):
        published[f"schema_view:{identifier}"] = measurement.publish_json(
            layout.schema_view(identifier), view.to_dict(), resume=resume
        )
    published["measurement_run_ref"] = measurement.publish_json(
        layout.measurement_run_ref, payloads["run_ref"], resume=resume
    )
    published["candidates"] = oracle.publish_json(
        layout.candidates, payloads["candidates"], resume=resume
    )
    # Publish the full manifest last: its existence means every freeze input and
    # public/private projection was atomically materialised at least once.
    published["run_manifest"] = frozen.publish_json(
        layout.run_manifest, payloads["run_manifest"], resume=resume
    )
    if published["frozen_measurement_contract"] != published["measurement_contract"]:
        raise V01CommandFailure("frozen/public measurement contracts are not byte-identical")
    return {
        "experiment_root": str(layout.experiment_root),
        "protocol_identifiers": payloads["identifiers"].to_dict(),
        "published": published,
        "resumed": bool(resume),
    }


def _validate_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_v01_experiment_config(args.config)
    base = _verify_base(config, args.base_artifacts_root)
    candidates = _load_candidates_for_freeze(config, base)
    return {
        "config_digest": config.config_digest,
        "experiment_id": config.experiment_id,
        "formal": _is_formal(config),
        "base_binding_digest": base.binding_digest,
        "candidate_count": len(candidates),
        "tasks": list(config.tasks.all),
        "shift_grid": list(config.shift.diagnostic_grid),
    }


def _freeze_run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_v01_experiment_config(args.config)
    base = _verify_base(config, args.base_artifacts_root)
    candidates = _load_candidates_for_freeze(config, base)
    git_state = _git_state(_project_root())
    if _is_formal(config) and not bool(git_state["clean"]):
        raise V01CommandFailure(
            "formal freeze-run requires a tracked-clean Git commit; use the smoke config while developing"
        )
    layout = V01ArtifactLayout(args.artifacts_root, config.experiment_id)
    existing = _object(layout.contexts, "private contexts") if args.resume else None
    _, payloads, variant_plan = _freeze_payloads(
        config=config,
        base=base,
        candidates=candidates,
        existing_contexts=existing,
        git_state=git_state,
    )
    with _run_lock(layout.run_lock):
        return _publish_freeze(
            artifacts_root=args.artifacts_root,
            config=config,
            payloads=payloads,
            variant_plan=variant_plan,
            resume=args.resume,
        )


def _full_layout(args: argparse.Namespace) -> V01ArtifactLayout:
    return V01ArtifactLayout(args.artifacts_root, args.experiment_id)


def _frozen_state(layout: V01ArtifactLayout) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    manifest = _object(layout.run_manifest, "v0.1 run manifest")
    if manifest.get("schema") != RUN_MANIFEST_SCHEMA:
        raise V01CommandFailure("unsupported v0.1 run manifest")
    if manifest.get("experiment_id") != layout.experiment_id:
        raise V01CommandFailure("run manifest experiment id mismatch")
    raw_config = manifest.get("config")
    if not isinstance(raw_config, Mapping):
        raise V01CommandFailure("run manifest lacks the frozen full config")
    config = V01ExperimentConfig.from_dict(raw_config)
    if config.config_digest != manifest.get("config_digest"):
        raise V01CommandFailure("run manifest config digest mismatch")
    if manifest.get("formal") is not _is_formal(config):
        raise V01CommandFailure("run manifest formal/smoke classification drifted")
    if manifest.get("analysis_contract") != config.analysis_projection():
        raise V01CommandFailure("run manifest analysis contract differs from config")
    if manifest.get("analysis_contract_digest") != config.analysis_config_digest:
        raise V01CommandFailure("run manifest analysis contract digest mismatch")
    source_digests = manifest.get("source_digests")
    # Authenticate the executor before touching any downstream measurement,
    # oracle or private-context artifact.  The manifest itself is the minimal
    # provenance root required to make this decision.
    _validate_live_provenance(
        formal=manifest.get("formal"),
        frozen_git=manifest.get("git"),
        frozen_runtime_versions=manifest.get("runtime_versions"),
        frozen_source_digests=source_digests,
        domains=("measurement", "oracle", "analysis"),
    )

    bound_files = {
        "base_ref_sha256": layout.base_protocol_ref,
        "registry_ref_sha256": layout.shift_registry_ref,
        "measurement_protocol_sha256": layout.measurement_protocol,
        "oracle_protocol_sha256": layout.oracle_protocol,
        "measurement_contract_sha256": layout.frozen_measurement_contract,
        "oracle_contract_sha256": layout.oracle_contract,
        "pair_plan_sha256": layout.pair_plan,
        "measurement_run_ref_sha256": layout.measurement_run_ref,
        "audit_plan_sha256": layout.audit_plan,
        "contexts_sha256": layout.contexts,
        "candidate_manifest_sha256": layout.candidates,
        "source_task_map_sha256": layout.source_task_map,
    }
    for field, path in bound_files.items():
        if not path.is_file() or manifest.get(field) != sha256_file(path):
            raise V01CommandFailure(f"frozen artifact binding mismatch: {field}")
    if (
        not layout.measurement_contract.is_file()
        or sha256_file(layout.measurement_contract)
        != sha256_file(layout.frozen_measurement_contract)
    ):
        raise V01CommandFailure("public/frozen measurement contracts differ")

    measurement_protocol = _object(
        layout.measurement_protocol, "measurement protocol"
    )
    oracle_protocol = _object(layout.oracle_protocol, "oracle protocol")
    run_ref = _object(layout.measurement_run_ref, "measurement run reference")
    if (
        not isinstance(source_digests, Mapping)
        or measurement_protocol.get("component_digests")
        != source_digests.get("measurement")
        or oracle_protocol.get("component_digests")
        != source_digests.get("oracle")
    ):
        raise V01CommandFailure(
            "protocol component digests differ from run provenance"
        )
    if (
        run_ref.get("formal") is not manifest.get("formal")
        or run_ref.get("git") != manifest.get("git")
        or run_ref.get("runtime_versions") != manifest.get("runtime_versions")
        or not isinstance(source_digests, Mapping)
        or run_ref.get("measurement_component_digests")
        != source_digests.get("measurement")
    ):
        raise V01CommandFailure(
            "measurement run reference provenance differs from the full run manifest"
        )
    identifiers = ProtocolIdentifiers.from_dict(manifest["protocol_identifiers"])
    if (
        measurement_protocol.get("measurement_protocol_id")
        != identifiers.measurement_protocol_id
        or oracle_protocol.get("oracle_protocol_id") != identifiers.oracle_protocol_id
        or run_ref.get("measurement_protocol_id") != identifiers.measurement_protocol_id
        or run_ref.get("measurement_run_id") != identifiers.measurement_run_id
    ):
        raise V01CommandFailure("frozen protocol identifiers disagree")
    if run_ref.get("measurement_protocol_sha256") != sha256_file(
        layout.measurement_protocol
    ):
        raise V01CommandFailure("measurement run ref protocol digest mismatch")
    if run_ref.get("measurement_contract_digest") != sha256_file(
        layout.measurement_contract
    ):
        raise V01CommandFailure("measurement run ref contract digest mismatch")
    if run_ref.get("pair_plan_digest") != verify_pair_plan(
        _object(layout.pair_plan, "pair plan")
    ):
        raise V01CommandFailure("measurement run ref pair-plan digest mismatch")
    schema_digests = run_ref.get("schema_view_digests")
    if not isinstance(schema_digests, Mapping) or not schema_digests:
        raise V01CommandFailure("measurement run ref lacks schema-view bindings")
    expected_schema_files = {str(value) for value in schema_digests.values()}
    actual_schema_files: set[str] = set()
    for path in sorted((layout.measurement_dir / "schema_views").glob("*.json")):
        view = MeasurementSchemaView.from_dict(
            _object(path, "measurement schema view")
        )
        if path.stem != view.schema_view_id:
            raise V01CommandFailure("measurement schema-view filename mismatch")
        actual_schema_files.add(view.digest)
    if actual_schema_files != expected_schema_files:
        raise V01CommandFailure("measurement schema-view coverage/digest mismatch")
    contexts = _object(layout.contexts, "private contexts")
    registry_ref = _object(layout.shift_registry_ref, "shift registry reference")
    return manifest, contexts, registry_ref


def _context_entries(contexts: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    entries = contexts.get("entries")
    if not isinstance(entries, list) or not entries:
        raise V01CommandFailure("private context map is empty or malformed")
    result: list[Mapping[str, Any]] = []
    for value in entries:
        if not isinstance(value, Mapping):
            raise V01CommandFailure("private context entry is not an object")
        context = PrivateContextRecord.from_dict(value["context"])
        shift = ShiftManifest.from_dict(value["shift_manifest"])
        if shift.private_context_id != context.private_context_id:
            raise V01CommandFailure("context and ShiftManifest ids differ")
        variant_id = str(value["variant_id"])
        if not variant_id.startswith("v01v-"):
            raise V01CommandFailure("invalid opaque variant id")
        result.append(value)
    return result


def _require_private_gate0(layout: V01ArtifactLayout) -> Mapping[str, Any]:
    from .gates import GATE_0_REQUIRED_CHECKS, evaluate_gate_0

    path = layout.benchmark_private_dir / "gate_0_attestation.json"
    payload = _object(path, "private Gate 0 attestation")
    expected_keys = {
        "schema", "gate", "hard_gate", "passed", "criteria",
        "run_manifest_sha256", "instance_count", "measurement_protocol_id",
        "task_audits", "fatal_error",
    }
    if set(payload) != expected_keys:
        raise V01CommandFailure("private Gate 0 attestation has unknown/missing fields")
    if payload.get("schema") != "policy-learnware.v01-private-gate0-attestation.v0":
        raise V01CommandFailure("unsupported private Gate 0 attestation")
    criteria = payload.get("criteria")
    if not isinstance(criteria, list) or len(criteria) != len(GATE_0_REQUIRED_CHECKS):
        raise V01CommandFailure("private Gate 0 criteria are incomplete")
    checks: dict[str, bool] = {}
    for item in criteria:
        if not isinstance(item, Mapping) or set(item) != {"name", "passed", "reason"}:
            raise V01CommandFailure("private Gate 0 criterion is malformed")
        name = str(item["name"])
        if name in checks or type(item["passed"]) is not bool:
            raise V01CommandFailure("private Gate 0 criteria are duplicated or untyped")
        checks[name] = bool(item["passed"])
    rebuilt = evaluate_gate_0(checks).to_dict()
    if any(payload.get(name) != rebuilt[name] for name in rebuilt):
        raise V01CommandFailure("private Gate 0 summary differs from its typed criteria")
    if payload.get("passed") is not True or payload.get("fatal_error") is not None:
        raise V01CommandFailure(f"Gate 0 did not pass: {path}")
    if payload.get("run_manifest_sha256") != sha256_file(layout.run_manifest):
        raise V01CommandFailure("Gate 0 attestation is not bound to frozen run")

    manifest = _object(layout.run_manifest, "v0.1 run manifest")
    identifiers = ProtocolIdentifiers.from_dict(manifest["protocol_identifiers"])
    contexts = _context_entries(_object(layout.contexts, "private contexts"))
    if (
        payload.get("measurement_protocol_id") != identifiers.measurement_protocol_id
        or payload.get("instance_count") != len(contexts)
    ):
        raise V01CommandFailure("Gate 0 coverage/protocol binding is invalid")
    raw_config = manifest.get("config")
    if not isinstance(raw_config, Mapping):
        raise V01CommandFailure("run manifest lacks config for Gate 0 validation")
    config = V01ExperimentConfig.from_dict(raw_config)
    task_audits = payload.get("task_audits")
    if not isinstance(task_audits, list) or [item.get("task") for item in task_audits] != sorted(config.tasks.all):
        raise V01CommandFailure("Gate 0 task-audit coverage is invalid")
    task_fields = {
        "task", "trajectory_identity", "five_factor_finite",
        "instance_isolation", "policy_return_identity",
    }
    for item in task_audits:
        if not isinstance(item, Mapping) or set(item) != task_fields:
            raise V01CommandFailure("Gate 0 task evidence is malformed")
        for field in task_fields - {"task"}:
            evidence = item[field]
            if not isinstance(evidence, Mapping) or evidence.get("passed") is not True:
                raise V01CommandFailure(f"Gate 0 task evidence did not pass: {field}")

    regression_path = layout.analysis_artifact("v0_regression_attestation.json")
    regression = _object(regression_path, "v0 regression attestation")
    regression_keys = {
        "schema", "command", "base_resume_command", "base_resume_config_sha256",
        "subprocess_environment",
        "backend_probe_command", "backend_probe_exit_code",
        "backend_probe_stdout_sha256", "backend_probe_stderr_sha256",
        "backend_probe_record", "backend_probe_passed",
        "test_stdout_sha256", "test_stderr_sha256", "test_record",
        "test_runner_passed",
        "cwd", "exit_code", "base_resume_exit_code", "base_resume_stdout_sha256",
        "base_resume_stderr_sha256", "base_resume_json_record", "passed_test_count",
        "failed_test_count", "error_test_count", "log_sha256", "base_resume_digest",
        "base_resume_passed", "base_binding_digest", "taskspec_semantic_source_digest",
        "frozen_taskspec_semantic_source_digest", "semantic_source_passed", "passed",
        "run_manifest_sha256",
    }
    if set(regression) != regression_keys:
        raise V01CommandFailure("v0 regression attestation has unknown/missing fields")
    base_ref = _object(layout.base_protocol_ref, "base protocol reference")
    regression_log = layout.benchmark_private_dir / "v0_regression.log"
    if (
        regression.get("schema") != "policy-learnware.v01-v0-regression-attestation.v2"
        or not _v0_regression_binding_evidence_valid(regression, base_ref)
        or regression.get("passed") is not True
        or regression.get("base_resume_passed") is not True
        or regression.get("semantic_source_passed") is not True
        or regression.get("exit_code") != 0
        or regression.get("base_resume_exit_code") != 0
        or regression.get("passed_test_count", 0) <= 0
        or regression.get("failed_test_count") != 0
        or regression.get("error_test_count") != 0
        or regression.get("run_manifest_sha256") != sha256_file(layout.run_manifest)
        or regression.get("base_binding_digest") != base_ref.get("binding_digest")
        or not regression_log.is_file()
        or regression.get("log_sha256") != sha256_file(regression_log)
    ):
        raise V01CommandFailure("v0 regression evidence is invalid or unbound")
    return payload


def _identity_candidates(
    candidates: Sequence[Any], task: str
) -> tuple[Any, Any]:
    """Return the unique frozen FPO-seed0 and PPO-seed0 records for a task."""

    selected: dict[str, list[Any]] = {"fpo": [], "ppo": []}
    for candidate in candidates:
        if (
            str(candidate.task_private) == task
            and int(candidate.training_seed) == 0
            and str(candidate.algorithm) in selected
        ):
            selected[str(candidate.algorithm)].append(candidate)
    for algorithm, values in selected.items():
        if len(values) != 1:
            raise V01CommandFailure(
                f"{task} requires exactly one {algorithm.upper()} seed0 candidate, got {len(values)}"
            )
    return selected["fpo"][0], selected["ppo"][0]


def _audit_policy_return_identity(
    *,
    task: str,
    candidates: Sequence[Any],
    nominal_adapter: Any,
    factor_one_adapter: Any,
    seed_plan: Any,
    fpo_root: Path,
    runs_root: Path | None,
    horizon: int,
    episodes: int,
    return_atol: float,
) -> dict[str, Any]:
    """Automatically execute frozen FPO/PPO seed0 parity and return identity."""

    from ..policy.evaluate import evaluate_frozen_policy_returns_batched
    from .oracle import prepare_candidate

    records = _identity_candidates(candidates, task)
    results: list[dict[str, Any]] = []
    for record in records:
        prepared = prepare_candidate(
            record,
            fpo_root=fpo_root,
            runs_root=runs_root,
            atol=return_atol,
            rtol=return_atol,
        )
        seeds = [
            seed_plan.oracle_episode(task, record.candidate_id, index)
            for index in range(episodes)
        ]
        reset = [item.reset_seed for item in seeds]
        policy = [item.policy_seed for item in seeds]
        kwargs = {
            "reset_seeds": reset,
            "policy_seeds": policy,
            "horizon": horizon,
            "observation_dim": int(factor_one_adapter.schema.observation_dim),
            "action_dim": int(factor_one_adapter.schema.action_dim),
        }
        nominal = np.asarray(
            evaluate_frozen_policy_returns_batched(
                prepared.policy, nominal_adapter.environment, **kwargs
            ),
            dtype=np.float64,
        )
        factor_one = np.asarray(
            evaluate_frozen_policy_returns_batched(
                prepared.policy, factor_one_adapter.environment, **kwargs
            ),
            dtype=np.float64,
        )
        if nominal.shape != (episodes,) or factor_one.shape != nominal.shape:
            raise V01CommandFailure("policy identity evaluator returned an invalid shape")
        if not np.all(np.isfinite(nominal)) or not np.all(np.isfinite(factor_one)):
            raise V01CommandFailure("policy identity evaluator returned non-finite values")
        raw_error = np.abs(nominal - factor_one)
        primary_error = raw_error / float(horizon)
        maximum_primary = float(np.max(primary_error, initial=0.0))
        results.append(
            {
                "candidate_id": record.candidate_id,
                "algorithm": record.algorithm,
                "training_seed": record.training_seed,
                "bundle_digest": prepared.metadata.bundle_digest,
                "golden_parity": dict(prepared.golden_parity),
                "compiled_parity": dict(prepared.compiled_parity),
                "episode_count": episodes,
                "reset_seeds": reset,
                "policy_seeds": policy,
                "nominal_raw_episodic_sums": nominal.tolist(),
                "factor_one_raw_episodic_sums": factor_one.tolist(),
                "maximum_raw_sum_absolute_error": float(
                    np.max(raw_error, initial=0.0)
                ),
                "maximum_primary_mean_absolute_error": maximum_primary,
                "return_atol_primary_mean": float(return_atol),
                "passed": maximum_primary <= float(return_atol),
            }
        )
    return {
        "schema": "policy-learnware.v01-policy-return-identity-audit.v0",
        "task": task,
        "episode_count_per_candidate": episodes,
        "candidates": results,
        "passed": all(item["passed"] for item in results),
    }


def _run_controlled_v0_regression(
    *, base_artifacts_root: Path, base_ref: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    """Run the fixed v0 suite and independently re-open the formal base run."""

    from ..cli import _taskspec_semantic_source_digest

    kwargs = {
        "pool_id": str(base_ref["pool_id"]),
        "expected_protocol_id": str(base_ref["protocol_id"]),
        "expected_protocol_draft_hash": str(base_ref["protocol_draft_hash"]),
    }
    first = verify_and_load_base_runtime(base_artifacts_root, **kwargs)
    current_semantic = _taskspec_semantic_source_digest()
    frozen_semantic = first.protocol.component_digests.get(
        "taskspec_semantic_source"
    )
    # Legacy protocols are already migration-checked by the two verified loads.
    semantic_identity = frozen_semantic in {None, current_semantic}
    command = _v0_regression_test_command()
    # Gate 0 has already exercised the pinned GPU runtime through trajectory
    # identity, finite rollouts and compiled FPO/PPO parity.  Keep the code
    # regression subprocess on the same pinned Python/JAX packages but isolate
    # it from the parent process' live CUDA contexts.  Otherwise a late
    # cuSolver handle allocation can fail solely because the audit parent still
    # owns several compiled MJX environments and their device memory.
    regression_environment = dict(_V0_REGRESSION_SUBPROCESS_ENVIRONMENT)
    subprocess_environment = os.environ.copy()
    subprocess_environment.update(regression_environment)
    backend_probe_command = _v0_regression_backend_probe_command()
    backend_probe_completed = subprocess.run(
        backend_probe_command,
        cwd=_project_root(),
        check=False,
        capture_output=True,
        text=True,
        env=subprocess_environment,
    )
    backend_probe_record: Mapping[str, Any] | None = None
    try:
        candidate = json.loads(backend_probe_completed.stdout.strip())
        if isinstance(candidate, Mapping):
            backend_probe_record = candidate
    except json.JSONDecodeError:
        pass
    backend_probe_passed = bool(
        _v0_regression_backend_probe_passed(
            backend_probe_record,
            returncode=int(backend_probe_completed.returncode),
        )
        and backend_probe_completed.stdout
        == json.dumps(dict(backend_probe_record), sort_keys=True) + "\n"
        and backend_probe_completed.stderr == ""
    )
    completed = subprocess.run(
        command,
        cwd=_project_root(),
        check=False,
        capture_output=True,
        text=True,
        env=subprocess_environment,
    )
    test_log = completed.stdout + completed.stderr
    test_record: Mapping[str, Any] | None = None
    try:
        candidate = json.loads(completed.stdout.strip())
        if isinstance(candidate, Mapping):
            test_record = candidate
    except json.JSONDecodeError:
        pass
    test_runner_passed = bool(
        _v0_regression_test_record_passed(
            test_record,
            returncode=int(completed.returncode),
        )
        and completed.stdout == json.dumps(dict(test_record), sort_keys=True) + "\n"
    )
    config_path = _project_root() / "configs" / "dmc6_outer006_v0.yaml"
    if not config_path.is_file():
        raise V01CommandFailure(f"controlled v0 config is missing: {config_path}")
    config_sha256 = sha256_file(config_path)
    resume_command = [
        sys.executable,
        "-m",
        "policy_learnware_v0.cli",
        "build-pool",
        "--config",
        str(config_path),
        "--artifacts-root",
        str(first.base_artifacts_root),
        "--resume",
    ]
    resume_completed = subprocess.run(
        resume_command,
        cwd=_project_root(),
        check=False,
        capture_output=True,
        text=True,
        env=subprocess_environment,
    )
    resume_log = resume_completed.stdout + resume_completed.stderr
    resume_record: Mapping[str, Any] | None = None
    decoder = json.JSONDecoder()
    offset = 0
    records: list[Mapping[str, Any]] = []
    while True:
        start = resume_completed.stdout.find("{", offset)
        if start < 0:
            break
        try:
            candidate, consumed = decoder.raw_decode(resume_completed.stdout[start:])
        except json.JSONDecodeError:
            offset = start + 1
            continue
        offset = start + max(consumed, 1)
        if (
            isinstance(candidate, Mapping)
            and candidate.get("schema") == "policy-learnware.cli-result.v0"
            and candidate.get("command") == "build-pool"
        ):
            records.append(candidate)
    if len(records) == 1:
        resume_record = records[0]
    resume_json_passed = _v0_regression_resume_json_passed(resume_record)
    second = verify_and_load_base_runtime(base_artifacts_root, **kwargs)
    log = (
        "[subprocess environment]\n"
        + json.dumps(regression_environment, sort_keys=True)
        + "\n[backend probe]\n"
        + backend_probe_completed.stdout
        + backend_probe_completed.stderr
        + "\n[v0 tests]\n"
        + test_log
        + "\n[v0 formal resume]\n"
        + resume_log
    )
    if test_record is None:
        passed_count = 0
        failed_count = 0
        error_count = 0
    else:
        passed_count = max(
            int(test_record.get("tests_run", 0))
            - int(test_record.get("failures", 0))
            - int(test_record.get("errors", 0))
            - int(test_record.get("skipped", 0)),
            0,
        )
        failed_count = int(test_record.get("failures", 0))
        error_count = int(test_record.get("errors", 0))
    base_resume_passed = first.binding_digest == second.binding_digest
    report = {
        "schema": "policy-learnware.v01-v0-regression-attestation.v2",
        "command": command,
        "subprocess_environment": regression_environment,
        "backend_probe_command": backend_probe_command,
        "backend_probe_exit_code": int(backend_probe_completed.returncode),
        "backend_probe_stdout_sha256": sha256_bytes(
            backend_probe_completed.stdout.encode("utf-8")
        ),
        "backend_probe_stderr_sha256": sha256_bytes(
            backend_probe_completed.stderr.encode("utf-8")
        ),
        "backend_probe_record": (
            None if backend_probe_record is None else dict(backend_probe_record)
        ),
        "backend_probe_passed": backend_probe_passed,
        "test_stdout_sha256": sha256_bytes(completed.stdout.encode("utf-8")),
        "test_stderr_sha256": sha256_bytes(completed.stderr.encode("utf-8")),
        "test_record": None if test_record is None else dict(test_record),
        "test_runner_passed": test_runner_passed,
        "base_resume_command": resume_command,
        "base_resume_config_sha256": config_sha256,
        "cwd": str(_project_root()),
        "exit_code": int(completed.returncode),
        "base_resume_exit_code": int(resume_completed.returncode),
        "base_resume_stdout_sha256": sha256_bytes(
            resume_completed.stdout.encode("utf-8")
        ),
        "base_resume_stderr_sha256": sha256_bytes(
            resume_completed.stderr.encode("utf-8")
        ),
        "base_resume_json_record": (
            None if resume_record is None else dict(resume_record)
        ),
        "passed_test_count": passed_count,
        "failed_test_count": failed_count,
        "error_test_count": error_count,
        "log_sha256": sha256_bytes(log.encode("utf-8")),
        "base_resume_passed": bool(
            base_resume_passed
            and resume_completed.returncode == 0
            and resume_json_passed
        ),
        "base_binding_digest": first.binding_digest,
        "taskspec_semantic_source_digest": current_semantic,
        "frozen_taskspec_semantic_source_digest": frozen_semantic,
        "semantic_source_passed": semantic_identity,
        "passed": bool(
            backend_probe_passed
            and test_runner_passed
            and passed_count > 0
            and failed_count == 0
            and error_count == 0
            and base_resume_passed
            and resume_completed.returncode == 0
            and resume_json_passed
            and semantic_identity
        ),
    }
    report["base_resume_digest"] = _v0_regression_base_resume_digest(
        report,
        {
            "binding_digest": first.binding_digest,
            "protocol_manifest_sha256": first.protocol_manifest_sha256,
            "pool_manifest_sha256": first.pool_manifest_sha256,
            "public_pool_manifest_sha256": first.public_pool_manifest_sha256,
        },
    )
    return report, log


def _audit_variants(args: argparse.Namespace) -> dict[str, Any]:
    """Execute every Gate-0 audit; no user-authored pass flag is accepted."""

    from ..envs.mujoco_playground import MujocoPlaygroundEnvAdapter
    from .gates import GATE_0_REQUIRED_CHECKS, evaluate_gate_0
    from .oracle import CandidateRecord
    from .probe import frozen_probe_action_tensor
    from .seeds import V01SeedPlan
    from .variant_env import (
        VariantEnvFactory,
        audit_five_factor_finite,
        audit_instance_isolation,
        audit_trajectory_identity,
    )

    layout = _full_layout(args)
    manifest, contexts, registry_ref = _frozen_state(layout)
    registry = ShiftRegistry.from_dict(registry_ref["registry"])
    if registry.digest != registry_ref.get("registry_digest"):
        raise V01CommandFailure("shift registry digest mismatch")
    identifiers = ProtocolIdentifiers.from_dict(manifest["protocol_identifiers"])
    base_ref = _object(layout.base_protocol_ref, "base protocol reference")
    if "project_seed" not in manifest:
        raise V01CommandFailure("run manifest lacks the frozen v0.1 project seed")
    seed_plan = V01SeedPlan(int(manifest["project_seed"]))
    factory = VariantEnvFactory(
        registry, expected_base_protocol_id=str(base_ref["protocol_id"])
    )
    entries = _context_entries(contexts)
    run_ref = _object(layout.measurement_run_ref, "measurement run reference")
    checks = {name: False for name in GATE_0_REQUIRED_CHECKS}
    checks["registry_factor_grid_valid"] = True
    instance_records: list[dict[str, Any]] = []
    task_audits: list[dict[str, Any]] = []
    fatal_error: str | None = None
    with _run_lock(layout.run_lock):
        private = layout.writer("benchmark_private")
        analysis = layout.writer("analysis")
        try:
            base = verify_and_load_base_runtime(
                args.base_artifacts_root,
                pool_id=str(base_ref["pool_id"]),
                expected_protocol_id=str(base_ref["protocol_id"]),
                expected_protocol_draft_hash=str(base_ref["protocol_draft_hash"]),
            )
            if base.binding_digest != base_ref.get("binding_digest"):
                raise V01CommandFailure(
                    "live base binding differs from frozen base reference"
                )
            raw_candidates = _object(
                layout.candidates, "candidate manifest"
            ).get("candidates")
            if not isinstance(raw_candidates, list):
                raise V01CommandFailure("candidate manifest is malformed")
            candidates = tuple(
                CandidateRecord(**record) for record in raw_candidates
            )
            frozen_config = V01ExperimentConfig.from_dict(manifest["config"])
            if _seed_collision_attestation(
                config=frozen_config,
                base=base,
                candidates=candidates,
            ) != manifest.get("seed_collision_attestation"):
                raise V01CommandFailure(
                    "live seed-collision audit differs from the frozen attestation"
                )
            by_task: dict[str, dict[float, Mapping[str, Any]]] = {}
            for raw in entries:
                context = PrivateContextRecord.from_dict(raw["context"])
                by_task.setdefault(context.task, {})[context.factor] = raw
            all_nominal = True
            all_diff = True
            all_schema = True
            all_trajectory_policy = True
            all_finite = True
            all_isolation = True
            for task in sorted(by_task):
                raw_by_factor = by_task[task]
                if tuple(sorted(raw_by_factor)) != (0.5, 0.75, 1.0, 1.5, 2.0):
                    raise V01CommandFailure(f"{task} does not contain the frozen five-factor grid")
                adapters: dict[float, Any] = {}
                for factor, raw in sorted(raw_by_factor.items()):
                    adapters[factor] = factory.create(
                        task=task,
                        shift_manifest=ShiftManifest.from_dict(raw["shift_manifest"]),
                        variant_id=str(raw["variant_id"]),
                        expected_horizon=1000,
                        expected_action_repeat=1,
                        jit=True,
                    )
                plain = MujocoPlaygroundEnvAdapter(
                    task,
                    expected_horizon=1000,
                    expected_action_repeat=1,
                    jit=True,
                )
                seed_records = [seed_plan.gate0_episode(task, index) for index in range(4)]
                reset_seeds = [item.reset_seed for item in seed_records]
                action_tensor = np.asarray(
                    frozen_probe_action_tensor(
                        plain.schema, [item.action_seed for item in seed_records]
                    ),
                    dtype=np.float32,
                )
                trajectory = audit_trajectory_identity(
                    plain,
                    adapters[1.0],
                    reset_seeds=reset_seeds,
                    action_tensor=action_tensor,
                    trajectory_atol=1.0e-6,
                    trajectory_rtol=1.0e-6,
                    reward_atol=1.0e-6,
                )
                finite = audit_five_factor_finite(
                    adapters,
                    reset_seeds=reset_seeds,
                    action_tensor=action_tensor,
                )
                isolation_sequence = [
                    (
                        factor,
                        factory.create(
                            task=task,
                            shift_manifest=ShiftManifest.from_dict(
                                raw_by_factor[factor]["shift_manifest"]
                            ),
                            variant_id=str(raw_by_factor[factor]["variant_id"]),
                            expected_horizon=1000,
                            expected_action_repeat=1,
                            jit=True,
                        ),
                    )
                    for factor in (1.0, 0.5, 2.0, 1.0)
                ]
                isolation = audit_instance_isolation(
                    isolation_sequence,
                    reset_seeds=reset_seeds,
                    action_tensor=action_tensor,
                    trajectory_atol=1.0e-6,
                    trajectory_rtol=1.0e-6,
                    reward_atol=1.0e-6,
                )
                policy_identity = _audit_policy_return_identity(
                    task=task,
                    candidates=candidates,
                    nominal_adapter=plain,
                    factor_one_adapter=adapters[1.0],
                    seed_plan=seed_plan,
                    fpo_root=args.fpo_root,
                    runs_root=args.runs_root,
                    horizon=1000,
                    episodes=4,
                    return_atol=1.0e-6,
                )
                private.publish_json(
                    layout.benchmark_private_dir / f"trajectory_identity_{task}.json",
                    trajectory.to_dict(),
                    resume=args.resume,
                )
                private.publish_json(
                    layout.benchmark_private_dir / f"five_factor_finite_{task}.json",
                    finite.to_dict(),
                    resume=args.resume,
                )
                private.publish_json(
                    layout.benchmark_private_dir / f"instance_isolation_{task}.json",
                    isolation.to_dict(),
                    resume=args.resume,
                )
                private.publish_json(
                    layout.benchmark_private_dir / f"policy_return_identity_{task}.json",
                    policy_identity,
                    resume=args.resume,
                )
                for factor, adapter in sorted(adapters.items()):
                    raw = raw_by_factor[factor]
                    variant_id = str(raw["variant_id"])
                    model_audit = adapter.model_diff_audit
                    expected_view = str(run_ref["schema_view_digests"][variant_id])
                    view_ok = adapter.measurement_schema_view.digest == expected_view
                    finite_item = finite.results[format(factor, ".17g")]
                    instance = adapter.create_instance_record(
                        finite_termination_audit_summary=finite_item.to_dict()
                    )
                    private.publish_json(
                        layout.model_diff(task, variant_id),
                        model_audit.to_dict(),
                        resume=args.resume,
                    )
                    private.publish_json(
                        layout.instance_record(task, variant_id),
                        instance.to_dict(),
                        resume=args.resume,
                    )
                    identity_payload = {
                        "schema": "policy-learnware.v01-variant-identity-audit.v0",
                        "variant_id": variant_id,
                        "nominal_model_identity": (
                            factor != 1.0
                            or model_audit.base_model_digest == model_audit.shifted_model_digest
                        ),
                        "schema_identity": view_ok,
                        "finite_no_early_termination": finite_item.passed,
                        "factor_one_trajectory_identity": (
                            trajectory.to_dict() if factor == 1.0 else None
                        ),
                    }
                    private.publish_json(
                        layout.identity_audit(task, variant_id),
                        identity_payload,
                        resume=args.resume,
                    )
                    instance_records.append(instance.to_dict())
                    all_nominal &= bool(identity_payload["nominal_model_identity"])
                    all_diff &= bool(
                        (factor == 1.0 and not model_audit.changed_leaves)
                        or model_audit.changed_leaves == ("_mjx_model.dof_damping",)
                    )
                    all_schema &= view_ok and adapter.schema.digest == plain.schema.digest
                all_trajectory_policy &= trajectory.passed and policy_identity["passed"]
                all_finite &= finite.passed
                all_isolation &= isolation.passed
                task_audits.append(
                    {
                        "task": task,
                        "trajectory_identity": trajectory.to_dict(),
                        "five_factor_finite": finite.to_dict(),
                        "instance_isolation": isolation.to_dict(),
                        "policy_return_identity": policy_identity,
                    }
                )
            checks["nominal_model_digest_identity"] = all_nominal
            checks["allowlisted_model_diff_only"] = all_diff
            checks["environment_contract_identity"] = all_schema
            checks["measurement_schema_identity"] = all_schema
            checks["identity_trajectory_and_policy_returns"] = all_trajectory_policy
            checks["non_nominal_finite_no_early_termination"] = all_finite
            checks["instance_isolation"] = all_isolation
            regression, regression_log = _run_controlled_v0_regression(
                base_artifacts_root=args.base_artifacts_root,
                base_ref=base_ref,
            )
            regression["run_manifest_sha256"] = sha256_file(layout.run_manifest)
            private.publish_text(
                layout.benchmark_private_dir / "v0_regression.log",
                regression_log,
                resume=args.resume,
            )
            analysis.publish_json(
                layout.analysis_artifact("v0_regression_attestation.json"),
                regression,
                resume=args.resume,
            )
            checks["v0_regression_attestation"] = bool(regression["passed"])
            checks["base_protocol_runtime_bundle_bindings"] = bool(
                base.binding_digest == base_ref["binding_digest"]
                and all_trajectory_policy
            )
        except Exception as error:
            fatal_error = f"{type(error).__name__}: {error}"
        report = evaluate_gate_0(checks).to_dict()
        attestation = {
            "schema": "policy-learnware.v01-private-gate0-attestation.v0",
            **report,
            "run_manifest_sha256": sha256_file(layout.run_manifest),
            "instance_count": len(instance_records),
            "measurement_protocol_id": identifiers.measurement_protocol_id,
            "task_audits": task_audits,
            "fatal_error": fatal_error,
        }
        private.publish_json(
            layout.benchmark_private_dir / "gate_0_attestation.json",
            attestation,
            resume=args.resume,
        )
    if not report["passed"]:
        raise V01CommandFailure(
            "Gate 0 failed; automatically generated diagnostics were retained"
            + (" (" + fatal_error + ")" if fatal_error else "")
        )
    return attestation


def _collect_probes(args: argparse.Namespace) -> dict[str, Any]:
    from .live_binding import (
        build_collection_binding_attestation,
        verify_collection_binding_attestation,
        verify_live_instance_binding,
    )
    from .probe import ProbeBatchExecutor, collect_probe_batch
    from .seeds import V01SeedPlan
    from .variant_env import VariantEnvFactory

    layout = _full_layout(args)
    manifest, contexts, registry_ref = _frozen_state(layout)
    _require_private_gate0(layout)
    registry = ShiftRegistry.from_dict(registry_ref["registry"])
    base_ref = _object(layout.base_protocol_ref, "base protocol reference")
    contract = _object(layout.measurement_contract, "measurement contract")
    contract_digest = sha256_file(layout.measurement_contract)
    run_ref = _object(layout.measurement_run_ref, "measurement run reference")
    if "project_seed" not in manifest:
        raise V01CommandFailure("run manifest lacks the frozen v0.1 project seed")
    project_seed = int(manifest["project_seed"])
    seed_plan = V01SeedPlan(project_seed)
    factory = VariantEnvFactory(registry, expected_base_protocol_id=str(base_ref["protocol_id"]))
    entries = _context_entries(contexts)
    work = [
        (entry, bank)
        for entry in entries
        for bank in range(int(contract["probe_banks"]))
    ]
    work = _shard(work, args.shard_index, args.shard_count)
    written = 0
    resumed = 0
    with _run_lock(layout.run_lock):
        measurement = layout.writer("measurement")
        private = layout.writer("benchmark_private")
        active_variant_id: str | None = None
        active_adapter: Any | None = None
        active_live_binding: Any | None = None
        active_expected_view: str | None = None
        active_instance_path: Path | None = None
        active_executor: ProbeBatchExecutor | None = None
        for raw, bank in work:
            context = PrivateContextRecord.from_dict(raw["context"])
            shift = ShiftManifest.from_dict(raw["shift_manifest"])
            variant_id = str(raw["variant_id"])
            seeds = [
                seed_plan.probe_episode(context.task, bank, index)
                for index in range(int(contract["episodes_per_bank"]))
            ]
            if variant_id != active_variant_id:
                active_adapter = factory.create(
                    task=context.task,
                    shift_manifest=shift,
                    variant_id=variant_id,
                    expected_horizon=1000,
                    expected_action_repeat=1,
                    # Batch executor owns the collection JITs.  Avoid compiling
                    # scalar reset/step wrappers that this command never calls.
                    jit=False,
                )
                active_expected_view = str(
                    run_ref["schema_view_digests"][variant_id]
                )
                if active_adapter.measurement_schema_view.digest != active_expected_view:
                    raise V01CommandFailure(
                        "live measurement schema differs from frozen view"
                    )
                active_instance_path = layout.instance_record(
                    context.task, variant_id
                )
                audited_instance = _object(
                    active_instance_path, "Gate-0 environment instance"
                )
                active_live_binding = verify_live_instance_binding(
                    active_adapter,
                    audited_instance,
                    audited_instance_record_sha256=sha256_file(
                        active_instance_path
                    ),
                )
                active_variant_id = variant_id
                active_executor = None
            if (
                active_adapter is None
                or active_live_binding is None
                or active_expected_view is None
                or active_instance_path is None
            ):
                raise AssertionError("active probe variant was not initialized")
            adapter = active_adapter
            live_binding = active_live_binding
            expected_view = active_expected_view
            instance_path = active_instance_path
            dataset_path = layout.dataset_npz(variant_id, bank)
            manifest_path = layout.dataset_manifest(variant_id, bank)
            attestation_path = layout.collection_attestation(variant_id, bank)
            existing = (
                dataset_path.is_file(),
                manifest_path.is_file(),
                attestation_path.is_file(),
            )
            if args.resume and any(existing):
                if not all(existing):
                    raise V01CommandFailure(
                        "resume found an incomplete probe dataset/manifest/private-"
                        f"attestation bundle for {variant_id}/bank_{bank:03d}"
                    )
                dataset = EpisodeDataset.load_npz(dataset_path)
                sidecar = VariantDatasetManifest.from_dict(
                    _object(manifest_path, "variant dataset manifest")
                )
                expected_sidecar = VariantDatasetManifest(
                    variant_id=variant_id,
                    bank=bank,
                    episode_count=dataset.episode_count,
                    transition_count=dataset.transition_count,
                    reset_seeds=tuple(int(value) for value in dataset.reset_seeds),
                    probe_seeds=tuple(int(value) for value in dataset.probe_seeds),
                    dataset_digest=dataset.digest,
                    base_protocol_id=str(base_ref["protocol_id"]),
                    measurement_contract_digest=contract_digest,
                    measurement_schema_view_digest=expected_view,
                )
                if (
                    sidecar.to_dict() != expected_sidecar.to_dict()
                    or sidecar.reset_seeds
                    != tuple(item.reset_seed for item in seeds)
                    or sidecar.probe_seeds
                    != tuple(item.probe_seed for item in seeds)
                ):
                    raise V01CommandFailure(
                        "resumed probe sidecar differs from raw data or frozen seeds/contract/schema"
                    )
                verify_collection_binding_attestation(
                    _object(
                        attestation_path,
                        "private collection binding attestation",
                    ),
                    audited_record=live_binding.audited_record,
                    audited_instance_record_sha256=sha256_file(instance_path),
                    dataset=dataset,
                    bank=bank,
                    expected_episode_count=int(contract["episodes_per_bank"]),
                    expected_horizon=int(adapter.measurement_schema_view.horizon),
                    run_manifest_sha256=sha256_file(layout.run_manifest),
                )
                resumed += 1
                continue
            if any(existing):
                raise V01CommandFailure(
                    "probe artifacts already exist; use --resume for immutable validation"
                )
            if active_executor is None:
                active_executor = ProbeBatchExecutor(
                    adapter,
                    episode_count=int(contract["episodes_per_bank"]),
                )
            dataset = collect_probe_batch(
                adapter,
                reset_seeds=[item.reset_seed for item in seeds],
                probe_seeds=[item.probe_seed for item in seeds],
                executor=active_executor,
            )
            attestation = build_collection_binding_attestation(
                live_binding,
                dataset,
                bank=bank,
                expected_episode_count=int(contract["episodes_per_bank"]),
                expected_horizon=int(adapter.measurement_schema_view.horizon),
                run_manifest_sha256=sha256_file(layout.run_manifest),
            )
            sidecar = VariantDatasetManifest(
                variant_id=variant_id,
                bank=bank,
                episode_count=dataset.episode_count,
                transition_count=dataset.transition_count,
                reset_seeds=tuple(int(value) for value in dataset.reset_seeds),
                probe_seeds=tuple(int(value) for value in dataset.probe_seeds),
                dataset_digest=dataset.digest,
                base_protocol_id=str(base_ref["protocol_id"]),
                measurement_contract_digest=contract_digest,
                measurement_schema_view_digest=expected_view,
            )
            private.publish_json(
                attestation_path,
                attestation,
                resume=args.resume,
            )
            measurement.publish_npz(dataset_path, dataset.to_arrays(copy=False), resume=args.resume)
            measurement.publish_json(manifest_path, sidecar.to_dict(), resume=args.resume)
            written += 1
    return {"work_units": len(work), "written": written, "resumed": resumed}


def _measurement_layout(root: Path) -> V01ArtifactLayout:
    measurement = root.expanduser().resolve()
    if measurement.name != "measurement":
        raise V01CommandFailure("--measurement-root must name the measurement directory")
    experiment = measurement.parent
    return V01ArtifactLayout(experiment.parent, experiment.name)


def _verify_taskspec_public_contracts_and_views(
    layout: V01ArtifactLayout,
    *,
    run_ref: Mapping[str, Any],
    contract: Mapping[str, Any],
    pair_plan: Mapping[str, Any],
    base: VerifiedBaseRuntime,
) -> None:
    """Verify every public, non-numeric input to a TaskSpec computation.

    This deliberately needs neither the encoder nor the Gaussian kernel.  It is
    therefore safe to run on the zero-recompute resume path before loading any
    measurement asset.
    """

    if run_ref.get("measurement_contract_digest") != sha256_file(
        layout.measurement_contract
    ):
        raise V01CommandFailure("measurement run ref contract digest mismatch")
    plan_digest = verify_pair_plan(pair_plan)
    if (
        run_ref.get("pair_plan_digest") != plan_digest
        or contract.get("pair_plan_digest") != plan_digest
    ):
        raise V01CommandFailure("measurement contract/pair-plan binding mismatch")
    if contract.get("measurement_protocol_id") != run_ref.get(
        "measurement_protocol_id"
    ):
        raise V01CommandFailure("measurement protocol id differs across public contracts")
    if contract.get("base_protocol_id") != base.protocol_id:
        raise V01CommandFailure("measurement contract base protocol id mismatch")

    variants = contract.get("variant_ids")
    contract_views = contract.get("schema_view_digests")
    run_views = run_ref.get("schema_view_digests")
    if (
        not isinstance(variants, list)
        or not variants
        or any(not isinstance(item, str) or not item for item in variants)
        or len(set(variants)) != len(variants)
        or not isinstance(contract_views, Mapping)
        or set(contract_views) != set(variants)
        or dict(contract_views) != run_views
    ):
        raise V01CommandFailure("measurement schema-view bindings are incomplete or inconsistent")

    expected_digests = {str(contract_views[variant_id]) for variant_id in variants}
    expected_filenames = {f"v01s-{digest[:20]}.json" for digest in expected_digests}
    schema_root = layout.measurement_dir / "schema_views"
    actual_paths = tuple(sorted(schema_root.glob("*.json")))
    if {path.name for path in actual_paths} != expected_filenames:
        raise V01CommandFailure("measurement schema-view file coverage mismatch")
    observed_digests: set[str] = set()
    for path in actual_paths:
        view = MeasurementSchemaView.from_dict(
            _object(path, "measurement schema view")
        )
        if path.stem != view.schema_view_id:
            raise V01CommandFailure("measurement schema-view filename/digest mismatch")
        observed_digests.add(view.digest)
    if observed_digests != expected_digests:
        raise V01CommandFailure("measurement schema-view content digest mismatch")


def _taskspec_output_material(
    payload: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], str, str]:
    """Derive every redundant matrix representation from canonical axes JSON."""

    pair_rows = list(payload["pair_rows"])
    arrays = {
        "family": np.asarray(
            [0 if row["family"] == "within" else 1 for row in pair_rows],
            dtype=np.int8,
        ),
        "d_phi": np.asarray([row["d_phi"] for row in pair_rows], dtype=np.float64),
        "raw_mmd2": np.asarray(
            [row["raw_mmd2"] for row in pair_rows], dtype=np.float64
        ),
        "mmd2": np.asarray([row["mmd2"] for row in pair_rows], dtype=np.float64),
    }
    pair_csv = _csv_text(
        pair_rows,
        (
            "family", "pair_index", "left_variant_id", "left_bank",
            "right_variant_id", "right_bank", "prefix", "raw_mmd2", "mmd2",
            "d_phi", "roundoff_clamped", "cross_term",
        ),
    )
    routing_flat = [
        {
            "routing_index": row["routing_index"],
            "variant_id": row["variant_id"],
            "bank": row["bank"],
            "prefix": row["prefix"],
            "selected_source_id": row["selected_source_id"],
        }
        for row in payload["routing_rows"]
    ]
    routing_csv = _csv_text(
        routing_flat,
        ("routing_index", "variant_id", "bank", "prefix", "selected_source_id"),
    )
    return arrays, pair_csv, routing_csv


def _resume_verified_taskspec_bundle(
    layout: V01ArtifactLayout,
    *,
    run_ref: Mapping[str, Any],
    contract: Mapping[str, Any],
    pair_plan: Mapping[str, Any],
    base: VerifiedBaseRuntime,
) -> dict[str, Any] | None:
    """Strictly reuse a complete TaskSpec bundle without encoder/kernel work.

    ``None`` means no matrix output exists and normal computation may proceed.
    Any non-empty strict subset of the five-output bundle is a torn publish and
    fails closed.  A complete bundle is accepted only after the whole public
    input/cache/primitive chain and every redundant serialization are checked.
    """

    outputs = (
        layout.taskspec_matrix_npz,
        layout.taskspec_matrix_axes,
        layout.taskspec_primitive_manifest,
        layout.taskspec_matrix_csv,
        layout.routing_matrix_csv,
    )
    with _run_lock(layout.run_lock):
        present = tuple(path.is_file() for path in outputs)
        if not any(present):
            return None
        if not all(present):
            missing = [path.name for path, exists in zip(outputs, present) if not exists]
            raise V01CommandFailure(
                f"resume found an incomplete TaskSpec output bundle; missing={missing}"
            )

        _verify_taskspec_public_contracts_and_views(
            layout,
            run_ref=run_ref,
            contract=contract,
            pair_plan=pair_plan,
            base=base,
        )
        source_support_sizes = _source_support_sizes(base)
        chain, aggregation, _primitive = _verified_measurement_aggregation(
            layout,
            verified_base_binding_digest=base.binding_digest,
            verified_base_asset_digests=base.asset_digests,
            require_success_attempt=False,
        )
        if not chain.passed or not aggregation.passed:
            raise V01CommandFailure("resumed TaskSpec input/aggregation audit did not pass")
        payload = dict(aggregation.rebuilt_payload)
        semantic_manifest_digests = {
            f"{variant_id}/bank_{bank:03d}": sha256_file(
                layout.semantic_cache_manifest(str(variant_id), bank)
            )
            for variant_id in contract["variant_ids"]
            for bank in range(int(contract["probe_banks"]))
        }
        semantic_content_digests = {
            f"{variant_id}/bank_{bank:03d}": sha256_ndarrays(
                chain.samples[(str(variant_id), bank)].cache_arrays()
            )
            for variant_id in contract["variant_ids"]
            for bank in range(int(contract["probe_banks"]))
        }
        primitive_manifest = {
            "schema": "policy-learnware.v01-taskspec-primitive-manifest.v0",
            "plan_digest": verify_pair_plan(pair_plan),
            "primitive_digest": aggregation.primitive_digest,
            "taskspec_matrix_sha256": _json_bytes_digest(payload),
            "semantic_manifest_sha256": semantic_manifest_digests,
            "semantic_content_digest": semantic_content_digests,
        }
        arrays, pair_csv, routing_csv = _taskspec_output_material(payload)
        writer = layout.writer("measurement")
        # The immutable publisher performs byte-for-byte checks.  These calls
        # do not write because the complete-bundle condition was established.
        writer.publish_npz(layout.taskspec_matrix_npz, arrays, resume=True)
        writer.publish_json(layout.taskspec_matrix_axes, payload, resume=True)
        writer.publish_json(
            layout.taskspec_primitive_manifest, primitive_manifest, resume=True
        )
        writer.publish_text(layout.taskspec_matrix_csv, pair_csv, resume=True)
        writer.publish_text(layout.routing_matrix_csv, routing_csv, resume=True)

        from .execution_profile import verify_any_successful_execution_attempt

        verified_profile = verify_any_successful_execution_attempt(
            layout.measurement_dir,
            source_support_sizes=source_support_sizes,
        )

        allowlist = assert_measurement_schema_allowlist(layout.measurement_dir)
        if not allowlist["passed"]:
            raise V01CommandFailure("TaskSpec outputs failed the measurement schema allowlist")
        isolation = assert_measurement_isolation(
            layout.measurement_dir, _default_forbidden_fields()
        )
        if not isolation["passed"]:
            raise V01CommandFailure("TaskSpec outputs failed the measurement visibility scan")
        return {
            "pair_count": aggregation.pair_count,
            "routing_count": aggregation.routing_count,
            "clamp_count": int(payload["clamp_count"]),
            "taskspec_matrix_sha256": sha256_file(layout.taskspec_matrix_axes),
            "taskspec_primitive_digest": aggregation.primitive_digest,
            "_resumed_execution": True,
            "_resumed_execution_profile": verified_profile,
        }


def _compute_taskspec_impl(args: argparse.Namespace) -> dict[str, Any]:
    from ..rkme.reducer import ReducedRKME
    from .execution_profile import estimate_taskspec_workload
    from .taskspec import (
        WeightedSemanticSample,
        compute_taskspec_matrix,
        encode_measurement_dataset,
        taskspec_primitive_digest,
    )

    assert_no_oracle_dependencies([args.base_artifacts_root, args.measurement_root])
    layout = _measurement_layout(args.measurement_root)
    run_ref = _object(layout.measurement_run_ref, "measurement run reference")
    measurement_components = run_ref.get("measurement_component_digests")
    _validate_live_provenance(
        formal=run_ref.get("formal"),
        frozen_git=run_ref.get("git"),
        frozen_runtime_versions=run_ref.get("runtime_versions"),
        frozen_source_digests={"measurement": measurement_components},
        domains=("measurement",),
    )
    contract = _object(layout.measurement_contract, "measurement contract")
    pair_plan = _object(layout.pair_plan, "frozen pair plan")
    if verify_pair_plan(pair_plan) != run_ref.get("pair_plan_digest"):
        raise V01CommandFailure("pair plan differs from measurement run reference")
    base_ref = run_ref.get("base_protocol_ref")
    if not isinstance(base_ref, Mapping):
        raise V01CommandFailure("measurement run ref lacks base protocol projection")
    base = verify_and_load_base_runtime(
        args.base_artifacts_root,
        pool_id=str(base_ref["pool_id"]),
        expected_protocol_id=str(base_ref["protocol_id"]),
        expected_protocol_draft_hash=str(base_ref["protocol_draft_hash"]),
    )
    if base.binding_digest != base_ref.get("binding_digest"):
        raise V01CommandFailure("base runtime binding differs from measurement run")
    if not isinstance(measurement_components, Mapping):
        raise V01CommandFailure("measurement run lacks component digests")
    if (
        measurement_components.get("base_binding") != base.binding_digest
        or measurement_components.get("base_assets")
        != sha256_json(dict(base.asset_digests))
    ):
        raise V01CommandFailure(
            "live base assets differ from measurement component provenance"
        )
    _verify_taskspec_public_contracts_and_views(
        layout,
        run_ref=run_ref,
        contract=contract,
        pair_plan=pair_plan,
        base=base,
    )
    if args.resume:
        resumed = _resume_verified_taskspec_bundle(
            layout,
            run_ref=run_ref,
            contract=contract,
            pair_plan=pair_plan,
            base=base,
        )
        if resumed is not None:
            return resumed
    assets = base.load_measurement_assets()
    views: dict[str, MeasurementSchemaView] = {}
    for path in sorted((layout.measurement_dir / "schema_views").glob("*.json")):
        view = MeasurementSchemaView.from_dict(_object(path, "measurement schema view"))
        if path.stem != view.schema_view_id:
            raise V01CommandFailure("measurement schema filename/digest mismatch")
        views[view.digest] = view
    samples: dict[tuple[str, int], WeightedSemanticSample] = {}
    semantic_cache_hits = 0
    semantic_cache_misses = 0
    writer = layout.writer("measurement")
    measurement_contract_digest = sha256_file(layout.measurement_contract)
    for variant_id in contract["variant_ids"]:
        view_digest = str(run_ref["schema_view_digests"][variant_id])
        try:
            view = views[view_digest]
        except KeyError as error:
            raise V01IncompleteArtifacts(f"missing schema view {view_digest}") from error
        for bank in range(int(contract["probe_banks"])):
            data_path = layout.dataset_npz(str(variant_id), bank)
            manifest_path = layout.dataset_manifest(str(variant_id), bank)
            sidecar = VariantDatasetManifest.from_dict(_object(manifest_path, "dataset manifest"))
            dataset = EpisodeDataset.load_npz(data_path)
            if dataset.digest != sidecar.dataset_digest:
                raise V01CommandFailure("dataset digest differs from sidecar")
            if (
                sidecar.variant_id != str(variant_id)
                or sidecar.bank != bank
                or sidecar.episode_count != dataset.episode_count
                or sidecar.transition_count != dataset.transition_count
                or sidecar.reset_seeds
                != tuple(int(value) for value in dataset.reset_seeds)
                or sidecar.probe_seeds
                != tuple(int(value) for value in dataset.probe_seeds)
                or sidecar.base_protocol_id != str(base_ref["protocol_id"])
                or sidecar.measurement_contract_digest
                != measurement_contract_digest
                or sidecar.measurement_schema_view_digest != view_digest
            ):
                raise V01CommandFailure(
                    "dataset sidecar differs from frozen measurement inputs"
                )
            cache_path = layout.semantic_cache(str(variant_id), bank)
            cache_manifest_path = layout.semantic_cache_manifest(
                str(variant_id), bank
            )
            cache_binding = {
                "schema": "policy-learnware.v01-semantic-cache-manifest.v0",
                "variant_id": str(variant_id),
                "bank": bank,
                "dataset_digest": dataset.digest,
                "measurement_schema_view_digest": view_digest,
                "base_binding_digest": base.binding_digest,
                "normalization_sha256": base.asset_digests["normalization"],
                "encoder_checkpoint_sha256": base.asset_digests[
                    "encoder_checkpoint"
                ],
                "encoder_config_sha256": base.asset_digests["encoder_config"],
            }
            if cache_path.is_file():
                semantic_cache_hits += 1
                cache_manifest = _object(
                    cache_manifest_path, "semantic cache manifest"
                )
                expected_manifest = {
                    **cache_binding,
                    "cache_sha256": sha256_file(cache_path),
                }
                if cache_manifest != expected_manifest:
                    raise V01CommandFailure(
                        "semantic cache is not bound to current measurement inputs"
                    )
                with np.load(cache_path, allow_pickle=False) as archive:
                    if set(archive.files) != {"points", "weights", "episode_offsets"}:
                        raise V01CommandFailure("semantic cache contains forbidden/unknown members")
                    sample = WeightedSemanticSample(
                        archive["points"], archive["weights"], archive["episode_offsets"]
                    )
            else:
                semantic_cache_misses += 1
                if cache_manifest_path.exists():
                    raise V01CommandFailure(
                        "semantic cache manifest exists without its cache"
                    )
                sample = encode_measurement_dataset(
                    dataset,
                    view,
                    variant_id=str(variant_id),
                    normalizer=assets.normalization,
                    encoder=assets.encoder,
                    max_action_dim=int(base.protocol.packed_layout["max_action_dim"]),
                )
                writer.publish_npz(cache_path, sample.cache_arrays(), resume=False)
                writer.publish_json(
                    cache_manifest_path,
                    {**cache_binding, "cache_sha256": sha256_file(cache_path)},
                    resume=False,
                )
            if sample.episode_count != dataset.episode_count:
                raise V01CommandFailure(
                    "semantic cache episode count differs from raw dataset"
                )
            samples[(str(variant_id), bank)] = sample
    sources: dict[str, ReducedRKME] = {}
    for entry in base.public_pool.entries:
        spec = entry.task_spec
        sources[entry.opaque_id] = ReducedRKME(
            supports=spec.supports,
            beta=spec.beta,
            bandwidth=spec.kernel_bandwidth,
            rkme_norm2=spec.rkme_norm2,
            empirical_norm2=spec.rkme_norm2,
            reduction_error=0.0,
            protocol_id=spec.protocol_id,
        )
    execution_workload = estimate_taskspec_workload(
        samples,
        pair_plan,
        sources,
        block_size=int(args.block_size),
        computation_backend=str(args.computation_backend),
        semantic_cache_hits=semantic_cache_hits,
        semantic_cache_misses=semantic_cache_misses,
    )
    result = compute_taskspec_matrix(
        samples,
        pair_plan,
        kernel=assets.kernel,
        sources=sources,
        block_size=args.block_size,
        computation_backend=args.computation_backend,
    )
    payload = result.to_dict()
    semantic_manifest_digests = {
        f"{variant_id}/bank_{bank:03d}": sha256_file(
            layout.semantic_cache_manifest(str(variant_id), bank)
        )
        for variant_id in contract["variant_ids"]
        for bank in range(int(contract["probe_banks"]))
    }
    primitive_manifest = {
        "schema": "policy-learnware.v01-taskspec-primitive-manifest.v0",
        "plan_digest": verify_pair_plan(pair_plan),
        "primitive_digest": taskspec_primitive_digest(pair_plan, result),
        "taskspec_matrix_sha256": _json_bytes_digest(payload),
        "semantic_manifest_sha256": semantic_manifest_digests,
        "semantic_content_digest": {
            f"{variant_id}/bank_{bank:03d}": sha256_ndarrays(
                samples[(str(variant_id), bank)].cache_arrays()
            )
            for variant_id in contract["variant_ids"]
            for bank in range(int(contract["probe_banks"]))
        },
    }
    arrays, pair_csv, routing_csv = _taskspec_output_material(payload)
    with _run_lock(layout.run_lock):
        writer.publish_npz(layout.taskspec_matrix_npz, arrays, resume=args.resume)
        writer.publish_json(layout.taskspec_matrix_axes, payload, resume=args.resume)
        writer.publish_json(
            layout.taskspec_primitive_manifest,
            primitive_manifest,
            resume=args.resume,
        )
        writer.publish_text(layout.taskspec_matrix_csv, pair_csv, resume=args.resume)
        writer.publish_text(layout.routing_matrix_csv, routing_csv, resume=args.resume)
    allowlist = assert_measurement_schema_allowlist(layout.measurement_dir)
    if not allowlist["passed"]:
        raise V01CommandFailure("TaskSpec outputs failed the measurement schema allowlist")
    isolation = assert_measurement_isolation(
        layout.measurement_dir, _default_forbidden_fields()
    )
    if not isolation["passed"]:
        raise V01CommandFailure("TaskSpec outputs failed the measurement visibility scan")
    return {
        "pair_count": len(result.pair_rows),
        "routing_count": len(result.routing_rows),
        "clamp_count": result.clamp_count,
        "taskspec_matrix_sha256": sha256_file(layout.taskspec_matrix_axes),
        "taskspec_primitive_digest": primitive_manifest["primitive_digest"],
        "_execution_workload": execution_workload,
    }


def _compute_taskspec(args: argparse.Namespace) -> dict[str, Any]:
    """Run TaskSpec under an immutable, non-semantic execution attempt."""

    from .execution_profile import (
        AttemptTimer,
        build_execution_attempt,
        build_initial_input_binding,
        classify_failure,
    )

    assert_no_oracle_dependencies([args.base_artifacts_root, args.measurement_root])
    layout = _measurement_layout(args.measurement_root)
    measurement_run_id, input_binding = build_initial_input_binding(
        layout.measurement_dir
    )
    timer = AttemptTimer.start()
    workload: Mapping[str, Any] | None = None
    try:
        raw_result = _compute_taskspec_impl(args)
        result = dict(raw_result)
        if result.pop("_resumed_execution", False):
            existing = result.pop("_resumed_execution_profile", None)
            if not isinstance(existing, Mapping):
                raise V01CommandFailure("resume did not return its verified execution profile")
            result.update(
                {
                    "resumed": True,
                    "execution_attempt_id": existing["execution_attempt_id"],
                    "execution_attempt_digest": existing["attempt_digest"],
                    "execution_attempt_sha256": sha256_file(
                        layout.execution_attempt(existing["execution_attempt_id"])
                    ),
                }
            )
            return result
        workload_value = result.pop("_execution_workload", None)
        if not isinstance(workload_value, Mapping):
            raise V01CommandFailure("TaskSpec execution did not report its exact workload")
        workload = workload_value
        attempt = build_execution_attempt(
            timer=timer,
            measurement_run_id=measurement_run_id,
            input_binding=input_binding,
            block_size=int(args.block_size),
            computation_backend=str(args.computation_backend),
            workload=workload,
            measurement_root=layout.measurement_dir,
            success=True,
            runtime=_runtime_versions(),
        )
        with _run_lock(layout.run_lock):
            attempt_sha256 = layout.writer("measurement").publish_json(
                layout.execution_attempt(timer.attempt_id), attempt, resume=False
            )
    except Exception as error:
        failed_attempt = build_execution_attempt(
            timer=timer,
            measurement_run_id=measurement_run_id,
            input_binding=input_binding,
            block_size=int(args.block_size),
            computation_backend=str(args.computation_backend),
            workload=None,
            measurement_root=layout.measurement_dir,
            success=False,
            failure=classify_failure(error),
            runtime=_runtime_versions(),
        )
        try:
            with _run_lock(layout.run_lock):
                layout.writer("measurement").publish_json(
                    layout.execution_attempt(timer.attempt_id),
                    failed_attempt,
                    resume=False,
                )
        except Exception as profile_error:
            raise V01CommandFailure(
                "TaskSpec failed and its bounded execution attempt could not be published"
            ) from profile_error
        raise
    result.update(
        {
            "resumed": False,
            "execution_attempt_id": timer.attempt_id,
            "execution_attempt_digest": attempt["attempt_digest"],
            "execution_attempt_sha256": attempt_sha256,
        }
    )
    return result


def _csv_text(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row[name] for name in fields})
    return stream.getvalue()


def _default_forbidden_fields() -> tuple[str, ...]:
    return (
        "lambda", "factor", "d_theta", "policy_id", "job_id", "algorithm",
        "training_seed", "return", "bundle_path", "bundle_digest", "task",
        "task_private", "source_task", "candidate_id", "raw_episodic_sum",
        "mean_step_return", "delta_return", "abs_transfer_gap",
        "environment_instance_digest", "base_model_digest", "shifted_model_digest",
    )


def _evaluate_oracle(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate immutable oracle shards; merge is deliberately all-or-nothing."""

    from .analysis import build_oracle_matrices
    from .live_binding import verify_live_instance_binding
    from .oracle import (
        CandidateRecord,
        evaluate_oracle_shard,
        prepare_candidate,
        validate_oracle_shard_payload,
    )
    from .seeds import V01SeedPlan
    from .variant_env import VariantEnvFactory

    frozen_root = args.frozen_root.expanduser().resolve()
    benchmark_root = args.benchmark_private_root.expanduser().resolve()
    oracle_root = args.oracle_root.expanduser().resolve()
    if frozen_root.name != "frozen" or benchmark_root.name != "benchmark_private" or oracle_root.name != "oracle_private":
        raise V01CommandFailure("oracle roots must be exact frozen/benchmark_private/oracle_private directories")
    if len({frozen_root.parent, benchmark_root.parent, oracle_root.parent}) != 1:
        raise V01CommandFailure("oracle roots must belong to one experiment")
    experiment = frozen_root.parent
    layout = V01ArtifactLayout(experiment.parent, experiment.name)
    manifest, contexts, registry_ref = _frozen_state(layout)
    _require_private_gate0(layout)
    contract = _object(layout.oracle_contract, "oracle contract")
    candidate_payload = _object(layout.candidates, "candidate manifest")
    raw_candidates = candidate_payload.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise V01CommandFailure("candidate manifest is empty")
    candidates = [CandidateRecord(**record) for record in raw_candidates]
    registry = ShiftRegistry.from_dict(registry_ref["registry"])
    base_ref = _object(layout.base_protocol_ref, "base protocol reference")
    factory = VariantEnvFactory(registry, expected_base_protocol_id=str(base_ref["protocol_id"]))
    seed_plan = V01SeedPlan(int(contract["project_seed"]))
    work = [
        (entry, candidate)
        for entry in _context_entries(contexts)
        for candidate in candidates
        if candidate.task_private == PrivateContextRecord.from_dict(entry["context"]).task
    ]
    work = _shard(work, args.shard_index, args.shard_count)
    prepared: dict[str, Any] = {}
    written = 0
    resumed = 0
    merged: dict[str, Any] | None = None
    with _run_lock(layout.run_lock):
        writer = layout.writer("oracle_private")
        for raw, candidate in work:
            context = PrivateContextRecord.from_dict(raw["context"])
            shift = ShiftManifest.from_dict(raw["shift_manifest"])
            variant_id = str(raw["variant_id"])
            path = layout.oracle_shard(context.task, variant_id, candidate.candidate_id)
            if candidate.candidate_id not in prepared:
                prepared[candidate.candidate_id] = prepare_candidate(
                    candidate,
                    fpo_root=args.fpo_root,
                    runs_root=args.runs_root,
                )
            adapter = factory.create(
                task=context.task,
                shift_manifest=shift,
                variant_id=variant_id,
                expected_horizon=int(contract["horizon"]),
                expected_action_repeat=1,
                jit=True,
            )
            instance_path = layout.instance_record(context.task, variant_id)
            audited_instance = _object(
                instance_path, "Gate-0 environment instance"
            )
            live_binding = verify_live_instance_binding(
                adapter,
                audited_instance,
                audited_instance_record_sha256=sha256_file(instance_path),
            )
            seeds = [
                seed_plan.oracle_episode(context.task, candidate.candidate_id, index)
                for index in range(int(contract["episodes_per_candidate_variant"]))
            ]
            # The shard label is the digest of the freshly reconstructed and
            # exactly Gate-0-matched live record, not an unchecked file label.
            instance_digest = live_binding.verified_instance_digest
            if path.is_file() and args.resume:
                validate_oracle_shard_payload(
                    _object(path, "oracle shard"),
                    prepared[candidate.candidate_id],
                    adapter,
                    task_private=context.task,
                    variant_id=variant_id,
                    instance_digest=instance_digest,
                    reset_seeds=[item.reset_seed for item in seeds],
                    policy_seeds=[item.policy_seed for item in seeds],
                    horizon=int(contract["horizon"]),
                )
                resumed += 1
                continue
            shard = evaluate_oracle_shard(
                prepared[candidate.candidate_id],
                adapter,
                task_private=context.task,
                variant_id=variant_id,
                instance_digest=instance_digest,
                reset_seeds=[item.reset_seed for item in seeds],
                policy_seeds=[item.policy_seed for item in seeds],
                horizon=int(contract["horizon"]),
            )
            writer.publish_json(path, shard.to_dict(), resume=False)
            written += 1
        if args.shard_index is None:
            shard_paths = sorted(
                (layout.oracle_private_dir / "shards").glob("*/*/*.json")
            )
            shard_payloads = [_object(path, "oracle shard") for path in shard_paths]
            config = V01ExperimentConfig.from_dict(manifest["config"])
            _, _, _, matrices = build_oracle_matrices(
                contexts,
                candidate_payload,
                shard_payloads,
                config=config,
                analysis_seed_namespace=str(
                    manifest["protocol_identifiers"]["experiment_protocol_id"]
                ),
            )
            episode_rows = list(matrices.episode_rows)
            aggregate_rows = [record.to_dict() for record in matrices.aggregates]
            writer.publish_npz(
                layout.oracle_episodes_npz,
                {
                    "episode_index": np.asarray(
                        [row["episode_index"] for row in episode_rows], dtype=np.int64
                    ),
                    "reset_seed": np.asarray(
                        [row["reset_seed"] for row in episode_rows], dtype=np.int64
                    ),
                    "policy_seed": np.asarray(
                        [row["policy_seed"] for row in episode_rows], dtype=np.int64
                    ),
                    "raw_episodic_sum": np.asarray(
                        [row["raw_episodic_sum"] for row in episode_rows],
                        dtype=np.float64,
                    ),
                    "mean_step_return": np.asarray(
                        [row["mean_step_return"] for row in episode_rows],
                        dtype=np.float64,
                    ),
                },
                resume=args.resume,
            )
            writer.publish_json(
                layout.oracle_episodes_axes,
                matrices.episodes_dict(),
                resume=args.resume,
            )
            episode_fields = tuple(episode_rows[0])
            writer.publish_text(
                layout.oracle_episodes_csv,
                _csv_text(episode_rows, episode_fields),
                resume=args.resume,
            )
            writer.publish_json(
                layout.oracle_aggregates_json,
                matrices.aggregates_dict(),
                resume=args.resume,
            )
            aggregate_fields = tuple(aggregate_rows[0])
            writer.publish_text(
                layout.oracle_aggregates_csv,
                _csv_text(aggregate_rows, aggregate_fields),
                resume=args.resume,
            )
            merged = {
                "oracle_shard_count": len(shard_paths),
                "oracle_episode_count": len(episode_rows),
                "oracle_aggregate_count": len(aggregate_rows),
                "oracle_aggregates_sha256": sha256_file(
                    layout.oracle_aggregates_json
                ),
            }
    return {
        "work_units": len(work),
        "written": written,
        "resumed": resumed,
        "merged": merged,
    }


def _analysis_layout(args: argparse.Namespace) -> V01ArtifactLayout:
    roots = [
        args.frozen_root.expanduser().resolve(),
        args.benchmark_private_root.expanduser().resolve(),
        args.measurement_root.expanduser().resolve(),
        args.oracle_root.expanduser().resolve(),
        args.analysis_root.expanduser().resolve(),
    ]
    expected = ("frozen", "benchmark_private", "measurement", "oracle_private", "analysis")
    if tuple(path.name for path in roots) != expected or len({path.parent for path in roots}) != 1:
        raise V01CommandFailure("analysis roots must be the five exact domains of one experiment")
    experiment = roots[0].parent
    return V01ArtifactLayout(experiment.parent, experiment.name)


def _verified_measurement_aggregation(
    layout: V01ArtifactLayout,
    *,
    verified_base_binding_digest: str | None = None,
    verified_base_asset_digests: Mapping[str, str] | None = None,
    require_success_attempt: bool = True,
    source_support_sizes: tuple[int, ...] | None = None,
) -> tuple[Any, Any, Mapping[str, Any]]:
    """Rebuild the complete matrix from independently frozen primitive terms."""

    from .recompute import (
        load_verified_measurement_samples,
        rebuild_taskspec_aggregation,
    )

    contract = _object(layout.measurement_contract, "measurement contract")
    pair_plan = _object(layout.pair_plan, "pair plan")
    matrix = _object(layout.taskspec_matrix_axes, "TaskSpec matrix")
    primitive = _object(
        layout.taskspec_primitive_manifest, "TaskSpec primitive manifest"
    )
    expected_keys = {
        "schema", "plan_digest", "primitive_digest", "taskspec_matrix_sha256",
        "semantic_manifest_sha256", "semantic_content_digest",
    }
    if set(primitive) != expected_keys:
        raise V01CommandFailure("TaskSpec primitive manifest has unknown/missing fields")
    if primitive.get("schema") != "policy-learnware.v01-taskspec-primitive-manifest.v0":
        raise V01CommandFailure("unsupported TaskSpec primitive manifest")
    plan_digest = verify_pair_plan(pair_plan)
    if primitive.get("plan_digest") != plan_digest:
        raise V01CommandFailure("TaskSpec primitive manifest pair-plan mismatch")
    if primitive.get("taskspec_matrix_sha256") != sha256_file(
        layout.taskspec_matrix_axes
    ):
        raise V01CommandFailure("TaskSpec primitive manifest matrix binding mismatch")
    expected_units = {
        f"{variant_id}/bank_{bank:03d}"
        for variant_id in contract["variant_ids"]
        for bank in range(int(contract["probe_banks"]))
    }
    semantic_manifests = primitive.get("semantic_manifest_sha256")
    semantic_content = primitive.get("semantic_content_digest")
    if (
        not isinstance(semantic_manifests, Mapping)
        or not isinstance(semantic_content, Mapping)
        or set(semantic_manifests) != expected_units
        or set(semantic_content) != expected_units
    ):
        raise V01CommandFailure("TaskSpec primitive manifest semantic coverage mismatch")
    for unit_id in sorted(expected_units):
        variant_id, bank_name = unit_id.split("/", 1)
        bank = int(bank_name.removeprefix("bank_"))
        if semantic_manifests[unit_id] != sha256_file(
            layout.semantic_cache_manifest(variant_id, bank)
        ):
            raise V01CommandFailure(
                f"TaskSpec primitive semantic-manifest binding mismatch: {unit_id}"
            )
    chain = load_verified_measurement_samples(
        layout.measurement_dir,
        contract,
        pair_plan,
        trusted_semantic_cache_digests={
            str(key): str(value) for key, value in semantic_content.items()
        },
        verified_base_binding_digest=verified_base_binding_digest,
        verified_base_asset_digests=verified_base_asset_digests,
    )
    aggregation = rebuild_taskspec_aggregation(
        pair_plan,
        matrix,
        trusted_primitive_digest=str(primitive["primitive_digest"]),
    )
    # A partial matrix left by an OOM/publication failure has no SUCCESS
    # attempt and is therefore ineligible for scientific merge.
    from .execution_profile import verify_any_successful_execution_attempt

    if require_success_attempt:
        if source_support_sizes is None:
            raise V01CommandFailure(
                "TaskSpec release verification requires live base support sizes"
            )
        verify_any_successful_execution_attempt(
            layout.measurement_dir,
            source_support_sizes=source_support_sizes,
        )
    return chain, aggregation, primitive


def _taskspec_dependency_probe_digest(
    *,
    base_artifacts_root: Path,
    measurement_root: Path,
    block_size: int,
    computation_backend: str,
) -> str:
    """Recompute a frozen, bounded TaskSpec primitive probe from raw inputs.

    This is deliberately smaller than Gate B and the later stratified audit.  It
    executes the real raw-dataset -> frozen encoder -> Gaussian self/cross and
    direct-routing primitives for the first canonical public pair/routing units
    at the smallest frozen prefix.  Its only capabilities are base assets and a
    measurement root, so staging that root beside missing/poisoned oracle trees
    provides an executable dependency test without repeating the 62.7B-entry
    formal matrix.
    """

    from ..rkme.reducer import ReducedRKME
    from .taskspec import (
        direct_routing_scores,
        empirical_mmd_with_raw,
        encode_measurement_dataset,
    )

    assert_no_oracle_dependencies([base_artifacts_root, measurement_root])
    layout = _measurement_layout(measurement_root)
    run_ref = _object(layout.measurement_run_ref, "dependency-probe run ref")
    contract = _object(layout.measurement_contract, "dependency-probe contract")
    pair_plan = _object(layout.pair_plan, "dependency-probe pair plan")
    plan_digest = verify_pair_plan(pair_plan)
    if (
        run_ref.get("pair_plan_digest") != plan_digest
        or contract.get("pair_plan_digest") != plan_digest
    ):
        raise V01CommandFailure("dependency-probe pair-plan binding mismatch")
    base_ref = run_ref.get("base_protocol_ref")
    if not isinstance(base_ref, Mapping):
        raise V01CommandFailure("dependency-probe run ref lacks base projection")
    base = verify_and_load_base_runtime(
        base_artifacts_root,
        pool_id=str(base_ref["pool_id"]),
        expected_protocol_id=str(base_ref["protocol_id"]),
        expected_protocol_draft_hash=str(base_ref["protocol_draft_hash"]),
    )
    if base.binding_digest != base_ref.get("binding_digest"):
        raise V01CommandFailure("dependency-probe base binding mismatch")
    assets = base.load_measurement_assets()
    views: dict[str, MeasurementSchemaView] = {}
    for path in sorted((layout.measurement_dir / "schema_views").glob("*.json")):
        view = MeasurementSchemaView.from_dict(_object(path, "measurement schema view"))
        if path.stem != view.schema_view_id:
            raise V01CommandFailure("dependency-probe schema filename mismatch")
        views[view.digest] = view

    prefix_grid = contract.get("prefix_grid")
    if not isinstance(prefix_grid, list) or not prefix_grid:
        raise V01CommandFailure("dependency-probe contract lacks a prefix grid")
    probe_prefix = int(min(prefix_grid))
    if probe_prefix <= 0:
        raise V01CommandFailure("dependency-probe prefix must be positive")
    if not pair_plan.get("within") or not pair_plan.get("routing"):
        raise V01CommandFailure("dependency-probe requires within and routing units")
    pair = pair_plan["within"][0]
    routing = pair_plan["routing"][0]
    units = {
        (str(pair["left_variant_id"]), int(pair["left_bank"])),
        (str(pair["right_variant_id"]), int(pair["right_bank"])),
        (str(routing["variant_id"]), int(routing["bank"])),
    }
    samples: dict[tuple[str, int], Any] = {}
    dataset_digests: dict[str, str] = {}
    contract_digest = sha256_file(layout.measurement_contract)
    schema_digests = contract.get("schema_view_digests")
    if not isinstance(schema_digests, Mapping):
        raise V01CommandFailure("dependency-probe schema map is malformed")
    for variant_id, bank in sorted(units):
        dataset = EpisodeDataset.load_npz(layout.dataset_npz(variant_id, bank))
        sidecar = VariantDatasetManifest.from_dict(
            _object(layout.dataset_manifest(variant_id, bank), "dataset manifest")
        )
        view_digest = str(schema_digests.get(variant_id, ""))
        if (
            sidecar.variant_id != variant_id
            or sidecar.bank != bank
            or sidecar.dataset_digest != dataset.digest
            or sidecar.episode_count != dataset.episode_count
            or sidecar.transition_count != dataset.transition_count
            or sidecar.reset_seeds != tuple(int(value) for value in dataset.reset_seeds)
            or sidecar.probe_seeds != tuple(int(value) for value in dataset.probe_seeds)
            or sidecar.base_protocol_id != str(base_ref["protocol_id"])
            or sidecar.measurement_contract_digest != contract_digest
            or sidecar.measurement_schema_view_digest != view_digest
        ):
            raise V01CommandFailure("dependency-probe dataset binding mismatch")
        try:
            view = views[view_digest]
        except KeyError as error:
            raise V01CommandFailure("dependency-probe schema view is missing") from error
        samples[(variant_id, bank)] = encode_measurement_dataset(
            dataset,
            view,
            variant_id=variant_id,
            normalizer=assets.normalization,
            encoder=assets.encoder,
            max_action_dim=int(base.protocol.packed_layout["max_action_dim"]),
        ).prefix(probe_prefix)
        dataset_digests[f"{variant_id}/bank_{bank:03d}"] = dataset.digest

    left = samples[(str(pair["left_variant_id"]), int(pair["left_bank"]))]
    right = samples[(str(pair["right_variant_id"]), int(pair["right_bank"]))]
    mmd = empirical_mmd_with_raw(
        left,
        right,
        assets.kernel,
        block_size=int(block_size),
        computation_backend=str(computation_backend),
    )
    sources: dict[str, ReducedRKME] = {}
    for entry in base.public_pool.entries:
        spec = entry.task_spec
        sources[entry.opaque_id] = ReducedRKME(
            supports=spec.supports,
            beta=spec.beta,
            bandwidth=spec.kernel_bandwidth,
            rkme_norm2=spec.rkme_norm2,
            empirical_norm2=spec.rkme_norm2,
            reduction_error=0.0,
            protocol_id=spec.protocol_id,
        )
    routing_sample = samples[(str(routing["variant_id"]), int(routing["bank"]))]
    scores = direct_routing_scores(
        routing_sample,
        sources,
        assets.kernel,
        block_size=int(block_size),
        computation_backend=str(computation_backend),
    )
    return sha256_json(
        {
            "schema": "policy-learnware.v01-taskspec-dependency-probe.v0",
            "plan_digest": plan_digest,
            "prefix": probe_prefix,
            "pair_selection": {
                key: pair[key]
                for key in (
                    "left_variant_id", "left_bank", "right_variant_id", "right_bank"
                )
            },
            "routing_selection": {
                key: routing[key] for key in ("variant_id", "bank")
            },
            "dataset_digests": dict(sorted(dataset_digests.items())),
            "raw_mmd2": mmd.raw_mmd2,
            "mmd2": mmd.mmd2,
            "d_phi": mmd.d_phi,
            "left_norm2": mmd.left_norm2,
            "right_norm2": mmd.right_norm2,
            "cross_term": mmd.cross_term,
            "routing_scores": dict(sorted(scores.items())),
        }
    )


def _compute_executable_gate_d(
    layout: V01ArtifactLayout,
    *,
    measurement_chain: Any,
    aggregation: Any,
    source_support_sizes: tuple[int, ...],
    base_artifacts_root: Path | None = None,
    block_size: int = 2048,
    computation_backend: str = "jax",
) -> tuple[dict[str, Any], Mapping[str, bool]]:
    """Compute every Gate-D criterion from executable, typed evidence."""

    from .recompute import (
        compute_gate_d_from_evidence,
        compute_measurement_visibility_evidence,
        compute_oracle_poison_evidence,
        compute_protocol_binding_evidence,
        compute_smoke_formal_separation_evidence,
        compute_taskspec_capability_evidence,
    )

    measurement_protocol = _object(
        layout.measurement_protocol, "measurement protocol"
    )
    try:
        forbidden = tuple(
            measurement_protocol["config_projection"]["measurement_gates"]
            ["leakage"]["forbidden_measurement_fields"]
        )
    except (KeyError, TypeError) as error:
        raise V01CommandFailure(
            "measurement protocol lacks the frozen leakage field list"
        ) from error
    module_root = Path(__file__).resolve().parent
    capability = compute_taskspec_capability_evidence(
        build_parser(),
        measurement_module_paths=(
            module_root / "taskspec.py",
            module_root / "plans.py",
            module_root / "execution_profile.py",
        ),
        orchestration_path=Path(__file__),
    )
    visibility = compute_measurement_visibility_evidence(
        layout.measurement_dir, forbidden
    )
    protocol = compute_protocol_binding_evidence(
        frozen_root=layout.frozen_dir,
        benchmark_private_root=layout.benchmark_private_dir,
        measurement_root=layout.measurement_dir,
        oracle_root=layout.oracle_private_dir,
        measurement_chain=measurement_chain,
        aggregation=aggregation,
    )
    separation = compute_smoke_formal_separation_evidence(layout.experiment_root)

    def resume_runner(measurement_root: Path) -> Mapping[str, Any]:
        if base_artifacts_root is None:
            raise V01CommandFailure(
                "executable oracle-poison audit requires --base-artifacts-root"
            )
        return _compute_taskspec(
            argparse.Namespace(
                base_artifacts_root=Path(base_artifacts_root),
                measurement_root=Path(measurement_root),
                block_size=int(block_size),
                computation_backend=str(computation_backend),
                resume=True,
            )
        )

    poison = compute_oracle_poison_evidence(
        resume_runner,
        measurement_root=layout.measurement_dir,
        source_support_sizes=source_support_sizes,
    )
    checks = {
        "measurement_artifacts_forbidden_fields_absent": visibility.passed,
        "taskspec_command_has_no_oracle_dependency": capability.passed,
        "oracle_poison_does_not_change_taskspec_digest": poison.passed,
        "context_confined_to_private_or_baseline": (
            visibility.passed and capability.passed
        ),
        "smoke_and_formal_runs_separated": separation.passed,
        "matrix_inputs_match_frozen_protocols": protocol.passed,
        "visibility_artifacts_untampered": visibility.passed and protocol.passed,
    }
    report = compute_gate_d_from_evidence(
        taskspec_capability=capability,
        measurement_visibility=visibility,
        protocol_binding=protocol,
        smoke_formal_separation=separation,
        oracle_poison_independence=poison,
    )
    return report, checks


def _evaluate_gates(args: argparse.Namespace) -> dict[str, Any]:
    """Rebuild every gate from frozen primitive rows and raw oracle shards."""

    from .analysis import assemble_scientific_analysis
    from .oracle import CandidateRecord

    layout = _analysis_layout(args)
    manifest, contexts, _ = _frozen_state(layout)
    gate0 = _require_private_gate0(layout)
    config = V01ExperimentConfig.from_dict(manifest["config"])
    base = _verify_analysis_base(layout, args.base_artifacts_root)
    support_sizes = _source_support_sizes(base)
    chain, aggregation, _ = _verified_measurement_aggregation(
        layout,
        verified_base_binding_digest=base.binding_digest,
        verified_base_asset_digests=base.asset_digests,
        source_support_sizes=support_sizes,
    )
    gate_d, gate_d_checks = _compute_executable_gate_d(
        layout,
        measurement_chain=chain,
        aggregation=aggregation,
        base_artifacts_root=args.base_artifacts_root,
        source_support_sizes=support_sizes,
        block_size=args.block_size,
        computation_backend=args.computation_backend,
    )
    candidates_payload = _object(layout.candidates, "candidate manifest")
    source_map_payload = _object(layout.source_task_map, "private source-task map")
    source_task_by_id = source_map_payload.get("source_task_by_id")
    if not isinstance(source_task_by_id, Mapping):
        raise V01CommandFailure("private source-task mapping is malformed")
    run_ref = _object(layout.measurement_run_ref, "measurement run reference")
    schema_by_variant = run_ref.get("schema_view_digests")
    if not isinstance(schema_by_variant, Mapping):
        raise V01CommandFailure("measurement run lacks schema-view bindings")
    shard_paths = sorted((layout.oracle_private_dir / "shards").glob("*/*/*.json"))
    if not shard_paths:
        raise V01IncompleteArtifacts("no raw oracle shards are available")
    shards = [_object(path, "oracle shard") for path in shard_paths]
    matrix = _object(layout.taskspec_matrix_axes, "TaskSpec matrix")
    identifiers = ProtocolIdentifiers.from_dict(manifest["protocol_identifiers"])
    result = assemble_scientific_analysis(
        contexts,
        candidates_payload,
        shards,
        matrix,
        config=config,
        analysis_seed_namespace=identifiers.experiment_protocol_id,
        source_task_by_id={
            str(key): str(value) for key, value in source_task_by_id.items()
        },
        schema_view_digest_by_variant={
            str(key): str(value) for key, value in schema_by_variant.items()
        },
        gate_d_checks=gate_d_checks,
    )
    analysis_payload = result.to_dict()
    if sha256_json(result.oracle.episodes_dict()) != sha256_json(
        _object(layout.oracle_episodes_axes, "published Oracle episode matrix")
    ):
        raise V01CommandFailure("published Oracle episode matrix differs from raw shards")
    if sha256_json(result.oracle.aggregates_dict()) != sha256_json(
        _object(layout.oracle_aggregates_json, "published Oracle aggregate matrix")
    ):
        raise V01CommandFailure("published Oracle aggregate matrix differs from raw shards")

    context_by_variant: dict[str, tuple[str, float, float]] = {}
    for entry in _context_entries(contexts):
        context = PrivateContextRecord.from_dict(entry["context"])
        context_by_variant[str(entry["variant_id"])] = (
            context.task,
            context.factor,
            context.d_theta,
        )
    joined_taskspec: list[dict[str, Any]] = []
    for row in matrix["pair_rows"]:
        left = context_by_variant[str(row["left_variant_id"])]
        right = context_by_variant[str(row["right_variant_id"])]
        joined_taskspec.append(
            {
                "family": row["family"],
                "pair_index": row["pair_index"],
                "task_private": left[0],
                "left_variant_id": row["left_variant_id"],
                "left_factor": left[1],
                "left_d_theta": left[2],
                "right_variant_id": row["right_variant_id"],
                "right_factor": right[1],
                "right_d_theta": right[2],
                "left_bank": row["left_bank"],
                "right_bank": row["right_bank"],
                "prefix": row["prefix"],
                "d_phi": row["d_phi"],
            }
        )
    raw_candidates = candidates_payload.get("candidates")
    if not isinstance(raw_candidates, list):
        raise V01CommandFailure("candidate manifest is malformed")
    candidate_by_id = {
        record.candidate_id: record
        for record in (CandidateRecord(**item) for item in raw_candidates)
    }
    joined_transfer: list[dict[str, Any]] = []
    for aggregate in result.oracle.aggregates:
        context = context_by_variant[aggregate.variant_id]
        candidate = candidate_by_id[aggregate.candidate_id]
        joined_transfer.append(
            {
                **aggregate.to_dict(),
                "factor": context[1],
                "d_theta": context[2],
                "algorithm": candidate.algorithm,
                "training_seed": candidate.training_seed,
            }
        )
    gate_a = analysis_payload["gate_a"]
    gate_b = analysis_payload["gate_b"]
    gate_c = {
        "schema": "policy-learnware.v01-gate-c-diagnostics.v0",
        "diagnostics": analysis_payload["gate_c_diagnostics"],
    }
    with _run_lock(layout.run_lock):
        writer = layout.writer("analysis")
        writer.publish_json(layout.analysis_artifact("gate_0.json"), dict(gate0), resume=args.resume)
        writer.publish_json(layout.analysis_artifact("gate_a.json"), gate_a, resume=args.resume)
        writer.publish_json(layout.analysis_artifact("gate_b.json"), gate_b, resume=args.resume)
        writer.publish_json(layout.analysis_artifact("gate_c_diagnostic.json"), gate_c, resume=args.resume)
        writer.publish_json(layout.analysis_artifact("gate_d.json"), gate_d, resume=args.resume)
        writer.publish_json(
            layout.analysis_artifact("join_audit.json"),
            analysis_payload["join_audit"],
            resume=args.resume,
        )
        writer.publish_text(
            layout.analysis_artifact("joined_taskspec_context.csv"),
            _csv_text(joined_taskspec, tuple(joined_taskspec[0])),
            resume=args.resume,
        )
        writer.publish_text(
            layout.analysis_artifact("joined_transfer_context.csv"),
            _csv_text(joined_transfer, tuple(joined_transfer[0])),
            resume=args.resume,
        )
    if not gate_d["passed"]:
        raise V01CommandFailure("Gate D failed; diagnostics were retained")
    # Scientific failure is a valid result and deliberately returns normally.
    return {
        "gate_0_passed": bool(gate0["passed"]),
        "gate_a_passed": bool(gate_a["passed"]),
        "gate_b_passed": bool(gate_b["passed"]),
        "gate_d_passed": bool(gate_d["passed"]),
        "scientific_failure_exit_zero": not (bool(gate_a["passed"]) and bool(gate_b["passed"])),
    }


def _recompute_audit(args: argparse.Namespace) -> dict[str, Any]:
    from ..rkme.reducer import ReducedRKME
    from .analysis import parse_context_bindings
    from .recompute import (
        assemble_stratified_recompute_result,
        load_verified_measurement_samples,
        recompute_oracle_gate_a_c,
        recompute_raw_numeric_subset,
    )
    from .taskspec import encode_measurement_dataset

    layout = _analysis_layout(args)
    manifest, contexts, _ = _frozen_state(layout)
    _require_private_gate0(layout)
    gate_d = _object(layout.analysis_artifact("gate_d.json"), "Gate D report")
    if gate_d.get("passed") is not True:
        raise V01CommandFailure("Gate D must pass before stratified recomputation")
    config = V01ExperimentConfig.from_dict(manifest["config"])
    base = _verify_analysis_base(layout, args.base_artifacts_root)
    assets = base.load_measurement_assets()
    contract = _object(layout.measurement_contract, "measurement contract")
    pair_plan = _object(layout.pair_plan, "pair plan")
    matrix = _object(layout.taskspec_matrix_axes, "TaskSpec matrix")
    audit_plan = _object(layout.audit_plan, "stratified audit plan")
    views: dict[str, MeasurementSchemaView] = {}
    for path in sorted((layout.measurement_dir / "schema_views").glob("*.json")):
        view = MeasurementSchemaView.from_dict(_object(path, "measurement schema view"))
        views[view.digest] = view

    def semantic_rebuilder(
        variant_id: str,
        _bank: int,
        dataset: EpisodeDataset,
        sidecar: VariantDatasetManifest,
    ) -> Any:
        try:
            view = views[sidecar.measurement_schema_view_digest]
        except KeyError as error:
            raise V01CommandFailure(
                f"audit cannot resolve schema view for {variant_id}"
            ) from error
        return encode_measurement_dataset(
            dataset,
            view,
            variant_id=variant_id,
            normalizer=assets.normalization,
            encoder=assets.encoder,
            max_action_dim=int(base.protocol.packed_layout["max_action_dim"]),
        )

    exact_chain = load_verified_measurement_samples(
        layout.measurement_dir,
        contract,
        pair_plan,
        semantic_rebuilder=semantic_rebuilder,
        verified_base_binding_digest=base.binding_digest,
        verified_base_asset_digests=base.asset_digests,
    )
    _, aggregation, _ = _verified_measurement_aggregation(
        layout,
        verified_base_binding_digest=base.binding_digest,
        verified_base_asset_digests=base.asset_digests,
        source_support_sizes=_source_support_sizes(base),
    )
    sources: dict[str, ReducedRKME] = {}
    for entry in base.public_pool.entries:
        spec = entry.task_spec
        sources[entry.opaque_id] = ReducedRKME(
            supports=spec.supports,
            beta=spec.beta,
            bandwidth=spec.kernel_bandwidth,
            rkme_norm2=spec.rkme_norm2,
            empirical_norm2=spec.rkme_norm2,
            reduction_error=0.0,
            protocol_id=spec.protocol_id,
        )
    bindings = parse_context_bindings(contexts, config=config)
    raw_subset = recompute_raw_numeric_subset(
        audit_plan,
        bindings=bindings,
        config=config,
        samples=exact_chain.samples,
        stored_matrix=matrix,
        kernel=assets.kernel,
        sources=sources,
        block_size=args.block_size,
        computation_backend=args.computation_backend,
        expected_source_count=6,
    )
    shard_paths = sorted((layout.oracle_private_dir / "shards").glob("*/*/*.json"))
    if not shard_paths:
        raise V01IncompleteArtifacts("no raw Oracle shards are available")
    candidates = _object(layout.candidates, "candidate manifest")
    identifiers = ProtocolIdentifiers.from_dict(manifest["protocol_identifiers"])
    oracle = recompute_oracle_gate_a_c(
        contexts,
        candidates,
        [_object(path, "oracle shard") for path in shard_paths],
        rebuilt_taskspec_matrix=aggregation.rebuilt_payload,
        config=config,
        analysis_seed_namespace=identifiers.experiment_protocol_id,
    )
    published_episode = _object(
        layout.oracle_episodes_axes, "published Oracle episode matrix"
    )
    published_aggregate = _object(
        layout.oracle_aggregates_json, "published Oracle aggregate matrix"
    )
    published_gate_a = _object(layout.analysis_artifact("gate_a.json"), "Gate A")
    published_gate_c = _object(
        layout.analysis_artifact("gate_c_diagnostic.json"), "Gate C"
    )
    oracle_consistency = {
        "episodes": sha256_json(oracle.oracle_episodes)
        == sha256_json(published_episode),
        "aggregates": sha256_json(oracle.oracle_aggregates)
        == sha256_json(published_aggregate),
        "gate_a": sha256_json(oracle.gate_a) == sha256_json(published_gate_a),
        "gate_c": sha256_json(list(oracle.gate_c_diagnostics))
        == sha256_json(published_gate_c.get("diagnostics")),
    }
    if not all(oracle_consistency.values()):
        raise V01CommandFailure(
            f"Oracle/Gate recomputation differs from published artifacts: {oracle_consistency}"
        )
    result = assemble_stratified_recompute_result(
        exact_chain, aggregation, raw_subset, oracle
    )
    payload = result.to_dict()
    payload["oracle_full_recompute"] = {
        **payload["oracle_full_recompute"],
        "published_output_consistency": oracle_consistency,
    }
    with _run_lock(layout.run_lock):
        layout.writer("analysis").publish_json(
            layout.analysis_artifact("recompute_audit.json"), payload, resume=args.resume
        )
    if not payload["passed"]:
        raise V01CommandFailure("stratified recomputation audit failed; diagnostics retained")
    return payload


def _build_report(args: argparse.Namespace) -> dict[str, Any]:
    from .report import V01Decision, decide_v01, render_summary

    layout = _analysis_layout(args)
    manifest, _, _ = _frozen_state(layout)
    is_formal = manifest.get("formal") is True
    if args.no_go_compute:
        from .execution_profile import (
            APPROVED_P5_SMOKE_CONFIG_DIGEST,
            verified_profile_projection,
            verify_execution_attempt,
        )

        if is_formal:
            raise V01CommandFailure(
                "NO_GO_COMPUTE preflight must be published from its smoke experiment root"
            )
        config = V01ExperimentConfig.from_dict(manifest["config"])
        if (
            manifest.get("config_digest") != APPROVED_P5_SMOKE_CONFIG_DIGEST
            or config.config_digest != APPROVED_P5_SMOKE_CONFIG_DIGEST
        ):
            raise V01CommandFailure(
                "NO_GO_COMPUTE requires the uniquely approved P5 smoke config"
            )
        # The digest above is the authority.  These redundant checks make the
        # resource-evidence coverage reviewable without copying task names or
        # private contexts into the completion artifact.
        if not (
            config.tasks.infrastructure == ("WalkerWalk",)
            and config.tasks.confirmatory == ()
            and len(config.shift.diagnostic_grid) == 5
            and config.probe.banks == 2
            and config.probe.max_episodes_per_bank == 16
            and config.probe.gate_b_unreduced_prefix == 16
        ):
            raise V01CommandFailure("approved P5 smoke coverage is inconsistent")
        smoke_coverage = {
            "config_digest": config.config_digest,
            "task_count": 1,
            "variant_count": 5,
            "probe_banks": 2,
            "semantic_dataset_count": 10,
            "episodes_per_bank": 16,
            "gate_prefix_episodes": 16,
            "expected_semantic_transitions": 160_000,
        }
        attempt_id = getattr(args, "execution_profile_attempt_id", None)
        if not isinstance(attempt_id, str) or not attempt_id:
            raise V01CommandFailure(
                "--no-go-compute requires --execution-profile-attempt-id"
            )
        base = _verify_analysis_base(layout, args.base_artifacts_root)
        profile = verify_execution_attempt(
            layout.measurement_dir,
            attempt_id,
            require_success=True,
            source_support_sizes=_source_support_sizes(base),
        )
        if (
            profile["execution"]["block_size"] != int(args.block_size)
            or profile["execution"]["computation_backend"]
            != str(args.computation_backend)
        ):
            raise V01CommandFailure(
                "NO_GO_COMPUTE CLI backend/block size differs from the selected profile"
            )
        workload = profile["workload"]
        if (
            workload["semantic_cache_hits"] != 0
            or workload["semantic_cache_misses"]
            != workload["semantic_dataset_count"]
        ):
            raise V01CommandFailure(
                "NO_GO_COMPUTE requires a fresh, full-coverage P5 smoke profile"
            )
        allowlist = assert_measurement_schema_allowlist(layout.measurement_dir)
        if allowlist.get("passed") is not True:
            raise V01CommandFailure(
                "NO_GO_COMPUTE measurement schema allowlist did not pass"
            )
        isolation = assert_measurement_isolation(
            layout.measurement_dir,
            config.gates.leakage.forbidden_measurement_fields,
        )
        if isolation.get("passed") is not True:
            raise V01CommandFailure(
                "NO_GO_COMPUTE measurement isolation audit did not pass"
            )
        profile_projection = verified_profile_projection(profile)
        measurement_audit_projection = {
            "schema_allowlist": {
                "schema": allowlist["schema"],
                "passed": True,
                "file_count": int(allowlist["file_count"]),
                "evidence_digest": sha256_json(allowlist),
            },
            "isolation": {
                "schema": isolation["schema"],
                "passed": True,
                "measurement_root_digest": isolation["measurement_root_digest"],
                "violation_count": 0,
                "evidence_digest": sha256_json(isolation),
            },
        }
        gate0 = _require_private_gate0(layout)
        decision = V01Decision(
            "NO_GO_COMPUTE",
            False,
            "The exact pre-registered TaskSpec computation was not approved or completed.",
        )
        preflight = {
            "schema": "policy-learnware.v01-preflight-completion-manifest.v0",
            "experiment_id": layout.experiment_id,
            "formal_run": is_formal,
            "decision": decision.to_dict(),
            "run_manifest_sha256": sha256_file(layout.run_manifest),
            "gate_0_attestation_sha256": sha256_file(
                layout.benchmark_private_dir / "gate_0_attestation.json"
            ),
            "gate_0_evidence_digest": sha256_json(gate0),
            "approved_smoke_coverage": smoke_coverage,
            "approved_smoke_coverage_digest": sha256_json(smoke_coverage),
            "execution_attempt_sha256": sha256_file(
                layout.execution_attempt(profile["execution_attempt_id"])
            ),
            "execution_profile": profile_projection,
            "execution_profile_digest": sha256_json(profile_projection),
            "measurement_audits": measurement_audit_projection,
            "measurement_audits_digest": sha256_json(measurement_audit_projection),
            "formal_completion_published": False,
        }
        with _run_lock(layout.run_lock):
            layout.writer("completion").publish_json(
                layout.preflight_completion_manifest, preflight, resume=args.resume
            )
        return {
            "decision": decision.to_dict(),
            "preflight": str(layout.preflight_completion_manifest),
        }
    if not is_formal:
        raise V01CommandFailure(
            "smoke runs cannot publish a formal v0.1 completion manifest"
        )

    # Release is a verification operation, not a trust boundary over mutable
    # JSON booleans.  All expected analysis artifacts must already exist; then
    # the typed scientific/Gate-D pipeline and stratified recomputation are run
    # again with immutable-resume semantics.  Any missing, minimal, forged or
    # drifted report fails byte-for-byte before a summary can be published.
    release_inputs = (
        layout.analysis_artifact("gate_0.json"),
        layout.analysis_artifact("gate_a.json"),
        layout.analysis_artifact("gate_b.json"),
        layout.analysis_artifact("gate_c_diagnostic.json"),
        layout.analysis_artifact("gate_d.json"),
        layout.analysis_artifact("join_audit.json"),
        layout.analysis_artifact("joined_taskspec_context.csv"),
        layout.analysis_artifact("joined_transfer_context.csv"),
        layout.analysis_artifact("recompute_audit.json"),
    )
    missing = [str(path) for path in release_inputs if not path.is_file()]
    if missing:
        raise V01IncompleteArtifacts(
            f"release revalidation requires pre-existing analysis artifacts: {missing}"
        )
    verification_args = argparse.Namespace(
        frozen_root=layout.frozen_dir,
        benchmark_private_root=layout.benchmark_private_dir,
        measurement_root=layout.measurement_dir,
        oracle_root=layout.oracle_private_dir,
        analysis_root=layout.analysis_dir,
        base_artifacts_root=args.base_artifacts_root,
        block_size=args.block_size,
        computation_backend=args.computation_backend,
        resume=True,
    )
    _evaluate_gates(verification_args)
    _recompute_audit(verification_args)

    identifiers = ProtocolIdentifiers.from_dict(manifest["protocol_identifiers"])
    gate0 = _object(layout.analysis_artifact("gate_0.json"), "Gate 0 report")
    gate_a = _object(layout.analysis_artifact("gate_a.json"), "Gate A report")
    gate_b = _object(layout.analysis_artifact("gate_b.json"), "Gate B report")
    gate_c = _object(layout.analysis_artifact("gate_c_diagnostic.json"), "Gate C diagnostic")
    gate_d = _object(layout.analysis_artifact("gate_d.json"), "Gate D report")
    recompute = _object(layout.analysis_artifact("recompute_audit.json"), "recompute audit")
    decision = decide_v01(
        gate_0=gate0,
        gate_a=gate_a,
        gate_b=gate_b,
        gate_d=gate_d,
        recompute_audit=recompute,
        no_go_compute=False,
    )
    summary = render_summary(
        experiment_id=layout.experiment_id,
        measurement_run_id=identifiers.measurement_run_id,
        oracle_protocol_id=identifiers.oracle_protocol_id,
        decision=decision,
        gate_0=gate0,
        gate_a=gate_a,
        gate_b=gate_b,
        gate_c=gate_c,
        gate_d=gate_d,
        recompute_audit=recompute,
    )
    analysis_inputs = {
        path.name: sha256_file(path)
        for path in (
            layout.analysis_artifact("gate_0.json"),
            layout.analysis_artifact("gate_a.json"),
            layout.analysis_artifact("gate_b.json"),
            layout.analysis_artifact("gate_c_diagnostic.json"),
            layout.analysis_artifact("gate_d.json"),
            layout.analysis_artifact("recompute_audit.json"),
        )
    }
    completion = {
        "schema": "policy-learnware.v01-completion-manifest.v0",
        "experiment_id": layout.experiment_id,
        "decision": decision.to_dict(),
        "run_manifest_sha256": sha256_file(layout.run_manifest),
        "analysis_inputs": analysis_inputs,
        "summary_sha256": sha256_bytes(summary.encode("utf-8")),
    }
    if not decision.formal_complete:
        raise V01CommandFailure(
            f"{decision.code} blocks completion manifest publication"
        )
    if not is_formal:
        raise V01CommandFailure(
            "a smoke run cannot be promoted to formal completion by its gate outcomes"
        )
    with _run_lock(layout.run_lock):
        layout.writer("analysis").publish_text(
            layout.analysis_artifact("summary.md"), summary, resume=args.resume
        )
        layout.writer("completion").publish_json(
            layout.completion_manifest, completion, resume=args.resume
        )
    return {"decision": decision.to_dict(), "completion": str(layout.completion_manifest)}


def _shard(values: Sequence[Any], index: int | None, count: int | None) -> list[Any]:
    if index is None and count is None:
        return list(values)
    if index is None or count is None:
        raise V01CommandFailure("--shard-index and --shard-count must be provided together")
    if count <= 0 or index < 0 or index >= count:
        raise V01CommandFailure("invalid shard coordinates")
    return [value for offset, value in enumerate(values) if offset % count == index]


def _dry_run(command: str, args: argparse.Namespace) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    if command in {"validate-config", "freeze-run"}:
        config = load_v01_experiment_config(args.config)
        inputs = {
            "config": str(args.config.expanduser().resolve()),
            "config_digest": config.config_digest,
            "experiment_id": config.experiment_id,
            "base_artifacts_root": str(args.base_artifacts_root.expanduser().resolve()),
            "formal": _is_formal(config),
            "registered_work": {
                "variants": len(config.tasks.all) * len(config.shift.diagnostic_grid),
                "probe_shards": len(config.tasks.all) * len(config.shift.diagnostic_grid) * config.probe.banks,
                "oracle_shards": len(config.tasks.all) * len(config.shift.diagnostic_grid) * config.base.candidates_per_task,
            },
        }
        if command == "freeze-run":
            inputs["artifacts_root"] = str(args.artifacts_root.expanduser().resolve())
    else:
        for name, value in sorted(vars(args).items()):
            if name in {"command", "dry_run", "resume", "handler"} or value is None:
                continue
            inputs[name] = str(value.expanduser().resolve()) if isinstance(value, Path) else value
    return {
        "schema": CLI_SCHEMA,
        "status": "dry-run",
        "command": command,
        "inputs": inputs,
        "writes_performed": False,
        "gpu_work_performed": False,
    }


def _execution_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true", help="validate and print the plan without writes/GPU work")
    parser.add_argument("--resume", action="store_true", help="verify and reuse byte-identical immutable work units")


def _parallel_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)


def _full_roots(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)


def _analysis_roots(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--benchmark-private-root", type=Path, required=True)
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="policy-learnware-v01")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config")
    validate.add_argument("--config", type=Path, required=True)
    validate.add_argument("--base-artifacts-root", type=Path, required=True)
    _execution_flags(validate)
    validate.set_defaults(handler=_validate_config)

    freeze = subparsers.add_parser("freeze-run")
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--base-artifacts-root", type=Path, required=True)
    freeze.add_argument("--artifacts-root", type=Path, required=True)
    _execution_flags(freeze)
    freeze.set_defaults(handler=_freeze_run)

    audit = subparsers.add_parser("audit-variants")
    _full_roots(audit)
    audit.add_argument("--base-artifacts-root", type=Path, required=True)
    audit.add_argument("--fpo-root", type=Path, required=True)
    audit.add_argument("--runs-root", type=Path)
    _execution_flags(audit)
    audit.set_defaults(handler=_audit_variants)

    probes = subparsers.add_parser("collect-probes")
    _full_roots(probes)
    _execution_flags(probes)
    _parallel_flags(probes)
    probes.set_defaults(handler=_collect_probes)

    taskspec = subparsers.add_parser("compute-taskspec-matrix")
    taskspec.add_argument("--base-artifacts-root", type=Path, required=True)
    taskspec.add_argument("--measurement-root", type=Path, required=True)
    taskspec.add_argument("--block-size", type=int, default=2048)
    taskspec.add_argument("--computation-backend", choices=("numpy", "jax"), default="jax")
    _execution_flags(taskspec)
    taskspec.set_defaults(handler=_compute_taskspec)

    oracle = subparsers.add_parser("evaluate-oracle")
    oracle.add_argument("--frozen-root", type=Path, required=True)
    oracle.add_argument("--benchmark-private-root", type=Path, required=True)
    oracle.add_argument("--oracle-root", type=Path, required=True)
    oracle.add_argument("--fpo-root", type=Path, required=True)
    oracle.add_argument("--runs-root", type=Path)
    _execution_flags(oracle)
    _parallel_flags(oracle)
    oracle.set_defaults(handler=_evaluate_oracle)

    gates = subparsers.add_parser("evaluate-gates")
    _analysis_roots(gates)
    gates.add_argument("--base-artifacts-root", type=Path, required=True)
    gates.add_argument("--block-size", type=int, default=2048)
    gates.add_argument(
        "--computation-backend", choices=("numpy", "jax"), default="jax"
    )
    _execution_flags(gates)
    gates.set_defaults(handler=_evaluate_gates)

    recompute = subparsers.add_parser("audit-recompute")
    _analysis_roots(recompute)
    recompute.add_argument("--base-artifacts-root", type=Path, required=True)
    recompute.add_argument("--block-size", type=int, default=2048)
    recompute.add_argument(
        "--computation-backend", choices=("numpy", "jax"), default="jax"
    )
    _execution_flags(recompute)
    recompute.set_defaults(handler=_recompute_audit)

    report = subparsers.add_parser("build-report")
    _analysis_roots(report)
    report.add_argument("--base-artifacts-root", type=Path, required=True)
    report.add_argument("--block-size", type=int, default=2048)
    report.add_argument(
        "--computation-backend", choices=("numpy", "jax"), default="jax"
    )
    report.add_argument("--no-go-compute", action="store_true")
    report.add_argument("--execution-profile-attempt-id")
    _execution_flags(report)
    report.set_defaults(handler=_build_report)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.dry_run and args.resume:
        raise V01CommandFailure("--dry-run and --resume are mutually exclusive")
    if hasattr(args, "block_size") and args.block_size <= 0:
        raise V01CommandFailure("--block-size must be positive")
    if hasattr(args, "shard_index"):
        _shard([], args.shard_index, args.shard_count)
    if hasattr(args, "devices") and args.devices != "auto":
        raise V01CommandFailure(
            "explicit device orchestration is not certified in v0.1; use --devices auto"
        )
    if getattr(args, "no_go_compute", False):
        attempt_id = getattr(args, "execution_profile_attempt_id", None)
        if not isinstance(attempt_id, str) or not attempt_id:
            raise V01CommandFailure(
                "--no-go-compute requires --execution-profile-attempt-id"
            )
    elif getattr(args, "execution_profile_attempt_id", None) is not None:
        raise V01CommandFailure(
            "--execution-profile-attempt-id is valid only with --no-go-compute"
        )


HANDLERS: Mapping[str, Callable[[argparse.Namespace], dict[str, Any]]] = {
    "validate-config": _validate_config,
    "freeze-run": _freeze_run,
    "audit-variants": _audit_variants,
    "collect-probes": _collect_probes,
    "compute-taskspec-matrix": _compute_taskspec,
    "evaluate-oracle": _evaluate_oracle,
    "evaluate-gates": _evaluate_gates,
    "audit-recompute": _recompute_audit,
    "build-report": _build_report,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = str(args.command)
    try:
        _validate_args(args)
        if args.dry_run:
            _emit(_dry_run(command, args))
            return 0
        result = HANDLERS[command](args)
        _emit(
            {
                "schema": CLI_SCHEMA,
                "status": "ok",
                "command": command,
                "result": result,
            }
        )
        return 0
    except (
        ArtifactExistsError,
        BaseRuntimeBindingError,
        V01ArtifactLayoutError,
        V01CommandFailure,
        V01ConfigError,
    ) as error:
        _emit(
            {
                "schema": CLI_SCHEMA,
                "status": "error",
                "command": command,
                "error_type": type(error).__name__,
                "message": str(error),
                "fail_closed": True,
            },
            stream=sys.stderr,
        )
        return 1
    except Exception as error:  # Every unexpected dependency/runtime error is fail closed.
        _emit(
            {
                "schema": CLI_SCHEMA,
                "status": "error",
                "command": command,
                "error_type": type(error).__name__,
                "message": str(error),
                "fail_closed": True,
            },
            stream=sys.stderr,
        )
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
