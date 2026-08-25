#!/usr/bin/env python3
"""Materialize one reviewed source-anchor specification into an immutable manifest.

This module contains no task, axis, factor, or index catalogue.  A nominal
specification and a shifted specification are a strict discriminated union;
every shifted scientific literal, including the exact flattened model
indices and the package-side axis binding, must already have been reviewed.
The live checkout/runtime are verified before the native registry is touched.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

try:  # Python >=3.11; the fallback keeps dependency-light local CI usable.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 test interpreter
    import tomli as tomllib

import numpy as np

try:  # Package import for tests/``python -m``; fallback for direct script use.
    from .anchor_binding import (
        ANCHOR_MANIFEST_SCHEMA,
        ANCHOR_OPERATOR_SCHEMA,
        SUPPORTED_MODEL_LEAVES,
        AnchorBindingError,
        AnchorManifest,
        array_digest,
        bind_model_to_anchor,
        derive_live_model_diff,
        finalize_anchor_manifest,
        snapshot_model,
    )
    from .provenance import (
        ContractError,
        NumericalIntegrityError,
        atomic_write_json,
        json_ready,
        load_strict_json,
        require_digest,
        require_exact_keys,
        require_git_commit,
        require_safe_id,
        runtime_contract_projection,
        sha256_json,
    )
except ImportError:  # pragma: no cover - direct executable use
    from anchor_binding import (
        ANCHOR_MANIFEST_SCHEMA,
        ANCHOR_OPERATOR_SCHEMA,
        SUPPORTED_MODEL_LEAVES,
        AnchorBindingError,
        AnchorManifest,
        array_digest,
        bind_model_to_anchor,
        derive_live_model_diff,
        finalize_anchor_manifest,
        snapshot_model,
    )
    from provenance import (
        ContractError,
        NumericalIntegrityError,
        atomic_write_json,
        json_ready,
        load_strict_json,
        require_digest,
        require_exact_keys,
        require_git_commit,
        require_safe_id,
        runtime_contract_projection,
        sha256_json,
    )


REVIEWED_ANCHOR_SPEC_SCHEMA = "policy-learnware.v02-reviewed-anchor-spec.v0"
AXIS_ANCHOR_BINDING_SCHEMA = "policy-learnware.v02-axis-anchor-binding.v0"
BACKEND = "mujoco_playground.registry"

_COMMON_SPEC_KEYS = {
    "schema",
    "kind",
    "task",
    "backend",
    "registry_config",
    "runtime",
}
_SHIFTED_SPEC_KEYS = _COMMON_SPEC_KEYS | {"axis", "operator", "axis_binding"}
_RUNTIME_KEYS = {
    "fpo_commit",
    "python_major_minor",
    "jax",
    "jaxlib",
    "mujoco",
    "playground",
}
_MASS_INERTIA_OPERATOR = "mass_inertia_scale_v02"


CommitResolver = Callable[[], str]
RuntimeResolver = Callable[[str], Mapping[str, Any]]


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{where} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ContractError(f"{where} keys must be strings")
    return dict(value)


def _positive_nonunit(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0 or result == 1.0:
        raise ContractError(f"{where} must be finite, positive, and non-unit")
    return result


def _validate_runtime(value: Any) -> dict[str, Any]:
    runtime = _mapping(value, "reviewed anchor runtime")
    require_exact_keys(runtime, _RUNTIME_KEYS, "reviewed anchor runtime")
    require_git_commit(runtime["fpo_commit"], "reviewed anchor runtime.fpo_commit")
    for name in _RUNTIME_KEYS - {"fpo_commit"}:
        if not isinstance(runtime[name], str) or not runtime[name]:
            raise ContractError(f"reviewed anchor runtime.{name} must be non-empty")
    return runtime


def _validate_axis_binding(
    value: Any, *, axis_id: str, factor_id: str, operator_source_digest: str
) -> dict[str, Any]:
    binding = _mapping(value, "reviewed axis binding")
    fields = {
        "schema",
        "axis_binding_digest",
        "axis_id",
        "factor_id",
        "operator_digest",
        "model_diff_digest",
    }
    require_exact_keys(binding, fields, "reviewed axis binding")
    if binding["schema"] != AXIS_ANCHOR_BINDING_SCHEMA:
        raise ContractError(f"unsupported axis binding schema: {binding['schema']!r}")
    require_safe_id(binding["axis_id"], "axis binding.axis_id")
    require_safe_id(binding["factor_id"], "axis binding.factor_id")
    require_digest(binding["operator_digest"], "axis binding.operator_digest")
    require_digest(binding["model_diff_digest"], "axis binding.model_diff_digest")
    actual = require_digest(binding["axis_binding_digest"], "axis_binding_digest")
    material = {name: binding[name] for name in fields - {"axis_binding_digest"}}
    if actual != sha256_json(material):
        raise ContractError("axis_binding_digest does not match its canonical fields")
    if binding["axis_id"] != axis_id:
        raise ContractError("axis binding and reviewed axis_id disagree")
    if binding["factor_id"] != factor_id:
        raise ContractError("axis binding and reviewed factor_id disagree")
    if binding["operator_digest"] != operator_source_digest:
        raise ContractError("axis binding and reviewed operator source digest disagree")
    return binding


def _validate_mutations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ContractError("shifted reviewed operator requires non-empty mutations")
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        item = _mapping(raw, f"reviewed mutation[{index}]")
        require_exact_keys(item, {"leaf", "flat_indices"}, f"reviewed mutation[{index}]")
        leaf = item["leaf"]
        if leaf not in SUPPORTED_MODEL_LEAVES:
            raise ContractError(f"unsupported/unallowlisted model leaf: {leaf!r}")
        indices = item["flat_indices"]
        if (
            not isinstance(indices, list)
            or not indices
            or any(isinstance(entry, bool) or not isinstance(entry, int) for entry in indices)
            or indices != sorted(set(indices))
            or indices[0] < 0
        ):
            raise ContractError(
                "reviewed mutation flat_indices must be sorted unique nonnegative integers"
            )
        result.append({"leaf": leaf, "flat_indices": list(indices)})
    leaves = [item["leaf"] for item in result]
    if leaves != sorted(set(leaves)):
        raise ContractError("reviewed mutations must have unique leaves in sorted order")
    return result


def validate_reviewed_anchor_spec(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the strict nominal/shifted input union without filling literals."""

    value = _mapping(raw, "reviewed anchor specification")
    kind = value.get("kind")
    if kind == "nominal":
        require_exact_keys(value, _COMMON_SPEC_KEYS, "nominal reviewed anchor specification")
    elif kind == "shifted":
        require_exact_keys(value, _SHIFTED_SPEC_KEYS, "shifted reviewed anchor specification")
    else:
        raise ContractError("reviewed anchor kind must be exactly 'nominal' or 'shifted'")
    if value["schema"] != REVIEWED_ANCHOR_SPEC_SCHEMA:
        raise ContractError(f"unsupported reviewed anchor schema: {value['schema']!r}")
    require_safe_id(value["task"], "reviewed anchor task")
    if value["backend"] != BACKEND:
        raise ContractError(f"reviewed anchor backend must be {BACKEND}")
    config = _mapping(value["registry_config"], "reviewed registry_config")
    config = json_ready(config)
    runtime = _validate_runtime(value["runtime"])
    result = {
        "schema": value["schema"],
        "kind": kind,
        "task": value["task"],
        "backend": value["backend"],
        "registry_config": config,
        "runtime": runtime,
    }
    if kind == "nominal":
        return result

    axis = _mapping(value["axis"], "reviewed axis")
    require_exact_keys(axis, {"axis_id", "axis_registry_digest"}, "reviewed axis")
    axis_id = require_safe_id(axis["axis_id"], "reviewed axis.axis_id")
    axis_registry_digest = require_digest(
        axis["axis_registry_digest"], "reviewed axis.axis_registry_digest"
    )

    operator = _mapping(value["operator"], "reviewed operator")
    require_exact_keys(
        operator,
        {"operator_id", "operator_source_digest", "factor_id", "factor", "mutations"},
        "reviewed operator",
    )
    operator_id = require_safe_id(operator["operator_id"], "reviewed operator.operator_id")
    operator_source_digest = require_digest(
        operator["operator_source_digest"], "reviewed operator.operator_source_digest"
    )
    factor_id = require_safe_id(operator["factor_id"], "reviewed operator.factor_id")
    factor = _positive_nonunit(operator["factor"], "reviewed operator.factor")
    mutations = _validate_mutations(operator["mutations"])
    binding = _validate_axis_binding(
        value["axis_binding"],
        axis_id=axis_id,
        factor_id=factor_id,
        operator_source_digest=operator_source_digest,
    )
    result.update(
        {
            "axis": {
                "axis_id": axis_id,
                "axis_registry_digest": axis_registry_digest,
            },
            "operator": {
                "operator_id": operator_id,
                "operator_source_digest": operator_source_digest,
                "factor_id": factor_id,
                "factor": factor,
                "mutations": mutations,
            },
            "axis_binding": binding,
        }
    )
    return result


def _host_array(value: Any, *, leaf: str) -> np.ndarray:
    try:
        import jax
    except ImportError:
        host = value
    else:  # pragma: no branch - exercised on the pinned production runtime
        host = jax.device_get(value)
    array = np.asarray(host)
    if array.dtype.hasobject or array.dtype.kind != "f" or array.size == 0:
        raise AnchorBindingError(f"{leaf} must be a non-empty floating array")
    if not bool(np.all(np.isfinite(array))):
        raise NumericalIntegrityError(f"{leaf} contains non-finite values")
    return np.ascontiguousarray(array)


def _scaled_leaf(value: Any, *, leaf: str, indices: Sequence[int], factor: float) -> Any:
    before = _host_array(value, leaf=leaf)
    selected = np.asarray(indices, dtype=np.int64)
    if int(selected[-1]) >= before.size:
        raise AnchorBindingError(f"{leaf} reviewed flat index exceeds leaf size")
    flat = before.reshape(-1).copy()
    selected_before = flat[selected].copy()
    flat[selected] *= np.asarray(factor, dtype=before.dtype)
    after = flat.reshape(before.shape)
    if not bool(np.all(np.isfinite(after))):
        raise NumericalIntegrityError(f"{leaf} scaling produced non-finite values")
    if not bool(np.all(np.not_equal(selected_before, after.reshape(-1)[selected]))):
        raise AnchorBindingError(f"{leaf} selected entries did not all change")
    mask = np.ones(before.size, dtype=bool)
    mask[selected] = False
    if not np.array_equal(before.reshape(-1)[mask], after.reshape(-1)[mask]):
        raise AnchorBindingError(f"{leaf} changed outside reviewed flat indices")
    if type(value).__module__.startswith(("jax", "jaxlib")):
        import jax.numpy as jnp

        return jnp.asarray(after)
    return after


def _validate_mass_inertia_coupling(model: Any, operator: Mapping[str, Any]) -> None:
    if operator["operator_id"] != _MASS_INERTIA_OPERATOR:
        return
    mutations = {item["leaf"]: item["flat_indices"] for item in operator["mutations"]}
    expected = {"_mjx_model.body_inertia", "_mjx_model.body_mass"}
    if set(mutations) != expected:
        raise ContractError("mass/inertia operator requires exactly body_mass and body_inertia")
    mass = _host_array(model.body_mass, leaf="_mjx_model.body_mass")
    inertia = _host_array(model.body_inertia, leaf="_mjx_model.body_inertia")
    if mass.ndim != 1 or inertia.ndim < 2 or inertia.shape[0] != mass.shape[0]:
        raise AnchorBindingError("live mass/inertia leaves have an incompatible coupled shape")
    mass_rows = tuple(mutations["_mjx_model.body_mass"])
    trailing = int(np.prod(inertia.shape[1:], dtype=np.int64))
    expected_inertia = tuple(
        row * trailing + component
        for row in mass_rows
        for component in range(trailing)
    )
    if tuple(mutations["_mjx_model.body_inertia"]) != expected_inertia:
        raise ContractError(
            "mass/inertia reviewed flat indices must couple every principal component "
            "for exactly the reviewed mass rows"
        )


def _environment_class(env: Any) -> str:
    return f"{type(env).__module__}.{type(env).__qualname__}"


def materialize_anchor_manifest(
    reviewed_spec: Mapping[str, Any],
    *,
    registry: Any,
    resolve_commit: CommitResolver,
    resolve_runtime: RuntimeResolver,
) -> dict[str, Any]:
    """Materialize and independently re-bind one reviewed specification.

    ``resolve_commit`` and ``resolve_runtime`` are mandatory dependency
    injection points so dependency-light tests can prove validation order.
    Production callers should use :func:`materialize_anchor_manifest_file`.
    """

    spec = validate_reviewed_anchor_spec(reviewed_spec)

    # Provenance must close before any registry config/environment call.
    actual_commit = require_git_commit(resolve_commit(), "live FPO commit")
    frozen_commit = spec["runtime"]["fpo_commit"]
    if actual_commit != frozen_commit:
        raise ContractError(
            f"upstream FPO commit mismatch: actual={actual_commit}, frozen={frozen_commit}"
        )
    actual_runtime = json_ready(resolve_runtime(actual_commit))
    if not isinstance(actual_runtime, dict):
        raise ContractError("runtime resolver must return an object")
    _validate_runtime(actual_runtime)
    if actual_runtime != spec["runtime"]:
        raise ContractError(
            f"native runtime contract mismatch: actual={actual_runtime}, frozen={spec['runtime']}"
        )

    config = registry.get_default_config(spec["task"])
    live_config = json_ready(config)
    if live_config != spec["registry_config"]:
        raise AnchorBindingError("live registry default config differs from reviewed config")
    env = registry.load(spec["task"], config=config)
    if not hasattr(env, "_mjx_model"):
        raise AnchorBindingError("registry environment lacks pinned _mjx_model")
    model = env._mjx_model
    nominal_snapshot = snapshot_model(model)

    if spec["kind"] == "nominal":
        _, live_model_diff_digest = derive_live_model_diff(model, model)
        raw = {
            "schema": ANCHOR_MANIFEST_SCHEMA,
            "task": spec["task"],
            "backend": spec["backend"],
            "nominal": True,
            "factor": 1.0,
            "environment_class": _environment_class(env),
            "registry_config": spec["registry_config"],
            "runtime": spec["runtime"],
            "expected_nominal_model_digest": nominal_snapshot.digest,
            "expected_bound_model_digest": nominal_snapshot.digest,
            "operator": None,
            "axis_binding_digest": None,
        }
    else:
        operator_review = spec["operator"]
        _validate_mass_inertia_coupling(model, operator_review)
        replacements: dict[str, Any] = {}
        mutation_rows: list[dict[str, Any]] = []
        for selection in operator_review["mutations"]:
            leaf = selection["leaf"]
            field = leaf.removeprefix("_mjx_model.")
            if "." in field or not hasattr(model, field):
                raise AnchorBindingError(f"live model lacks direct allowlisted field {field!r}")
            before = getattr(model, field)
            after = _scaled_leaf(
                before,
                leaf=leaf,
                indices=selection["flat_indices"],
                factor=operator_review["factor"],
            )
            replacements[field] = after
            mutation_rows.append(
                {
                    "leaf": leaf,
                    "flat_indices": selection["flat_indices"],
                    "multiplier": operator_review["factor"],
                    "expected_before_digest": array_digest(before),
                    "expected_after_digest": array_digest(after),
                }
            )
        tree_replace = getattr(model, "tree_replace", None)
        if not callable(tree_replace):
            raise AnchorBindingError("native model lacks immutable tree_replace")
        bound_model = tree_replace(replacements)
        if bound_model is model:
            raise AnchorBindingError("tree_replace returned the nominal model in place")
        if snapshot_model(model).digest != nominal_snapshot.digest:
            raise AnchorBindingError("reviewed operator mutated the nominal model in place")
        bound_snapshot = snapshot_model(bound_model)
        live_model_diff, live_model_diff_digest = derive_live_model_diff(
            model, bound_model
        )
        before_by_path = nominal_snapshot.by_path
        after_by_path = bound_snapshot.by_path
        if set(before_by_path) != set(after_by_path):
            raise AnchorBindingError("model pytree structure changed while materializing anchor")
        changed = tuple(
            sorted(path for path in before_by_path if before_by_path[path] != after_by_path[path])
        )
        expected_changed = tuple(
            sorted(item["leaf"].removeprefix("_mjx_model.") for item in mutation_rows)
        )
        if changed != expected_changed:
            raise AnchorBindingError(
                f"materialized model diff escaped allowlist: expected={expected_changed}, "
                f"observed={changed}"
            )
        expected_indices = {
            item["leaf"]: item["flat_indices"] for item in mutation_rows
        }
        actual_indices = {
            item["leaf"]: item["changed_flat_indices"]
            for item in live_model_diff["changes"]
        }
        if actual_indices != expected_indices:
            raise AnchorBindingError(
                "materialized live model diff indices disagree with reviewed indices"
            )
        if live_model_diff_digest != spec["axis_binding"]["model_diff_digest"]:
            raise ContractError(
                "reviewed axis binding model_diff_digest disagrees with the live model diff"
            )
        operator = {
            "schema": ANCHOR_OPERATOR_SCHEMA,
            "operator_id": operator_review["operator_id"],
            "axis_id": spec["axis"]["axis_id"],
            "axis_registry_digest": spec["axis"]["axis_registry_digest"],
            "factor": operator_review["factor"],
            "mutations": mutation_rows,
        }
        raw = {
            "schema": ANCHOR_MANIFEST_SCHEMA,
            "task": spec["task"],
            "backend": spec["backend"],
            "nominal": False,
            "factor": operator_review["factor"],
            "environment_class": _environment_class(env),
            "registry_config": spec["registry_config"],
            "runtime": spec["runtime"],
            "expected_nominal_model_digest": nominal_snapshot.digest,
            "expected_bound_model_digest": bound_snapshot.digest,
            "operator": operator,
            "axis_binding_digest": spec["axis_binding"]["axis_binding_digest"],
        }

    manifest_raw = finalize_anchor_manifest(raw)
    if manifest_raw["model_diff_digest"] != live_model_diff_digest:
        raise AnchorBindingError(
            "finalized anchor model_diff_digest disagrees with the live model diff"
        )
    manifest = AnchorManifest.from_dict(manifest_raw)
    rebound, audit = bind_model_to_anchor(model, manifest)
    if snapshot_model(model).digest != nominal_snapshot.digest or not audit.source_unchanged:
        raise AnchorBindingError("independent anchor re-bind changed the nominal source")
    if snapshot_model(rebound).digest != manifest.expected_bound_model_digest:
        raise AnchorBindingError("independent anchor re-bind did not reproduce bound digest")
    return manifest.to_dict()


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ContractError(f"cannot verify pinned FPO checkout {root}: {error}") from error
    return result.stdout.strip()


def _checkout_resolver(fpo_root: Path) -> CommitResolver:
    def resolve() -> str:
        dirty = _git(fpo_root, "status", "--porcelain", "--untracked-files=no")
        if dirty:
            raise ContractError("upstream FPO tracked files are dirty")
        return _git(fpo_root, "rev-parse", "HEAD")

    return resolve


def verify_pinned_playground_dependency(fpo_root: Path) -> str:
    """Require the checkout to pin the exact installed registry distribution."""

    pyproject = fpo_root / "playground" / "pyproject.toml"
    if pyproject.is_symlink() or not pyproject.is_file():
        raise ContractError(f"FPO checkout lacks a regular playground/pyproject.toml: {fpo_root}")
    try:
        payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        dependencies = payload["project"]["dependencies"]
    except (OSError, UnicodeError, tomllib.TOMLDecodeError, KeyError, TypeError) as error:
        raise ContractError(f"cannot read FPO playground dependency pin: {error}") from error
    if not isinstance(dependencies, list) or any(
        not isinstance(item, str) for item in dependencies
    ):
        raise ContractError("FPO playground dependencies must be a TOML string array")
    try:
        installed = importlib.metadata.version("playground")
    except importlib.metadata.PackageNotFoundError as error:
        raise ContractError("installed playground distribution is unavailable") from error
    exact = f"playground=={installed}"
    if exact not in dependencies:
        raise ContractError(
            "FPO checkout does not pin the exact installed playground version: "
            f"expected {exact!r}"
        )
    return installed


def _load_native_registry(fpo_root: Path) -> Any:
    source_dir = fpo_root / "playground" / "src"
    if not (source_dir / "flow_policy").is_dir():
        raise ContractError(f"not a pinned FPO/GoRL checkout: {fpo_root}")
    verify_pinned_playground_dependency(fpo_root)
    source_text = str(source_dir)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    # The FPO commit pins ``playground==X`` while the distribution supplies the
    # registry package; importing the suite performs its native registrations.
    from mujoco_playground import dm_control_suite, registry

    del dm_control_suite
    return registry


def materialize_anchor_manifest_file(
    *,
    spec_path: Path | str,
    output_path: Path | str,
    fpo_root: Path | str,
    registry_loader: Callable[[Path], Any] = _load_native_registry,
) -> dict[str, Any]:
    """Verify production provenance, materialize, then write without overwrite."""

    reviewed = load_strict_json(spec_path)
    spec = validate_reviewed_anchor_spec(reviewed)
    root = Path(fpo_root).resolve()
    commit_resolver = _checkout_resolver(root)

    # Resolve commit/runtime before importing or touching the native registry.
    commit = require_git_commit(commit_resolver(), "live FPO commit")
    if commit != spec["runtime"]["fpo_commit"]:
        raise ContractError(
            f"upstream FPO commit mismatch: actual={commit}, "
            f"frozen={spec['runtime']['fpo_commit']}"
        )
    actual_runtime = runtime_contract_projection(fpo_commit=commit)
    if actual_runtime != spec["runtime"]:
        raise ContractError(
            f"native runtime contract mismatch: actual={actual_runtime}, frozen={spec['runtime']}"
        )
    registry = registry_loader(root)
    manifest = materialize_anchor_manifest(
        spec,
        registry=registry,
        resolve_commit=lambda: commit,
        resolve_runtime=lambda _: actual_runtime,
    )
    atomic_write_json(output_path, manifest, overwrite=False)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="Reviewed strict anchor spec JSON")
    parser.add_argument("--output", type=Path, required=True, help="New immutable AnchorManifest")
    parser.add_argument("--fpo-root", type=Path, required=True, help="Clean pinned FPO/GoRL checkout")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = materialize_anchor_manifest_file(
        spec_path=args.spec,
        output_path=args.output,
        fpo_root=args.fpo_root,
    )
    print(
        f"wrote immutable anchor manifest; anchor_id={manifest['anchor_id']} "
        f"manifest_digest={manifest['manifest_digest']} output={args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AXIS_ANCHOR_BINDING_SCHEMA",
    "REVIEWED_ANCHOR_SPEC_SCHEMA",
    "materialize_anchor_manifest",
    "materialize_anchor_manifest_file",
    "validate_reviewed_anchor_spec",
]
