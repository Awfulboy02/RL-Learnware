"""Strict relocation-aware replay of the frozen v0.2 exact-90 handoff.

Historical receipts remain byte-identical.  The only permitted relaxation is
path *location*: paths are opened, and the selected bundle path is compared,
through the verified root relocation manifest.  All non-path semantics and
all content digests retain the original strict equality contract.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from policy_learnware_v0.v02.artifacts import (
    RelocationResolver,
    V02AssetError,
)

from .pool_acceptance import accept_policy_pool
from .provenance import (
    ContractError,
    load_strict_json,
    validate_self_digest,
)


_VOLATILE_ACCEPTANCE_KEYS = frozenset({"accepted_at", "report_digest"})
_PATH_CELL_KEYS = frozenset({"bundle_path"})


def _strict_explicit_acceptance_path(
    value: str | Path | None,
    *,
    canonical: Path,
) -> Path:
    """Accept only the canonical receipt, never an alternate evidence tree."""

    if value is None:
        return canonical
    raw = os.fspath(value)
    if (
        not raw.startswith("/")
        or "\\" in raw
        or "\0" in raw
        or "\n" in raw
        or "\r" in raw
        or any(part in {"", ".", ".."} for part in raw.split("/")[1:])
    ):
        raise ContractError("explicit policy-pool acceptance path is not normalized")
    candidate = Path(raw)
    if candidate != canonical:
        raise ContractError(
            "explicit policy-pool acceptance is outside the canonical exact90 asset"
        )
    return candidate


def _canonical_path(
    value: Any,
    *,
    resolver: RelocationResolver,
    where: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{where} must be a non-empty absolute path")
    try:
        return resolver.resolve(value, must_exist=True)
    except (OSError, V02AssetError, ValueError) as error:
        raise ContractError(f"{where} does not have a verified relocation: {error}") from error


def _require_acceptance_semantic_equality(
    stored: Mapping[str, Any],
    recomputed: Mapping[str, Any],
    *,
    resolver: RelocationResolver,
) -> None:
    """Compare one frozen receipt with one replay result, path-aware only once."""

    stored_keys = set(stored) - _VOLATILE_ACCEPTANCE_KEYS
    recomputed_keys = set(recomputed) - _VOLATILE_ACCEPTANCE_KEYS
    if stored_keys != recomputed_keys:
        raise ContractError("replayed policy-pool acceptance field inventory differs")
    for key in sorted(stored_keys - {"cells"}):
        if stored[key] != recomputed[key]:
            raise ContractError(f"replayed policy-pool acceptance differs at {key}")

    stored_cells = stored.get("cells")
    recomputed_cells = recomputed.get("cells")
    if not isinstance(stored_cells, Mapping) or not isinstance(
        recomputed_cells, Mapping
    ):
        raise ContractError("policy-pool acceptance cells must be objects")
    if set(stored_cells) != set(recomputed_cells):
        raise ContractError("replayed policy-pool cell inventory differs")
    for job_id in sorted(stored_cells):
        frozen_cell = stored_cells[job_id]
        replayed_cell = recomputed_cells[job_id]
        if not isinstance(frozen_cell, Mapping) or not isinstance(
            replayed_cell, Mapping
        ):
            raise ContractError(f"policy-pool cell {job_id} must be an object")
        if set(frozen_cell) != set(replayed_cell):
            raise ContractError(f"replayed policy-pool cell {job_id} fields differ")
        for key in sorted(frozen_cell):
            where = f"policy-pool cell {job_id}.{key}"
            if key in _PATH_CELL_KEYS:
                frozen_path = _canonical_path(
                    frozen_cell[key], resolver=resolver, where=where
                )
                replayed_path = _canonical_path(
                    replayed_cell[key], resolver=resolver, where=where
                )
                if frozen_path != replayed_path:
                    raise ContractError(f"replayed {where} resolves elsewhere")
            elif frozen_cell[key] != replayed_cell[key]:
                raise ContractError(f"replayed {where} differs")


def replay_relocated_policy_pool_acceptance(
    *,
    artifacts_root: str | Path | None = None,
    acceptance_path: str | Path | None = None,
    repository_root: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Revalidate and return the immutable exact-90 acceptance receipt.

    The resolver always reads ``relocation_manifest.json`` at the resolved
    artifacts root.  There is intentionally no manifest-path parameter and no
    per-version fallback.  ``acceptance_path`` may be supplied by a caller such
    as v0.4, but it must name the canonical receipt within the verified exact90
    target.
    """

    try:
        resolver = RelocationResolver.load(
            artifacts_root=artifacts_root,
            repository_root=repository_root,
            environ=environ,
        )
        exact90 = resolver.ensure_verified_asset("exact90", verify_bytes=True)
        formal_inputs = resolver.ensure_verified_asset(
            "formal_inputs", verify_bytes=True
        )
    except (OSError, V02AssetError, ValueError) as error:
        raise ContractError(f"cannot verify relocated exact90 asset: {error}") from error

    layout = resolver.layout
    if exact90 != layout.exact90.resolve():
        raise ContractError("verified exact90 mapping does not target the canonical layout")
    if formal_inputs != layout.formal_inputs.resolve():
        raise ContractError(
            "verified formal_inputs mapping does not target the canonical layout"
        )
    acceptance = _strict_explicit_acceptance_path(
        acceptance_path,
        canonical=layout.frozen_acceptance,
    )
    plan_path = layout.server_plan
    runs_root = layout.runs_root
    promotions_path = layout.promotions
    for where, path, expected_kind in (
        ("policy-pool acceptance", acceptance, "file"),
        ("server training plan", plan_path, "file"),
        ("compiled-parity promotions", promotions_path, "file"),
        ("server runs", runs_root, "directory"),
    ):
        if path.is_symlink():
            raise ContractError(f"{where} cannot be a symlink")
        exists = path.is_file() if expected_kind == "file" else path.is_dir()
        if not exists:
            raise ContractError(f"canonical {where} is missing")

    stored = load_strict_json(acceptance)
    validate_self_digest(
        stored,
        key="report_digest",
        where="frozen policy-pool acceptance",
    )
    plan = load_strict_json(plan_path)
    promotions = load_strict_json(promotions_path)
    recomputed = accept_policy_pool(
        server_plan=plan,
        runs_root=runs_root,
        promotion_manifest=promotions,
        path_resolver=resolver.resolve,
    )
    _require_acceptance_semantic_equality(
        stored,
        recomputed,
        resolver=resolver,
    )
    return dict(stored)


__all__ = ["replay_relocated_policy_pool_acceptance"]
