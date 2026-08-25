"""Independent raw-evidence recomputation for the v0.2 sidecar.

The entry point in this module deliberately accepts published results only as
hash-bound comparison snapshots.  Championization, selector outputs, oracle
metrics, hierarchical statistics, costs, and information-isolation audits are
rebuilt from their primitive inputs.  A caller-provided aggregate or ``passed``
flag is never used as scientific evidence.

The module is intentionally dependency-light: NumPy is the only dependency
outside the Python standard library and the v0.2 pure-data contracts.

Formal authority is deliberately unavailable until source-owned loaders can
reconstruct every typed input section from canonical raw artifacts and compare
exact input-projection digests.  A caller-created ``IndependentRecomputeInputs``
object plus plausible or decoy artifact references is not sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np

from ..hashing import canonicalize, sha256_json, sha256_ndarrays
from ..io import read_json
from .audit import (
    PublicArtifactRule,
    audit_evidence_contract,
    audit_oracle_independence,
    audit_public_artifacts,
    audit_public_market_entries,
)
from .axes import DynamicsOperatorAudit
from .baselines import FrozenSelectorArtifact, TargetQueryView
from .competence import (
    ChampionizationResult,
    SourceEpisodeRow,
    championize_by_anchor,
)
from .config import COMPETENCE_MODES, CompetenceMode
from .costs import CostRecord, reconcile_cold_warm_costs
from .gates import (
    CanonicalEvidenceRef,
    FormalGateEvidenceError,
    build_canonical_evidence_ref,
    verify_canonical_evidence_ref,
)
from .metrics import (
    HierarchicalValue,
    aggregate_hierarchy,
    compute_ranking_metrics,
    compute_selection_metrics,
)
from .oracle import (
    OracleEpisodeRow,
    PublishedSelection,
    aggregate_full_pool_oracle,
    minimum_executable_set,
)
from .representation import (
    ProbeTraceView,
    RepresentationBuildContract,
    build_environment_spec,
)
from .extensions.representation import (
    EncodedEpisodeDataset,
    SemanticEncoderProtocol,
)
from .schemas import ExecutionABIRecord, PublicMarketEntry
from .selectors import LMinSelector, PublicMarketView, SelectionRecord
from .statistics import (
    bootstrap_max_t_intervals,
    centered_one_sided_p_value,
    derive_bootstrap_seed,
    evaluate_noninferiority,
    hierarchical_bootstrap,
    hierarchical_paired_difference_bootstrap,
    holm_bonferroni,
)
from .variant_env import Gate0Audit, RolloutAudit


RECOMPUTE_SCHEMA = "policy-learnware.v02-independent-recompute-report.v0"
FORMAL_RECOMPUTE_SCHEMA = "policy-learnware.v02-formal-independent-recompute-report.v0"
FORMAL_RECOMPUTE_SOURCE_MANIFEST_SCHEMA = (
    "policy-learnware.v02-formal-recompute-source-manifest.v0"
)
FORMAL_RECOMPUTE_PROVENANCE_SCHEMA = (
    "policy-learnware.v02-formal-recompute-provenance.v0"
)
FORMAL_RECOMPUTE_DERIVATION_ID = (
    "policy_learnware_v0.v02.recompute:run_formal_independent_recompute/v0"
)
RECOMPUTE_SECTIONS = frozenset(
    {
        "source",
        "gate0",
        "representations",
        "selectors",
        "oracle",
        "statistics",
        "costs",
        "information",
    }
)
FORMAL_RECOMPUTE_SECTION_SOURCE_SCHEMAS: Mapping[str, str] = MappingProxyType(
    {
        "source": "policy-learnware.v02-formal-source-recompute-input.v0",
        "gate0": "policy-learnware.v02-formal-gate0-recompute-input.v0",
        "oracle": "policy-learnware.v02-formal-oracle-recompute-input.v0",
        "statistics": "policy-learnware.v02-formal-statistics-recompute-input.v0",
        "costs": "policy-learnware.v02-formal-costs-recompute-input.v0",
    }
)
RTOL = 0.0
ATOL = 0.0


class RecomputeContractError(ValueError):
    """A primitive, frozen coverage contract, or published snapshot disagrees."""


# Persisted JSON is evidence, not an execution capability.  Only the source-
# owned formal runner below can attach this live, deliberately unserializable
# authority to a report.
_TRUSTED_RECOMPUTE_AUTHORITY = object()


def _nonempty(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise RecomputeContractError(f"{where} must be a non-empty canonical string")
    return value


def _digest(value: Any, where: str) -> str:
    result = _nonempty(value, where).lower()
    if len(result) != 64:
        raise RecomputeContractError(f"{where} must be a SHA-256 digest")
    try:
        int(result, 16)
    except ValueError as error:
        raise RecomputeContractError(f"{where} must be a SHA-256 digest") from error
    return result


def _strict_json_mapping(
    value: object, expected: set[str], where: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecomputeContractError(f"{where} must be a JSON object")
    observed = set(value)
    if observed != expected:
        raise RecomputeContractError(
            f"{where} fields differ: missing={sorted(expected-observed)}, "
            f"unknown={sorted(observed-expected)}"
        )
    return value


def _experiment_id(value: Any, where: str = "experiment_id") -> str:
    result = _nonempty(value, where)
    if any(
        character
        not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-"
        for character in result
    ):
        raise RecomputeContractError(f"{where} is not a safe path segment")
    if result in {".", ".."}:
        raise RecomputeContractError(f"{where} is not a safe path segment")
    return result


def _finite(value: Any, where: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise RecomputeContractError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RecomputeContractError(f"{where} must be finite")
    return result


def _nonnegative_int(value: Any, where: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise RecomputeContractError(f"{where} must be an integer")
    result = int(value)
    if result < 0:
        raise RecomputeContractError(f"{where} must be non-negative")
    return result


def _canonical_tuple(values: Sequence[str], where: str) -> tuple[str, ...]:
    parsed = tuple(_nonempty(value, f"{where}[]") for value in values)
    if not parsed or len(parsed) != len(set(parsed)):
        raise RecomputeContractError(f"{where} must be non-empty and unique")
    return tuple(sorted(parsed))


def _deep_freeze(value: Any) -> Any:
    canonical = canonicalize(value)
    if isinstance(canonical, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in canonical.items()}
        )
    if isinstance(canonical, list):
        return tuple(_deep_freeze(item) for item in canonical)
    return canonical


def _jsonable(value: Any) -> Any:
    """Return ordinary JSON containers for report serialization."""

    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _first_mismatch(left: Any, right: Any, path: str = "$") -> str | None:
    """Locate an exact structural or numeric mismatch for actionable diagnostics."""

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return (
                f"{path} keys mismatch: expected={sorted(left)}, "
                f"recomputed={sorted(right)}"
            )
        for key in sorted(left):
            mismatch = _first_mismatch(left[key], right[key], f"{path}.{key}")
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return (
                f"{path} length mismatch: expected={len(left)}, recomputed={len(right)}"
            )
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            mismatch = _first_mismatch(left_item, right_item, f"{path}[{index}]")
            if mismatch is not None:
                return mismatch
        return None
    numeric = (
        isinstance(left, (int, float, np.integer, np.floating))
        and not isinstance(left, (bool, np.bool_))
        and isinstance(right, (int, float, np.integer, np.floating))
        and not isinstance(right, (bool, np.bool_))
    )
    if numeric:
        left_value = float(left)
        right_value = float(right)
        if (
            not math.isfinite(left_value)
            or not math.isfinite(right_value)
            or not math.isclose(left_value, right_value, rel_tol=RTOL, abs_tol=ATOL)
        ):
            return (
                f"{path} numeric mismatch: expected={left_value!r}, "
                f"recomputed={right_value!r}"
            )
        return None
    if type(left) is not type(right) or left != right:
        return f"{path} mismatch: expected={left!r}, recomputed={right!r}"
    return None


@dataclass(frozen=True)
class PublishedSnapshot:
    """A frozen payload and the digest supplied by an immutable manifest.

    The constructor intentionally does not bless the pair.  Independent
    recompute verifies both the snapshot's own digest and equality with the
    rebuilt payload.
    """

    payload: Mapping[str, Any]
    digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.payload, Mapping):
            raise RecomputeContractError("published snapshot payload must be a mapping")
        object.__setattr__(self, "payload", _deep_freeze(self.payload))
        object.__setattr__(
            self, "digest", _digest(self.digest, "published snapshot digest")
        )

    @classmethod
    def create(cls, payload: Mapping[str, Any]) -> "PublishedSnapshot":
        frozen = _deep_freeze(payload)
        return cls(payload=frozen, digest=sha256_json(frozen))


def _verify_snapshot(
    rebuilt: Mapping[str, Any], snapshot: PublishedSnapshot, *, where: str
) -> str:
    stored_payload_digest = sha256_json(snapshot.payload)
    if stored_payload_digest != snapshot.digest:
        raise RecomputeContractError(
            f"{where} published payload digest mismatch: "
            f"stored={snapshot.digest}, payload={stored_payload_digest}"
        )
    mismatch = _first_mismatch(snapshot.payload, rebuilt)
    if mismatch is not None:
        raise RecomputeContractError(f"{where} {mismatch}")
    rebuilt_digest = sha256_json(rebuilt)
    if rebuilt_digest != snapshot.digest:
        raise RecomputeContractError(
            f"{where} rebuilt digest mismatch: "
            f"published={snapshot.digest}, recomputed={rebuilt_digest}"
        )
    return rebuilt_digest


def selector_unit_id(method_id: str, query_id: str) -> str:
    return f"{_nonempty(method_id, 'method_id')}::{_nonempty(query_id, 'query_id')}"


@dataclass(frozen=True)
class FullCoverageContract:
    """Frozen work-unit universe; omission is a recompute failure."""

    source_anchor_ids: tuple[str, ...]
    source_market_bindings: Mapping[str, str]
    public_market_ids: tuple[str, ...]
    gate0_audit_ids: tuple[str, ...]
    representation_unit_ids: tuple[str, ...]
    selector_unit_ids: tuple[str, ...]
    oracle_query_ids: tuple[str, ...]
    oracle_unit_ids: tuple[str, ...]
    cost_query_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "source_anchor_ids",
            "public_market_ids",
            "gate0_audit_ids",
            "representation_unit_ids",
            "selector_unit_ids",
            "oracle_query_ids",
            "oracle_unit_ids",
            "cost_query_ids",
        ):
            object.__setattr__(self, name, _canonical_tuple(getattr(self, name), name))
        if not isinstance(self.source_market_bindings, Mapping):
            raise RecomputeContractError("source_market_bindings must be a mapping")
        bindings = {
            _nonempty(anchor, "source_market_bindings key"): _nonempty(
                opaque_id, f"source_market_bindings[{anchor!r}]"
            )
            for anchor, opaque_id in self.source_market_bindings.items()
        }
        if not bindings or not set(bindings).issubset(self.source_anchor_ids):
            raise RecomputeContractError(
                "source_market_bindings keys must be a non-empty source-anchor subset"
            )
        if set(bindings.values()) != set(self.public_market_ids):
            raise RecomputeContractError(
                "source_market_bindings must cover every public market ID exactly once"
            )
        if len(bindings.values()) != len(set(bindings.values())):
            raise RecomputeContractError("source_market_bindings must be one-to-one")
        if set(self.selector_unit_ids) != set(self.oracle_unit_ids):
            raise RecomputeContractError(
                "full statistical coverage requires one oracle unit per selector unit"
            )
        object.__setattr__(
            self,
            "source_market_bindings",
            MappingProxyType(dict(sorted(bindings.items()))),
        )

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v02-full-recompute-coverage.v0",
            "source_anchor_ids": list(self.source_anchor_ids),
            "source_market_bindings": dict(self.source_market_bindings),
            "public_market_ids": list(self.public_market_ids),
            "gate0_audit_ids": list(self.gate0_audit_ids),
            "representation_unit_ids": list(self.representation_unit_ids),
            "selector_unit_ids": list(self.selector_unit_ids),
            "oracle_query_ids": list(self.oracle_query_ids),
            "oracle_unit_ids": list(self.oracle_unit_ids),
            "cost_query_ids": list(self.cost_query_ids),
        }


def formal_recompute_source_manifest_relative_path() -> str:
    """Canonical provenance-manifest location for the formal raw replay."""

    return "analysis/formal_recompute_sources.json"


@dataclass(frozen=True)
class FormalRecomputeSourceManifest:
    """Exact config/run-bound source-artifact census for all replay sections."""

    experiment_id: str
    config_digest: str
    config_file_sha256: str
    coverage_contract_digest: str
    section_sources: Mapping[str, tuple[CanonicalEvidenceRef, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "experiment_id", _experiment_id(self.experiment_id))
        for name in (
            "config_digest",
            "config_file_sha256",
            "coverage_contract_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if (
            not isinstance(self.section_sources, Mapping)
            or set(self.section_sources) != RECOMPUTE_SECTIONS
        ):
            raise RecomputeContractError(
                "formal recompute source manifest must cover every section exactly"
            )
        normalized: dict[str, tuple[CanonicalEvidenceRef, ...]] = {}
        forbidden_schemas = {
            RECOMPUTE_SCHEMA,
            FORMAL_RECOMPUTE_SCHEMA,
            FORMAL_RECOMPUTE_SOURCE_MANIFEST_SCHEMA,
            FORMAL_RECOMPUTE_PROVENANCE_SCHEMA,
        }
        manifest_path = formal_recompute_source_manifest_relative_path()
        for section in sorted(RECOMPUTE_SECTIONS):
            refs = tuple(self.section_sources[section])
            if not refs or any(
                not isinstance(item, CanonicalEvidenceRef) for item in refs
            ):
                raise RecomputeContractError(
                    f"formal recompute section {section!r} has no typed sources"
                )
            paths = tuple(item.canonical_path for item in refs)
            if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
                raise RecomputeContractError(
                    f"formal recompute section {section!r} sources must be sorted unique"
                )
            for reference in refs:
                if reference.config_digest != self.config_digest:
                    raise RecomputeContractError(
                        "every formal recompute source must carry the exact config binding"
                    )
                if reference.canonical_path == manifest_path:
                    raise RecomputeContractError(
                        "formal recompute source manifest cannot cite itself"
                    )
                if reference.artifact_schema in forbidden_schemas:
                    raise RecomputeContractError(
                        "formal recompute reports/manifests cannot serve as raw sources"
                    )
            normalized[section] = refs
        object.__setattr__(self, "section_sources", MappingProxyType(normalized))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": FORMAL_RECOMPUTE_SOURCE_MANIFEST_SCHEMA,
            "experiment_id": self.experiment_id,
            "config_digest": self.config_digest,
            "config_file_sha256": self.config_file_sha256,
            "coverage_contract_digest": self.coverage_contract_digest,
            "section_sources": {
                section: [
                    reference.to_dict() for reference in self.section_sources[section]
                ]
                for section in sorted(RECOMPUTE_SECTIONS)
            },
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "FormalRecomputeSourceManifest":
        data = _strict_json_mapping(
            value,
            {
                "schema",
                "experiment_id",
                "config_digest",
                "config_file_sha256",
                "coverage_contract_digest",
                "section_sources",
            },
            "formal recompute source manifest",
        )
        if data["schema"] != FORMAL_RECOMPUTE_SOURCE_MANIFEST_SCHEMA:
            raise RecomputeContractError(
                "unsupported formal recompute source manifest schema"
            )
        raw_sections = data["section_sources"]
        if (
            not isinstance(raw_sections, Mapping)
            or set(raw_sections) != RECOMPUTE_SECTIONS
        ):
            raise RecomputeContractError(
                "formal recompute source manifest section coverage differs"
            )
        parsed: dict[str, tuple[CanonicalEvidenceRef, ...]] = {}
        for section in RECOMPUTE_SECTIONS:
            raw_refs = raw_sections[section]
            if not isinstance(raw_refs, list):
                raise RecomputeContractError(
                    f"formal recompute section {section!r} sources must be a list"
                )
            parsed[section] = tuple(
                CanonicalEvidenceRef.from_dict(item) for item in raw_refs
            )
        return cls(
            experiment_id=data["experiment_id"],
            config_digest=data["config_digest"],
            config_file_sha256=data["config_file_sha256"],
            coverage_contract_digest=data["coverage_contract_digest"],
            section_sources=parsed,
        )


def formal_recompute_evaluator_descriptor() -> Mapping[str, Any]:
    """Source-owned identity of the complete typed recomputation algorithm."""

    return MappingProxyType(
        {
            "schema": "policy-learnware.v02-formal-recompute-evaluator.v0",
            "derivation_id": FORMAL_RECOMPUTE_DERIVATION_ID,
            "sections": sorted(RECOMPUTE_SECTIONS),
            "checks": [
                "cost_recompute",
                "full_digest_coverage",
                "full_selector_replay",
                "full_statistical_recompute",
                "information_isolation",
                "raw_numeric_subset_coverage",
            ],
            "numeric_tolerance": {"relative": RTOL, "absolute": ATOL},
            "caller_aggregates_or_gates_consumed": False,
        }
    )


def formal_recompute_evaluator_digest() -> str:
    return sha256_json(formal_recompute_evaluator_descriptor())


# Loader registration lives below the typed input definitions.  Keeping the
# declaration there prevents a schema name from being mistaken for a complete
# inverse: registry membership requires executable parsing *and* an exact
# live-input projection comparison.


@dataclass(frozen=True)
class FormalRecomputeSourceBinding:
    """Verified live binding between a source manifest and its artifact tree."""

    experiment_root: Path
    manifest: FormalRecomputeSourceManifest
    manifest_ref: CanonicalEvidenceRef

    def __post_init__(self) -> None:
        root = Path(self.experiment_root).expanduser().resolve()
        if not root.is_dir():
            raise RecomputeContractError(
                f"formal recompute experiment root is not a directory: {root}"
            )
        object.__setattr__(self, "experiment_root", root)
        if not isinstance(self.manifest, FormalRecomputeSourceManifest):
            raise RecomputeContractError("formal recompute manifest has the wrong type")
        if not isinstance(self.manifest_ref, CanonicalEvidenceRef):
            raise RecomputeContractError(
                "formal recompute manifest ref has the wrong type"
            )
        if (
            self.manifest_ref.canonical_path
            != formal_recompute_source_manifest_relative_path()
            or self.manifest_ref.artifact_schema
            != FORMAL_RECOMPUTE_SOURCE_MANIFEST_SCHEMA
            or self.manifest_ref.artifact_digest != self.manifest.digest
            or self.manifest_ref.config_digest != self.manifest.config_digest
        ):
            raise RecomputeContractError(
                "formal recompute manifest reference is not canonically bound"
            )

    def verify_sources(self) -> Mapping[str, str]:
        """Re-read every byte and return per-section source-census digests."""

        result: dict[str, str] = {}
        for section in sorted(RECOMPUTE_SECTIONS):
            refs = self.manifest.section_sources[section]
            for reference in refs:
                try:
                    verify_canonical_evidence_ref(
                        reference,
                        experiment_root=self.experiment_root,
                        expected_config_digest=self.manifest.config_digest,
                        require_config_binding=True,
                        source_artifact=True,
                    )
                except FormalGateEvidenceError as error:
                    raise RecomputeContractError(
                        f"formal recompute source verification failed for {section}: {error}"
                    ) from error
            result[section] = sha256_json([reference.to_dict() for reference in refs])
        return MappingProxyType(result)


def load_formal_recompute_source_binding(
    manifest_path: str | Path,
    *,
    experiment_root: str | Path,
    expected_experiment_id: str,
    expected_config_digest: str,
    expected_config_file_sha256: str,
    expected_coverage_contract_digest: str,
) -> FormalRecomputeSourceBinding:
    """Load the canonical manifest and verify all exact source bytes."""

    root = Path(experiment_root).expanduser().resolve()
    supplied = Path(manifest_path).expanduser().resolve()
    expected = root.joinpath(
        *PurePosixPath(formal_recompute_source_manifest_relative_path()).parts
    ).resolve()
    if supplied != expected:
        raise RecomputeContractError(
            f"formal recompute source manifest must use canonical path {expected}"
        )
    try:
        reference = build_canonical_evidence_ref(
            supplied,
            experiment_root=root,
            expected_config_digest=expected_config_digest,
        )
        raw = verify_canonical_evidence_ref(
            reference,
            experiment_root=root,
            expected_config_digest=expected_config_digest,
            require_config_binding=True,
        )
    except FormalGateEvidenceError as error:
        raise RecomputeContractError(
            f"formal recompute manifest verification failed: {error}"
        ) from error
    manifest = FormalRecomputeSourceManifest.from_dict(raw)
    if canonicalize(raw) != manifest.to_dict():
        raise RecomputeContractError(
            "formal recompute source manifest is not canonical"
        )
    expected_values = {
        "experiment_id": _experiment_id(expected_experiment_id),
        "config_digest": _digest(expected_config_digest, "expected config_digest"),
        "config_file_sha256": _digest(
            expected_config_file_sha256, "expected config_file_sha256"
        ),
        "coverage_contract_digest": _digest(
            expected_coverage_contract_digest,
            "expected coverage_contract_digest",
        ),
    }
    for name, value in expected_values.items():
        if getattr(manifest, name) != value:
            raise RecomputeContractError(
                f"formal recompute source manifest {name} binding differs"
            )
    binding = FormalRecomputeSourceBinding(
        experiment_root=root,
        manifest=manifest,
        manifest_ref=reference,
    )
    binding.verify_sources()
    return binding


@dataclass(frozen=True)
class FormalRecomputeProvenance:
    """Persistable provenance emitted only after the live raw replay runs."""

    experiment_id: str
    config_digest: str
    config_file_sha256: str
    coverage_contract_digest: str
    source_manifest_ref: CanonicalEvidenceRef
    source_manifest_digest: str
    section_source_digests: Mapping[str, str]
    derivation_id: str
    evaluator_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "experiment_id", _experiment_id(self.experiment_id))
        for name in (
            "config_digest",
            "config_file_sha256",
            "coverage_contract_digest",
            "source_manifest_digest",
            "evaluator_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if not isinstance(self.source_manifest_ref, CanonicalEvidenceRef):
            raise RecomputeContractError(
                "formal provenance manifest ref has wrong type"
            )
        if (
            self.source_manifest_ref.canonical_path
            != formal_recompute_source_manifest_relative_path()
            or self.source_manifest_ref.artifact_schema
            != FORMAL_RECOMPUTE_SOURCE_MANIFEST_SCHEMA
            or self.source_manifest_ref.artifact_digest != self.source_manifest_digest
            or self.source_manifest_ref.config_digest != self.config_digest
        ):
            raise RecomputeContractError(
                "formal provenance source manifest binding differs"
            )
        if self.derivation_id != FORMAL_RECOMPUTE_DERIVATION_ID:
            raise RecomputeContractError(
                "formal provenance derivation_id is not canonical"
            )
        if self.evaluator_digest != formal_recompute_evaluator_digest():
            raise RecomputeContractError(
                "formal provenance evaluator digest differs from source registry"
            )
        if (
            not isinstance(self.section_source_digests, Mapping)
            or set(self.section_source_digests) != RECOMPUTE_SECTIONS
        ):
            raise RecomputeContractError(
                "formal provenance must bind every section source census"
            )
        object.__setattr__(
            self,
            "section_source_digests",
            MappingProxyType(
                {
                    section: _digest(
                        self.section_source_digests[section],
                        f"section_source_digests[{section!r}]",
                    )
                    for section in sorted(RECOMPUTE_SECTIONS)
                }
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": FORMAL_RECOMPUTE_PROVENANCE_SCHEMA,
            "experiment_id": self.experiment_id,
            "config_digest": self.config_digest,
            "config_file_sha256": self.config_file_sha256,
            "coverage_contract_digest": self.coverage_contract_digest,
            "source_manifest_ref": self.source_manifest_ref.to_dict(),
            "source_manifest_digest": self.source_manifest_digest,
            "section_source_digests": dict(self.section_source_digests),
            "derivation_id": self.derivation_id,
            "evaluator_digest": self.evaluator_digest,
            "in_process_authority_persisted": False,
        }

    @classmethod
    def from_dict(cls, value: object) -> "FormalRecomputeProvenance":
        data = _strict_json_mapping(
            value,
            {
                "schema",
                "experiment_id",
                "config_digest",
                "config_file_sha256",
                "coverage_contract_digest",
                "source_manifest_ref",
                "source_manifest_digest",
                "section_source_digests",
                "derivation_id",
                "evaluator_digest",
                "in_process_authority_persisted",
            },
            "formal recompute provenance",
        )
        if data["schema"] != FORMAL_RECOMPUTE_PROVENANCE_SCHEMA:
            raise RecomputeContractError(
                "unsupported formal recompute provenance schema"
            )
        if data["in_process_authority_persisted"] is not False:
            raise RecomputeContractError(
                "formal recompute authority cannot be persisted"
            )
        if not isinstance(data["section_source_digests"], Mapping):
            raise RecomputeContractError(
                "formal recompute section_source_digests must be a mapping"
            )
        return cls(
            experiment_id=data["experiment_id"],
            config_digest=data["config_digest"],
            config_file_sha256=data["config_file_sha256"],
            coverage_contract_digest=data["coverage_contract_digest"],
            source_manifest_ref=CanonicalEvidenceRef.from_dict(
                data["source_manifest_ref"]
            ),
            source_manifest_digest=data["source_manifest_digest"],
            section_source_digests=data["section_source_digests"],
            derivation_id=data["derivation_id"],
            evaluator_digest=data["evaluator_digest"],
        )


@dataclass(frozen=True)
class SourceRecomputeInputs:
    selection_rows: tuple[SourceEpisodeRow, ...]
    attestation_rows: tuple[SourceEpisodeRow, ...]
    competence_floors: Mapping[str, float]
    mean_tolerance: float
    lcb_z: float | None
    return_contract_id: str
    competence_mode: CompetenceMode
    published: PublishedSnapshot

    def __post_init__(self) -> None:
        selection = tuple(self.selection_rows)
        attestation = tuple(self.attestation_rows)
        if not selection or not attestation:
            raise RecomputeContractError(
                "source recompute requires selection and attestation episode rows"
            )
        if any(
            not isinstance(row, SourceEpisodeRow) for row in selection + attestation
        ):
            raise RecomputeContractError("source rows must be SourceEpisodeRow objects")
        floors = {
            _nonempty(anchor, "competence floor anchor"): _finite(
                value, f"competence_floors[{anchor!r}]"
            )
            for anchor, value in self.competence_floors.items()
        }
        if not floors or any(value < 0.0 or value > 1.0 for value in floors.values()):
            raise RecomputeContractError(
                "competence floors must cover anchors on [0, 1]"
            )
        if (
            not isinstance(self.competence_mode, str)
            or self.competence_mode not in COMPETENCE_MODES
        ):
            raise RecomputeContractError(
                "source competence_mode must be OBSERVE or ENFORCE"
            )
        object.__setattr__(self, "selection_rows", selection)
        object.__setattr__(self, "attestation_rows", attestation)
        object.__setattr__(
            self, "competence_floors", MappingProxyType(dict(sorted(floors.items())))
        )
        object.__setattr__(
            self,
            "return_contract_id",
            _nonempty(self.return_contract_id, "return_contract_id"),
        )


def _ordered_source_rows(
    rows: Sequence[SourceEpisodeRow],
) -> tuple[SourceEpisodeRow, ...]:
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row.source_anchor_id,
                row.candidate_id,
                row.bundle_digest,
                row.reset_seed,
            ),
        )
    )


def championization_payload(
    result: ChampionizationResult,
    *,
    selection_rows: Sequence[SourceEpisodeRow],
    attestation_rows: Sequence[SourceEpisodeRow],
) -> dict[str, Any]:
    """Canonical full source result, including raw row digests omitted by summaries."""

    ordered_selection = _ordered_source_rows(selection_rows)
    ordered_attestation = _ordered_source_rows(attestation_rows)
    return {
        "schema": "policy-learnware.v02-source-independent-recompute.v0",
        "selection_rows_digest": sha256_json(
            [row.to_dict() for row in ordered_selection]
        ),
        "attestation_rows_digest": sha256_json(
            [row.to_dict() for row in ordered_attestation]
        ),
        "selection_attestation_seed_blocks": {
            "selection": sorted({row.reset_seed for row in ordered_selection}),
            "attestation": sorted({row.reset_seed for row in ordered_attestation}),
            "overlap": sorted(
                {row.reset_seed for row in ordered_selection}
                & {row.reset_seed for row in ordered_attestation}
            ),
        },
        "selected_by_anchor": dict(sorted(result.selected_by_anchor.items())),
        "competence_mode": result.competence_mode,
        "selection_summaries": [item.to_dict() for item in result.selection_summaries],
        "competence_records": {
            anchor: record.to_dict()
            for anchor, record in sorted(result.competence_records.items())
        },
        "rejected_anchors": dict(sorted(result.rejected_anchors.items())),
        "selection_digest": result.selection_digest,
        "normalization_contract": "dmc_fixed_horizon_[0,1]",
        "precomputed_aggregate_consumed": False,
    }


def recompute_source(
    inputs: SourceRecomputeInputs,
    *,
    coverage: FullCoverageContract,
    market: PublicMarketView,
) -> tuple[ChampionizationResult, Mapping[str, Any], str]:
    ordered_selection = _ordered_source_rows(inputs.selection_rows)
    ordered_attestation = _ordered_source_rows(inputs.attestation_rows)
    anchors = {row.source_anchor_id for row in ordered_selection}
    if anchors != set(coverage.source_anchor_ids):
        raise RecomputeContractError(
            "source selection rows do not exactly cover frozen source anchors"
        )
    if {row.source_anchor_id for row in ordered_attestation} != anchors:
        raise RecomputeContractError(
            "source attestation rows do not exactly cover frozen source anchors"
        )
    if set(inputs.competence_floors) != anchors:
        raise RecomputeContractError(
            "source competence floors do not exactly cover frozen source anchors"
        )
    result = championize_by_anchor(
        ordered_selection,
        ordered_attestation,
        competence_floors=inputs.competence_floors,
        mean_tolerance=inputs.mean_tolerance,
        lcb_z=inputs.lcb_z,
        return_contract_id=inputs.return_contract_id,
        competence_mode=inputs.competence_mode,
    )
    if set(result.competence_records) != set(coverage.source_market_bindings):
        raise RecomputeContractError(
            "accepted source competence records differ from frozen market bindings"
        )
    if set(market.entries) != set(coverage.public_market_ids):
        raise RecomputeContractError("public market IDs differ from frozen coverage")
    for anchor, opaque_id in coverage.source_market_bindings.items():
        recomputed = result.competence_records[anchor].normalized_competence
        public = market.entries[opaque_id].normalized_source_competence
        if not math.isclose(recomputed, public, rel_tol=RTOL, abs_tol=ATOL):
            raise RecomputeContractError(
                f"public competence mismatch for source anchor {anchor!r}: "
                f"market={public!r}, recomputed={recomputed!r}"
            )
    payload = championization_payload(
        result,
        selection_rows=ordered_selection,
        attestation_rows=ordered_attestation,
    )
    return result, payload, _verify_snapshot(payload, inputs.published, where="source")


@dataclass(frozen=True)
class Gate0AuditUnit:
    audit_id: str
    operator_audit: DynamicsOperatorAudit
    gate0_audit: Gate0Audit
    published_operator: PublishedSnapshot
    published_gate0: PublishedSnapshot

    def __post_init__(self) -> None:
        object.__setattr__(self, "audit_id", _nonempty(self.audit_id, "audit_id"))
        if not isinstance(self.operator_audit, DynamicsOperatorAudit):
            raise RecomputeContractError("operator_audit has the wrong type")
        if not isinstance(self.gate0_audit, Gate0Audit):
            raise RecomputeContractError("gate0_audit has the wrong type")


def _rebuild_operator_audit(audit: DynamicsOperatorAudit) -> Mapping[str, Any]:
    for name in (
        "axis_id",
        "operator_id",
        "operator_version",
        "task_id",
    ):
        _nonempty(getattr(audit, name), f"operator audit {name}")
    factor = _finite(audit.factor, "operator audit factor")
    if factor <= 0.0:
        raise RecomputeContractError("operator audit factor must be positive")
    for name in ("base_model_digest", "shifted_model_digest"):
        _digest(getattr(audit, name), f"operator audit {name}")
    changed = tuple(audit.changed_leaves)
    unchanged = tuple(audit.unchanged_leaves)
    if (
        len(changed) != len(set(changed))
        or len(unchanged) != len(set(unchanged))
        or set(changed) & set(unchanged)
    ):
        raise RecomputeContractError(
            "operator changed/unchanged leaf summary is invalid"
        )
    selected_count = _nonnegative_int(
        audit.selected_element_count, "selected_element_count"
    )
    changed_count = _nonnegative_int(
        audit.changed_element_count, "changed_element_count"
    )
    expected_changed_count = 0 if factor == 1.0 else selected_count
    count_ok = changed_count == expected_changed_count
    if factor == 1.0 and changed:
        count_ok = False
    for name in (
        "source_object_unchanged",
        "exact_allowlist",
        "coupling_check",
        "finite",
        "passed",
    ):
        if type(getattr(audit, name)) is not bool:
            raise RecomputeContractError(f"operator audit {name} must be boolean")
    expected_passed = bool(
        audit.source_object_unchanged
        and audit.exact_allowlist
        and audit.coupling_check
        and audit.finite
        and count_ok
        and audit.reason is None
    )
    if audit.passed != expected_passed:
        raise RecomputeContractError(
            "operator audit passed field disagrees with primitive checks"
        )
    rebuilt = dict(audit.to_dict())
    rebuilt["passed"] = expected_passed
    return rebuilt


def _rebuild_rollout_audit(audit: RolloutAudit) -> RolloutAudit:
    if not isinstance(audit, RolloutAudit):
        raise RecomputeContractError("scalar rollout audit has the wrong type")
    if audit.episode_count <= 0 or audit.steps_per_episode <= 0:
        raise RecomputeContractError("rollout audit counts must be positive")
    for name in ("all_finite", "no_early_termination", "passed"):
        if type(getattr(audit, name)) is not bool:
            raise RecomputeContractError(f"rollout audit {name} must be boolean")
    paired_values = (
        audit.paired_observation_identity,
        audit.paired_reward_identity,
        audit.paired_flag_identity,
    )
    if not (
        all(value is None for value in paired_values)
        or all(type(value) is bool for value in paired_values)
    ):
        raise RecomputeContractError("rollout paired-identity fields are incoherent")
    for name in (
        "maximum_observation_absolute_error",
        "maximum_reward_absolute_error",
    ):
        value = getattr(audit, name)
        if value is not None and _finite(value, f"rollout audit {name}") < 0.0:
            raise RecomputeContractError(f"rollout audit {name} must be non-negative")
    paired_ok = all(value is None for value in paired_values) or all(paired_values)
    expected_passed = bool(
        audit.all_finite
        and audit.no_early_termination
        and paired_ok
        and audit.reason is None
    )
    if audit.passed != expected_passed:
        raise RecomputeContractError(
            "rollout audit passed field disagrees with primitive checks"
        )
    return RolloutAudit(
        episode_count=audit.episode_count,
        steps_per_episode=audit.steps_per_episode,
        all_finite=audit.all_finite,
        no_early_termination=audit.no_early_termination,
        paired_observation_identity=audit.paired_observation_identity,
        paired_reward_identity=audit.paired_reward_identity,
        paired_flag_identity=audit.paired_flag_identity,
        maximum_observation_absolute_error=audit.maximum_observation_absolute_error,
        maximum_reward_absolute_error=audit.maximum_reward_absolute_error,
        passed=expected_passed,
        reason=audit.reason,
    )


def _rebuild_gate0_audit(
    gate: Gate0Audit, operator: DynamicsOperatorAudit
) -> Mapping[str, Any]:
    scalar = _rebuild_rollout_audit(gate.scalar_rollout)
    if gate.operator_audit_digest != operator.digest:
        raise RecomputeContractError(
            "Gate0 operator digest does not bind its raw audit"
        )
    if gate.environment_instance_digest != _digest(
        gate.environment_instance_digest, "environment_instance_digest"
    ):
        raise RecomputeContractError("invalid Gate0 environment digest")
    for name in (
        "schema_contract_identity",
        "factor_role_valid",
        "jit_finite",
        "batched_rollout_finite",
        "source_object_unchanged",
        "exact_allowlist",
        "coupled_physics",
        "passed",
    ):
        if type(getattr(gate, name)) is not bool:
            raise RecomputeContractError(f"Gate0 {name} must be boolean")
    mirrored = {
        "source_object_unchanged": operator.source_object_unchanged,
        "exact_allowlist": operator.exact_allowlist,
        "coupled_physics": operator.coupling_check,
    }
    if any(getattr(gate, name) != value for name, value in mirrored.items()):
        raise RecomputeContractError("Gate0 summary disagrees with operator primitives")
    checks = {
        "schema_contract_mismatch": gate.schema_contract_identity,
        "factor_role_mismatch": gate.factor_role_valid,
        "scalar_rollout_failed": scalar.passed,
        "jit_rollout_failed": gate.jit_finite,
        "batched_rollout_failed": gate.batched_rollout_finite,
        "source_object_modified": operator.source_object_unchanged,
        "operator_allowlist_failed": operator.exact_allowlist,
        "coupled_physics_failed": operator.coupling_check,
        "operator_audit_failed": operator.passed,
    }
    reasons = tuple(name for name, passed in checks.items() if not passed)
    expected_passed = not reasons
    if gate.passed != expected_passed or tuple(gate.reasons) != reasons:
        raise RecomputeContractError(
            "Gate0 passed/reasons disagree with independently rebuilt checks"
        )
    rebuilt = Gate0Audit(
        environment_instance_digest=gate.environment_instance_digest,
        operator_audit_digest=operator.digest,
        schema_contract_identity=gate.schema_contract_identity,
        factor_role_valid=gate.factor_role_valid,
        scalar_rollout=scalar,
        jit_finite=gate.jit_finite,
        batched_rollout_finite=gate.batched_rollout_finite,
        source_object_unchanged=operator.source_object_unchanged,
        exact_allowlist=operator.exact_allowlist,
        coupled_physics=operator.coupling_check,
        passed=expected_passed,
        reasons=reasons,
    )
    return rebuilt.to_dict()


def verify_gate0_audits(
    units: Sequence[Gate0AuditUnit], *, expected_ids: Sequence[str]
) -> tuple[Mapping[str, Any], str]:
    rows = tuple(units)
    keyed = {unit.audit_id: unit for unit in rows}
    if len(keyed) != len(rows) or set(keyed) != set(expected_ids):
        raise RecomputeContractError("Gate0 audit units differ from frozen coverage")
    payload_units: dict[str, Any] = {}
    for audit_id, unit in sorted(keyed.items()):
        operator_payload = _rebuild_operator_audit(unit.operator_audit)
        operator_digest = _verify_snapshot(
            operator_payload,
            unit.published_operator,
            where=f"Gate0[{audit_id}].operator",
        )
        gate_payload = _rebuild_gate0_audit(unit.gate0_audit, unit.operator_audit)
        gate_digest = _verify_snapshot(
            gate_payload, unit.published_gate0, where=f"Gate0[{audit_id}]"
        )
        payload_units[audit_id] = {
            "operator_digest": operator_digest,
            "gate0_digest": gate_digest,
            "gate0_passed": gate_payload["passed"],
        }
    payload = {
        "schema": "policy-learnware.v02-gate0-summary-recompute.v0",
        "unit_count": len(payload_units),
        "units": payload_units,
        "caller_gate_aggregate_consumed": False,
    }
    return payload, sha256_json(payload)


@dataclass(frozen=True)
class RepresentationReplayUnit:
    """One frozen raw trace -> encoder -> RKME/EnvironmentSpec work unit."""

    unit_id: str
    trace: ProbeTraceView
    encoder: SemanticEncoderProtocol
    contract: RepresentationBuildContract
    published_encoded_cache: PublishedSnapshot
    published_environment_spec: PublishedSnapshot

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "unit_id", _nonempty(self.unit_id, "representation unit_id")
        )
        if not isinstance(self.trace, ProbeTraceView):
            raise RecomputeContractError("representation trace has the wrong type")
        if not isinstance(self.encoder, SemanticEncoderProtocol):
            raise RecomputeContractError("representation encoder violates its protocol")
        if not isinstance(self.contract, RepresentationBuildContract):
            raise RecomputeContractError(
                "representation build contract has the wrong type"
            )


def encoded_cache_payload(
    unit: RepresentationReplayUnit,
    encoded: EncodedEpisodeDataset,
) -> dict[str, Any]:
    trace = unit.trace
    assert trace.probe_dataset_digest is not None
    raw_arrays_digest = sha256_ndarrays(
        {
            "packed": trace.dataset.packed,
            "episode_offsets": trace.dataset.episode_offsets,
            "reset_seeds": trace.dataset.reset_seeds,
            "probe_seeds": trace.dataset.probe_seeds,
        }
    )
    return {
        "schema": "policy-learnware.v02-independent-encoded-cache.v0",
        "unit_id": unit.unit_id,
        "raw_probe_arrays_digest": raw_arrays_digest,
        "probe_dataset_digest": trace.probe_dataset_digest,
        "encoder_metadata": unit.encoder.metadata.to_dict(),
        "encoder_metadata_digest": sha256_json(unit.encoder.metadata.to_dict()),
        "representation_build_contract_digest": unit.contract.digest,
        "encoded_arrays_digest": sha256_ndarrays(
            {
                "points": encoded.points,
                "episode_offsets": encoded.episode_offsets,
            }
        ),
        "transition_count": int(encoded.points.shape[0]),
        "latent_dim": int(encoded.points.shape[1]),
        "episode_count": encoded.episode_count,
        "representation_protocol_id": encoded.representation_protocol_id,
        "precomputed_cache_consumed": False,
    }


def replay_representations(
    units: Sequence[RepresentationReplayUnit], *, expected_ids: Sequence[str]
) -> tuple[Mapping[str, Any], str]:
    """Re-encode raw packed probes and rebuild every frozen EnvironmentSpec."""

    rows = tuple(units)
    keyed = {unit.unit_id: unit for unit in rows}
    if len(keyed) != len(rows) or set(keyed) != set(expected_ids):
        raise RecomputeContractError(
            "representation replay units differ from frozen coverage"
        )
    payload_units: dict[str, Any] = {}
    for unit_id, unit in sorted(keyed.items()):
        encoded_first = unit.encoder.encode(
            unit.trace.dataset, batch_size=unit.contract.batch_size
        )
        encoded_second = unit.encoder.encode(
            unit.trace.dataset, batch_size=unit.contract.batch_size
        )
        if not isinstance(encoded_first, EncodedEpisodeDataset) or not isinstance(
            encoded_second, EncodedEpisodeDataset
        ):
            raise RecomputeContractError(
                f"representation encoder {unit_id!r} returned an invalid cache"
            )
        first_cache = encoded_cache_payload(unit, encoded_first)
        second_cache = encoded_cache_payload(unit, encoded_second)
        if _first_mismatch(first_cache, second_cache) is not None:
            raise RecomputeContractError(
                f"representation encoder {unit_id!r} is not deterministic"
            )
        cache_digest = _verify_snapshot(
            first_cache,
            unit.published_encoded_cache,
            where=f"representation[{unit_id}].encoded_cache",
        )
        rebuilt_spec = build_environment_spec(unit.trace, unit.encoder, unit.contract)
        spec_payload = rebuilt_spec.to_dict()
        spec_digest = _verify_snapshot(
            spec_payload,
            unit.published_environment_spec,
            where=f"representation[{unit_id}].environment_spec",
        )
        payload_units[unit_id] = {
            "encoded_cache_digest": cache_digest,
            "environment_spec_snapshot_digest": spec_digest,
            "environment_spec_digest": rebuilt_spec.environment_spec_digest,
            "probe_dataset_digest": unit.trace.probe_dataset_digest,
            "representation_protocol_id": unit.contract.representation_protocol_id,
        }
    payload = {
        "schema": "policy-learnware.v02-full-representation-replay.v0",
        "unit_count": len(payload_units),
        "units": payload_units,
        "raw_probe_traces_reencoded": True,
        "precomputed_environment_specs_consumed": False,
    }
    return payload, sha256_json(payload)


@dataclass(frozen=True)
class SelectorReplayUnit:
    query_id: str
    selector: Any
    query: TargetQueryView
    artifact: FrozenSelectorArtifact | None
    published: PublishedSnapshot

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", _nonempty(self.query_id, "query_id"))
        if not isinstance(self.query, TargetQueryView):
            raise RecomputeContractError("selector query must be a TargetQueryView")
        if not hasattr(self.selector, "method_id") or not hasattr(
            self.selector, "select"
        ):
            raise RecomputeContractError("selector does not expose method_id/select")
        if self.artifact is not None and not isinstance(
            self.artifact, FrozenSelectorArtifact
        ):
            raise RecomputeContractError("selector artifact has the wrong type")

    @property
    def method_id(self) -> str:
        return _nonempty(self.selector.method_id, "selector method_id")

    @property
    def unit_id(self) -> str:
        return selector_unit_id(self.method_id, self.query_id)


def _select_once(unit: SelectorReplayUnit, market: PublicMarketView) -> SelectionRecord:
    if unit.artifact is None:
        if not isinstance(unit.selector, LMinSelector):
            raise RecomputeContractError(
                "artifact-free replay is permitted only for typed LMinSelector"
            )
        result = unit.selector.select(
            query_spec=unit.query.query_spec,
            market=market,
            target_evidence_digest=unit.query.target_evidence_digest,
            cost_digest=unit.query.cost_digest,
        )
    else:
        result = unit.selector.select(unit.query, market, unit.artifact)
    if not isinstance(result, SelectionRecord):
        raise RecomputeContractError("selector replay did not return SelectionRecord")
    if result.method_id != unit.method_id:
        raise RecomputeContractError("selection method_id differs from selector")
    if result.target_evidence_digest != unit.query.target_evidence_digest:
        raise RecomputeContractError("selection target evidence binding mismatch")
    if result.cost_digest != unit.query.cost_digest:
        raise RecomputeContractError("selection cost binding mismatch")
    ranked_ids = tuple(row.opaque_learnware_id for row in result.ranking)
    if len(ranked_ids) != len(market.entries) or set(ranked_ids) != set(market.entries):
        raise RecomputeContractError(
            "selection ranking does not cover the full public market"
        )
    return result


def replay_selectors(
    units: Sequence[SelectorReplayUnit],
    *,
    market: PublicMarketView,
    expected_ids: Sequence[str],
) -> tuple[Mapping[str, SelectionRecord], Mapping[str, Any], str]:
    rows = tuple(units)
    keyed = {unit.unit_id: unit for unit in rows}
    if len(keyed) != len(rows) or set(keyed) != set(expected_ids):
        raise RecomputeContractError(
            "selector replay units differ from frozen coverage"
        )
    records: dict[str, SelectionRecord] = {}
    payload_units: dict[str, Any] = {}
    for unit_id, unit in sorted(keyed.items()):
        first = _select_once(unit, market)
        second = _select_once(unit, market)
        if (
            first.digest != second.digest
            or _first_mismatch(first.to_dict(), second.to_dict()) is not None
        ):
            raise RecomputeContractError(
                f"selector {unit_id!r} is not deterministic under exact replay"
            )
        evidence_audit = audit_evidence_contract(first.evidence_contract)
        if (
            not evidence_audit.passed
            or not first.evidence_contract.is_public_zero_update
        ):
            raise RecomputeContractError(
                f"selector {unit_id!r} violates the zero-target-update evidence contract"
            )
        digest = _verify_snapshot(
            first.to_dict(), unit.published, where=f"selection[{unit_id}]"
        )
        records[unit_id] = first
        payload_units[unit_id] = {
            "selection_digest": digest,
            "selected_id": first.selected_id,
            "ranking_count": len(first.ranking),
            "evidence_audit": evidence_audit.to_dict(),
        }
    payload = {
        "schema": "policy-learnware.v02-full-selector-replay.v0",
        "policy_market_id": market.policy_market_id,
        "representation_index_id": market.representation_index.representation_index_id,
        "unit_count": len(payload_units),
        "units": payload_units,
        "published_selector_outputs_consumed": False,
    }
    return MappingProxyType(records), payload, sha256_json(payload)


@dataclass(frozen=True)
class OracleMetricUnit:
    query_id: str
    task_id: str
    axis_id: str
    context_id: str
    method_id: str
    market_ids: tuple[str, ...]
    deployment_registry: Mapping[str, Any]
    target_execution_abi: ExecutionABIRecord
    private_target_instance_digest: str
    evaluation_protocol_id: str
    failure_floor: float
    epsilon: float
    tie_atol: float
    candidate_paired_seeds: bool
    published: PublishedSnapshot

    def __post_init__(self) -> None:
        for name in ("query_id", "task_id", "axis_id", "context_id", "method_id"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "market_ids",
            _canonical_tuple(self.market_ids, "market_ids"),
        )
        if not isinstance(self.deployment_registry, Mapping):
            raise RecomputeContractError("deployment_registry must be a mapping")
        registry = dict(self.deployment_registry)
        if set(registry) != set(self.market_ids):
            raise RecomputeContractError(
                "deployment_registry must cover the full anonymous market"
            )
        if not isinstance(self.target_execution_abi, ExecutionABIRecord):
            raise RecomputeContractError("target_execution_abi has the wrong type")
        executable = minimum_executable_set(
            self.market_ids, registry, self.target_execution_abi
        )
        if len(executable) < 2:
            raise RecomputeContractError(
                "ranking recompute requires at least two ABI-executable policies"
            )
        object.__setattr__(self, "deployment_registry", MappingProxyType(registry))
        for name in ("private_target_instance_digest", "evaluation_protocol_id"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        floor = _finite(self.failure_floor, "failure_floor")
        epsilon = _finite(self.epsilon, "epsilon")
        tolerance = _finite(self.tie_atol, "tie_atol")
        if not 0.0 <= floor <= 1.0 or not 0.0 <= epsilon <= 1.0 or tolerance < 0.0:
            raise RecomputeContractError("invalid oracle floor/epsilon/tie tolerance")
        if type(self.candidate_paired_seeds) is not bool:
            raise RecomputeContractError("candidate_paired_seeds must be boolean")
        object.__setattr__(self, "failure_floor", floor)
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "tie_atol", tolerance)

    @property
    def unit_id(self) -> str:
        return selector_unit_id(self.method_id, self.query_id)


def _execution_abi(value: Any) -> ExecutionABIRecord:
    if isinstance(value, ExecutionABIRecord):
        return value
    abi = getattr(value, "execution_abi", None)
    if isinstance(abi, ExecutionABIRecord):
        return abi
    raise RecomputeContractError("deployment entry lacks an ExecutionABIRecord")


def _oracle_unit_contract_digest(unit: OracleMetricUnit) -> str:
    return sha256_json(
        {
            "schema": "policy-learnware.v02-oracle-recompute-query-contract.v0",
            "query_id": unit.query_id,
            "task_id": unit.task_id,
            "axis_id": unit.axis_id,
            "context_id": unit.context_id,
            "market_ids": list(unit.market_ids),
            "deployment_execution_abis": {
                opaque_id: _execution_abi(unit.deployment_registry[opaque_id]).to_dict()
                for opaque_id in unit.market_ids
            },
            "target_execution_abi": unit.target_execution_abi.to_dict(),
            "private_target_instance_digest": unit.private_target_instance_digest,
            "evaluation_protocol_id": unit.evaluation_protocol_id,
            "failure_floor": unit.failure_floor,
            "epsilon": unit.epsilon,
            "tie_atol": unit.tie_atol,
            "candidate_paired_seeds": unit.candidate_paired_seeds,
        }
    )


def recompute_oracle_metrics(
    rows: Sequence[OracleEpisodeRow],
    units: Sequence[OracleMetricUnit],
    *,
    selections: Mapping[str, SelectionRecord],
    expected_unit_ids: Sequence[str],
    expected_query_ids: Sequence[str],
    expected_market_ids: Sequence[str],
) -> tuple[Mapping[str, Mapping[str, Any]], Mapping[str, Any], str]:
    raw_rows = tuple(rows)
    if not raw_rows or any(not isinstance(row, OracleEpisodeRow) for row in raw_rows):
        raise RecomputeContractError("oracle rows must be a non-empty typed sequence")
    if {row.opaque_query_id for row in raw_rows} != set(expected_query_ids):
        raise RecomputeContractError("oracle raw query IDs differ from frozen coverage")
    typed_units = tuple(units)
    keyed = {unit.unit_id: unit for unit in typed_units}
    if len(keyed) != len(typed_units) or set(keyed) != set(expected_unit_ids):
        raise RecomputeContractError("oracle metric units differ from frozen coverage")
    if set(selections) != set(expected_unit_ids):
        raise RecomputeContractError(
            "oracle metrics lack full replayed-selection coverage"
        )

    # All methods evaluating one query must share the exact private ABI census,
    # target instance, seed, and evaluation contract.
    query_contracts: dict[str, str] = {}
    for unit in typed_units:
        if set(unit.market_ids) != set(expected_market_ids):
            raise RecomputeContractError(
                "oracle unit market IDs differ from full anonymous market coverage"
            )
        contract = _oracle_unit_contract_digest(unit)
        previous = query_contracts.setdefault(unit.query_id, contract)
        if previous != contract:
            raise RecomputeContractError(
                f"oracle metric methods disagree on query contract {unit.query_id!r}"
            )

    metrics: dict[str, Mapping[str, Any]] = {}
    payload_units: dict[str, Any] = {}
    query_results: dict[str, Any] = {}
    for query_id in sorted(expected_query_ids):
        query_units = tuple(
            sorted(
                (unit for unit in typed_units if unit.query_id == query_id),
                key=lambda item: item.method_id,
            )
        )
        if not query_units:
            raise RecomputeContractError(
                f"oracle query has no metric units: {query_id!r}"
            )
        exemplar = query_units[0]
        result = aggregate_full_pool_oracle(
            opaque_query_id=query_id,
            private_target_instance_digest=exemplar.private_target_instance_digest,
            evaluation_protocol_id=exemplar.evaluation_protocol_id,
            market_ids=exemplar.market_ids,
            deployment_registry=exemplar.deployment_registry,
            target_execution_abi=exemplar.target_execution_abi,
            episode_rows=tuple(
                row for row in raw_rows if row.opaque_query_id == query_id
            ),
            published_selections=tuple(
                PublishedSelection.from_selection_record(selections[unit.unit_id])
                for unit in query_units
            ),
            failure_floor=exemplar.failure_floor,
            tie_atol=exemplar.tie_atol,
            candidate_paired_seeds=exemplar.candidate_paired_seeds,
        )
        query_results[query_id] = result

    for unit_id, unit in sorted(keyed.items()):
        result = query_results[unit.query_id]
        values = {
            opaque_id: float(result.normalized_value_vector[opaque_id])
            for opaque_id in result.executable_ids
        }
        selection = selections[unit_id]
        selection_metrics = compute_selection_metrics(
            selected_policy_id=selection.selected_id,
            normalized_returns_by_policy=values,
            executable_policy_ids=result.executable_ids,
            incompatible_failure_value=unit.failure_floor,
            epsilon=unit.epsilon,
            tie_tolerance=unit.tie_atol,
        )
        outcome = result.outcomes[unit.method_id]
        if (
            not math.isclose(
                selection_metrics.selected_normalized_return,
                outcome.selected_value,
                rel_tol=RTOL,
                abs_tol=ATOL,
            )
            or not math.isclose(
                selection_metrics.pool_regret,
                outcome.regret,
                rel_tol=RTOL,
                abs_tol=ATOL,
            )
            or selection_metrics.top1_agreement != outcome.oracle_top1_agreement
        ):
            raise RecomputeContractError(
                f"oracle metrics disagree with full-pool outcome for {unit_id!r}"
            )
        predicted = tuple(
            row.opaque_learnware_id
            for row in selection.ranking
            if row.opaque_learnware_id in set(result.executable_ids)
        )
        ranking_metrics = compute_ranking_metrics(
            predicted,
            values,
            tie_tolerance=unit.tie_atol,
        )
        payload = {
            "schema": "policy-learnware.v02-oracle-query-metrics-recompute.v0",
            "unit_id": unit_id,
            "query_id": unit.query_id,
            "method_id": unit.method_id,
            "task_id": unit.task_id,
            "axis_id": unit.axis_id,
            "context_id": unit.context_id,
            "full_anonymous_market_ids": list(result.market_ids),
            "full_pool_oracle_digest": result.digest,
            "full_pool_oracle": result.to_private_dict(),
            "selection_metrics": selection_metrics.to_dict(),
            "ranking_metrics": ranking_metrics.to_dict(),
            "ranking_scope": "executable_pool_only",
            "published_oracle_aggregate_consumed": False,
        }
        digest = _verify_snapshot(
            payload, unit.published, where=f"oracle_metrics[{unit_id}]"
        )
        metrics[unit_id] = MappingProxyType(
            {
                "unit": unit,
                "selection": selection_metrics,
                "ranking": ranking_metrics,
            }
        )
        payload_units[unit_id] = {"metrics_digest": digest, **payload}
    payload = {
        "schema": "policy-learnware.v02-full-oracle-metrics-recompute.v0",
        "unit_count": len(payload_units),
        "query_count": len(expected_query_ids),
        "full_pool_oracle_digests": {
            query_id: result.digest
            for query_id, result in sorted(query_results.items())
        },
        "units": payload_units,
    }
    return MappingProxyType(metrics), payload, sha256_json(payload)


Endpoint = Literal["selected_normalized_return", "pool_regret"]


@dataclass(frozen=True)
class PairedComparisonPlan:
    comparison_id: str
    left_method_id: str
    right_method_id: str
    endpoint: Endpoint
    bootstrap_seed: int
    holm_family_id: str | None = None
    noninferiority_margin: float | None = None
    null_boundary: float = 0.0

    def __post_init__(self) -> None:
        for name in ("comparison_id", "left_method_id", "right_method_id"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if self.left_method_id == self.right_method_id:
            raise RecomputeContractError("paired comparison methods must differ")
        if self.endpoint not in {"selected_normalized_return", "pool_regret"}:
            raise RecomputeContractError("unsupported paired comparison endpoint")
        object.__setattr__(
            self,
            "bootstrap_seed",
            _nonnegative_int(self.bootstrap_seed, "bootstrap_seed"),
        )
        if self.holm_family_id is not None:
            object.__setattr__(
                self,
                "holm_family_id",
                _nonempty(self.holm_family_id, "holm_family_id"),
            )
        if self.noninferiority_margin is not None:
            margin = _finite(self.noninferiority_margin, "noninferiority_margin")
            if margin < 0.0:
                raise RecomputeContractError(
                    "noninferiority margin must be non-negative"
                )
            object.__setattr__(self, "noninferiority_margin", margin)
        object.__setattr__(
            self, "null_boundary", _finite(self.null_boundary, "null_boundary")
        )


@dataclass(frozen=True)
class StatisticsPlan:
    resamples: int
    bootstrap_seed: int
    confidence_level: float
    comparisons: tuple[PairedComparisonPlan, ...]
    published: PublishedSnapshot

    def __post_init__(self) -> None:
        resamples = _nonnegative_int(self.resamples, "resamples")
        if resamples <= 0:
            raise RecomputeContractError("resamples must be positive")
        seed = _nonnegative_int(self.bootstrap_seed, "bootstrap_seed")
        if seed >= 2**64:
            raise RecomputeContractError("bootstrap_seed must lie below 2**64")
        level = _finite(self.confidence_level, "confidence_level")
        if not 0.0 < level < 1.0:
            raise RecomputeContractError("confidence_level must lie in (0, 1)")
        comparisons = tuple(self.comparisons)
        if any(not isinstance(item, PairedComparisonPlan) for item in comparisons):
            raise RecomputeContractError("comparisons have the wrong type")
        ids = tuple(item.comparison_id for item in comparisons)
        if len(ids) != len(set(ids)):
            raise RecomputeContractError("comparison IDs must be unique")
        family_seeds: dict[str, int] = {}
        for item in comparisons:
            if item.holm_family_id is None:
                continue
            previous = family_seeds.setdefault(item.holm_family_id, item.bootstrap_seed)
            if previous != item.bootstrap_seed:
                raise RecomputeContractError(
                    "one simultaneous/Holm family must share a paired bootstrap seed"
                )
        object.__setattr__(self, "resamples", resamples)
        object.__setattr__(self, "bootstrap_seed", seed)
        object.__setattr__(self, "confidence_level", level)
        object.__setattr__(self, "comparisons", comparisons)


def _bootstrap_payload(result: Any) -> dict[str, Any]:
    return {
        **result.to_summary_dict(),
        "replicates_digest": sha256_ndarrays({"replicates": result.replicates}),
    }


def _metric_leaves(
    metrics: Mapping[str, Mapping[str, Any]], method_id: str, endpoint: Endpoint
) -> tuple[HierarchicalValue, ...]:
    leaves: list[HierarchicalValue] = []
    for item in metrics.values():
        unit = item["unit"]
        if unit.method_id != method_id:
            continue
        selection = item["selection"]
        leaves.append(
            HierarchicalValue(
                task_id=unit.task_id,
                axis_id=unit.axis_id,
                context_id=unit.context_id,
                observation_id=unit.query_id,
                value=float(getattr(selection, endpoint)),
            )
        )
    if not leaves:
        raise RecomputeContractError(f"statistics method has no leaves: {method_id!r}")
    return tuple(sorted(leaves, key=lambda row: row.key))


def recompute_statistics(
    metrics: Mapping[str, Mapping[str, Any]],
    plan: StatisticsPlan,
) -> tuple[Mapping[str, Any], str]:
    methods = tuple(sorted({item["unit"].method_id for item in metrics.values()}))
    if not methods:
        raise RecomputeContractError("statistics require recomputed oracle metrics")
    method_payloads: dict[str, Any] = {}
    leaves_by_method_endpoint: dict[tuple[str, str], tuple[HierarchicalValue, ...]] = {}
    for method_id in methods:
        endpoint_payloads: dict[str, Any] = {}
        for endpoint in ("selected_normalized_return", "pool_regret"):
            leaves = _metric_leaves(metrics, method_id, endpoint)  # type: ignore[arg-type]
            leaves_by_method_endpoint[(method_id, endpoint)] = leaves
            seed = derive_bootstrap_seed(plan.bootstrap_seed, method_id, endpoint)
            bootstrap = hierarchical_bootstrap(
                leaves,
                resamples=plan.resamples,
                seed=seed,
                confidence_level=plan.confidence_level,
            )
            endpoint_payloads[endpoint] = {
                "aggregate": aggregate_hierarchy(leaves).to_dict(),
                "bootstrap": _bootstrap_payload(bootstrap),
            }
        method_payloads[method_id] = endpoint_payloads

    comparisons: dict[str, Any] = {}
    p_value_families: dict[str, dict[str, float]] = {}
    bootstrap_families: dict[str, dict[str, Any]] = {}
    for comparison in sorted(plan.comparisons, key=lambda item: item.comparison_id):
        if (
            comparison.left_method_id not in methods
            or comparison.right_method_id not in methods
        ):
            raise RecomputeContractError(
                f"comparison {comparison.comparison_id!r} references an absent method"
            )
        left = leaves_by_method_endpoint[
            (comparison.left_method_id, comparison.endpoint)
        ]
        right = leaves_by_method_endpoint[
            (comparison.right_method_id, comparison.endpoint)
        ]
        # Positive always means the left/proposed method is better.
        paired = hierarchical_paired_difference_bootstrap(
            left if comparison.endpoint == "selected_normalized_return" else right,
            right if comparison.endpoint == "selected_normalized_return" else left,
            resamples=plan.resamples,
            seed=comparison.bootstrap_seed,
            confidence_level=plan.confidence_level,
        )
        raw_p = centered_one_sided_p_value(
            paired.observed,
            paired.replicates,
            null_boundary=comparison.null_boundary,
        )
        payload: dict[str, Any] = {
            "comparison_id": comparison.comparison_id,
            "left_method_id": comparison.left_method_id,
            "right_method_id": comparison.right_method_id,
            "endpoint": comparison.endpoint,
            "positive_direction": (
                "left_minus_right"
                if comparison.endpoint == "selected_normalized_return"
                else "right_minus_left"
            ),
            "null_boundary": comparison.null_boundary,
            "bootstrap": _bootstrap_payload(paired),
            "raw_p_value": raw_p,
            "holm_family_id": comparison.holm_family_id,
            "noninferiority": None,
        }
        if comparison.noninferiority_margin is not None:
            if comparison.null_boundary != -comparison.noninferiority_margin:
                raise RecomputeContractError(
                    "noninferiority comparison null boundary must equal negative margin"
                )
            payload["noninferiority"] = evaluate_noninferiority(
                paired, margin=comparison.noninferiority_margin
            ).to_dict()
        if comparison.holm_family_id is not None:
            p_value_families.setdefault(comparison.holm_family_id, {})[
                comparison.comparison_id
            ] = raw_p
            bootstrap_families.setdefault(comparison.holm_family_id, {})[
                comparison.comparison_id
            ] = paired
        comparisons[comparison.comparison_id] = payload
    holm = {
        family_id: {
            hypothesis_id: result.to_dict()
            for hypothesis_id, result in sorted(holm_bonferroni(p_values).items())
        }
        for family_id, p_values in sorted(p_value_families.items())
    }
    simultaneous = {
        family_id: bootstrap_max_t_intervals(family).to_dict()
        for family_id, family in sorted(bootstrap_families.items())
    }
    payload = {
        "schema": "policy-learnware.v02-full-statistical-recompute.v0",
        "resamples": plan.resamples,
        "bootstrap_seed": plan.bootstrap_seed,
        "confidence_level": plan.confidence_level,
        "methods": method_payloads,
        "comparisons": comparisons,
        "holm_families": holm,
        "simultaneous_max_t_families": simultaneous,
        "published_statistical_aggregate_consumed": False,
    }
    return payload, _verify_snapshot(payload, plan.published, where="statistics")


def recompute_costs(
    records: Sequence[CostRecord],
    *,
    expected_query_ids: Sequence[str],
    published: PublishedSnapshot,
) -> tuple[Mapping[str, Any], str]:
    typed = tuple(records)
    if not typed or any(not isinstance(record, CostRecord) for record in typed):
        raise RecomputeContractError("cost recompute requires typed CostRecord rows")
    result = reconcile_cold_warm_costs(
        typed, expected_query_ids=tuple(expected_query_ids)
    )
    payload = {
        "schema": "policy-learnware.v02-cost-independent-recompute.v0",
        "raw_record_count": len(typed),
        "raw_records_digest": sha256_json(
            [
                record.to_dict()
                for record in sorted(typed, key=lambda row: (row.query_id, row.mode))
            ]
        ),
        "reconciliation": result.to_dict(),
        "published_cost_aggregate_consumed": False,
    }
    return payload, _verify_snapshot(payload, published, where="costs")


@dataclass(frozen=True)
class InformationAuditInputs:
    public_artifact_root: str | Path
    public_artifact_rules: tuple[PublicArtifactRule, ...]
    replay_selector: Callable[[Path, Path, Path], str]
    market_public_root: str | Path
    measurement_root: str | Path
    selector_outputs_root: str | Path
    published: PublishedSnapshot

    def __post_init__(self) -> None:
        rules = tuple(self.public_artifact_rules)
        if not rules or any(not isinstance(rule, PublicArtifactRule) for rule in rules):
            raise RecomputeContractError(
                "information audit requires public artifact rules"
            )
        if not callable(self.replay_selector):
            raise RecomputeContractError("replay_selector must be callable")
        object.__setattr__(self, "public_artifact_rules", rules)


def recompute_information_audit(
    inputs: InformationAuditInputs,
    *,
    market: PublicMarketView,
    selections: Mapping[str, SelectionRecord],
) -> tuple[Mapping[str, Any], str]:
    public_projection = {
        opaque_id: {
            "opaque_learnware_id": entry.opaque_learnware_id,
            "normalized_source_competence": entry.normalized_source_competence,
            "tie_break_token": entry.tie_break_token,
        }
        for opaque_id, entry in sorted(market.entries.items())
    }
    market_audit = audit_public_market_entries(public_projection)
    artifact_audit = audit_public_artifacts(
        inputs.public_artifact_root, inputs.public_artifact_rules
    )
    evidence_audits = {
        unit_id: audit_evidence_contract(record.evidence_contract)
        for unit_id, record in sorted(selections.items())
    }
    oracle_audit = audit_oracle_independence(
        inputs.replay_selector,
        market_public_root=inputs.market_public_root,
        measurement_root=inputs.measurement_root,
        selector_outputs_root=inputs.selector_outputs_root,
    )
    evidence_safe = all(
        audit.passed and selections[unit_id].evidence_contract.is_public_zero_update
        for unit_id, audit in evidence_audits.items()
    )
    passed = bool(
        market_audit.passed
        and artifact_audit.passed
        and evidence_safe
        and oracle_audit.passed
    )
    payload = {
        "schema": "policy-learnware.v02-information-independent-recompute.v0",
        "passed": passed,
        "public_market": market_audit.to_dict(),
        "public_artifacts": artifact_audit.to_dict(),
        "evidence_contracts": {
            unit_id: audit.to_dict() for unit_id, audit in evidence_audits.items()
        },
        "oracle_independence": oracle_audit.to_dict(),
        "selector_oracle_root_capability": False,
        "precomputed_audit_pass_fields_consumed": False,
    }
    if not passed:
        raise RecomputeContractError(
            "information isolation recompute found one or more audit violations"
        )
    return payload, _verify_snapshot(
        payload, inputs.published, where="information audit"
    )


@dataclass(frozen=True)
class IndependentRecomputeInputs:
    coverage: FullCoverageContract
    source: SourceRecomputeInputs
    market: PublicMarketView
    gate0_units: tuple[Gate0AuditUnit, ...]
    representation_units: tuple[RepresentationReplayUnit, ...]
    selector_units: tuple[SelectorReplayUnit, ...]
    oracle_rows: tuple[OracleEpisodeRow, ...]
    oracle_units: tuple[OracleMetricUnit, ...]
    statistics: StatisticsPlan
    cost_records: tuple[CostRecord, ...]
    published_costs: PublishedSnapshot
    information_audit: InformationAuditInputs

    def __post_init__(self) -> None:
        if not isinstance(self.coverage, FullCoverageContract):
            raise RecomputeContractError("coverage has the wrong type")
        if not isinstance(self.source, SourceRecomputeInputs):
            raise RecomputeContractError("source recompute inputs have the wrong type")
        if not isinstance(self.market, PublicMarketView):
            raise RecomputeContractError("market has the wrong type")
        if not isinstance(self.statistics, StatisticsPlan):
            raise RecomputeContractError("statistics plan has the wrong type")
        if not isinstance(self.information_audit, InformationAuditInputs):
            raise RecomputeContractError("information audit inputs have the wrong type")
        for name, expected_type in (
            ("gate0_units", Gate0AuditUnit),
            ("representation_units", RepresentationReplayUnit),
            ("selector_units", SelectorReplayUnit),
            ("oracle_rows", OracleEpisodeRow),
            ("oracle_units", OracleMetricUnit),
            ("cost_records", CostRecord),
        ):
            values = tuple(getattr(self, name))
            if not values or any(
                not isinstance(item, expected_type) for item in values
            ):
                raise RecomputeContractError(f"{name} is empty or has an invalid type")
            object.__setattr__(self, name, values)


@dataclass(frozen=True)
class LoadedFormalSourceSection:
    """Typed source/championization inputs plus its frozen public projection."""

    source: SourceRecomputeInputs
    source_anchor_ids: tuple[str, ...]
    source_market_bindings: Mapping[str, str]
    public_market_ids: tuple[str, ...]
    policy_market_id: str
    protocol_family_id: str
    public_entries: Mapping[str, PublicMarketEntry]


@dataclass(frozen=True)
class LoadedFormalOracleSection:
    rows: tuple[OracleEpisodeRow, ...]
    units: tuple[OracleMetricUnit, ...]


@dataclass(frozen=True)
class LoadedFormalCostsSection:
    records: tuple[CostRecord, ...]
    published: PublishedSnapshot


@dataclass(frozen=True)
class LoadedFormalRecomputeSection:
    """Result of one source-owned inverse and its canonical input projection."""

    section: str
    value: object
    projection: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.section not in FORMAL_RECOMPUTE_SECTION_SOURCE_SCHEMAS:
            raise RecomputeContractError("loaded formal recompute section is unknown")
        if not isinstance(self.projection, Mapping):
            raise RecomputeContractError("loaded formal section projection is invalid")
        object.__setattr__(self, "projection", _deep_freeze(self.projection))


def _snapshot_projection(snapshot: PublishedSnapshot) -> dict[str, Any]:
    if not isinstance(snapshot, PublishedSnapshot):
        raise RecomputeContractError("formal source contains an invalid snapshot")
    return {"payload": canonicalize(snapshot.payload), "digest": snapshot.digest}


def _load_snapshot(value: object, where: str) -> PublishedSnapshot:
    data = _strict_json_mapping(value, {"payload", "digest"}, where)
    if not isinstance(data["payload"], Mapping):
        raise RecomputeContractError(f"{where}.payload must be a JSON object")
    return PublishedSnapshot(payload=data["payload"], digest=data["digest"])


def _source_row(value: object, where: str) -> SourceEpisodeRow:
    fields = set(SourceEpisodeRow.__dataclass_fields__)
    data = _strict_json_mapping(value, fields, where)
    try:
        return SourceEpisodeRow(**{name: data[name] for name in fields})
    except (TypeError, ValueError) as error:
        raise RecomputeContractError(f"{where} is invalid: {error}") from error


def _source_projection_from_inputs(inputs: IndependentRecomputeInputs) -> dict[str, Any]:
    return {
        "coverage": {
            "source_anchor_ids": list(inputs.coverage.source_anchor_ids),
            "source_market_bindings": dict(inputs.coverage.source_market_bindings),
            "public_market_ids": list(inputs.coverage.public_market_ids),
        },
        "source": {
            "selection_rows": [
                row.to_dict() for row in _ordered_source_rows(inputs.source.selection_rows)
            ],
            "attestation_rows": [
                row.to_dict()
                for row in _ordered_source_rows(inputs.source.attestation_rows)
            ],
            "competence_floors": dict(inputs.source.competence_floors),
            "competence_mode": inputs.source.competence_mode,
            "mean_tolerance": inputs.source.mean_tolerance,
            "lcb_z": inputs.source.lcb_z,
            "return_contract_id": inputs.source.return_contract_id,
            "published": _snapshot_projection(inputs.source.published),
        },
        "market": {
            "policy_market_id": inputs.market.policy_market_id,
            "protocol_family_id": inputs.market.protocol_family_id,
            "entries": {
                opaque_id: entry.to_dict()
                for opaque_id, entry in sorted(inputs.market.entries.items())
            },
        },
    }


def _source_projection_from_loaded(value: LoadedFormalSourceSection) -> dict[str, Any]:
    return {
        "coverage": {
            "source_anchor_ids": list(value.source_anchor_ids),
            "source_market_bindings": dict(value.source_market_bindings),
            "public_market_ids": list(value.public_market_ids),
        },
        "source": {
            "selection_rows": [
                row.to_dict() for row in _ordered_source_rows(value.source.selection_rows)
            ],
            "attestation_rows": [
                row.to_dict()
                for row in _ordered_source_rows(value.source.attestation_rows)
            ],
            "competence_floors": dict(value.source.competence_floors),
            "competence_mode": value.source.competence_mode,
            "mean_tolerance": value.source.mean_tolerance,
            "lcb_z": value.source.lcb_z,
            "return_contract_id": value.source.return_contract_id,
            "published": _snapshot_projection(value.source.published),
        },
        "market": {
            "policy_market_id": value.policy_market_id,
            "protocol_family_id": value.protocol_family_id,
            "entries": {
                opaque_id: entry.to_dict()
                for opaque_id, entry in sorted(value.public_entries.items())
            },
        },
    }


def _load_source_projection(value: object) -> LoadedFormalSourceSection:
    data = _strict_json_mapping(value, {"coverage", "source", "market"}, "source projection")
    coverage = _strict_json_mapping(
        data["coverage"],
        {"source_anchor_ids", "source_market_bindings", "public_market_ids"},
        "source coverage projection",
    )
    if not isinstance(coverage["source_anchor_ids"], list) or not isinstance(
        coverage["public_market_ids"], list
    ) or not isinstance(coverage["source_market_bindings"], Mapping):
        raise RecomputeContractError("source coverage projection has invalid containers")
    source = _strict_json_mapping(
        data["source"],
        {
            "selection_rows",
            "attestation_rows",
            "competence_floors",
            "competence_mode",
            "mean_tolerance",
            "lcb_z",
            "return_contract_id",
            "published",
        },
        "source input projection",
    )
    if not isinstance(source["selection_rows"], list) or not isinstance(
        source["attestation_rows"], list
    ) or not isinstance(source["competence_floors"], Mapping):
        raise RecomputeContractError("source input projection has invalid containers")
    try:
        typed_source = SourceRecomputeInputs(
            selection_rows=tuple(
                _source_row(row, f"source.selection_rows[{index}]")
                for index, row in enumerate(source["selection_rows"])
            ),
            attestation_rows=tuple(
                _source_row(row, f"source.attestation_rows[{index}]")
                for index, row in enumerate(source["attestation_rows"])
            ),
            competence_floors=source["competence_floors"],
            competence_mode=source["competence_mode"],
            mean_tolerance=source["mean_tolerance"],
            lcb_z=source["lcb_z"],
            return_contract_id=source["return_contract_id"],
            published=_load_snapshot(source["published"], "source published snapshot"),
        )
    except (TypeError, ValueError) as error:
        raise RecomputeContractError(f"source typed reconstruction failed: {error}") from error
    market = _strict_json_mapping(
        data["market"],
        {"policy_market_id", "protocol_family_id", "entries"},
        "source market projection",
    )
    if not isinstance(market["entries"], Mapping):
        raise RecomputeContractError("source market entries must be an object")
    try:
        entries = {
            str(opaque_id): PublicMarketEntry.from_dict(entry)
            for opaque_id, entry in market["entries"].items()
        }
        anchor_ids = _canonical_tuple(
            tuple(coverage["source_anchor_ids"]), "source_anchor_ids"
        )
        public_ids = _canonical_tuple(
            tuple(coverage["public_market_ids"]), "public_market_ids"
        )
        bindings = {
            _nonempty(anchor, "source_market_bindings key"): _nonempty(
                opaque_id, "source_market_bindings value"
            )
            for anchor, opaque_id in coverage["source_market_bindings"].items()
        }
        if set(bindings) != set(anchor_ids) or set(bindings.values()) != set(public_ids):
            raise RecomputeContractError(
                "source market bindings do not exactly cover source/public IDs"
            )
        if len(set(bindings.values())) != len(bindings) or set(entries) != set(public_ids):
            raise RecomputeContractError("source public market projection is not one-to-one")
        loaded = LoadedFormalSourceSection(
            source=typed_source,
            source_anchor_ids=anchor_ids,
            source_market_bindings=MappingProxyType(dict(sorted(bindings.items()))),
            public_market_ids=public_ids,
            policy_market_id=_nonempty(market["policy_market_id"], "policy_market_id"),
            protocol_family_id=_nonempty(
                market["protocol_family_id"], "protocol_family_id"
            ),
            public_entries=MappingProxyType(dict(sorted(entries.items()))),
        )
    except (TypeError, ValueError) as error:
        if isinstance(error, RecomputeContractError):
            raise
        raise RecomputeContractError(f"source market reconstruction failed: {error}") from error
    return loaded


def _operator_audit(value: object, where: str) -> DynamicsOperatorAudit:
    fields = set(DynamicsOperatorAudit.__dataclass_fields__)
    data = _strict_json_mapping(value, fields | {"schema"}, where)
    if data["schema"] != "policy-learnware.v02-dynamics-operator-audit.v0":
        raise RecomputeContractError(f"{where} has another schema")
    kwargs = {name: data[name] for name in fields}
    kwargs["changed_leaves"] = tuple(kwargs["changed_leaves"])
    kwargs["unchanged_leaves"] = tuple(kwargs["unchanged_leaves"])
    try:
        result = DynamicsOperatorAudit(**kwargs)
    except (TypeError, ValueError) as error:
        raise RecomputeContractError(f"{where} is invalid: {error}") from error
    if canonicalize(data) != result.to_dict():
        raise RecomputeContractError(f"{where} is not canonical")
    return result


def _rollout_audit(value: object, where: str) -> RolloutAudit:
    fields = set(RolloutAudit.__dataclass_fields__)
    data = _strict_json_mapping(value, fields | {"schema"}, where)
    if data["schema"] != "policy-learnware.v02-rollout-audit.v0":
        raise RecomputeContractError(f"{where} has another schema")
    try:
        result = RolloutAudit(**{name: data[name] for name in fields})
    except (TypeError, ValueError) as error:
        raise RecomputeContractError(f"{where} is invalid: {error}") from error
    if canonicalize(data) != result.to_dict():
        raise RecomputeContractError(f"{where} is not canonical")
    return result


def _gate0_audit(value: object, where: str) -> Gate0Audit:
    fields = set(Gate0Audit.__dataclass_fields__)
    data = _strict_json_mapping(value, fields | {"schema"}, where)
    if data["schema"] != "policy-learnware.v02-gate0-audit.v0":
        raise RecomputeContractError(f"{where} has another schema")
    kwargs = {name: data[name] for name in fields}
    kwargs["scalar_rollout"] = _rollout_audit(
        kwargs["scalar_rollout"], f"{where}.scalar_rollout"
    )
    kwargs["reasons"] = tuple(kwargs["reasons"])
    try:
        result = Gate0Audit(**kwargs)
    except (TypeError, ValueError) as error:
        raise RecomputeContractError(f"{where} is invalid: {error}") from error
    if canonicalize(data) != result.to_dict():
        raise RecomputeContractError(f"{where} is not canonical")
    return result


def _gate0_projection(units: Sequence[Gate0AuditUnit]) -> dict[str, Any]:
    return {
        "units": [
            {
                "audit_id": unit.audit_id,
                "operator_audit": unit.operator_audit.to_dict(),
                "gate0_audit": unit.gate0_audit.to_dict(),
                "published_operator": _snapshot_projection(unit.published_operator),
                "published_gate0": _snapshot_projection(unit.published_gate0),
            }
            for unit in sorted(units, key=lambda item: item.audit_id)
        ]
    }


def _load_gate0_projection(value: object) -> tuple[Gate0AuditUnit, ...]:
    data = _strict_json_mapping(value, {"units"}, "gate0 projection")
    if not isinstance(data["units"], list) or not data["units"]:
        raise RecomputeContractError("gate0 projection requires non-empty units")
    units: list[Gate0AuditUnit] = []
    for index, raw in enumerate(data["units"]):
        row = _strict_json_mapping(
            raw,
            {
                "audit_id",
                "operator_audit",
                "gate0_audit",
                "published_operator",
                "published_gate0",
            },
            f"gate0.units[{index}]",
        )
        units.append(
            Gate0AuditUnit(
                audit_id=row["audit_id"],
                operator_audit=_operator_audit(
                    row["operator_audit"], f"gate0.units[{index}].operator_audit"
                ),
                gate0_audit=_gate0_audit(
                    row["gate0_audit"], f"gate0.units[{index}].gate0_audit"
                ),
                published_operator=_load_snapshot(
                    row["published_operator"],
                    f"gate0.units[{index}].published_operator",
                ),
                published_gate0=_load_snapshot(
                    row["published_gate0"], f"gate0.units[{index}].published_gate0"
                ),
            )
        )
    if tuple(unit.audit_id for unit in units) != tuple(
        sorted(unit.audit_id for unit in units)
    ) or len({unit.audit_id for unit in units}) != len(units):
        raise RecomputeContractError("gate0 units must be sorted and unique")
    return tuple(units)


@dataclass(frozen=True)
class _FormalDeploymentEntry:
    execution_abi: ExecutionABIRecord
    bundle_digest: str | None = None


def _deployment_projection(value: Any) -> dict[str, Any]:
    abi = _execution_abi(value)
    bundle = getattr(value, "bundle_digest", None)
    if bundle is not None:
        bundle = _digest(bundle, "deployment bundle_digest")
    return {"execution_abi": abi.to_dict(), "bundle_digest": bundle}


def _oracle_row(value: object, where: str) -> OracleEpisodeRow:
    fields = set(OracleEpisodeRow.__dataclass_fields__)
    data = _strict_json_mapping(value, fields | {"schema"}, where)
    if data["schema"] != "policy-learnware.v02-oracle-episode-row.v0":
        raise RecomputeContractError(f"{where} has another schema")
    try:
        result = OracleEpisodeRow(**{name: data[name] for name in fields})
    except (TypeError, ValueError) as error:
        raise RecomputeContractError(f"{where} is invalid: {error}") from error
    if canonicalize(data) != result.to_private_dict():
        raise RecomputeContractError(f"{where} is not canonical")
    return result


def _oracle_unit_projection(unit: OracleMetricUnit) -> dict[str, Any]:
    return {
        "query_id": unit.query_id,
        "task_id": unit.task_id,
        "axis_id": unit.axis_id,
        "context_id": unit.context_id,
        "method_id": unit.method_id,
        "market_ids": list(unit.market_ids),
        "deployment_registry": {
            opaque_id: _deployment_projection(unit.deployment_registry[opaque_id])
            for opaque_id in unit.market_ids
        },
        "target_execution_abi": unit.target_execution_abi.to_dict(),
        "private_target_instance_digest": unit.private_target_instance_digest,
        "evaluation_protocol_id": unit.evaluation_protocol_id,
        "failure_floor": unit.failure_floor,
        "epsilon": unit.epsilon,
        "tie_atol": unit.tie_atol,
        "candidate_paired_seeds": unit.candidate_paired_seeds,
        "published": _snapshot_projection(unit.published),
    }


def _oracle_projection(
    rows: Sequence[OracleEpisodeRow], units: Sequence[OracleMetricUnit]
) -> dict[str, Any]:
    return {
        "rows": [
            row.to_private_dict()
            for row in sorted(
                rows,
                key=lambda item: (
                    item.opaque_query_id,
                    item.opaque_learnware_id,
                    item.episode_index,
                ),
            )
        ],
        "units": [
            _oracle_unit_projection(unit)
            for unit in sorted(units, key=lambda item: item.unit_id)
        ],
    }


def _load_oracle_projection(value: object) -> LoadedFormalOracleSection:
    data = _strict_json_mapping(value, {"rows", "units"}, "oracle projection")
    if not isinstance(data["rows"], list) or not data["rows"] or not isinstance(
        data["units"], list
    ) or not data["units"]:
        raise RecomputeContractError("oracle projection requires raw rows and units")
    rows = tuple(
        _oracle_row(row, f"oracle.rows[{index}]")
        for index, row in enumerate(data["rows"])
    )
    units: list[OracleMetricUnit] = []
    unit_fields = {
        "query_id",
        "task_id",
        "axis_id",
        "context_id",
        "method_id",
        "market_ids",
        "deployment_registry",
        "target_execution_abi",
        "private_target_instance_digest",
        "evaluation_protocol_id",
        "failure_floor",
        "epsilon",
        "tie_atol",
        "candidate_paired_seeds",
        "published",
    }
    for index, raw in enumerate(data["units"]):
        row = _strict_json_mapping(raw, unit_fields, f"oracle.units[{index}]")
        if not isinstance(row["market_ids"], list) or not isinstance(
            row["deployment_registry"], Mapping
        ):
            raise RecomputeContractError("oracle unit market containers are invalid")
        registry: dict[str, _FormalDeploymentEntry] = {}
        for opaque_id, raw_entry in row["deployment_registry"].items():
            entry = _strict_json_mapping(
                raw_entry,
                {"execution_abi", "bundle_digest"},
                f"oracle.units[{index}].deployment_registry[{opaque_id!r}]",
            )
            try:
                registry[str(opaque_id)] = _FormalDeploymentEntry(
                    execution_abi=ExecutionABIRecord.from_dict(entry["execution_abi"]),
                    bundle_digest=(
                        None
                        if entry["bundle_digest"] is None
                        else _digest(entry["bundle_digest"], "deployment bundle_digest")
                    ),
                )
            except (TypeError, ValueError) as error:
                raise RecomputeContractError(
                    f"oracle deployment entry is invalid: {error}"
                ) from error
        try:
            units.append(
                OracleMetricUnit(
                    query_id=row["query_id"],
                    task_id=row["task_id"],
                    axis_id=row["axis_id"],
                    context_id=row["context_id"],
                    method_id=row["method_id"],
                    market_ids=tuple(row["market_ids"]),
                    deployment_registry=registry,
                    target_execution_abi=ExecutionABIRecord.from_dict(
                        row["target_execution_abi"]
                    ),
                    private_target_instance_digest=row[
                        "private_target_instance_digest"
                    ],
                    evaluation_protocol_id=row["evaluation_protocol_id"],
                    failure_floor=row["failure_floor"],
                    epsilon=row["epsilon"],
                    tie_atol=row["tie_atol"],
                    candidate_paired_seeds=row["candidate_paired_seeds"],
                    published=_load_snapshot(
                        row["published"], f"oracle.units[{index}].published"
                    ),
                )
            )
        except (TypeError, ValueError) as error:
            raise RecomputeContractError(f"oracle unit is invalid: {error}") from error
    loaded = LoadedFormalOracleSection(rows=rows, units=tuple(units))
    if canonicalize(data) != _oracle_projection(loaded.rows, loaded.units):
        raise RecomputeContractError("oracle projection is not canonical")
    return loaded


def _comparison_projection(value: PairedComparisonPlan) -> dict[str, Any]:
    return {
        name: getattr(value, name) for name in value.__dataclass_fields__
    }


def _statistics_projection(value: StatisticsPlan) -> dict[str, Any]:
    return {
        "resamples": value.resamples,
        "bootstrap_seed": value.bootstrap_seed,
        "confidence_level": value.confidence_level,
        "comparisons": [
            _comparison_projection(item)
            for item in sorted(value.comparisons, key=lambda item: item.comparison_id)
        ],
        "published": _snapshot_projection(value.published),
    }


def _load_statistics_projection(value: object) -> StatisticsPlan:
    data = _strict_json_mapping(
        value,
        {"resamples", "bootstrap_seed", "confidence_level", "comparisons", "published"},
        "statistics projection",
    )
    if not isinstance(data["comparisons"], list):
        raise RecomputeContractError("statistics comparisons must be a list")
    comparison_fields = set(PairedComparisonPlan.__dataclass_fields__)
    comparisons: list[PairedComparisonPlan] = []
    for index, raw in enumerate(data["comparisons"]):
        row = _strict_json_mapping(
            raw, comparison_fields, f"statistics.comparisons[{index}]"
        )
        try:
            comparisons.append(
                PairedComparisonPlan(**{name: row[name] for name in comparison_fields})
            )
        except (TypeError, ValueError) as error:
            raise RecomputeContractError(
                f"statistics comparison is invalid: {error}"
            ) from error
    try:
        result = StatisticsPlan(
            resamples=data["resamples"],
            bootstrap_seed=data["bootstrap_seed"],
            confidence_level=data["confidence_level"],
            comparisons=tuple(comparisons),
            published=_load_snapshot(data["published"], "statistics published"),
        )
    except (TypeError, ValueError) as error:
        raise RecomputeContractError(f"statistics plan is invalid: {error}") from error
    if canonicalize(data) != _statistics_projection(result):
        raise RecomputeContractError("statistics projection is not canonical")
    return result


def _costs_projection(
    records: Sequence[CostRecord], published: PublishedSnapshot
) -> dict[str, Any]:
    return {
        "records": [
            row.to_dict()
            for row in sorted(records, key=lambda item: (item.query_id, item.mode))
        ],
        "published": _snapshot_projection(published),
    }


def _load_costs_projection(value: object) -> LoadedFormalCostsSection:
    data = _strict_json_mapping(value, {"records", "published"}, "costs projection")
    if not isinstance(data["records"], list) or not data["records"]:
        raise RecomputeContractError("costs projection requires raw records")
    try:
        records = tuple(CostRecord.from_dict(row) for row in data["records"])
        result = LoadedFormalCostsSection(
            records=records,
            published=_load_snapshot(data["published"], "costs published"),
        )
    except (TypeError, ValueError) as error:
        raise RecomputeContractError(f"cost records are invalid: {error}") from error
    if canonicalize(data) != _costs_projection(result.records, result.published):
        raise RecomputeContractError("costs projection is not canonical")
    return result


def _live_section_projection(
    section: str, inputs: IndependentRecomputeInputs
) -> Mapping[str, Any]:
    if section == "source":
        return _source_projection_from_inputs(inputs)
    if section == "gate0":
        return _gate0_projection(inputs.gate0_units)
    if section == "oracle":
        return _oracle_projection(inputs.oracle_rows, inputs.oracle_units)
    if section == "statistics":
        return _statistics_projection(inputs.statistics)
    if section == "costs":
        return _costs_projection(inputs.cost_records, inputs.published_costs)
    raise RecomputeContractError(
        f"formal section {section!r} requires an unavailable runtime loader"
    )


def build_formal_recompute_section_source(
    section: str,
    inputs: IndependentRecomputeInputs,
    *,
    config_digest: str,
) -> Mapping[str, Any]:
    """Build one canonical source artifact for a fully serializable section.

    This is a serializer, not an authority mint.  Formal authority is attached
    only after the persisted bytes are re-read, reconstructed into typed
    objects, compared to the live inputs, and the entire replay succeeds.
    """

    if not isinstance(inputs, IndependentRecomputeInputs):
        raise RecomputeContractError("formal section serializer requires typed inputs")
    schema = FORMAL_RECOMPUTE_SECTION_SOURCE_SCHEMAS.get(section)
    if schema is None:
        dependency = formal_recompute_loader_dependencies().get(section, "unknown")
        raise RecomputeContractError(
            f"formal section {section!r} is not serializable: {dependency}"
        )
    projection = canonicalize(_live_section_projection(section, inputs))
    return MappingProxyType(
        {
            "schema": schema,
            "config_digest": _digest(config_digest, "config_digest"),
            "input_projection": projection,
            "input_projection_digest": sha256_json(projection),
        }
    )


def _load_section_envelope(
    payloads: tuple[Mapping[str, Any], ...],
    *,
    section: str,
    expected_config_digest: str,
) -> Mapping[str, Any]:
    if len(payloads) != 1:
        raise RecomputeContractError(
            f"formal {section} loader requires exactly one canonical source artifact"
        )
    data = _strict_json_mapping(
        payloads[0],
        {"schema", "config_digest", "input_projection", "input_projection_digest"},
        f"formal {section} source",
    )
    if data["schema"] != FORMAL_RECOMPUTE_SECTION_SOURCE_SCHEMAS[section]:
        raise RecomputeContractError(f"formal {section} source schema differs")
    if data["config_digest"] != _digest(
        expected_config_digest, "expected config_digest"
    ):
        raise RecomputeContractError(f"formal {section} source is config-misbound")
    if not isinstance(data["input_projection"], Mapping):
        raise RecomputeContractError(f"formal {section} input projection is invalid")
    if data["input_projection_digest"] != sha256_json(data["input_projection"]):
        raise RecomputeContractError(
            f"formal {section} input projection digest disagrees"
        )
    return data["input_projection"]


def _load_formal_source_section(
    payloads: tuple[Mapping[str, Any], ...], expected_config_digest: str
) -> LoadedFormalRecomputeSection:
    projection = _load_section_envelope(
        payloads, section="source", expected_config_digest=expected_config_digest
    )
    value = _load_source_projection(projection)
    rebuilt = _source_projection_from_loaded(value)
    mismatch = _first_mismatch(projection, rebuilt)
    if mismatch is not None:
        raise RecomputeContractError(f"formal source typed projection {mismatch}")
    return LoadedFormalRecomputeSection("source", value, rebuilt)


def _load_formal_gate0_section(
    payloads: tuple[Mapping[str, Any], ...], expected_config_digest: str
) -> LoadedFormalRecomputeSection:
    projection = _load_section_envelope(
        payloads, section="gate0", expected_config_digest=expected_config_digest
    )
    value = _load_gate0_projection(projection)
    rebuilt = _gate0_projection(value)
    mismatch = _first_mismatch(projection, rebuilt)
    if mismatch is not None:
        raise RecomputeContractError(f"formal gate0 typed projection {mismatch}")
    return LoadedFormalRecomputeSection("gate0", value, rebuilt)


def _load_formal_oracle_section(
    payloads: tuple[Mapping[str, Any], ...], expected_config_digest: str
) -> LoadedFormalRecomputeSection:
    projection = _load_section_envelope(
        payloads, section="oracle", expected_config_digest=expected_config_digest
    )
    value = _load_oracle_projection(projection)
    rebuilt = _oracle_projection(value.rows, value.units)
    mismatch = _first_mismatch(projection, rebuilt)
    if mismatch is not None:
        raise RecomputeContractError(f"formal oracle typed projection {mismatch}")
    return LoadedFormalRecomputeSection("oracle", value, rebuilt)


def _load_formal_statistics_section(
    payloads: tuple[Mapping[str, Any], ...], expected_config_digest: str
) -> LoadedFormalRecomputeSection:
    projection = _load_section_envelope(
        payloads, section="statistics", expected_config_digest=expected_config_digest
    )
    value = _load_statistics_projection(projection)
    rebuilt = _statistics_projection(value)
    mismatch = _first_mismatch(projection, rebuilt)
    if mismatch is not None:
        raise RecomputeContractError(f"formal statistics typed projection {mismatch}")
    return LoadedFormalRecomputeSection("statistics", value, rebuilt)


def _load_formal_costs_section(
    payloads: tuple[Mapping[str, Any], ...], expected_config_digest: str
) -> LoadedFormalRecomputeSection:
    projection = _load_section_envelope(
        payloads, section="costs", expected_config_digest=expected_config_digest
    )
    value = _load_costs_projection(projection)
    rebuilt = _costs_projection(value.records, value.published)
    mismatch = _first_mismatch(projection, rebuilt)
    if mismatch is not None:
        raise RecomputeContractError(f"formal costs typed projection {mismatch}")
    return LoadedFormalRecomputeSection("costs", value, rebuilt)


FormalRecomputeSourceLoader = Callable[
    [tuple[Mapping[str, Any], ...], str], LoadedFormalRecomputeSection
]


_STRUCTURAL_RECOMPUTE_SOURCE_LOADERS: Mapping[
    str, FormalRecomputeSourceLoader
] = MappingProxyType(
    {
        "source": _load_formal_source_section,
        "gate0": _load_formal_gate0_section,
        "oracle": _load_formal_oracle_section,
        "statistics": _load_formal_statistics_section,
        "costs": _load_formal_costs_section,
    }
)

# Structural inversion is necessary but not sufficient for formal authority.
# None of the five inverses above is registered here until its scientific or
# runtime contract is independently bound to the reviewed freeze (see
# ``formal_recompute_loader_dependencies``).  In particular, serializing a
# caller-created live object and reading it back is not source ownership.
_FORMAL_RECOMPUTE_SOURCE_LOADERS: Mapping[
    str, FormalRecomputeSourceLoader
] = MappingProxyType({})


def formal_recompute_loader_dependencies() -> Mapping[str, str]:
    """Exact reason each still-unregistered section cannot be reconstructed."""

    return MappingProxyType(
        {
            "source": (
                "typed rows are structurally invertible, but the reviewed anchor grid, "
                "admitted server candidates, mean_tolerance, lcb_z, and return contract "
                "are not jointly bound by the formal config/freeze"
            ),
            "gate0": (
                "typed audit summaries are structurally invertible, but raw rollout "
                "arrays and a config-derived axis/environment runtime replay are absent"
            ),
            "representations": (
                "requires a frozen source-owned SemanticEncoderProtocol loader and "
                "raw probe-array receipt; JSON metadata cannot reconstruct encode()"
            ),
            "selectors": (
                "requires frozen source-owned selector/model loaders for all nine "
                "methods; a published SelectionRecord cannot reconstruct select()"
            ),
            "oracle": (
                "raw episode rows are structurally invertible, but query/task/axis, "
                "deployment ABI, evaluation protocol, floor, and tie rules are not "
                "exactly derived from the reviewed config/freeze"
            ),
            "statistics": (
                "the plan is structurally invertible, but comparison families, seeds, "
                "null boundaries, and noninferiority rules are not fully config-derived"
            ),
            "costs": (
                "raw CostRecords are structurally invertible, but their opaque "
                "cost_contract_digest has no reviewed contract payload in the freeze"
            ),
            "information": (
                "requires a source-owned three-root replay callback/capability "
                "launcher; a persisted isolation pass bit cannot reconstruct it"
            ),
        }
    )


def missing_formal_recompute_source_loaders() -> tuple[str, ...]:
    """Sections that cannot yet reconstruct live inputs from canonical refs."""

    return tuple(sorted(RECOMPUTE_SECTIONS - set(_FORMAL_RECOMPUTE_SOURCE_LOADERS)))


def structurally_reconstructable_formal_recompute_sections() -> tuple[str, ...]:
    """Sections with strict typed inverses but no formal authority yet."""

    return tuple(sorted(_STRUCTURAL_RECOMPUTE_SOURCE_LOADERS))


def verify_structural_formal_recompute_source_sections(
    inputs: IndependentRecomputeInputs,
    *,
    source_binding: FormalRecomputeSourceBinding,
) -> Mapping[str, str]:
    """Diagnostic: reconstruct and exact-compare every structural inverse.

    Passing this diagnostic does not confer formal authority because the
    dependencies reported by :func:`formal_recompute_loader_dependencies`
    remain outside the reviewed freeze.
    """

    if not isinstance(inputs, IndependentRecomputeInputs):
        raise RecomputeContractError("formal source verification requires typed inputs")
    if not isinstance(source_binding, FormalRecomputeSourceBinding):
        raise RecomputeContractError("formal source verification requires a binding")
    result: dict[str, str] = {}
    for section, loader in sorted(_STRUCTURAL_RECOMPUTE_SOURCE_LOADERS.items()):
        payloads: list[Mapping[str, Any]] = []
        for reference in source_binding.manifest.section_sources[section]:
            try:
                payloads.append(
                    verify_canonical_evidence_ref(
                        reference,
                        experiment_root=source_binding.experiment_root,
                        expected_config_digest=source_binding.manifest.config_digest,
                        require_config_binding=True,
                        source_artifact=True,
                    )
                )
            except FormalGateEvidenceError as error:
                raise RecomputeContractError(
                    f"formal {section} source verification failed: {error}"
                ) from error
        loaded = loader(tuple(payloads), source_binding.manifest.config_digest)
        live = _live_section_projection(section, inputs)
        mismatch = _first_mismatch(loaded.projection, live)
        if mismatch is not None:
            raise RecomputeContractError(
                f"formal {section} source differs from live typed inputs: {mismatch}"
            )
        result[section] = sha256_json(loaded.projection)
    return MappingProxyType(result)


@dataclass(frozen=True)
class IndependentRecomputeReport:
    coverage_contract_digest: str
    checks: Mapping[str, bool]
    section_digests: Mapping[str, str]
    errors: tuple[str, ...]
    formal_provenance: FormalRecomputeProvenance | None = None
    _execution_authority: object | None = field(default=None, repr=False, compare=False)
    _source_binding: FormalRecomputeSourceBinding | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coverage_contract_digest",
            _digest(self.coverage_contract_digest, "coverage_contract_digest"),
        )
        expected = {
            "full_digest_coverage",
            "full_selector_replay",
            "full_statistical_recompute",
            "raw_numeric_subset_coverage",
            "cost_recompute",
            "information_isolation",
        }
        if set(self.checks) != expected or any(
            type(value) is not bool for value in self.checks.values()
        ):
            raise RecomputeContractError(
                "IndependentRecomputeReport checks are invalid"
            )
        object.__setattr__(self, "checks", MappingProxyType(dict(self.checks)))
        object.__setattr__(
            self,
            "section_digests",
            MappingProxyType(
                {
                    _nonempty(name, "section digest name"): _digest(
                        value, f"section_digests[{name!r}]"
                    )
                    for name, value in self.section_digests.items()
                }
            ),
        )
        if not set(self.section_digests).issubset(RECOMPUTE_SECTIONS):
            raise RecomputeContractError("report contains an unknown section digest")
        object.__setattr__(self, "errors", tuple(str(item) for item in self.errors))
        if self.formal_provenance is not None:
            if not isinstance(self.formal_provenance, FormalRecomputeProvenance):
                raise RecomputeContractError("formal_provenance has the wrong type")
            if (
                self.formal_provenance.coverage_contract_digest
                != self.coverage_contract_digest
            ):
                raise RecomputeContractError(
                    "formal provenance and report coverage digests differ"
                )
        if self._source_binding is not None and not isinstance(
            self._source_binding, FormalRecomputeSourceBinding
        ):
            raise RecomputeContractError("formal source binding has the wrong type")
        if self.passed and set(self.section_digests) != RECOMPUTE_SECTIONS:
            raise RecomputeContractError(
                "passing recompute report requires exact section digest coverage"
            )

    @property
    def full_digest_coverage(self) -> bool:
        return self.checks["full_digest_coverage"]

    @property
    def full_selector_replay(self) -> bool:
        return self.checks["full_selector_replay"]

    @property
    def full_statistical_recompute(self) -> bool:
        return self.checks["full_statistical_recompute"]

    @property
    def raw_numeric_subset_coverage(self) -> bool:
        return self.checks["raw_numeric_subset_coverage"]

    @property
    def cost_recompute(self) -> bool:
        return self.checks["cost_recompute"]

    @property
    def information_isolation(self) -> bool:
        return self.checks["information_isolation"]

    @property
    def passed(self) -> bool:
        return all(self.checks.values()) and not self.errors

    @property
    def is_formally_authoritative(self) -> bool:
        return bool(
            self._execution_authority is _TRUSTED_RECOMPUTE_AUTHORITY
            and self.formal_provenance is not None
            and self._source_binding is not None
        )

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def require_passed(self) -> None:
        if not self.passed:
            detail = "; ".join(self.errors) or "one or more recompute checks failed"
            raise RecomputeContractError(
                f"independent recompute failed closed: {detail}"
            )

    def require_formal_authority(
        self,
        *,
        expected_experiment_id: str,
        expected_config_digest: str,
        expected_config_file_sha256: str,
        expected_coverage_contract_digest: str | None = None,
    ) -> None:
        """Require a live trusted run and recheck its exact source bytes."""

        self.require_passed()
        if not self.is_formally_authoritative:
            raise RecomputeContractError(
                "persisted/reconstructed recompute reports lack trusted "
                "in-process execution authority"
            )
        assert self.formal_provenance is not None
        assert self._source_binding is not None
        expected = {
            "experiment_id": _experiment_id(expected_experiment_id),
            "config_digest": _digest(expected_config_digest, "expected config_digest"),
            "config_file_sha256": _digest(
                expected_config_file_sha256, "expected config_file_sha256"
            ),
        }
        for name, value in expected.items():
            if getattr(self.formal_provenance, name) != value:
                raise RecomputeContractError(
                    f"formal recompute report {name} binding differs"
                )
        if expected_coverage_contract_digest is not None and (
            self.coverage_contract_digest
            != _digest(
                expected_coverage_contract_digest,
                "expected coverage_contract_digest",
            )
        ):
            raise RecomputeContractError(
                "formal recompute report coverage contract differs"
            )
        binding = self._source_binding
        if (
            binding.manifest.experiment_id != expected["experiment_id"]
            or binding.manifest.config_digest != expected["config_digest"]
            or binding.manifest.config_file_sha256 != expected["config_file_sha256"]
            or binding.manifest.coverage_contract_digest
            != self.coverage_contract_digest
            or binding.manifest.digest != self.formal_provenance.source_manifest_digest
            or binding.manifest_ref != self.formal_provenance.source_manifest_ref
        ):
            raise RecomputeContractError(
                "formal recompute live source binding differs from report provenance"
            )
        current_sources = binding.verify_sources()
        if dict(current_sources) != dict(self.formal_provenance.section_source_digests):
            raise RecomputeContractError(
                "formal recompute source census changed after execution"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "IndependentRecomputeReport":
        """Strictly load a persisted report without trusting its ``passed`` flag.

        The six primitive recompute checks and the absence of errors derive
        ``passed``.  The canonical report digest remains the :attr:`digest`
        property of the reconstructed object and is never accepted as an
        uploaded assertion.
        """

        if not isinstance(value, Mapping):
            raise RecomputeContractError("IndependentRecomputeReport must be a mapping")
        check_names = {
            "full_digest_coverage",
            "full_selector_replay",
            "full_statistical_recompute",
            "raw_numeric_subset_coverage",
            "cost_recompute",
            "information_isolation",
        }
        base_expected = {
            "schema",
            "passed",
            "coverage_contract_digest",
            "section_digests",
            "errors",
            "precomputed_aggregates_or_gates_consumed",
            *check_names,
        }
        schema = value.get("schema")
        if schema == RECOMPUTE_SCHEMA:
            expected = base_expected
            formal_provenance = None
        elif schema == FORMAL_RECOMPUTE_SCHEMA:
            expected = base_expected | {"formal_provenance"}
            formal_provenance = FormalRecomputeProvenance.from_dict(
                value.get("formal_provenance")
            )
        else:
            raise RecomputeContractError(
                "unsupported independent recompute report schema"
            )
        missing = expected - set(value)
        unknown = set(value) - expected
        if missing or unknown:
            raise RecomputeContractError(
                "invalid IndependentRecomputeReport keys; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        if value["precomputed_aggregates_or_gates_consumed"] is not False:
            raise RecomputeContractError(
                "persisted recompute report claims consumption of untrusted aggregates/gates"
            )
        if type(value["passed"]) is not bool:
            raise RecomputeContractError("persisted recompute passed must be boolean")
        if not isinstance(value["section_digests"], Mapping):
            raise RecomputeContractError("section_digests must be a mapping")
        if not isinstance(value["errors"], list) or any(
            not isinstance(item, str) for item in value["errors"]
        ):
            raise RecomputeContractError("errors must be a list of strings")
        report = cls(
            coverage_contract_digest=value["coverage_contract_digest"],
            checks={name: value[name] for name in check_names},
            section_digests=value["section_digests"],
            errors=tuple(value["errors"]),
            formal_provenance=formal_provenance,
        )
        if value["passed"] != report.passed:
            raise RecomputeContractError(
                "persisted recompute passed disagrees with primitive checks/errors"
            )
        return report

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema": (
                FORMAL_RECOMPUTE_SCHEMA
                if self.formal_provenance is not None
                else RECOMPUTE_SCHEMA
            ),
            "passed": self.passed,
            **dict(self.checks),
            "coverage_contract_digest": self.coverage_contract_digest,
            "section_digests": dict(sorted(self.section_digests.items())),
            "errors": list(self.errors),
            "precomputed_aggregates_or_gates_consumed": False,
        }
        if self.formal_provenance is not None:
            result["formal_provenance"] = self.formal_provenance.to_dict()
        return result


def run_independent_recompute(
    inputs: IndependentRecomputeInputs,
) -> IndependentRecomputeReport:
    """Run every independent section and return a fail-closed typed report."""

    if not isinstance(inputs, IndependentRecomputeInputs):
        raise RecomputeContractError("inputs must be IndependentRecomputeInputs")
    errors: list[str] = []
    section_digests: dict[str, str] = {}
    section_ok = {
        "source": False,
        "gate0": False,
        "representations": False,
        "selectors": False,
        "oracle": False,
        "statistics": False,
        "costs": False,
        "information": False,
    }

    try:
        _, _, source_digest = recompute_source(
            inputs.source, coverage=inputs.coverage, market=inputs.market
        )
        section_digests["source"] = source_digest
        section_ok["source"] = True
    except Exception as error:  # every contract/library error becomes a closed report
        errors.append(f"source: {type(error).__name__}: {error}")

    try:
        _, gate0_digest = verify_gate0_audits(
            inputs.gate0_units, expected_ids=inputs.coverage.gate0_audit_ids
        )
        section_digests["gate0"] = gate0_digest
        section_ok["gate0"] = True
    except Exception as error:
        errors.append(f"gate0: {type(error).__name__}: {error}")

    try:
        _, representation_digest = replay_representations(
            inputs.representation_units,
            expected_ids=inputs.coverage.representation_unit_ids,
        )
        section_digests["representations"] = representation_digest
        section_ok["representations"] = True
    except Exception as error:
        errors.append(f"representations: {type(error).__name__}: {error}")

    selections: Mapping[str, SelectionRecord] = MappingProxyType({})
    try:
        selections, _, selector_digest = replay_selectors(
            inputs.selector_units,
            market=inputs.market,
            expected_ids=inputs.coverage.selector_unit_ids,
        )
        section_digests["selectors"] = selector_digest
        section_ok["selectors"] = True
    except Exception as error:
        errors.append(f"selectors: {type(error).__name__}: {error}")

    metrics: Mapping[str, Mapping[str, Any]] = MappingProxyType({})
    if section_ok["selectors"]:
        try:
            metrics, _, oracle_digest = recompute_oracle_metrics(
                inputs.oracle_rows,
                inputs.oracle_units,
                selections=selections,
                expected_unit_ids=inputs.coverage.oracle_unit_ids,
                expected_query_ids=inputs.coverage.oracle_query_ids,
                expected_market_ids=inputs.coverage.public_market_ids,
            )
            section_digests["oracle"] = oracle_digest
            section_ok["oracle"] = True
        except Exception as error:
            errors.append(f"oracle: {type(error).__name__}: {error}")
    else:
        errors.append("oracle: blocked by failed selector replay")

    if section_ok["oracle"]:
        try:
            _, statistics_digest = recompute_statistics(metrics, inputs.statistics)
            section_digests["statistics"] = statistics_digest
            section_ok["statistics"] = True
        except Exception as error:
            errors.append(f"statistics: {type(error).__name__}: {error}")
    else:
        errors.append("statistics: blocked by failed oracle recompute")

    try:
        _, cost_digest = recompute_costs(
            inputs.cost_records,
            expected_query_ids=inputs.coverage.cost_query_ids,
            published=inputs.published_costs,
        )
        section_digests["costs"] = cost_digest
        section_ok["costs"] = True
    except Exception as error:
        errors.append(f"costs: {type(error).__name__}: {error}")

    if section_ok["selectors"]:
        try:
            _, audit_digest = recompute_information_audit(
                inputs.information_audit,
                market=inputs.market,
                selections=selections,
            )
            section_digests["information"] = audit_digest
            section_ok["information"] = True
        except Exception as error:
            errors.append(f"information: {type(error).__name__}: {error}")
    else:
        errors.append("information: blocked by failed selector replay")

    substantive = all(section_ok.values())
    checks = {
        "full_digest_coverage": substantive,
        "full_selector_replay": section_ok["selectors"],
        "full_statistical_recompute": section_ok["oracle"] and section_ok["statistics"],
        "raw_numeric_subset_coverage": (
            section_ok["source"]
            and section_ok["gate0"]
            and section_ok["representations"]
        ),
        "cost_recompute": section_ok["costs"],
        "information_isolation": section_ok["information"],
    }
    return IndependentRecomputeReport(
        coverage_contract_digest=inputs.coverage.digest,
        checks=checks,
        section_digests=section_digests,
        errors=tuple(errors),
    )


def run_formal_independent_recompute(
    inputs: IndependentRecomputeInputs,
    *,
    source_binding: FormalRecomputeSourceBinding,
) -> IndependentRecomputeReport:
    """Run the full replay and attach non-persistable formal authority.

    The source manifest must already bind the exact run, config bytes, frozen
    coverage contract, and at least one config-bound primitive artifact for
    every replay section.  Sources are re-read both before and after execution
    so a concurrent mutation cannot be hidden by the report.
    """

    if not isinstance(inputs, IndependentRecomputeInputs):
        raise RecomputeContractError("inputs must be IndependentRecomputeInputs")
    if not isinstance(source_binding, FormalRecomputeSourceBinding):
        raise RecomputeContractError("formal recompute requires a typed source binding")
    missing_loaders = missing_formal_recompute_source_loaders()
    if missing_loaders:
        dependencies = formal_recompute_loader_dependencies()
        raise RecomputeContractError(
            "formal recompute authority is unavailable: source-owned loaders "
            "cannot yet reconstruct and projection-bind every typed input "
            f"section; missing={list(missing_loaders)}; dependencies="
            f"{ {name: dependencies[name] for name in missing_loaders} }"
        )
    # This becomes reachable only once every loader is formally registered.
    # Registered inverses must still reconstruct and exact-compare live inputs;
    # structural diagnostic loaders alone never enter this authority path.
    for section, loader in sorted(_FORMAL_RECOMPUTE_SOURCE_LOADERS.items()):
        payloads = tuple(
            verify_canonical_evidence_ref(
                reference,
                experiment_root=source_binding.experiment_root,
                expected_config_digest=source_binding.manifest.config_digest,
                require_config_binding=True,
                source_artifact=True,
            )
            for reference in source_binding.manifest.section_sources[section]
        )
        loaded = loader(payloads, source_binding.manifest.config_digest)
        mismatch = _first_mismatch(
            loaded.projection, _live_section_projection(section, inputs)
        )
        if mismatch is not None:
            raise RecomputeContractError(
                f"formal {section} source differs from live typed inputs: {mismatch}"
            )
    manifest = source_binding.manifest
    if manifest.coverage_contract_digest != inputs.coverage.digest:
        raise RecomputeContractError(
            "formal source manifest and live input coverage contracts differ"
        )
    before = source_binding.verify_sources()
    base = run_independent_recompute(inputs)
    after = source_binding.verify_sources()
    if dict(before) != dict(after):
        raise RecomputeContractError(
            "formal recompute sources changed during execution"
        )
    provenance = FormalRecomputeProvenance(
        experiment_id=manifest.experiment_id,
        config_digest=manifest.config_digest,
        config_file_sha256=manifest.config_file_sha256,
        coverage_contract_digest=manifest.coverage_contract_digest,
        source_manifest_ref=source_binding.manifest_ref,
        source_manifest_digest=manifest.digest,
        section_source_digests=after,
        derivation_id=FORMAL_RECOMPUTE_DERIVATION_ID,
        evaluator_digest=formal_recompute_evaluator_digest(),
    )
    return IndependentRecomputeReport(
        coverage_contract_digest=base.coverage_contract_digest,
        checks=base.checks,
        section_digests=base.section_digests,
        errors=base.errors,
        formal_provenance=provenance,
        _execution_authority=_TRUSTED_RECOMPUTE_AUTHORITY,
        _source_binding=source_binding,
    )


def validate_formal_recompute_report_payload(
    value: object,
    *,
    experiment_root: str | Path,
    expected_experiment_id: str,
    expected_config_digest: str,
    expected_config_file_sha256: str,
) -> IndependentRecomputeReport:
    """Verify an archival formal report and all referenced current bytes.

    The returned object intentionally has no in-process execution authority.
    A persisted report can be inspected and compared, but cannot by itself
    authorize completion; callers needing authority must retain the live object
    returned by :func:`run_formal_independent_recompute`.
    """

    if not isinstance(value, Mapping):
        raise RecomputeContractError("formal recompute report must be a mapping")
    report = IndependentRecomputeReport.from_dict(value)
    provenance = report.formal_provenance
    if provenance is None:
        raise RecomputeContractError(
            "formal completion rejects legacy recompute reports without provenance"
        )
    if (
        provenance.experiment_id != _experiment_id(expected_experiment_id)
        or provenance.config_digest
        != _digest(expected_config_digest, "expected config_digest")
        or provenance.config_file_sha256
        != _digest(expected_config_file_sha256, "expected config_file_sha256")
    ):
        raise RecomputeContractError(
            "formal recompute report is bound to another run/config"
        )
    root = Path(experiment_root).expanduser().resolve()
    manifest_path = root.joinpath(
        *PurePosixPath(provenance.source_manifest_ref.canonical_path).parts
    )
    binding = load_formal_recompute_source_binding(
        manifest_path,
        experiment_root=root,
        expected_experiment_id=provenance.experiment_id,
        expected_config_digest=provenance.config_digest,
        expected_config_file_sha256=provenance.config_file_sha256,
        expected_coverage_contract_digest=report.coverage_contract_digest,
    )
    if (
        binding.manifest_ref != provenance.source_manifest_ref
        or binding.manifest.digest != provenance.source_manifest_digest
        or dict(binding.verify_sources()) != dict(provenance.section_source_digests)
    ):
        raise RecomputeContractError(
            "formal recompute report source provenance is stale or misbound"
        )
    if canonicalize(value) != report.to_dict():
        raise RecomputeContractError("formal recompute report is not canonical")
    return report


__all__ = [
    "ATOL",
    "RECOMPUTE_SCHEMA",
    "RECOMPUTE_SECTIONS",
    "RTOL",
    "FullCoverageContract",
    "FORMAL_RECOMPUTE_DERIVATION_ID",
    "FORMAL_RECOMPUTE_PROVENANCE_SCHEMA",
    "FORMAL_RECOMPUTE_SCHEMA",
    "FORMAL_RECOMPUTE_SECTION_SOURCE_SCHEMAS",
    "FORMAL_RECOMPUTE_SOURCE_MANIFEST_SCHEMA",
    "FormalRecomputeProvenance",
    "FormalRecomputeSourceBinding",
    "FormalRecomputeSourceManifest",
    "Gate0AuditUnit",
    "IndependentRecomputeInputs",
    "IndependentRecomputeReport",
    "InformationAuditInputs",
    "OracleEpisodeRow",
    "OracleMetricUnit",
    "PairedComparisonPlan",
    "PublishedSnapshot",
    "RecomputeContractError",
    "RepresentationReplayUnit",
    "SelectorReplayUnit",
    "SourceRecomputeInputs",
    "StatisticsPlan",
    "build_formal_recompute_section_source",
    "championization_payload",
    "encoded_cache_payload",
    "recompute_costs",
    "recompute_information_audit",
    "recompute_oracle_metrics",
    "recompute_source",
    "recompute_statistics",
    "formal_recompute_evaluator_descriptor",
    "formal_recompute_evaluator_digest",
    "formal_recompute_loader_dependencies",
    "formal_recompute_source_manifest_relative_path",
    "load_formal_recompute_source_binding",
    "missing_formal_recompute_source_loaders",
    "replay_selectors",
    "replay_representations",
    "run_independent_recompute",
    "run_formal_independent_recompute",
    "selector_unit_id",
    "structurally_reconstructable_formal_recompute_sections",
    "validate_formal_recompute_report_payload",
    "verify_structural_formal_recompute_source_sections",
    "verify_gate0_audits",
]
