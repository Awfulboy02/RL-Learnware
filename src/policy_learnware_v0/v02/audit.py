"""Executable information-isolation audits for v0.2 public selection.

The routines here inspect bytes and typed contracts.  They deliberately do not
accept a caller-provided ``passed`` field.  Public artifacts are governed by an
exact path/type/schema allowlist, public market entries have a three-field data
projection, and oracle independence is exercised in both missing-oracle and
poison-oracle sandboxes while the replay callback receives public roots only.
"""

from __future__ import annotations

from dataclasses import dataclass
import fnmatch
import inspect
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
from types import MappingProxyType
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import numpy as np

from ..hashing import sha256_file, sha256_json
from ..io import atomic_write_json
from .selectors import EvidenceContract


PUBLIC_MARKET_ENTRY_FIELDS = frozenset(
    {"opaque_learnware_id", "normalized_source_competence", "tie_break_token"}
)

# Exact keys that must not occur anywhere in a public JSON object unless a
# versioned artifact rule explicitly grants an exception.  Runtime/schema ABI
# information is private and selected-only in v0.2; it is not a public hard
# filter.
DEFAULT_PUBLIC_FORBIDDEN_KEYS = frozenset(
    {
        "task",
        "task_id",
        "source_task",
        "private_benchmark_task_ref",
        "task_contract_digest",
        "reward",
        "reward_contract",
        "reward_semantics",
        "axis",
        "axis_id",
        "factor",
        "factor_id",
        "target_factor",
        "seed",
        "reset_seed",
        "reset_seeds",
        "probe_seed",
        "probe_seeds",
        "policy_seed",
        "policy_seeds",
        "training_seed",
        "candidate_id",
        "bundle",
        "bundle_path",
        "bundle_digest",
        "execution_abi",
        "execution_abi_record",
        "runtime_contract",
        "observation_schema_digest",
        "action_schema_digest",
        "schema_fingerprint",
        "candidate_target_return",
        "target_return",
        "oracle_return",
        "policy_return",
        "path",
    }
)
DEFAULT_PUBLIC_FORBIDDEN_STRING_TOKENS = frozenset(
    {"task", "reward", "axis", "factor", "oracle", "bundle", "seed", "runtime", "abi"}
)
DEFAULT_PUBLIC_FORBIDDEN_NPZ_MEMBERS = frozenset(DEFAULT_PUBLIC_FORBIDDEN_KEYS)

_OPAQUE_LEARNWARE_ID = re.compile(r"^(?:lw-|v02lw-)[0-9a-f]{20,64}$")
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


@dataclass(frozen=True)
class AuditViolation:
    path: str
    location: str
    reason: str
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "location": self.location,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class PublicArtifactRule:
    """One exact public artifact path pattern and payload allowlist."""

    pattern: str
    kind: Literal["json", "npz", "text", "bytes"]
    json_keys: frozenset[str] = frozenset()
    npz_members: frozenset[str] = frozenset()
    permitted_forbidden_keys: frozenset[str] = frozenset()
    permitted_forbidden_string_tokens: frozenset[str] = frozenset()
    permitted_forbidden_npz_members: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.pattern, str) or not self.pattern:
            raise ValueError("public artifact rule pattern must be non-empty")
        pattern_path = Path(self.pattern)
        if pattern_path.is_absolute() or ".." in pattern_path.parts or "\\" in self.pattern:
            raise ValueError("public artifact rule pattern must be relative and traversal-free")
        if self.kind not in {"json", "npz", "text", "bytes"}:
            raise ValueError(f"unsupported public artifact kind: {self.kind!r}")
        if self.kind == "json" and not self.json_keys:
            raise ValueError("JSON public artifact rules require an exact top-level key allowlist")
        if self.kind != "json" and self.json_keys:
            raise ValueError("json_keys are valid only for JSON artifact rules")
        if self.kind == "npz" and not self.npz_members:
            raise ValueError("NPZ public artifact rules require an exact member allowlist")
        if self.kind != "npz" and self.npz_members:
            raise ValueError("npz_members are valid only for NPZ artifact rules")
        for name in (
            "json_keys",
            "npz_members",
            "permitted_forbidden_keys",
            "permitted_forbidden_string_tokens",
            "permitted_forbidden_npz_members",
        ):
            values = getattr(self, name)
            if any(not isinstance(item, str) or not item for item in values):
                raise ValueError(f"{name} must contain non-empty strings")


@dataclass(frozen=True)
class PublicArtifactAudit:
    root: str
    file_count: int
    tree_digest: str | None
    violations: tuple[AuditViolation, ...]

    @property
    def passed(self) -> bool:
        return self.tree_digest is not None and not self.violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v02-public-artifact-audit.v0",
            "passed": self.passed,
            "root": self.root,
            "file_count": self.file_count,
            "tree_digest": self.tree_digest,
            "violations": [item.to_dict() for item in self.violations],
        }


@dataclass(frozen=True)
class PublicMarketAudit:
    entry_count: int
    violations: tuple[AuditViolation, ...]

    @property
    def passed(self) -> bool:
        return self.entry_count > 0 and not self.violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v02-public-market-allowlist-audit.v0",
            "passed": self.passed,
            "entry_count": self.entry_count,
            "allowed_entry_fields": sorted(PUBLIC_MARKET_ENTRY_FIELDS),
            "violations": [item.to_dict() for item in self.violations],
        }


def audit_public_market_entries(
    entries: Mapping[str, Mapping[str, Any]],
) -> PublicMarketAudit:
    """Enforce the minimal selector-visible market projection.

    The mapping key and ``opaque_learnware_id`` must agree.  Task/schema/runtime
    compatibility, bundle identity, and all ABI fields are rejected rather than
    silently ignored.
    """

    violations: list[AuditViolation] = []
    if not isinstance(entries, Mapping) or not entries:
        violations.append(AuditViolation("market_public", "$", "empty_or_invalid_market"))
        return PublicMarketAudit(0, tuple(violations))
    tokens: set[str] = set()
    for key, raw in entries.items():
        location = f"entries[{key!r}]"
        if not isinstance(key, str) or not isinstance(raw, Mapping):
            violations.append(AuditViolation("market_public", location, "invalid_entry_type"))
            continue
        observed = set(raw)
        if observed != PUBLIC_MARKET_ENTRY_FIELDS:
            violations.append(
                AuditViolation(
                    "market_public",
                    location,
                    "entry_field_allowlist_mismatch",
                    f"missing={sorted(PUBLIC_MARKET_ENTRY_FIELDS-observed)}, "
                    f"unknown={sorted(observed-PUBLIC_MARKET_ENTRY_FIELDS)}",
                )
            )
        if not PUBLIC_MARKET_ENTRY_FIELDS.issubset(observed):
            continue
        opaque_id = raw["opaque_learnware_id"]
        competence = raw["normalized_source_competence"]
        tie_break = raw["tie_break_token"]
        if (
            not isinstance(opaque_id, str)
            or _OPAQUE_LEARNWARE_ID.fullmatch(opaque_id) is None
            or key != opaque_id
        ):
            violations.append(AuditViolation("market_public", location, "invalid_opaque_id"))
        if (
            isinstance(competence, bool)
            or not isinstance(competence, (int, float))
            or not math.isfinite(float(competence))
            or not 0.0 <= float(competence) <= 1.0
        ):
            violations.append(AuditViolation("market_public", location, "invalid_competence"))
        if not isinstance(tie_break, str) or not tie_break:
            violations.append(AuditViolation("market_public", location, "invalid_tie_break_token"))
        elif tie_break in tokens:
            violations.append(AuditViolation("market_public", location, "tie_break_token_collision"))
        else:
            tokens.add(tie_break)
    return PublicMarketAudit(len(entries), tuple(violations))


def _tokens(value: str) -> frozenset[str]:
    return frozenset(
        item.lower() for item in re.split(r"[^A-Za-z0-9]+", value) if item
    )


def _looks_like_path(value: str) -> bool:
    return bool(
        value.startswith(("/", "~/", "file://"))
        or _WINDOWS_ABSOLUTE.match(value)
        or "../" in value
        or "..\\" in value
    )


def _scan_string(
    value: str,
    *,
    relative: str,
    location: str,
    rule: PublicArtifactRule,
    forbidden_tokens: frozenset[str],
    violations: list[AuditViolation],
) -> None:
    blocked = (
        _tokens(value)
        & forbidden_tokens
        - rule.permitted_forbidden_string_tokens
    )
    for token in sorted(blocked):
        violations.append(
            AuditViolation(relative, location, "forbidden_string_token", token)
        )
    if _looks_like_path(value):
        violations.append(AuditViolation(relative, location, "path_like_public_string", value))


def _walk_json(
    value: Any,
    *,
    relative: str,
    location: str,
    rule: PublicArtifactRule,
    forbidden_keys: frozenset[str],
    forbidden_tokens: frozenset[str],
    violations: list[AuditViolation],
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            child = f"{location}.{key_text}" if location else key_text
            if (
                key_text.lower() in forbidden_keys
                and key_text.lower() not in rule.permitted_forbidden_keys
            ):
                violations.append(
                    AuditViolation(relative, child, "forbidden_json_key", key_text)
                )
            _walk_json(
                item,
                relative=relative,
                location=child,
                rule=rule,
                forbidden_keys=forbidden_keys,
                forbidden_tokens=forbidden_tokens,
                violations=violations,
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _walk_json(
                item,
                relative=relative,
                location=f"{location}[{index}]",
                rule=rule,
                forbidden_keys=forbidden_keys,
                forbidden_tokens=forbidden_tokens,
                violations=violations,
            )
    elif isinstance(value, str):
        _scan_string(
            value,
            relative=relative,
            location=location or "$",
            rule=rule,
            forbidden_tokens=forbidden_tokens,
            violations=violations,
        )


def _kind_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix == ".npz":
        return "npz"
    if suffix in {".txt", ".md", ".csv"}:
        return "text"
    return "bytes"


def artifact_tree_digest(root: str | Path) -> str:
    """Hash a tree while rejecting traversal and every symlink component."""

    base = Path(root).expanduser()
    if base.is_symlink():
        raise ValueError("artifact tree root cannot be a symlink")
    base = base.resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"artifact tree does not exist: {base}")
    files: dict[str, str] = {}
    for path in sorted(base.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"artifact tree contains a symlink: {path}")
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(base)
        except ValueError as exc:
            raise ValueError(f"artifact path escapes root: {path}") from exc
        if path.is_file():
            files[relative.as_posix()] = sha256_file(path)
    return sha256_json(files)


def audit_public_artifacts(
    root: str | Path,
    rules: Sequence[PublicArtifactRule],
    *,
    forbidden_keys: Iterable[str] = DEFAULT_PUBLIC_FORBIDDEN_KEYS,
    forbidden_string_tokens: Iterable[str] = DEFAULT_PUBLIC_FORBIDDEN_STRING_TOKENS,
    forbidden_npz_members: Iterable[str] = DEFAULT_PUBLIC_FORBIDDEN_NPZ_MEMBERS,
) -> PublicArtifactAudit:
    """Scan one public tree against exact path, type, JSON, and NPZ contracts."""

    raw_root = Path(root).expanduser()
    violations: list[AuditViolation] = []
    if raw_root.is_symlink():
        violations.append(AuditViolation(str(raw_root), "$", "symlink_root_forbidden"))
        return PublicArtifactAudit(str(raw_root), 0, None, tuple(violations))
    resolved_root = raw_root.resolve()
    if not resolved_root.is_dir():
        violations.append(AuditViolation(str(resolved_root), "$", "public_root_missing"))
        return PublicArtifactAudit(str(resolved_root), 0, None, tuple(violations))
    if not rules:
        violations.append(AuditViolation(str(resolved_root), "$", "empty_artifact_allowlist"))
        return PublicArtifactAudit(str(resolved_root), 0, None, tuple(violations))

    blocked_keys = frozenset(str(item).lower() for item in forbidden_keys)
    blocked_tokens = frozenset(str(item).lower() for item in forbidden_string_tokens)
    blocked_members = frozenset(str(item).lower() for item in forbidden_npz_members)
    file_count = 0
    safe_files: dict[str, str] = {}
    for lexical in sorted(resolved_root.rglob("*")):
        lexical_relative = lexical.relative_to(resolved_root).as_posix()
        if lexical.is_symlink():
            try:
                lexical.resolve().relative_to(resolved_root)
            except ValueError:
                reason = "symlink_escape"
            else:
                reason = "symlink_forbidden"
            violations.append(AuditViolation(lexical_relative, "$", reason))
            continue
        resolved = lexical.resolve()
        try:
            resolved.relative_to(resolved_root)
        except ValueError:
            violations.append(AuditViolation(lexical_relative, "$", "path_traversal"))
            continue
        if not lexical.is_file():
            continue
        file_count += 1
        safe_files[lexical_relative] = sha256_file(lexical)
        matches = tuple(
            rule for rule in rules if fnmatch.fnmatchcase(lexical_relative, rule.pattern)
        )
        if len(matches) != 1:
            violations.append(
                AuditViolation(
                    lexical_relative,
                    "$",
                    "unregistered_artifact" if not matches else "ambiguous_artifact_rule",
                )
            )
            continue
        rule = matches[0]
        observed_kind = _kind_for(lexical)
        if observed_kind != rule.kind:
            violations.append(
                AuditViolation(
                    lexical_relative,
                    "$",
                    "artifact_type_mismatch",
                    f"expected={rule.kind}, observed={observed_kind}",
                )
            )
            continue
        if rule.kind == "json":
            try:
                payload = json.loads(lexical.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                violations.append(
                    AuditViolation(lexical_relative, "$", "invalid_json", type(exc).__name__)
                )
                continue
            if not isinstance(payload, Mapping) or set(payload) != set(rule.json_keys):
                observed = sorted(payload) if isinstance(payload, Mapping) else []
                violations.append(
                    AuditViolation(
                        lexical_relative,
                        "$",
                        "json_top_level_allowlist_mismatch",
                        f"expected={sorted(rule.json_keys)}, observed={observed}",
                    )
                )
            _walk_json(
                payload,
                relative=lexical_relative,
                location="",
                rule=rule,
                forbidden_keys=blocked_keys,
                forbidden_tokens=blocked_tokens,
                violations=violations,
            )
        elif rule.kind == "npz":
            try:
                with np.load(lexical, allow_pickle=False) as archive:
                    observed_members = set(archive.files)
                    if observed_members != set(rule.npz_members):
                        violations.append(
                            AuditViolation(
                                lexical_relative,
                                "$",
                                "npz_member_allowlist_mismatch",
                                f"expected={sorted(rule.npz_members)}, "
                                f"observed={sorted(observed_members)}",
                            )
                        )
                    for member in archive.files:
                        lowered = member.lower()
                        member_tokens = _tokens(member)
                        path_like = (
                            "/" in member or "\\" in member or ".." in Path(member).parts
                        )
                        forbidden = bool(
                            lowered in blocked_members
                            or member_tokens & blocked_members
                        ) and member not in rule.permitted_forbidden_npz_members
                        if path_like:
                            violations.append(
                                AuditViolation(lexical_relative, f"member:{member}", "npz_member_traversal")
                            )
                        if forbidden:
                            violations.append(
                                AuditViolation(lexical_relative, f"member:{member}", "forbidden_npz_member")
                            )
                        array = archive[member]
                        if array.dtype.hasobject:
                            violations.append(
                                AuditViolation(lexical_relative, f"member:{member}", "object_npz_member")
                            )
                        elif array.dtype.kind in {"U", "S"}:
                            for index, value in np.ndenumerate(array):
                                text = (
                                    value.decode("utf-8", errors="strict")
                                    if isinstance(value, bytes)
                                    else str(value)
                                )
                                _scan_string(
                                    text,
                                    relative=lexical_relative,
                                    location=f"member:{member}{index}",
                                    rule=rule,
                                    forbidden_tokens=blocked_tokens,
                                    violations=violations,
                                )
            except (OSError, ValueError, UnicodeError) as exc:
                violations.append(
                    AuditViolation(lexical_relative, "$", "invalid_npz", type(exc).__name__)
                )
        elif rule.kind == "text":
            try:
                text = lexical.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                violations.append(
                    AuditViolation(lexical_relative, "$", "invalid_text", type(exc).__name__)
                )
                continue
            _scan_string(
                text,
                relative=lexical_relative,
                location="$",
                rule=rule,
                forbidden_tokens=blocked_tokens,
                violations=violations,
            )

    return PublicArtifactAudit(
        root=str(resolved_root),
        file_count=file_count,
        tree_digest=sha256_json(safe_files),
        violations=tuple(violations),
    )


@dataclass(frozen=True)
class EvidenceContractAudit:
    checks: Mapping[str, bool]
    violations: tuple[AuditViolation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", MappingProxyType(dict(self.checks)))

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(self.checks.values()) and not self.violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v02-evidence-contract-audit.v0",
            "passed": self.passed,
            "checks": dict(self.checks),
            "violations": [item.to_dict() for item in self.violations],
        }


def audit_evidence_contract(
    contract: EvidenceContract | Mapping[str, Any],
) -> EvidenceContractAudit:
    """Recompute zero target-policy rollout/reward/update authorization."""

    expected = set(EvidenceContract.__dataclass_fields__)
    if isinstance(contract, EvidenceContract):
        payload: Mapping[str, Any] = contract.to_dict()
    elif isinstance(contract, Mapping):
        payload = contract
    else:
        return EvidenceContractAudit(
            {}, (AuditViolation("EvidenceContract", "$", "invalid_contract_type"),)
        )
    violations: list[AuditViolation] = []
    observed = set(payload)
    if observed != expected:
        violations.append(
            AuditViolation(
                "EvidenceContract",
                "$",
                "contract_field_allowlist_mismatch",
                f"missing={sorted(expected-observed)}, unknown={sorted(observed-expected)}",
            )
        )
        return EvidenceContractAudit({}, tuple(violations))
    try:
        parsed = EvidenceContract(**{name: payload[name] for name in expected})
    except (TypeError, ValueError) as exc:
        violations.append(
            AuditViolation("EvidenceContract", "$", "invalid_contract_value", str(exc))
        )
        return EvidenceContractAudit({}, tuple(violations))
    checks = {
        "target_parameters_hidden": not parsed.reads_target_parameters,
        "target_task_reward_schema_identity_hidden": (
            not parsed.reads_target_task_reward_schema_identity
        ),
        "candidate_target_rollouts_zero": not parsed.reads_candidate_target_rollouts,
        "candidate_policy_target_rewards_zero": (
            not parsed.reads_candidate_policy_target_rewards
        ),
        "target_gradient_updates_zero": parsed.target_gradient_updates == 0,
        "submit_side_profiles_hidden": not parsed.reads_submit_side_profiles,
    }
    for name, passed in checks.items():
        if not passed:
            violations.append(
                AuditViolation("EvidenceContract", name, "public_selector_permission_violation")
            )
    return EvidenceContractAudit(checks, tuple(violations))


@dataclass(frozen=True)
class OracleIndependenceAudit:
    baseline_selection_digest: str | None
    scenarios: Mapping[str, Mapping[str, Any]]
    violations: tuple[AuditViolation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scenarios",
            MappingProxyType({name: MappingProxyType(dict(value)) for name, value in self.scenarios.items()}),
        )

    @property
    def passed(self) -> bool:
        return bool(
            self.baseline_selection_digest
            and set(self.scenarios) == {"missing", "poison"}
            and all(value.get("passed") is True for value in self.scenarios.values())
            and not self.violations
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v02-oracle-independence-audit.v0",
            "passed": self.passed,
            "baseline_selection_digest": self.baseline_selection_digest,
            "replay_callback_capabilities": [
                "market_public_root",
                "measurement_root",
                "selector_outputs_root",
            ],
            "oracle_root_passed_to_replay": False,
            "scenarios": {name: dict(value) for name, value in self.scenarios.items()},
            "violations": [item.to_dict() for item in self.violations],
        }


SelectorReplay = Callable[[Path, Path, Path], str]


def audit_oracle_independence(
    replay_selector: SelectorReplay,
    *,
    market_public_root: str | Path,
    measurement_root: str | Path,
    selector_outputs_root: str | Path,
) -> OracleIndependenceAudit:
    """Replay selection beside absent and poisoned oracle roots.

    ``replay_selector`` is intentionally passed exactly three public roots.  It
    must return the digest of the immutable selection tree and must not mutate
    any supplied tree.  The poison scenario catches hidden sibling lookups.
    """

    violations: list[AuditViolation] = []
    scenarios: dict[str, Mapping[str, Any]] = {}
    baseline: str | None = None
    roots = {
        "market_public": Path(market_public_root).expanduser().resolve(),
        "measurement": Path(measurement_root).expanduser().resolve(),
        "selector_outputs": Path(selector_outputs_root).expanduser().resolve(),
    }
    try:
        signature = inspect.signature(replay_selector)
        parameters = tuple(signature.parameters.values())
        if (
            len(parameters) != 3
            or any(
                item.kind
                not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for item in parameters
            )
        ):
            raise ValueError("selector replay must expose exactly three positional public-root capabilities")
        initial = {name: artifact_tree_digest(path) for name, path in roots.items()}
        baseline = initial["selector_outputs"]
        with tempfile.TemporaryDirectory(prefix=".policy-learnware-v02-oracle-audit-") as raw:
            temp = Path(raw)
            for label in ("missing", "poison"):
                scenario_root = temp / label
                staged: dict[str, Path] = {}
                for name, source in roots.items():
                    destination = scenario_root / name
                    shutil.copytree(source, destination, symlinks=True)
                    staged[name] = destination
                oracle = scenario_root / "confirmatory_oracle_private"
                poison_path = oracle / "tempting_oracle.json"
                poison_digest: str | None = None
                if label == "poison":
                    poison_digest = atomic_write_json(
                        poison_path,
                        {
                            "schema": "poison",
                            "passed": True,
                            "oracle_return": 1.0e99,
                            "selected_id": "attacker",
                        },
                    )
                before = {name: artifact_tree_digest(path) for name, path in staged.items()}
                returned = replay_selector(
                    staged["market_public"],
                    staged["measurement"],
                    staged["selector_outputs"],
                )
                after = {name: artifact_tree_digest(path) for name, path in staged.items()}
                state_unchanged = (
                    not oracle.exists()
                    if label == "missing"
                    else poison_path.is_file() and sha256_file(poison_path) == poison_digest
                )
                passed = bool(
                    type(returned) is str
                    and returned == baseline
                    and before == initial
                    and after == initial
                    and state_unchanged
                )
                scenarios[label] = {
                    "passed": passed,
                    "returned_selection_digest": returned if isinstance(returned, str) else None,
                    "before_tree_digests": before,
                    "after_tree_digests": after,
                    "oracle_state_unchanged": state_unchanged,
                }
                if not passed:
                    violations.append(
                        AuditViolation(label, "$", "oracle_changed_selector_replay_or_public_inputs")
                    )
    except Exception as exc:
        violations.append(
            AuditViolation("oracle_independence", "$", "audit_execution_failed", f"{type(exc).__name__}: {exc}")
        )
    return OracleIndependenceAudit(baseline, scenarios, tuple(violations))


__all__ = [
    "AuditViolation",
    "DEFAULT_PUBLIC_FORBIDDEN_KEYS",
    "DEFAULT_PUBLIC_FORBIDDEN_NPZ_MEMBERS",
    "DEFAULT_PUBLIC_FORBIDDEN_STRING_TOKENS",
    "EvidenceContractAudit",
    "OracleIndependenceAudit",
    "PUBLIC_MARKET_ENTRY_FIELDS",
    "PublicArtifactAudit",
    "PublicArtifactRule",
    "PublicMarketAudit",
    "SelectorReplay",
    "artifact_tree_digest",
    "audit_evidence_contract",
    "audit_oracle_independence",
    "audit_public_artifacts",
    "audit_public_market_entries",
]
