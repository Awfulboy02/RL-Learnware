"""Strict P0 foundation configuration for the v0.3 sidecar.

No scientific defaults are supplied here.  A development configuration may
leave the human review-decision digest open; the formal-freeze stage requires
that a digest be declared, but this local parser does not verify or grant its
authority.  Every nested mapping is closed to unknown fields and the canonical
projection is digest-bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..hashing import sha256_json
from .windowing import WindowingProtocol
from .schemas import (
    ANONYMOUS_SELECTOR_ENTRY_ALLOWLIST,
    PUBLIC_FORBIDDEN_FIELDS,
    V03SchemaError,
    checked_digest,
    checked_ids,
    checked_safe_id,
    strict_mapping,
)


V03_FOUNDATION_CONFIG_SCHEMA = "policy-learnware.v03-foundation-config.v0"
V03_FOUNDATION_STAGES = frozenset({"foundation_development", "formal_freeze"})
V04_ENCODER_EXTENSION_MIGRATION_TARGET = "v0.4"
_REVIEW_MARKERS = frozenset(
    {"TBD", "REVIEW_REQUIRED", "[REVIEW REQUIRED]", "TODO", "UNRESOLVED"}
)


class V03ConfigError(V03SchemaError):
    """The v0.3 foundation scope is incomplete or internally inconsistent."""


class _StrictSafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _StrictSafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise V03ConfigError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _strict_config(
    value: Any, expected: set[str] | frozenset[str], where: str
) -> Mapping[str, Any]:
    try:
        return strict_mapping(value, expected, where)
    except V03SchemaError as exc:
        raise V03ConfigError(str(exc)) from exc


def _reject_review_markers(value: Any, path: str = "$") -> None:
    if isinstance(value, str) and value.strip().upper() in _REVIEW_MARKERS:
        raise V03ConfigError(f"{path} remains unresolved")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_review_markers(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_review_markers(item, f"{path}[{index}]")


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise V03ConfigError(f"{where} must be a positive integer")
    return value


@dataclass(frozen=True)
class WindowProtocolConfig:
    window_length: int
    stride: int
    pooling: str
    pad_final_window: bool
    protocol_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "window_length", _positive_int(self.window_length, "window_length")
        )
        object.__setattr__(self, "stride", _positive_int(self.stride, "stride"))
        if self.pooling not in {"mean", "last", "attention"}:
            raise V03ConfigError("pooling must be one of mean, last, attention")
        if not isinstance(self.pad_final_window, bool):
            raise V03ConfigError("pad_final_window must be boolean")
        supplied = checked_digest(self.protocol_id, "window protocol_id")
        derived = WindowingProtocol(
            window_length=self.window_length,
            stride=self.stride,
            pooling=self.pooling,
            pad_final_window=self.pad_final_window,
        ).window_protocol_digest
        if supplied != derived:
            raise V03ConfigError(
                "window protocol_id must equal the digest derived from window semantics"
            )
        object.__setattr__(self, "protocol_id", supplied)

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_length": self.window_length,
            "stride": self.stride,
            "pooling": self.pooling,
            "pad_final_window": self.pad_final_window,
            "protocol_id": self.protocol_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WindowProtocolConfig":
        fields = {"window_length", "stride", "pooling", "pad_final_window", "protocol_id"}
        data = _strict_config(value, fields, "window_protocol")
        return cls(**{name: data[name] for name in fields})


@dataclass(frozen=True)
class PrimaryFreezeScaffold:
    query_mode: str
    selector_mode: str
    pool_scope: str
    opaque_learnware_field: str
    opaque_query_field: str
    oracle_owner: str

    def __post_init__(self) -> None:
        if self.query_mode not in {"QUERY_EMPIRICAL", "QUERY_REDUCED"}:
            raise V03ConfigError("query_mode must be QUERY_EMPIRICAL or QUERY_REDUCED")
        if self.selector_mode != "distance_only":
            raise V03ConfigError("primary selector_mode must be distance_only")
        if self.pool_scope != "anonymous_global":
            raise V03ConfigError("primary pool_scope must be anonymous_global")
        if self.opaque_learnware_field != "opaque_learnware_id":
            raise V03ConfigError("canonical learnware field must be opaque_learnware_id")
        if self.opaque_query_field != "opaque_query_id":
            raise V03ConfigError("canonical query field must be opaque_query_id")
        if self.oracle_owner != "policy-learnware-paper1":
            raise V03ConfigError("joint oracle owner must be policy-learnware-paper1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_mode": self.query_mode,
            "selector_mode": self.selector_mode,
            "pool_scope": self.pool_scope,
            "opaque_learnware_field": self.opaque_learnware_field,
            "opaque_query_field": self.opaque_query_field,
            "oracle_owner": self.oracle_owner,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PrimaryFreezeScaffold":
        fields = {
            "query_mode",
            "selector_mode",
            "pool_scope",
            "opaque_learnware_field",
            "opaque_query_field",
            "oracle_owner",
        }
        data = _strict_config(value, fields, "primary_freeze")
        return cls(**{name: data[name] for name in fields})


@dataclass(frozen=True)
class EncoderExtensionGateConfig:
    """Opt-in boundary for encoder-family experiments migrated out of v0.3.

    The gate is deliberately disabled when its entire configuration block is
    omitted.  Activating it only acknowledges a separately digest-bound v0.4
    migration decision; it does not constitute formal scientific authority
    for v0.3 or v0.4.
    """

    enabled: bool = False
    migration_target: str | None = None
    authority_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise V03ConfigError("encoder_extension_gate.enabled must be boolean")
        if not self.enabled:
            if self.migration_target is not None or self.authority_digest is not None:
                raise V03ConfigError(
                    "disabled encoder_extension_gate cannot declare migration authority"
                )
            return

        if self.migration_target != V04_ENCODER_EXTENSION_MIGRATION_TARGET:
            raise V03ConfigError(
                "enabled encoder_extension_gate requires migration_target='v0.4'"
            )
        object.__setattr__(
            self,
            "authority_digest",
            checked_digest(
                self.authority_digest,
                "encoder_extension_gate.authority_digest",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "migration_target": self.migration_target,
            "authority_digest": self.authority_digest,
        }

    @classmethod
    def disabled(cls) -> "EncoderExtensionGateConfig":
        return cls(enabled=False)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EncoderExtensionGateConfig":
        if not isinstance(value, Mapping) or not all(
            isinstance(key, str) for key in value
        ):
            raise V03ConfigError("encoder_extension_gate must be a string-keyed mapping")
        allowed = {"enabled", "migration_target", "authority_digest"}
        unknown = set(value) - allowed
        if unknown:
            raise V03ConfigError(
                "invalid encoder_extension_gate keys; "
                f"missing=[], unknown={sorted(unknown)}"
            )
        if "enabled" not in value:
            raise V03ConfigError(
                "invalid encoder_extension_gate keys; missing=['enabled'], unknown=[]"
            )
        enabled = value["enabled"]
        if not isinstance(enabled, bool):
            raise V03ConfigError("encoder_extension_gate.enabled must be boolean")
        if not enabled:
            return cls(
                enabled=False,
                migration_target=value.get("migration_target"),
                authority_digest=value.get("authority_digest"),
            )
        missing = {"migration_target", "authority_digest"} - set(value)
        if missing:
            raise V03ConfigError(
                "invalid encoder_extension_gate keys; "
                f"missing={sorted(missing)}, unknown=[]"
            )
        return cls(
            enabled=True,
            migration_target=value["migration_target"],
            authority_digest=value["authority_digest"],
        )


@dataclass(frozen=True)
class V03FoundationConfig:
    schema: str
    development_id: str
    stage: str
    protocol_id: str
    task_private_ids: tuple[str, ...]
    artifact_root: str
    anonymous_public_allowlist: tuple[str, ...]
    window_protocol: WindowProtocolConfig
    primary_freeze: PrimaryFreezeScaffold
    review_decisions_digest: str | None
    encoder_extension_gate: EncoderExtensionGateConfig = (
        EncoderExtensionGateConfig.disabled()
    )

    def __post_init__(self) -> None:
        if self.schema != V03_FOUNDATION_CONFIG_SCHEMA:
            raise V03ConfigError("unknown v0.3 foundation config schema")
        object.__setattr__(
            self, "development_id", checked_safe_id(self.development_id, "development_id")
        )
        if self.stage not in V03_FOUNDATION_STAGES:
            raise V03ConfigError(f"unknown v0.3 foundation stage: {self.stage!r}")
        object.__setattr__(self, "protocol_id", checked_digest(self.protocol_id, "protocol_id"))
        tasks = checked_ids(self.task_private_ids, "task_private_ids")
        if len(tasks) < 2:
            raise V03ConfigError(
                "signal-attribution foundation requires at least two registered tasks"
            )
        object.__setattr__(self, "task_private_ids", tasks)

        root = Path(self.artifact_root).expanduser()
        if not root.is_absolute() or ".." in root.parts:
            raise V03ConfigError("artifact_root must be an absolute path without traversal")
        object.__setattr__(self, "artifact_root", str(root))

        allowlist = checked_ids(
            self.anonymous_public_allowlist, "anonymous_public_allowlist"
        )
        forbidden = set(allowlist) & set(PUBLIC_FORBIDDEN_FIELDS)
        if forbidden:
            raise V03ConfigError(
                f"anonymous public allowlist contains forbidden fields: {sorted(forbidden)}"
            )
        expected = set(ANONYMOUS_SELECTOR_ENTRY_ALLOWLIST)
        if set(allowlist) != expected:
            raise V03ConfigError(
                "anonymous public allowlist must exactly match the frozen selector-entry schema"
            )
        object.__setattr__(self, "anonymous_public_allowlist", tuple(sorted(allowlist)))

        if not isinstance(self.window_protocol, WindowProtocolConfig):
            raise V03ConfigError("window_protocol must be WindowProtocolConfig")
        if not isinstance(self.primary_freeze, PrimaryFreezeScaffold):
            raise V03ConfigError("primary_freeze must be PrimaryFreezeScaffold")
        if not isinstance(self.encoder_extension_gate, EncoderExtensionGateConfig):
            raise V03ConfigError(
                "encoder_extension_gate must be EncoderExtensionGateConfig"
            )
        if self.stage == "formal_freeze" and self.encoder_extension_gate.enabled:
            raise V03ConfigError(
                "formal_freeze requires encoder_extension_gate.enabled=false"
            )

        if self.review_decisions_digest is None:
            if self.stage == "formal_freeze":
                raise V03ConfigError(
                    "formal_freeze requires a declared review_decisions_digest"
                )
        else:
            object.__setattr__(
                self,
                "review_decisions_digest",
                checked_digest(self.review_decisions_digest, "review_decisions_digest"),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "development_id": self.development_id,
            "stage": self.stage,
            "protocol_id": self.protocol_id,
            "task_private_ids": list(self.task_private_ids),
            "artifact_root": self.artifact_root,
            "anonymous_public_allowlist": list(self.anonymous_public_allowlist),
            "window_protocol": self.window_protocol.to_dict(),
            "primary_freeze": self.primary_freeze.to_dict(),
            "review_decisions_digest": self.review_decisions_digest,
            "encoder_extension_gate": self.encoder_extension_gate.to_dict(),
        }

    @property
    def config_digest(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "V03FoundationConfig":
        fields = {
            "schema",
            "development_id",
            "stage",
            "protocol_id",
            "task_private_ids",
            "artifact_root",
            "anonymous_public_allowlist",
            "window_protocol",
            "primary_freeze",
            "review_decisions_digest",
        }
        _reject_review_markers(value)
        if not isinstance(value, Mapping) or not all(
            isinstance(key, str) for key in value
        ):
            raise V03ConfigError("v0.3 foundation config must be a string-keyed mapping")
        optional_fields = {"encoder_extension_gate"}
        missing = fields - set(value)
        unknown = set(value) - fields - optional_fields
        if missing or unknown:
            raise V03ConfigError(
                "invalid v0.3 foundation config keys; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        data = value
        try:
            return cls(
                schema=data["schema"],
                development_id=data["development_id"],
                stage=data["stage"],
                protocol_id=data["protocol_id"],
                task_private_ids=tuple(data["task_private_ids"]),
                artifact_root=data["artifact_root"],
                anonymous_public_allowlist=tuple(data["anonymous_public_allowlist"]),
                window_protocol=WindowProtocolConfig.from_dict(data["window_protocol"]),
                primary_freeze=PrimaryFreezeScaffold.from_dict(data["primary_freeze"]),
                review_decisions_digest=data["review_decisions_digest"],
                encoder_extension_gate=EncoderExtensionGateConfig.from_dict(
                    data["encoder_extension_gate"]
                )
                if "encoder_extension_gate" in data
                else EncoderExtensionGateConfig.disabled(),
            )
        except V03ConfigError:
            raise
        except V03SchemaError as exc:
            raise V03ConfigError(str(exc)) from exc
        except (TypeError, KeyError) as exc:
            raise V03ConfigError("invalid v0.3 foundation config value") from exc


def load_v03_foundation_config(path: str | Path) -> V03FoundationConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.load(handle, Loader=_StrictSafeLoader)
    if not isinstance(payload, Mapping):
        raise V03ConfigError("v0.3 foundation config must be a mapping")
    return V03FoundationConfig.from_dict(payload)


__all__ = [
    "EncoderExtensionGateConfig",
    "PrimaryFreezeScaffold",
    "V03ConfigError",
    "V03FoundationConfig",
    "V03_FOUNDATION_CONFIG_SCHEMA",
    "V03_FOUNDATION_STAGES",
    "V04_ENCODER_EXTENSION_MIGRATION_TARGET",
    "WindowProtocolConfig",
    "load_v03_foundation_config",
]
