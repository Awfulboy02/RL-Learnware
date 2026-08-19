"""Read-only binding of v0.1 measurements to the completed v0 runtime.

No v0 asset is regenerated here.  The loader follows the existing v0 manifest
chain, delegates semantic-source and package verification to the original
``_verify_frozen_protocol_runtime`` implementation (including its audited
legacy migration), verifies the public selector pool, and only then exposes
the frozen normalizer, encoder and Gaussian kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from ..artifacts import ArtifactLayout, ArtifactLayoutError
from ..cli import CommandFailure, _verify_frozen_protocol_runtime
from ..hashing import sha256_file, sha256_json
from ..io import read_json
from ..pool.learnware import LearnwarePool, PoolValidationError, load_public_pool
from ..representation.encoder import EncoderCheckpoint, TransitionSemanticEncoder
from ..representation.normalization import NormalizationStats
from ..rkme.gaussian import GaussianKernel
from ..schemas import EnvSchema, FrozenProtocol


class BaseRuntimeBindingError(RuntimeError):
    """The selected directory is not the exact completed v0 formal runtime."""


@dataclass(frozen=True)
class FrozenMeasurementAssets:
    """Already-verified v0 mathematical components, loaded read-only."""

    normalization: NormalizationStats
    encoder_checkpoint: EncoderCheckpoint
    encoder: TransitionSemanticEncoder
    kernel: GaussianKernel


@dataclass(frozen=True)
class VerifiedBaseRuntime:
    """Hash-addressed view of the completed formal v0 run."""

    base_artifacts_root: Path
    base_run_dir: Path
    protocol: FrozenProtocol
    public_pool: LearnwarePool
    protocol_manifest: Mapping[str, Any]
    pool_manifest: Mapping[str, Any]
    source_task_specs: Mapping[str, Any]
    asset_digests: Mapping[str, str]
    protocol_manifest_sha256: str
    pool_manifest_sha256: str
    public_pool_manifest_sha256: str

    def __post_init__(self) -> None:
        root = Path(self.base_artifacts_root).resolve()
        run = Path(self.base_run_dir).resolve()
        if run.parent != root:
            raise ValueError("base_run_dir must be a direct child of base_artifacts_root")
        if run.name != self.protocol.config["pool"]["pool_id"]:
            raise ValueError("base runtime pool id differs from FrozenProtocol")
        object.__setattr__(self, "base_artifacts_root", root)
        object.__setattr__(self, "base_run_dir", run)
        object.__setattr__(
            self, "protocol_manifest", MappingProxyType(dict(self.protocol_manifest))
        )
        object.__setattr__(
            self, "pool_manifest", MappingProxyType(dict(self.pool_manifest))
        )
        object.__setattr__(
            self, "source_task_specs", MappingProxyType(dict(self.source_task_specs))
        )
        object.__setattr__(
            self, "asset_digests", MappingProxyType(dict(self.asset_digests))
        )

    @property
    def pool_id(self) -> str:
        return self.public_pool.pool_id

    @property
    def protocol_id(self) -> str:
        return self.protocol.protocol_id

    @property
    def env_schemas(self) -> Mapping[str, EnvSchema]:
        return self.protocol.env_schemas

    @property
    def binding_digest(self) -> str:
        """Compact provenance identity used by the v0.1 freeze step."""

        return sha256_json(
            {
                "pool_id": self.pool_id,
                "protocol_id": self.protocol_id,
                "protocol_manifest_sha256": self.protocol_manifest_sha256,
                "pool_manifest_sha256": self.pool_manifest_sha256,
                "public_pool_manifest_sha256": self.public_pool_manifest_sha256,
                "asset_digests": dict(self.asset_digests),
            }
        )

    def load_measurement_assets(self) -> FrozenMeasurementAssets:
        """Load the frozen representation after rechecking the bound files."""

        paths = {
            "normalization": self.base_run_dir / "protocol" / "normalization.npz",
            "encoder_checkpoint": self.base_run_dir / "protocol" / "encoder.msgpack",
            "encoder_config": self.base_run_dir / "protocol" / "encoder_config.json",
            "kernel": self.base_run_dir / "protocol" / "kernel.json",
        }
        for name, path in paths.items():
            expected = self.asset_digests[name]
            actual = sha256_file(path)
            if actual != expected:
                raise BaseRuntimeBindingError(
                    f"frozen {name} digest changed: expected {expected}, got {actual}"
                )
        normalization = NormalizationStats.load_npz(paths["normalization"])
        encoder_checkpoint = EncoderCheckpoint.load(
            paths["encoder_checkpoint"], read_json(paths["encoder_config"])
        )
        encoder = TransitionSemanticEncoder(encoder_checkpoint)
        kernel = GaussianKernel.load_json(paths["kernel"])
        expected_bandwidth = float(self.protocol.packed_layout["kernel_bandwidth"])
        if not np.isclose(
            kernel.bandwidth, expected_bandwidth, rtol=0.0, atol=0.0
        ):
            raise BaseRuntimeBindingError(
                "frozen kernel bandwidth differs from FrozenProtocol"
            )
        return FrozenMeasurementAssets(
            normalization=normalization,
            encoder_checkpoint=encoder_checkpoint,
            encoder=encoder,
            kernel=kernel,
        )


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BaseRuntimeBindingError(f"{where} must be a JSON object")
    return value


def _require_sha256(value: Any, where: str) -> str:
    digest = str(value)
    if len(digest) != 64:
        raise BaseRuntimeBindingError(f"{where} is not a SHA-256 digest")
    try:
        int(digest, 16)
    except ValueError as error:
        raise BaseRuntimeBindingError(f"{where} is not a SHA-256 digest") from error
    return digest.lower()


def _verify_top_level_pool_manifest(
    layout: ArtifactLayout,
    protocol: FrozenProtocol,
    public_pool: LearnwarePool,
) -> Mapping[str, Any]:
    payload = _require_mapping(read_json(layout.pool_manifest), "pool_manifest.json")
    if payload.get("schema") != "policy-learnware.pool-build.v0":
        raise BaseRuntimeBindingError("unsupported base pool manifest schema")
    if payload.get("complete") is not True:
        raise BaseRuntimeBindingError("base pool is not complete")
    if payload.get("pool_id") != public_pool.pool_id:
        raise BaseRuntimeBindingError("base pool manifest pool id mismatch")
    if payload.get("protocol_id") != protocol.protocol_id:
        raise BaseRuntimeBindingError("base pool manifest protocol id mismatch")
    expected_protocol_file = _require_sha256(
        payload.get("frozen_protocol_sha256"), "frozen_protocol_sha256"
    )
    if sha256_file(layout.frozen_protocol) != expected_protocol_file:
        raise BaseRuntimeBindingError("base frozen protocol file digest mismatch")
    expected_public_manifest = _require_sha256(
        payload.get("public_pool_manifest_sha256"),
        "public_pool_manifest_sha256",
    )
    public_manifest_path = layout.selector_pool_dir / "pool_manifest.json"
    if sha256_file(public_manifest_path) != expected_public_manifest:
        raise BaseRuntimeBindingError("selector-pool manifest digest mismatch")
    if payload.get("public_pool_digest") != sha256_json(public_pool.public_manifest()):
        raise BaseRuntimeBindingError("selector-pool semantic digest mismatch")
    entries = _require_mapping(payload.get("entries"), "pool_manifest.entries")
    if set(entries) != set(protocol.env_schemas):
        raise BaseRuntimeBindingError("base pool task coverage differs from protocol")
    if int(payload.get("entry_count", -1)) != len(protocol.env_schemas):
        raise BaseRuntimeBindingError("base pool entry count mismatch")
    return payload


def _task_specs_by_private_task(
    pool_manifest: Mapping[str, Any], public_pool: LearnwarePool
) -> dict[str, Any]:
    public_by_id = {entry.opaque_id: entry.task_spec for entry in public_pool.entries}
    result: dict[str, Any] = {}
    entries = _require_mapping(pool_manifest.get("entries"), "pool_manifest.entries")
    for task, raw in entries.items():
        record = _require_mapping(raw, f"pool_manifest.entries.{task}")
        opaque_id = str(record.get("opaque_id", ""))
        try:
            spec = public_by_id[opaque_id]
        except KeyError as error:
            raise BaseRuntimeBindingError(
                f"private task mapping references unknown public id: {opaque_id!r}"
            ) from error
        expected = str(record.get("task_spec_digest", ""))
        if spec.task_spec_digest != expected:
            raise BaseRuntimeBindingError(
                f"TaskSpec digest mismatch in private task mapping for {task}"
            )
        result[str(task)] = spec
    if len(result) != len(public_by_id):
        raise BaseRuntimeBindingError("private/public source TaskSpec coverage differs")
    return result


def verify_and_load_base_runtime(
    base_artifacts_root: str | Path,
    *,
    pool_id: str,
    expected_protocol_id: str,
    expected_protocol_draft_hash: str,
) -> VerifiedBaseRuntime:
    """Verify and load the formal v0 run without modifying any base artifact.

    ``base_artifacts_root`` must be the parent directory (for example
    ``artifacts_retry_roundoff``), not the already-expanded pool directory.
    This catches the easy-to-miss duplicate ``pool_id/pool_id`` invocation.
    """

    root = Path(base_artifacts_root).expanduser().resolve()
    if root.name == pool_id and (root / "protocol").is_dir():
        raise BaseRuntimeBindingError(
            "base_artifacts_root must be the parent of pool_id, not base_run_dir"
        )
    try:
        layout = ArtifactLayout(root, pool_id)
    except ArtifactLayoutError as error:
        raise BaseRuntimeBindingError(str(error)) from error
    if not layout.pool_root.is_dir():
        raise BaseRuntimeBindingError(f"base run directory is missing: {layout.pool_root}")

    try:
        protocol_manifest = layout.verify_manifest_files(layout.protocol_manifest)
        # The protocol manifest binds manifests.  Verify each manifest's actual
        # payloads as well so a nested model/checkpoint edit cannot pass.
        for manifest_path in (
            layout.environment_manifest,
            layout.normalization_manifest,
            layout.encoder_manifest,
            layout.kernel_manifest,
        ):
            layout.verify_manifest_files(manifest_path)
        protocol = FrozenProtocol.load(layout.frozen_protocol)
        _verify_frozen_protocol_runtime(protocol)
        public_pool = load_public_pool(layout.selector_pool_dir)
    except (ArtifactLayoutError, CommandFailure, PoolValidationError, OSError, ValueError) as error:
        raise BaseRuntimeBindingError(f"base v0 verification failed: {error}") from error

    if protocol.protocol_id != expected_protocol_id:
        raise BaseRuntimeBindingError(
            f"base protocol id mismatch: {protocol.protocol_id} != {expected_protocol_id}"
        )
    if protocol_manifest.get("protocol_id") != protocol.protocol_id:
        raise BaseRuntimeBindingError("protocol completion manifest id mismatch")
    actual_draft_hash = sha256_json(protocol.config)
    if actual_draft_hash != expected_protocol_draft_hash:
        raise BaseRuntimeBindingError(
            "base protocol draft hash differs from the approved v0.1 binding"
        )
    if protocol_manifest.get("protocol_draft_hash") != expected_protocol_draft_hash:
        raise BaseRuntimeBindingError("protocol manifest draft hash mismatch")
    if protocol_manifest.get("pool_id") != pool_id:
        raise BaseRuntimeBindingError("protocol manifest pool id mismatch")
    if public_pool.protocol_id != protocol.protocol_id:
        raise BaseRuntimeBindingError("public pool protocol differs from FrozenProtocol")
    if public_pool.pool_id != pool_id:
        raise BaseRuntimeBindingError("public pool id differs from requested pool")
    expected_bandwidth = float(protocol.packed_layout["kernel_bandwidth"])
    if not np.isclose(
        public_pool.kernel_bandwidth, expected_bandwidth, rtol=0.0, atol=0.0
    ):
        raise BaseRuntimeBindingError("public pool bandwidth differs from protocol")

    pool_manifest = _verify_top_level_pool_manifest(layout, protocol, public_pool)
    source_specs = _task_specs_by_private_task(pool_manifest, public_pool)
    encoder_manifest = layout.verify_manifest_files(layout.encoder_manifest)
    normalization_manifest = layout.verify_manifest_files(layout.normalization_manifest)
    kernel_manifest = layout.verify_manifest_files(layout.kernel_manifest)
    selector_manifest = _require_mapping(
        read_json(layout.selector_pool_dir / "pool_manifest.json"),
        "selector_pool/pool_manifest.json",
    )
    selector_entries = selector_manifest.get("entries")
    if not isinstance(selector_entries, list):
        raise BaseRuntimeBindingError("selector pool entries must be a list")
    task_spec_digests = {
        str(record["opaque_id"]): _require_sha256(
            record["file_sha256"], "selector TaskSpec file_sha256"
        )
        for record in selector_entries
        if isinstance(record, Mapping)
    }
    if len(task_spec_digests) != len(public_pool.entries):
        raise BaseRuntimeBindingError("selector TaskSpec digest coverage is incomplete")

    asset_digests = {
        "protocol": sha256_file(layout.frozen_protocol),
        "normalization": sha256_file(layout.normalization),
        "encoder_checkpoint": sha256_file(layout.encoder_checkpoint),
        "encoder_config": sha256_file(layout.encoder_config),
        "kernel": sha256_file(layout.kernel),
        "normalization_manifest": sha256_file(layout.normalization_manifest),
        "encoder_manifest": sha256_file(layout.encoder_manifest),
        "kernel_manifest": sha256_file(layout.kernel_manifest),
        "source_rkmes": sha256_json(task_spec_digests),
    }
    # Explicitly tie the primary files to their already-verified nested records.
    nested_expected = {
        "normalization": normalization_manifest["files"]["normalization"]["sha256"],
        "encoder_checkpoint": encoder_manifest["files"]["checkpoint"]["sha256"],
        "encoder_config": encoder_manifest["files"]["config"]["sha256"],
        "kernel": kernel_manifest["files"]["kernel"]["sha256"],
    }
    for name, expected in nested_expected.items():
        if asset_digests[name] != expected:
            raise BaseRuntimeBindingError(f"nested manifest digest mismatch for {name}")

    return VerifiedBaseRuntime(
        base_artifacts_root=root,
        base_run_dir=layout.pool_root,
        protocol=protocol,
        public_pool=public_pool,
        protocol_manifest=protocol_manifest,
        pool_manifest=pool_manifest,
        source_task_specs=source_specs,
        asset_digests=asset_digests,
        protocol_manifest_sha256=sha256_file(layout.protocol_manifest),
        pool_manifest_sha256=sha256_file(layout.pool_manifest),
        public_pool_manifest_sha256=sha256_file(
            layout.selector_pool_dir / "pool_manifest.json"
        ),
    )


# More concise spelling for orchestration code.
load_verified_base_runtime = verify_and_load_base_runtime


__all__ = [
    "BaseRuntimeBindingError",
    "FrozenMeasurementAssets",
    "VerifiedBaseRuntime",
    "load_verified_base_runtime",
    "verify_and_load_base_runtime",
]
