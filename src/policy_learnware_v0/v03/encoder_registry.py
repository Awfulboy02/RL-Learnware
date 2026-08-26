"""Encoder factory registry plus a deterministic fake adapter for conformance."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
from typing import Callable

import numpy as np

from ..hashing import canonical_json_bytes, sha256_bytes, sha256_json
from .access import AccessTier, EncoderAccessCard
from .checkpoints import EncoderCheckpointManifest
from .encoder_protocol import (
    AdapterFitData,
    AdapterTrainingContract,
    CostRecord,
    EncoderFitResult,
    EncoderInferenceContract,
    EncoderProtocolError,
    EncoderTrainingContract,
    EncoderTrainingData,
    EncoderValidationData,
    SanitizedEncoderInputBatch,
    SemanticEncoderProtocol,
    SemanticSampleBatch,
    adapter_training_manifest_digest,
    project_adapter_fit,
    validate_adapter_fit_invocation,
    validate_encoder_input_access,
    validate_fit_result_bindings,
    validate_semantic_output,
)
from .schemas import EncoderProtocolRecord, checked_digest, checked_safe_id


EncoderFactory = Callable[[], SemanticEncoderProtocol]

_FAKE_CHECKPOINT_SCHEMA = "policy-learnware.v03-fake-encoder-checkpoint.v1"
_FAKE_CHECKPOINT_FIELDS = {
    "schema",
    "encoder_id",
    "training_manifest_digest",
    "protocol_record_digest",
    "training_contract_digest",
    "access_card_digest",
    "input_view_digest",
    "semantic_output_protocol_digest",
    "runtime_digest",
    "fold_id",
    "seed",
    "latent_dim",
    "training_cost",
}
_COST_RECORD_FIELDS = {
    "wall_seconds",
    "peak_memory_bytes",
    "device",
    "trial_count",
}


class EncoderRegistryError(ValueError):
    """An encoder registration or adapter factory is invalid."""


@dataclass(frozen=True)
class EncoderRegistration:
    encoder_id: str
    family: str
    access_tier: AccessTier
    implementation_digest: str
    protocol_record_digest: str
    access_card_digest: str
    input_view_digest: str
    window_protocol_digest: str
    semantic_output_protocol_digest: str
    latent_dim: int
    factory: EncoderFactory

    def __post_init__(self) -> None:
        object.__setattr__(self, "encoder_id", checked_safe_id(self.encoder_id, "encoder_id"))
        object.__setattr__(self, "family", checked_safe_id(self.family, "family"))
        if self.access_tier not in {
            "E0_UNSUPERVISED",
            "E1_CATEGORICAL_SOURCE",
            "E2_PRIVILEGED_PARAMETER",
        }:
            raise EncoderRegistryError("unknown registration access tier")
        object.__setattr__(
            self,
            "implementation_digest",
            checked_digest(self.implementation_digest, "implementation_digest"),
        )
        for name in (
            "protocol_record_digest",
            "access_card_digest",
            "input_view_digest",
            "window_protocol_digest",
            "semantic_output_protocol_digest",
        ):
            object.__setattr__(self, name, checked_digest(getattr(self, name), name))
        if isinstance(self.latent_dim, bool) or not isinstance(self.latent_dim, int) or self.latent_dim <= 0:
            raise EncoderRegistryError("registration latent_dim must be positive")
        if not callable(self.factory):
            raise EncoderRegistryError("encoder factory must be callable")

    @property
    def registration_digest(self) -> str:
        return sha256_json(
            {
                "schema": "policy-learnware.v03-encoder-registration.v0",
                "encoder_id": self.encoder_id,
                "family": self.family,
                "access_tier": self.access_tier,
                "implementation_digest": self.implementation_digest,
                "protocol_record_digest": self.protocol_record_digest,
                "access_card_digest": self.access_card_digest,
                "input_view_digest": self.input_view_digest,
                "window_protocol_digest": self.window_protocol_digest,
                "semantic_output_protocol_digest": self.semantic_output_protocol_digest,
                "latent_dim": self.latent_dim,
            }
        )


class EncoderRegistry:
    def __init__(self) -> None:
        self._registrations: dict[str, EncoderRegistration] = {}

    def register(self, registration: EncoderRegistration) -> None:
        if not isinstance(registration, EncoderRegistration):
            raise EncoderRegistryError("registration must be EncoderRegistration")
        if registration.encoder_id in self._registrations:
            raise EncoderRegistryError(
                f"duplicate encoder registration: {registration.encoder_id!r}"
            )
        adapter = registration.factory()
        if not isinstance(adapter, SemanticEncoderProtocol):
            raise EncoderRegistryError("factory result does not implement SemanticEncoderProtocol")
        if adapter.encoder_family != registration.family:
            raise EncoderRegistryError("factory encoder_family differs from registration")
        adapter_bindings = {
            "protocol_record_digest": registration.protocol_record_digest,
            "access_card_digest": registration.access_card_digest,
            "semantic_output_protocol_digest": (
                registration.semantic_output_protocol_digest
            ),
        }
        for name, expected in adapter_bindings.items():
            if getattr(adapter, name, None) != expected:
                raise EncoderRegistryError(
                    f"factory {name} differs from registration"
                )
        signature = inspect.signature(adapter.encode_windows)
        forbidden = {"task_id", "anchor_id", "factor", "probe_style", "candidate_id"}
        leaked = forbidden & set(signature.parameters)
        if leaked:
            raise EncoderRegistryError(
                f"encode_windows exposes forbidden inference identities: {sorted(leaked)}"
            )
        self._registrations[registration.encoder_id] = registration

    def registration(self, encoder_id: str) -> EncoderRegistration:
        try:
            return self._registrations[encoder_id]
        except KeyError as exc:
            raise EncoderRegistryError(f"unknown encoder ID: {encoder_id!r}") from exc

    def create(self, encoder_id: str) -> SemanticEncoderProtocol:
        registration = self.registration(encoder_id)
        adapter = registration.factory()
        if not isinstance(adapter, SemanticEncoderProtocol):
            raise EncoderRegistryError("factory drifted from SemanticEncoderProtocol")
        if adapter.encoder_family != registration.family:
            raise EncoderRegistryError("factory family drifted after registration")
        for name in (
            "protocol_record_digest",
            "access_card_digest",
            "semantic_output_protocol_digest",
        ):
            if getattr(adapter, name, None) != getattr(registration, name):
                raise EncoderRegistryError(f"factory {name} drifted after registration")
        return adapter

    def manifest(self) -> dict[str, object]:
        entries = [
            {
                "encoder_id": registration.encoder_id,
                "family": registration.family,
                "access_tier": registration.access_tier,
                "implementation_digest": registration.implementation_digest,
                "protocol_record_digest": registration.protocol_record_digest,
                "access_card_digest": registration.access_card_digest,
                "input_view_digest": registration.input_view_digest,
                "window_protocol_digest": registration.window_protocol_digest,
                "semantic_output_protocol_digest": registration.semantic_output_protocol_digest,
                "latent_dim": registration.latent_dim,
                "registration_digest": registration.registration_digest,
            }
            for registration in sorted(
                self._registrations.values(), key=lambda item: item.encoder_id
            )
        ]
        material: dict[str, object] = {
            "schema": "policy-learnware.v03-encoder-registry.v0",
            "entries": entries,
        }
        return {**material, "registry_digest": sha256_json(material)}


class FakeSemanticEncoder:
    """Non-scientific deterministic adapter used only to test the shared contract."""

    def __init__(
        self,
        *,
        protocol_record: EncoderProtocolRecord,
        access_card: EncoderAccessCard,
        semantic_output_protocol_digest: str,
    ) -> None:
        if protocol_record.encoder_id != access_card.encoder_id:
            raise EncoderRegistryError("fake protocol/access-card encoder IDs disagree")
        if protocol_record.access_card_digest != access_card.access_card_digest:
            raise EncoderRegistryError("fake protocol/access-card digest mismatch")
        self.encoder_family = protocol_record.family
        self._protocol = protocol_record
        self._access = access_card
        self._semantic_output_protocol_digest = checked_digest(
            semantic_output_protocol_digest, "semantic_output_protocol_digest"
        )
        self.protocol_record_digest = protocol_record.protocol_record_digest
        self.access_card_digest = access_card.access_card_digest
        self.semantic_output_protocol_digest = self._semantic_output_protocol_digest
        self._frozen: EncoderFitResult | None = None

    @staticmethod
    def _checkpoint_material(
        *,
        encoder_id: str,
        training_manifest_digest: str,
        protocol_record_digest: str,
        training_contract_digest: str,
        access_card_digest: str,
        input_view_digest: str,
        semantic_output_protocol_digest: str,
        runtime_digest: str,
        fold_id: str,
        seed: int,
        latent_dim: int,
        training_cost: CostRecord,
    ) -> dict[str, object]:
        return {
            "schema": _FAKE_CHECKPOINT_SCHEMA,
            "encoder_id": encoder_id,
            "training_manifest_digest": training_manifest_digest,
            "protocol_record_digest": protocol_record_digest,
            "training_contract_digest": training_contract_digest,
            "access_card_digest": access_card_digest,
            "input_view_digest": input_view_digest,
            "semantic_output_protocol_digest": semantic_output_protocol_digest,
            "runtime_digest": runtime_digest,
            "fold_id": fold_id,
            "seed": seed,
            "latent_dim": latent_dim,
            "training_cost": training_cost.to_dict(),
        }

    @classmethod
    def _checkpoint_material_from_fit(
        cls, fit: EncoderFitResult
    ) -> dict[str, object]:
        return cls._checkpoint_material(
            encoder_id=fit.encoder_id,
            training_manifest_digest=fit.training_manifest_digest,
            protocol_record_digest=fit.protocol_record_digest,
            training_contract_digest=fit.training_contract_digest,
            access_card_digest=fit.access_card_digest,
            input_view_digest=fit.input_view_digest,
            semantic_output_protocol_digest=fit.semantic_output_protocol_digest,
            runtime_digest=fit.runtime_digest,
            fold_id=fit.fold_id,
            seed=fit.seed,
            latent_dim=fit.latent_dim,
            training_cost=fit.training_cost,
        )

    def _assert_fit_matches_protocol(self, fit: EncoderFitResult) -> None:
        if not isinstance(fit, EncoderFitResult):
            raise EncoderProtocolError("fake checkpoint requires EncoderFitResult")
        expected = {
            "encoder_id": self._protocol.encoder_id,
            "protocol_record_digest": self._protocol.protocol_record_digest,
            "access_card_digest": self._access.access_card_digest,
            "input_view_digest": self._protocol.input_view_digest,
            "semantic_output_protocol_digest": (
                self._semantic_output_protocol_digest
            ),
            "latent_dim": self._protocol.latent_dim,
        }
        drift = {
            name
            for name, value in expected.items()
            if getattr(fit, name) != value
        }
        if drift:
            raise EncoderProtocolError(
                f"fit result does not match fake encoder protocol: {sorted(drift)}"
            )

    def fit(
        self,
        train: AdapterFitData,
        validation: AdapterFitData,
        contract: AdapterTrainingContract,
    ) -> EncoderFitResult:
        if not isinstance(train, AdapterFitData) or not isinstance(
            validation, AdapterFitData
        ):
            raise EncoderProtocolError("fake fit requires sanitized train/validation data")
        if not isinstance(contract, AdapterTrainingContract):
            raise EncoderProtocolError("fake fit requires AdapterTrainingContract")
        if contract.protocol_record != self._protocol or contract.access_card != self._access:
            raise EncoderProtocolError("fake fit contract does not match registered protocol")
        if (
            contract.semantic_output_protocol_digest
            != self._semantic_output_protocol_digest
        ):
            raise EncoderProtocolError("training semantic-output protocol mismatch")
        validate_adapter_fit_invocation(train, validation, contract)
        training_manifest_digest = adapter_training_manifest_digest(
            train, validation, contract
        )
        training_cost = CostRecord(
            wall_seconds=0.0,
            peak_memory_bytes=0,
            device="synthetic-cpu",
            trial_count=1,
        )
        checkpoint_material = self._checkpoint_material(
            encoder_id=self._protocol.encoder_id,
            training_manifest_digest=training_manifest_digest,
            protocol_record_digest=self._protocol.protocol_record_digest,
            training_contract_digest=contract.training_contract_digest,
            access_card_digest=self._access.access_card_digest,
            input_view_digest=self._protocol.input_view_digest,
            semantic_output_protocol_digest=self._semantic_output_protocol_digest,
            runtime_digest=contract.runtime_digest,
            fold_id=contract.abstract_fold_id,
            seed=contract.seed,
            latent_dim=self._protocol.latent_dim,
            training_cost=training_cost,
        )
        checkpoint_digest = sha256_bytes(canonical_json_bytes(checkpoint_material))
        return EncoderFitResult(
            encoder_id=self._protocol.encoder_id,
            checkpoint_digest=checkpoint_digest,
            training_manifest_digest=training_manifest_digest,
            protocol_record_digest=self._protocol.protocol_record_digest,
            training_contract_digest=contract.training_contract_digest,
            access_card_digest=self._access.access_card_digest,
            input_view_digest=self._protocol.input_view_digest,
            semantic_output_protocol_digest=self._semantic_output_protocol_digest,
            runtime_digest=contract.runtime_digest,
            fold_id=contract.abstract_fold_id,
            seed=contract.seed,
            latent_dim=self._protocol.latent_dim,
            training_cost=training_cost,
        )

    def load_frozen(self, fit: EncoderFitResult) -> None:
        if not isinstance(fit, EncoderFitResult):
            raise EncoderProtocolError("load_frozen requires EncoderFitResult")
        self._assert_fit_matches_protocol(fit)
        self._frozen = fit

    def export_frozen_checkpoint_bytes(self, fit: EncoderFitResult) -> bytes:
        """Return the one canonical byte representation committed by ``fit``."""

        self._assert_fit_matches_protocol(fit)
        payload = canonical_json_bytes(self._checkpoint_material_from_fit(fit))
        if sha256_bytes(payload) != fit.checkpoint_digest:
            raise EncoderProtocolError(
                "fit checkpoint digest does not match canonical fake checkpoint bytes"
            )
        return payload

    def load_frozen_checkpoint_bytes(
        self,
        *,
        manifest: EncoderCheckpointManifest,
        checkpoint_bytes: bytes,
    ) -> EncoderFitResult:
        """Reconstruct a frozen fit on a fresh adapter from verified bytes.

        The caller is expected to obtain ``manifest`` and ``checkpoint_bytes``
        through :func:`load_frozen_encoder_checkpoint`.  This adapter boundary
        nevertheless rechecks the exact content hash, canonical encoding, all
        manifest bindings, and the local protocol before accepting state.
        """

        if not isinstance(manifest, EncoderCheckpointManifest):
            raise EncoderProtocolError(
                "fresh checkpoint load requires EncoderCheckpointManifest"
            )
        if type(checkpoint_bytes) is not bytes:
            raise EncoderProtocolError("fresh checkpoint payload must be exact bytes")
        payload = checkpoint_bytes
        if len(payload) != manifest.checkpoint_size_bytes:
            raise EncoderProtocolError(
                "checkpoint byte size differs from checkpoint manifest"
            )
        if sha256_bytes(payload) != manifest.checkpoint_digest:
            raise EncoderProtocolError(
                "checkpoint bytes digest differs from checkpoint manifest"
            )
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EncoderProtocolError(
                "fake checkpoint bytes are not canonical JSON"
            ) from exc
        if not isinstance(decoded, dict) or set(decoded) != _FAKE_CHECKPOINT_FIELDS:
            raise EncoderProtocolError("fake checkpoint fields differ from schema")
        if decoded.get("schema") != _FAKE_CHECKPOINT_SCHEMA:
            raise EncoderProtocolError("unknown fake checkpoint schema")
        if canonical_json_bytes(decoded) != payload:
            raise EncoderProtocolError("fake checkpoint JSON is not canonical")
        raw_cost = decoded.get("training_cost")
        if not isinstance(raw_cost, dict) or set(raw_cost) != _COST_RECORD_FIELDS:
            raise EncoderProtocolError("fake checkpoint training_cost is malformed")
        try:
            cost = CostRecord(
                wall_seconds=raw_cost["wall_seconds"],
                peak_memory_bytes=raw_cost["peak_memory_bytes"],
                device=raw_cost["device"],
                trial_count=raw_cost["trial_count"],
            )
            fit = EncoderFitResult(
                encoder_id=decoded["encoder_id"],
                checkpoint_digest=manifest.checkpoint_digest,
                training_manifest_digest=decoded["training_manifest_digest"],
                protocol_record_digest=decoded["protocol_record_digest"],
                training_contract_digest=decoded["training_contract_digest"],
                access_card_digest=decoded["access_card_digest"],
                input_view_digest=decoded["input_view_digest"],
                semantic_output_protocol_digest=decoded[
                    "semantic_output_protocol_digest"
                ],
                runtime_digest=decoded["runtime_digest"],
                fold_id=decoded["fold_id"],
                seed=decoded["seed"],
                latent_dim=decoded["latent_dim"],
                training_cost=cost,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EncoderProtocolError("fake checkpoint payload is malformed") from exc

        manifest_bindings = {
            "encoder_id": manifest.encoder_id,
            "checkpoint_digest": manifest.checkpoint_digest,
            "training_manifest_digest": manifest.training_manifest_digest,
            "protocol_record_digest": manifest.protocol_record_digest,
            "training_contract_digest": manifest.training_contract_digest,
            "access_card_digest": manifest.access_card_digest,
            "input_view_digest": manifest.input_view_digest,
            "semantic_output_protocol_digest": (
                manifest.semantic_output_protocol_digest
            ),
            "runtime_digest": manifest.runtime_digest,
            "fold_id": manifest.fold_id,
            "seed": manifest.seed,
            "latent_dim": manifest.latent_dim,
        }
        drift = {
            name
            for name, expected in manifest_bindings.items()
            if getattr(fit, name) != expected
        }
        if drift:
            raise EncoderProtocolError(
                f"fake checkpoint payload differs from manifest: {sorted(drift)}"
            )
        local_manifest_bindings = {
            "encoder_id": self._protocol.encoder_id,
            "protocol_record_digest": self._protocol.protocol_record_digest,
            "access_card_digest": self._access.access_card_digest,
            "input_view_digest": self._protocol.input_view_digest,
            "window_protocol_digest": self._protocol.window_protocol_digest,
            "semantic_output_protocol_digest": (
                self._semantic_output_protocol_digest
            ),
            "latent_dim": self._protocol.latent_dim,
        }
        local_drift = {
            name
            for name, expected in local_manifest_bindings.items()
            if getattr(manifest, name) != expected
        }
        if local_drift:
            raise EncoderProtocolError(
                f"checkpoint manifest differs from fresh adapter: {sorted(local_drift)}"
            )
        self.load_frozen(fit)
        return fit

    def encode_windows(
        self,
        inputs: SanitizedEncoderInputBatch,
        *,
        inference_contract: EncoderInferenceContract,
    ) -> SemanticSampleBatch:
        if self._frozen is None:
            raise EncoderProtocolError("encoder checkpoint has not been loaded and frozen")
        if not isinstance(inputs, SanitizedEncoderInputBatch):
            raise EncoderProtocolError(
                "encode_windows requires SanitizedEncoderInputBatch"
            )
        if not isinstance(inference_contract, EncoderInferenceContract):
            raise EncoderProtocolError("missing typed inference contract")
        expected_pairs = {
            "checkpoint": (
                inference_contract.checkpoint_digest, self._frozen.checkpoint_digest
            ),
            "input view": (
                inference_contract.input_view_digest, self._protocol.input_view_digest
            ),
            "window protocol": (
                inference_contract.window_protocol_digest,
                self._protocol.window_protocol_digest,
            ),
            "semantic output": (
                inference_contract.semantic_output_protocol_digest,
                self._semantic_output_protocol_digest,
            ),
            "runtime": (
                inference_contract.runtime_digest,
                self._frozen.runtime_digest,
            ),
        }
        for name, (actual, expected) in expected_pairs.items():
            if actual != expected:
                raise EncoderProtocolError(f"{name} digest mismatch")
        if inputs.window_protocol_digest != self._protocol.window_protocol_digest:
            raise EncoderProtocolError("inference windows use an unknown protocol")
        if inputs.input_view_digest != self._protocol.input_view_digest:
            raise EncoderProtocolError("inference inputs use an unknown view")
        validate_encoder_input_access(inputs, self._access)

        observation = inputs.channel("observation").reshape(
            len(inputs.window_ids), inputs.window_length, -1
        )
        action = inputs.channel("action").reshape(
            len(inputs.window_ids), inputs.window_length, -1
        )
        next_observation = inputs.channel("next_observation").reshape(
            len(inputs.window_ids), inputs.window_length, -1
        )
        features = np.concatenate(
            [observation, action, next_observation - observation], axis=2
        )
        mask = inputs.window_mask[:, :, None].astype(np.float64)
        pooled = np.sum(features * mask, axis=1) / np.sum(mask, axis=1)
        seed_material = sha256_json(
            {
                "checkpoint_digest": self._frozen.checkpoint_digest,
                "feature_dim": int(pooled.shape[1]),
                "latent_dim": self._protocol.latent_dim,
            }
        )
        rng = np.random.default_rng(int(seed_material[:16], 16))
        projection = rng.standard_normal(
            (pooled.shape[1], self._protocol.latent_dim), dtype=np.float64
        ) / np.sqrt(float(pooled.shape[1]))
        values = np.asarray(pooled @ projection, dtype=np.float64)
        output = SemanticSampleBatch(
            values=values,
            valid_mask=np.ones(values.shape[0], dtype=np.bool_),
            window_ids=inputs.window_ids,
        )
        validate_semantic_output(output, inputs, latent_dim=self._protocol.latent_dim)
        return output


@dataclass(frozen=True)
class AdapterConformanceReport:
    encoder_id: str
    checkpoint_digest: str
    semantic_batch_digest: str
    deterministic: bool
    passed: bool


def run_adapter_conformance(
    adapter: SemanticEncoderProtocol,
    *,
    train: EncoderTrainingData,
    validation: EncoderValidationData,
    training_contract: EncoderTrainingContract,
    inference_contract_factory: Callable[[EncoderFitResult], EncoderInferenceContract],
) -> AdapterConformanceReport:
    if not isinstance(adapter, SemanticEncoderProtocol):
        raise EncoderProtocolError(
            "adapter does not implement SemanticEncoderProtocol"
        )
    if not isinstance(training_contract, EncoderTrainingContract):
        raise EncoderProtocolError(
            "conformance requires EncoderTrainingContract"
        )
    expected_adapter_bindings = {
        "encoder_family": training_contract.protocol_record.family,
        "protocol_record_digest": training_contract.protocol_record.protocol_record_digest,
        "access_card_digest": training_contract.access_card.access_card_digest,
        "semantic_output_protocol_digest": (
            training_contract.semantic_output_protocol_digest
        ),
    }
    drifted_adapter = {
        name
        for name, expected in expected_adapter_bindings.items()
        if getattr(adapter, name, None) != expected
    }
    if drifted_adapter:
        raise EncoderProtocolError(
            f"adapter declaration bindings drifted: {sorted(drifted_adapter)}"
        )
    adapter_train, adapter_validation, adapter_contract = project_adapter_fit(
        train, validation, training_contract
    )
    fit = adapter.fit(adapter_train, adapter_validation, adapter_contract)
    validate_fit_result_bindings(
        fit, adapter_train, adapter_validation, adapter_contract
    )
    adapter.load_frozen(fit)
    inference_contract = inference_contract_factory(fit)
    first = adapter.encode_windows(
        adapter_validation.inputs, inference_contract=inference_contract
    )
    validate_semantic_output(
        first,
        adapter_validation.inputs,
        latent_dim=training_contract.protocol_record.latent_dim,
    )
    second = adapter.encode_windows(
        adapter_validation.inputs, inference_contract=inference_contract
    )
    validate_semantic_output(
        second,
        adapter_validation.inputs,
        latent_dim=training_contract.protocol_record.latent_dim,
    )
    deterministic = first.semantic_batch_digest == second.semantic_batch_digest
    if not deterministic:
        raise EncoderProtocolError("adapter output is not deterministic")
    return AdapterConformanceReport(
        encoder_id=fit.encoder_id,
        checkpoint_digest=fit.checkpoint_digest,
        semantic_batch_digest=first.semantic_batch_digest,
        deterministic=True,
        passed=True,
    )


def fake_registration(
    protocol_record: EncoderProtocolRecord,
    access_card: EncoderAccessCard,
    *,
    semantic_output_protocol_digest: str,
) -> EncoderRegistration:
    if protocol_record.implementation_digest != checked_digest(
        protocol_record.implementation_digest, "implementation_digest"
    ):
        raise EncoderRegistryError("invalid fake implementation digest")
    return EncoderRegistration(
        encoder_id=protocol_record.encoder_id,
        family=protocol_record.family,
        access_tier=access_card.access_tier,
        implementation_digest=protocol_record.implementation_digest,
        protocol_record_digest=protocol_record.protocol_record_digest,
        access_card_digest=access_card.access_card_digest,
        input_view_digest=protocol_record.input_view_digest,
        window_protocol_digest=protocol_record.window_protocol_digest,
        semantic_output_protocol_digest=semantic_output_protocol_digest,
        latent_dim=protocol_record.latent_dim,
        factory=lambda: FakeSemanticEncoder(
            protocol_record=protocol_record,
            access_card=access_card,
            semantic_output_protocol_digest=semantic_output_protocol_digest,
        ),
    )


__all__ = [
    "AdapterConformanceReport",
    "EncoderFactory",
    "EncoderRegistration",
    "EncoderRegistry",
    "EncoderRegistryError",
    "FakeSemanticEncoder",
    "fake_registration",
    "run_adapter_conformance",
]
