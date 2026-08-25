from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from policy_learnware_v0.envs.base import SyntheticEnvAdapter
from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.v02.extensions import (
    CONTINUOUS_VECTOR_MDP_V02,
    ConformanceError,
    DuplicateEnvironmentBackendError,
    DuplicatePolicyRuntimeError,
    DuplicateSemanticEncoderError,
    EnvironmentBackendRegistry,
    EnvironmentCapabilities,
    EnvironmentCapabilityError,
    EnvironmentHandle,
    EvaluationContract,
    MujocoPlaygroundBackendPlugin,
    PolicyRuntimeRegistry,
    PolicyStep,
    ProtocolFamilyMismatch,
    RawTransitionEncoder,
    RepresentationBindingError,
    RuntimeContract,
    ScalarPolicyRuntimePlugin,
    SemanticEncoderRegistry,
    SyntheticEncoderAdapter,
    V02ExtensionRegistries,
    ValidatedPolicyBundle,
    check_environment_backend,
    check_policy_runtime,
    check_semantic_encoder,
    compatibility_partition,
    default_metadata,
)


def _sha(label: str) -> str:
    return sha256_json({"label": label})


class _FakeEnvironmentBackend:
    backend_id = "synthetic.test-only"
    capabilities = EnvironmentCapabilities(
        protocol_family_id=CONTINUOUS_VECTOR_MDP_V02,
        supports_training=False,
        supports_probe=True,
        supports_scalar_oracle=True,
        supports_compiled_oracle=False,
        provides_native_env=False,
    )

    def __init__(self, task: str = "SyntheticTask") -> None:
        self.adapter = SyntheticEnvAdapter(
            task=task, observation_dim=3, action_dim=2, horizon=4
        )

    def inspect(self, task_ref: str):
        if task_ref != self.adapter.schema.task:
            raise ValueError("unknown fake task")
        return self.adapter.schema

    def make_handle(self, instance_manifest, *, purpose):
        self.capabilities.require(purpose)
        if instance_manifest["protocol_family_id"] != CONTINUOUS_VECTOR_MDP_V02:
            raise ProtocolFamilyMismatch("fake family mismatch")
        return EnvironmentHandle(
            adapter=self.adapter,
            native_env=None,
            instance_digest=instance_manifest["instance_digest"],
            audit_digest=instance_manifest["audit_digest"],
            capabilities=self.capabilities,
        )

    def axis_operators(self):
        return {}


def _environment_fixture():
    plugin = _FakeEnvironmentBackend()
    manifest = {
        "protocol_family_id": CONTINUOUS_VECTOR_MDP_V02,
        "instance_digest": _sha("instance"),
        "audit_digest": _sha("audit"),
    }
    return plugin, manifest, plugin.make_handle(manifest, purpose="probe")


def _runtime_contract(handle: EnvironmentHandle, runtime_id: str) -> RuntimeContract:
    return RuntimeContract(
        protocol_family_id=handle.protocol_family_id,
        task_contract_digest=handle.task_contract_digest,
        observation_schema_digest=handle.observation_schema_digest,
        action_schema_digest=handle.action_schema_digest,
        observation_dim=handle.adapter.schema.observation_dim,
        action_dim=handle.adapter.schema.action_dim,
        action_transform_id="identity",
        policy_runtime_id=runtime_id,
        state_schema_id="counter-v0",
    )


class _FakePolicy:
    def __init__(self, contract: RuntimeContract) -> None:
        self.runtime_contract = contract

    def initial_state(self, seed: int) -> int:
        return 0

    def act(self, observation, state, key, *, deterministic):
        del observation, deterministic
        return PolicyStep(
            action=np.zeros(self.runtime_contract.action_dim, dtype=np.float32),
            state=int(state) + 1,
            next_key=int(key) + 1,
        )


def _scalar_runtime(root: Path, handle: EnvironmentHandle, *, runtime_id="fake-runtime"):
    contract = _runtime_contract(handle, runtime_id)

    def validate(path: Path) -> ValidatedPolicyBundle:
        if path.resolve() != root.resolve():
            raise ValueError("unexpected bundle path")
        return ValidatedPolicyBundle(
            path=path,
            bundle_schema="fake-policy-bundle.v0",
            bundle_digest=_sha("fake-bundle"),
            runtime_contract=contract,
            payload={"fake": True},
        )

    return ScalarPolicyRuntimePlugin(
        runtime_id=runtime_id,
        supported_bundle_schemas=("fake-policy-bundle.v0",),
        validator=validate,
        loader=lambda bundle: _FakePolicy(bundle.runtime_contract),
        parity_checker=lambda bundle, policy: {"passed": True},
    )


class EnvironmentExtensionTests(unittest.TestCase):
    def test_fake_backend_registry_and_conformance(self) -> None:
        plugin, manifest, handle = _environment_fixture()
        registry = EnvironmentBackendRegistry()
        registry.register(plugin)
        self.assertIs(
            registry.resolve(
                plugin.backend_id,
                protocol_family_id=CONTINUOUS_VECTOR_MDP_V02,
                purpose="probe",
            ),
            plugin,
        )
        with self.assertRaises(DuplicateEnvironmentBackendError):
            registry.register(plugin)
        with self.assertRaises(EnvironmentCapabilityError):
            registry.resolve(plugin.backend_id, purpose="train")
        with self.assertRaises(ProtocolFamilyMismatch):
            registry.resolve(plugin.backend_id, protocol_family_id="pixels-v0")

        report = check_environment_backend(
            plugin,
            task_ref=handle.adapter.schema.task,
            instance_manifest=manifest,
        )
        self.assertTrue(report.passed, report.to_dict())
        self.assertIs(report.require(), report)

    def test_official_wrapper_rejects_shifted_manifest_without_binding(self) -> None:
        def forbidden_factory(*args, **kwargs):
            raise AssertionError("nominal adapter must not be loaded")

        plugin = MujocoPlaygroundBackendPlugin(adapter_factory=forbidden_factory)
        with self.assertRaises(EnvironmentCapabilityError):
            plugin.make_handle(
                {
                    "protocol_family_id": CONTINUOUS_VECTOR_MDP_V02,
                    "axis_binding_digest": _sha("shifted-axis"),
                },
                purpose="train",
            )


class PolicyExtensionTests(unittest.TestCase):
    def test_scalar_runtime_conformance_and_stateful_evaluation(self) -> None:
        _, _, handle = _environment_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plugin = _scalar_runtime(root, handle)
            report = check_policy_runtime(
                plugin,
                bundle=root,
                environment=handle,
                seeds=(3, 7),
                evaluation_contract=EvaluationContract(
                    horizon=4, observation_dim=3, action_dim=2
                ),
            )
            self.assertTrue(report.passed, report.to_dict())

            validated = plugin.validate(root)
            policy = plugin.load(validated)
            rows = plugin.evaluate_batched(
                policy,
                handle,
                (3,),
                EvaluationContract(horizon=4, observation_dim=3, action_dim=2),
            )
            self.assertEqual(rows[0].steps, 4)
            self.assertTrue(rows[0].truncated)
            self.assertTrue(np.isfinite(rows[0].return_sum))

    def test_runtime_registry_rejects_duplicate_and_ambiguous_schema(self) -> None:
        _, _, handle = _environment_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _scalar_runtime(root, handle, runtime_id="fake-one")
            second = _scalar_runtime(root, handle, runtime_id="fake-two")
            registry = PolicyRuntimeRegistry()
            registry.register(first)
            with self.assertRaises(DuplicatePolicyRuntimeError):
                registry.register(first)
            registry.register(second)
            with self.assertRaisesRegex(ValueError, "ambiguous runtime routing"):
                registry.resolve_bundle_schema(
                    "fake-policy-bundle.v0",
                    protocol_family_id=CONTINUOUS_VECTOR_MDP_V02,
                )


class RepresentationExtensionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = SimpleNamespace(
            packed=np.arange(24, dtype=np.float32).reshape(6, 4) / 24.0,
            episode_offsets=np.asarray([0, 3, 6], dtype=np.int64),
        )
        self.event_digest = _sha("canonical-event-view")

    def test_raw_and_synthetic_encoders_share_conformance_protocol(self) -> None:
        raw = RawTransitionEncoder(
            default_metadata(
                representation_id="raw_transition_v02",
                family="raw",
                input_dim=4,
                output_dim=4,
                canonical_event_view_digest=self.event_digest,
            )
        )
        synthetic = SyntheticEncoderAdapter(
            default_metadata(
                representation_id="synthetic_future_v02",
                family="synthetic-test-only",
                input_dim=4,
                output_dim=2,
                canonical_event_view_digest=self.event_digest,
            ),
            lambda values: values[:, :2] * 2.0,
        )
        for encoder in (raw, synthetic):
            report = check_semantic_encoder(encoder, self.dataset, batch_size=2)
            self.assertTrue(report.passed, report.to_dict())
        self.assertNotEqual(
            raw.metadata.representation_protocol_id,
            synthetic.metadata.representation_protocol_id,
        )

    def test_encoder_registry_fails_closed_on_duplicates_and_view_drift(self) -> None:
        raw = RawTransitionEncoder(
            default_metadata(
                representation_id="raw_transition_v02",
                family="raw",
                input_dim=4,
                output_dim=4,
                canonical_event_view_digest=self.event_digest,
            )
        )
        registry = SemanticEncoderRegistry()
        registry.register(raw)
        with self.assertRaises(DuplicateSemanticEncoderError):
            registry.register(raw)
        with self.assertRaises(RepresentationBindingError):
            registry.resolve(
                raw.metadata.representation_id,
                canonical_event_view_digest=_sha("different-view"),
            )

    def test_conformance_reports_nonfinite_encoder_without_publishing_pass(self) -> None:
        bad = SyntheticEncoderAdapter(
            default_metadata(
                representation_id="bad-future-v02",
                family="synthetic-test-only",
                input_dim=4,
                output_dim=1,
                canonical_event_view_digest=self.event_digest,
            ),
            lambda values: np.full((values.shape[0], 1), np.nan, dtype=np.float32),
        )
        report = check_semantic_encoder(bad, self.dataset)
        self.assertFalse(report.passed)
        with self.assertRaises(ConformanceError):
            report.require()


class PartitionTests(unittest.TestCase):
    def test_cross_plugin_partition_and_capability_filter(self) -> None:
        environment, _, handle = _environment_fixture()
        event_digest = _sha("canonical-event-view")
        encoder = RawTransitionEncoder(
            default_metadata(
                representation_id="raw_transition_v02",
                family="raw",
                input_dim=4,
                output_dim=4,
                canonical_event_view_digest=event_digest,
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            runtime = _scalar_runtime(Path(directory), handle)
            registries = V02ExtensionRegistries()
            registries.environments.register(environment)
            registries.policies.register(runtime)
            registries.representations.register(encoder)
            resolved = registries.resolve_partition(
                backend_id=environment.backend_id,
                runtime_id=runtime.runtime_id,
                representation_id=encoder.metadata.representation_id,
                protocol_family_id=CONTINUOUS_VECTOR_MDP_V02,
                purpose="oracle",
            )
            self.assertEqual(resolved, (environment, runtime, encoder))
            validated = runtime.validate(Path(directory))
            partition = compatibility_partition(
                handle,
                validated.runtime_contract,
                encoder_metadata=encoder.metadata,
            )
            self.assertEqual(partition.protocol_family_id, CONTINUOUS_VECTOR_MDP_V02)
            self.assertEqual(len(partition.digest), 64)
            cross_task = replace(
                validated.runtime_contract,
                task_contract_digest=_sha("another-private-task-contract"),
            )
            self.assertEqual(
                compatibility_partition(handle, cross_task).digest,
                compatibility_partition(handle, validated.runtime_contract).digest,
            )
            with self.assertRaises(EnvironmentCapabilityError):
                registries.resolve_partition(
                    backend_id=environment.backend_id,
                    runtime_id=runtime.runtime_id,
                    representation_id=encoder.metadata.representation_id,
                    protocol_family_id=CONTINUOUS_VECTOR_MDP_V02,
                    purpose="oracle",
                    compiled_oracle=True,
                )


if __name__ == "__main__":
    unittest.main()
