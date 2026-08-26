"""Frozen, candidate-independent probe policies for v0.3.

Probe policies emit public normalized actions in ``[-1, 1]^d``.  Mapping to
native actuator bounds is an explicit backend operation, making differences in
action ABI auditable rather than silently part of the probe implementation.
The formal CP2 choice still requires the joint-freeze authority described by
the coding plan; the literal implementation below is a pre-bakeoff holdout and
cannot appear in an encoder training manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

import numpy as np

from ..hashing import canonicalize, sha256_json
from ..probe.gaussian import GaussianRandomProbe
from ..schemas import EnvSchema


CP0_EXACT_COMMON = "CP0_EXACT_COMMON"
CP1_FAMILY_SHIFT = "CP1_FAMILY_SHIFT"
CP2_UNSEEN_PROBE = "CP2_UNSEEN_PROBE"
PROBE_REGIMES = frozenset(
    {CP0_EXACT_COMMON, CP1_FAMILY_SHIFT, CP2_UNSEEN_PROBE}
)
PROBE_POLICY_PROTOCOL_ID = sha256_json(
    {
        "schema": "policy-learnware.v03-probe-policy-protocol.v0",
        "action_domain": "normalized[-1,1]",
        "native_mapping": "affine-per-coordinate",
        "candidate_independent": True,
        "style_visibility": "training-private-not-semantic",
    }
)

CP0_STYLE_ID = "gaussian_white_v0"
CP1_OU_STYLE_ID = "ou_colored_v0"
CP1_SWEEP_STYLE_ID = "coordinate_sweep_v0"
CP2_STYLE_ID = "impulse_hold_literal_v0"

FORBIDDEN_PROBE_TOKENS = frozenset(
    {
        "candidate",
        "policy",
        "bundle",
        "q_value",
        "qvalue",
        "oracle",
        "return",
        "regret",
        "task_id",
        "axis",
        "factor",
        "anchor",
    }
)


class ProbeContractError(ValueError):
    """A probe policy, ABI, seed binding, or holdout manifest is invalid."""


def _digest(value: Any, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
    ):
        raise ProbeContractError(f"{where} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ProbeContractError(
            f"{where} must be a lowercase SHA-256 digest"
        ) from error
    return value


def _nonempty(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ProbeContractError(f"{where} must be a non-empty canonical string")
    return value


def _readonly_vector(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1 or array.size == 0:
        raise ProbeContractError(f"{name} must be a non-empty vector")
    if not np.all(np.isfinite(array)):
        raise ProbeContractError(f"{name} contains non-finite values")
    result = np.ascontiguousarray(array).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ActionABI:
    """Only the actuator geometry required by a public probe."""

    low: np.ndarray
    high: np.ndarray
    dtype: str = "float32"

    def __post_init__(self) -> None:
        low = _readonly_vector(self.low, name="ActionABI.low")
        high = _readonly_vector(self.high, name="ActionABI.high")
        if high.shape != low.shape or np.any(low >= high):
            raise ProbeContractError("ActionABI bounds must be same-shape and ordered")
        if np.dtype(self.dtype) != np.dtype(np.float32):
            raise ProbeContractError("v0.3 public probe ABI uses frozen float32")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    @classmethod
    def from_env_schema(cls, schema: EnvSchema) -> "ActionABI":
        if not isinstance(schema, EnvSchema):
            raise ProbeContractError("schema must be an EnvSchema")
        return cls(low=schema.action_low, high=schema.action_high)

    @property
    def action_dim(self) -> int:
        return int(self.low.size)

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schema": "policy-learnware.v03-action-abi.v0",
                "low": self.low.tolist(),
                "high": self.high.tolist(),
                "dtype": self.dtype,
            }
        )

    def map_normalized(self, normalized_action: Any) -> np.ndarray:
        action = np.asarray(normalized_action, dtype=np.float32)
        if action.shape != self.low.shape:
            raise ProbeContractError(
                f"normalized action shape {action.shape} != {(self.action_dim,)}"
            )
        if not np.all(np.isfinite(action)):
            raise ProbeContractError("normalized action contains non-finite values")
        tolerance = np.float32(8.0 * np.finfo(np.float32).eps)
        if np.any(action < -1.0 - tolerance) or np.any(action > 1.0 + tolerance):
            raise ProbeContractError("normalized action lies outside [-1, 1]")
        clipped = np.clip(action, -1.0, 1.0)
        native = self.low + np.float32(0.5) * (clipped + 1.0) * (
            self.high - self.low
        )
        native = np.asarray(native, dtype=np.float32)
        native.setflags(write=False)
        return native


@dataclass(frozen=True)
class ProbeStyle:
    probe_family_id: str
    probe_style_id: str
    regime: Literal[
        "CP0_EXACT_COMMON", "CP1_FAMILY_SHIFT", "CP2_UNSEEN_PROBE"
    ]
    implementation: Literal[
        "gaussian_white", "ou_colored", "coordinate_sweep", "impulse_hold"
    ]
    parameters: Mapping[str, float | int]
    freeze_authority: str
    eligible_for_encoder_training: bool

    def __post_init__(self) -> None:
        _nonempty(self.probe_family_id, "probe_family_id")
        _nonempty(self.probe_style_id, "probe_style_id")
        _nonempty(self.freeze_authority, "freeze_authority")
        if self.regime not in PROBE_REGIMES:
            raise ProbeContractError(f"unknown probe regime: {self.regime!r}")
        if self.implementation not in {
            "gaussian_white",
            "ou_colored",
            "coordinate_sweep",
            "impulse_hold",
        }:
            raise ProbeContractError(f"unknown probe implementation: {self.implementation!r}")
        if not isinstance(self.parameters, Mapping):
            raise ProbeContractError("probe parameters must be a mapping")
        parameters = canonicalize(dict(self.parameters))
        if not isinstance(parameters, dict):  # pragma: no cover - canonicalize contract
            raise ProbeContractError("probe parameters must canonicalize to a mapping")
        for name, value in parameters.items():
            if name.lower() in FORBIDDEN_PROBE_TOKENS:
                raise ProbeContractError(f"forbidden candidate-dependent parameter: {name}")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ProbeContractError(f"probe parameter {name} must be numeric")
            if not np.isfinite(float(value)):
                raise ProbeContractError(f"probe parameter {name} is non-finite")
        if self.regime == CP2_UNSEEN_PROBE and self.eligible_for_encoder_training:
            raise ProbeContractError("CP2 style cannot be eligible for encoder training")
        object.__setattr__(self, "parameters", MappingProxyType(parameters))
        assert_candidate_independent(self)

    @property
    def digest(self) -> str:
        return sha256_json(self.to_private_dict())

    def to_private_dict(self) -> dict[str, Any]:
        """Training-private record.  Style ID is not a semantic input channel."""

        return {
            "probe_family_id": self.probe_family_id,
            "probe_style_id": self.probe_style_id,
            "regime": self.regime,
            "implementation": self.implementation,
            "parameters": dict(self.parameters),
            "freeze_authority": self.freeze_authority,
            "eligible_for_encoder_training": self.eligible_for_encoder_training,
            "probe_policy_protocol_id": PROBE_POLICY_PROTOCOL_ID,
        }


def assert_candidate_independent(style: ProbeStyle) -> None:
    """Reject any public style definition that embeds forbidden evidence."""

    payload = style.to_private_dict()

    def walk(value: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                lowered = str(key).lower()
                if lowered in FORBIDDEN_PROBE_TOKENS:
                    raise ProbeContractError(
                        "probe definition is candidate-dependent at " + ".".join((*path, str(key)))
                    )
                walk(child, (*path, str(key)))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, (*path, str(index)))

    walk(payload)


FROZEN_PROBE_STYLES: Mapping[str, ProbeStyle] = MappingProxyType(
    {
        style.probe_style_id: style
        for style in (
            ProbeStyle(
                probe_family_id="public_normalized_noise_v0",
                probe_style_id=CP0_STYLE_ID,
                regime=CP0_EXACT_COMMON,
                implementation="gaussian_white",
                parameters={"sigma": 1.0},
                freeze_authority="legacy-family-compatible-implementation-literal",
                eligible_for_encoder_training=True,
            ),
            ProbeStyle(
                probe_family_id="public_normalized_noise_v0",
                probe_style_id=CP1_OU_STYLE_ID,
                regime=CP1_FAMILY_SHIFT,
                implementation="ou_colored",
                parameters={"theta": 0.18, "sigma": 0.35},
                freeze_authority="v03-development-implementation-literal",
                eligible_for_encoder_training=True,
            ),
            ProbeStyle(
                probe_family_id="public_normalized_excitation_v0",
                probe_style_id=CP1_SWEEP_STYLE_ID,
                regime=CP1_FAMILY_SHIFT,
                implementation="coordinate_sweep",
                parameters={"amplitude": 0.65, "base_period": 12, "chirp_rate": 0.015},
                freeze_authority="v03-development-implementation-literal",
                eligible_for_encoder_training=True,
            ),
            ProbeStyle(
                probe_family_id="public_normalized_excitation_v0",
                probe_style_id=CP2_STYLE_ID,
                regime=CP2_UNSEEN_PROBE,
                implementation="impulse_hold",
                parameters={"amplitude": 0.7, "hold_steps": 5, "zero_steps": 3},
                freeze_authority="pre-bakeoff-implementation-literal-awaiting-joint-freeze",
                eligible_for_encoder_training=False,
            ),
        )
    }
)


@dataclass(frozen=True)
class ProbeState:
    seed: int
    action_abi_digest: str
    action_dim: int
    normalized_memory: np.ndarray
    step_count: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ProbeContractError("probe seed must be a nonnegative integer")
        if self.action_dim <= 0:
            raise ProbeContractError("action_dim must be positive")
        memory = _readonly_vector(self.normalized_memory, name="normalized_memory")
        if memory.shape != (self.action_dim,):
            raise ProbeContractError("normalized_memory shape disagrees with action_dim")
        if np.any(np.abs(memory) > 1.0):
            raise ProbeContractError("normalized_memory lies outside [-1, 1]")
        if self.step_count < 0:
            raise ProbeContractError("step_count must be nonnegative")
        object.__setattr__(self, "normalized_memory", memory)


@runtime_checkable
class ProbePolicyProtocol(Protocol):
    probe_family_id: str
    probe_style_id: str

    def reset(self, seed: int, native_action_abi: ActionABI) -> ProbeState: ...

    def act(
        self,
        observation: np.ndarray,
        state: ProbeState,
        *,
        step: int,
    ) -> tuple[np.ndarray, ProbeState]: ...


def _step_rng(seed: int, step: int, style_digest: str) -> np.random.Generator:
    material = sha256_json(
        {"seed": seed, "step": step, "style_digest": style_digest}
    )
    return np.random.default_rng(int(material[:16], 16))


@dataclass(frozen=True)
class FrozenProbePolicy:
    style: ProbeStyle

    @property
    def probe_family_id(self) -> str:
        return self.style.probe_family_id

    @property
    def probe_style_id(self) -> str:
        return self.style.probe_style_id

    def reset(self, seed: int, native_action_abi: ActionABI) -> ProbeState:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ProbeContractError("probe seed must be a nonnegative integer")
        if not isinstance(native_action_abi, ActionABI):
            raise ProbeContractError("native_action_abi must be an ActionABI")
        return ProbeState(
            seed=seed,
            action_abi_digest=native_action_abi.digest,
            action_dim=native_action_abi.action_dim,
            normalized_memory=np.zeros(native_action_abi.action_dim, dtype=np.float32),
        )

    def act(
        self,
        observation: np.ndarray,
        state: ProbeState,
        *,
        step: int,
    ) -> tuple[np.ndarray, ProbeState]:
        observation_array = np.asarray(observation)
        if observation_array.ndim != 1 or not np.all(np.isfinite(observation_array)):
            raise ProbeContractError("probe observation must be a finite vector")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ProbeContractError("step must be a nonnegative integer")
        if step != state.step_count:
            raise ProbeContractError("probe steps must be contiguous and state-aligned")
        dimension = state.action_dim
        parameters = self.style.parameters
        rng = _step_rng(state.seed, step, self.style.digest)
        implementation = self.style.implementation
        if implementation == "gaussian_white":
            # The backend-neutral reference follows the existing NumPy probe
            # sequence exactly.  Production Threefry collection can use
            # ``legacy_cp0_normalized_action_tensor`` below.
            action = GaussianRandomProbe(
                sigma=float(parameters["sigma"])
            ).sample_sequence_numpy(
                seed=state.seed,
                steps=step + 1,
                action_low=-np.ones(dimension, dtype=np.float32),
                action_high=np.ones(dimension, dtype=np.float32),
            )[-1]
        elif implementation == "ou_colored":
            noise = rng.normal(0.0, float(parameters["sigma"]), size=dimension)
            theta = float(parameters["theta"])
            action = state.normalized_memory + theta * (-state.normalized_memory) + noise
        elif implementation == "coordinate_sweep":
            action = np.zeros(dimension, dtype=np.float64)
            coordinate = step % dimension
            phase = (
                2.0
                * np.pi
                * step
                / float(parameters["base_period"])
                + float(parameters["chirp_rate"]) * step * step
            )
            action[coordinate] = float(parameters["amplitude"]) * np.sin(phase)
        elif implementation == "impulse_hold":
            hold = int(parameters["hold_steps"])
            zero = int(parameters["zero_steps"])
            cycle = hold + zero
            block = step // cycle
            within = step % cycle
            action = np.zeros(dimension, dtype=np.float64)
            if within < hold:
                coordinate = block % dimension
                sign = 1.0 if (block // dimension) % 2 == 0 else -1.0
                action[coordinate] = sign * float(parameters["amplitude"])
        else:  # pragma: no cover - ProbeStyle rejects unknown implementations
            raise AssertionError(implementation)
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        action.setflags(write=False)
        next_state = ProbeState(
            seed=state.seed,
            action_abi_digest=state.action_abi_digest,
            action_dim=dimension,
            normalized_memory=action,
            step_count=step + 1,
        )
        return action, next_state


def registered_probe(style_id: str) -> FrozenProbePolicy:
    try:
        return FrozenProbePolicy(FROZEN_PROBE_STYLES[style_id])
    except KeyError as error:
        raise ProbeContractError(f"unregistered probe style: {style_id!r}") from error


def legacy_cp0_normalized_action_tensor(
    *,
    probe_seeds: tuple[int, ...] | list[int] | np.ndarray,
    horizon: int,
    action_dim: int,
) -> Any:
    """Reuse the v0.1 Threefry full-episode generator in normalized space.

    The returned ``[N,H,A]`` tensor remains a JAX array.  Native action mapping
    must happen afterwards through :class:`ActionABI`; this helper never sees
    task, axis, candidate, bundle, return, or oracle information.
    """

    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ProbeContractError("horizon must be a positive integer")
    if isinstance(action_dim, bool) or not isinstance(action_dim, int) or action_dim <= 0:
        raise ProbeContractError("action_dim must be a positive integer")
    # Delayed import keeps dependency-light CPU contract tests free of JAX.
    from types import SimpleNamespace

    from ..v01.probe import frozen_probe_action_tensor

    schema = SimpleNamespace(
        horizon=horizon,
        action_dim=action_dim,
        action_low=-np.ones(action_dim, dtype=np.float32),
        action_high=np.ones(action_dim, dtype=np.float32),
    )
    return frozen_probe_action_tensor(
        schema,
        probe_seeds,
        sigma=float(FROZEN_PROBE_STYLES[CP0_STYLE_ID].parameters["sigma"]),
    )


ProbeRole = Literal["source_reference", "target_query"]


@dataclass(frozen=True)
class ProbeSeedBinding:
    role: ProbeRole
    style_id: str
    namespace: str
    nonce: str
    episode_id: int
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.role not in {"source_reference", "target_query"}:
            raise ProbeContractError(f"invalid probe role: {self.role!r}")
        if self.style_id not in FROZEN_PROBE_STYLES:
            raise ProbeContractError(f"unregistered probe style: {self.style_id!r}")
        _nonempty(self.namespace, "namespace")
        _nonempty(self.nonce, "nonce")
        if isinstance(self.episode_id, bool) or not isinstance(self.episode_id, int) or self.episode_id < 0:
            raise ProbeContractError("episode_id must be a nonnegative integer")
        expected = int(
            sha256_json(
                {
                    "protocol_id": PROBE_POLICY_PROTOCOL_ID,
                    "role": self.role,
                    "style_id": self.style_id,
                    "namespace": self.namespace,
                    "nonce": self.nonce,
                    "episode_id": self.episode_id,
                }
            )[:8],
            16,
        )
        if self.seed is None:
            object.__setattr__(self, "seed", expected)
        elif self.seed != expected:
            raise ProbeContractError("probe seed does not match the frozen binding")

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v03-probe-seed-binding.v0",
            "probe_policy_protocol_id": PROBE_POLICY_PROTOCOL_ID,
            "role": self.role,
            "style_id": self.style_id,
            "namespace": self.namespace,
            "nonce": self.nonce,
            "episode_id": self.episode_id,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class ProbeCollectionReceipt:
    """Typed proof that one bank was collected by replaying a frozen probe.

    This record is emitted only after the collector has replayed every
    normalized action, mapped it through the declared native Action ABI, and
    compared the result with the stored transition rows.  It is development
    evidence, not a formal G03-Probe authority signature.
    """

    role: ProbeRole
    probe_style_id: str
    style_digest: str
    action_abi_digest: str
    collection_implementation_digest: str
    seed_binding_digests: tuple[str, ...]
    seed_sequence_digest: str
    dataset_digest: str
    normalized_actions_digest: str
    native_actions_digest: str
    episode_count: int
    transition_count: int
    replay_verification_digest: str | None = None

    def __post_init__(self) -> None:
        if self.role not in {"source_reference", "target_query"}:
            raise ProbeContractError(f"invalid probe receipt role: {self.role!r}")
        try:
            style = FROZEN_PROBE_STYLES[self.probe_style_id]
        except KeyError as error:
            raise ProbeContractError(
                f"unregistered receipt probe style: {self.probe_style_id!r}"
            ) from error
        if _digest(self.style_digest, "style_digest") != style.digest:
            raise ProbeContractError("receipt style digest disagrees with registry")
        for name in (
            "action_abi_digest",
            "collection_implementation_digest",
            "seed_sequence_digest",
            "dataset_digest",
            "normalized_actions_digest",
            "native_actions_digest",
        ):
            _digest(getattr(self, name), name)
        bindings = tuple(self.seed_binding_digests)
        if not bindings or len(bindings) != self.episode_count:
            raise ProbeContractError(
                "receipt must bind exactly one seed record per episode"
            )
        if len(set(bindings)) != len(bindings):
            raise ProbeContractError("receipt seed binding digests must be unique")
        for digest in bindings:
            _digest(digest, "seed_binding_digest")
        if (
            isinstance(self.episode_count, bool)
            or not isinstance(self.episode_count, int)
            or self.episode_count <= 0
        ):
            raise ProbeContractError("receipt episode_count must be positive")
        if (
            isinstance(self.transition_count, bool)
            or not isinstance(self.transition_count, int)
            or self.transition_count < self.episode_count
        ):
            raise ProbeContractError(
                "receipt transition_count must cover all episodes"
            )
        object.__setattr__(self, "seed_binding_digests", bindings)
        expected = sha256_json(self._verification_payload())
        if self.replay_verification_digest is None:
            object.__setattr__(self, "replay_verification_digest", expected)
        elif (
            _digest(
                self.replay_verification_digest, "replay_verification_digest"
            )
            != expected
        ):
            raise ProbeContractError(
                "replay verification digest disagrees with receipt contents"
            )

    def _verification_payload(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v03-probe-collection-replay.v0",
            "probe_policy_protocol_id": PROBE_POLICY_PROTOCOL_ID,
            "role": self.role,
            "probe_style_id": self.probe_style_id,
            "style_digest": self.style_digest,
            "action_abi_digest": self.action_abi_digest,
            "collection_implementation_digest": self.collection_implementation_digest,
            "seed_binding_digests": list(self.seed_binding_digests),
            "seed_sequence_digest": self.seed_sequence_digest,
            "dataset_digest": self.dataset_digest,
            "normalized_actions_digest": self.normalized_actions_digest,
            "native_actions_digest": self.native_actions_digest,
            "episode_count": self.episode_count,
            "transition_count": self.transition_count,
            "verification": "FROZEN_PROBE_REPLAY_AND_NATIVE_ACTION_EQUALITY",
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    @property
    def candidate_independence_pass(self) -> bool:
        """Derived from the frozen style/seed/action replay contract.

        This is deliberately not a caller-provided boolean.  It says only
        that this receipt is internally bound to the candidate-independent
        public probe protocol; it is not a formal authority attestation.
        """

        style = FROZEN_PROBE_STYLES[self.probe_style_id]
        try:
            assert_candidate_independent(style)
        except ProbeContractError:  # pragma: no cover - frozen registry invariant
            return False
        return self.replay_verification_digest == sha256_json(
            self._verification_payload()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._verification_payload(),
            "replay_verification_digest": self.replay_verification_digest,
        }


@dataclass(frozen=True)
class ProbeTrainingManifest:
    training_style_ids: tuple[str, ...]
    confirmatory_style_id: str
    fold_ids: tuple[str, ...]
    freeze_authority: str

    def __post_init__(self) -> None:
        if not self.training_style_ids or len(set(self.training_style_ids)) != len(
            self.training_style_ids
        ):
            raise ProbeContractError("training style IDs must be unique and non-empty")
        unknown = set(self.training_style_ids) - set(FROZEN_PROBE_STYLES)
        if unknown:
            raise ProbeContractError(f"unregistered training styles: {sorted(unknown)}")
        if self.confirmatory_style_id not in FROZEN_PROBE_STYLES:
            raise ProbeContractError("confirmatory style is unregistered")
        if not self.fold_ids or len(set(self.fold_ids)) != len(self.fold_ids):
            raise ProbeContractError("fold IDs must be unique and non-empty")
        _nonempty(self.freeze_authority, "freeze_authority")
        validate_cp2_holdout(self)

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schema": "policy-learnware.v03-probe-training-manifest.v0",
                "training_style_ids": list(self.training_style_ids),
                "confirmatory_style_id": self.confirmatory_style_id,
                "fold_ids": list(self.fold_ids),
                "freeze_authority": self.freeze_authority,
            }
        )


def validate_cp2_holdout(manifest: ProbeTrainingManifest) -> None:
    if manifest.confirmatory_style_id in manifest.training_style_ids:
        raise ProbeContractError("CP2 style appears in the encoder training manifest")
    style = FROZEN_PROBE_STYLES[manifest.confirmatory_style_id]
    if style.regime != CP2_UNSEEN_PROBE or style.eligible_for_encoder_training:
        raise ProbeContractError("confirmatory style is not a frozen CP2 holdout")
    for style_id in manifest.training_style_ids:
        if not FROZEN_PROBE_STYLES[style_id].eligible_for_encoder_training:
            raise ProbeContractError(f"non-training style appears in manifest: {style_id}")
