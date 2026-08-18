"""Optional JAX/Flax transition encoder used by the v0 TaskSpec pipeline.

Importing this module never requires JAX.  Calls that initialize, train, or run
the encoder fail with an actionable dependency message when JAX/Flax/Optax are
not installed, allowing NumPy-only inspection and RKME tests to remain usable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contrastive import TaskBalancedBatchSampler, supervised_contrastive_loss_jax


class EncoderDependencyError(ImportError):
    """Raised only when an operation actually needs JAX/Flax/Optax."""


def _require_jax_stack(*, require_optax: bool = False) -> tuple[Any, ...]:
    try:
        import jax
        import jax.numpy as jnp
        from flax import linen as nn
        from flax import serialization
    except ImportError as exc:
        raise EncoderDependencyError(
            "TransitionSemanticEncoder requires JAX and Flax. Use the GoRL "
            "training environment or install compatible jax/flax packages."
        ) from exc
    if not require_optax:
        return jax, jnp, nn, serialization
    try:
        import optax
    except ImportError as exc:
        raise EncoderDependencyError(
            "Encoder training additionally requires Optax. Use the GoRL training "
            "environment or install a compatible optax package."
        ) from exc
    return jax, jnp, nn, serialization, optax


def jax_encoder_available(*, require_optax: bool = False) -> bool:
    try:
        _require_jax_stack(require_optax=require_optax)
    except EncoderDependencyError:
        return False
    return True


@dataclass(frozen=True)
class EncoderConfig:
    input_dim: int = 109
    hidden_dims: tuple[int, ...] = (256, 256)
    latent_dim: int = 32
    activation: str = "relu"
    l2_normalize_output: bool = True
    temperature: float = 0.1
    batch_size: int = 1024
    train_steps: int = 20_000
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.0
    validation_interval: int = 500
    validation_batches: int = 8
    seed: int = 0

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.latent_dim <= 0:
            raise ValueError("input_dim and latent_dim must be positive")
        if not self.hidden_dims or any(value <= 0 for value in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        if self.activation != "relu":
            raise ValueError("v0 supports only relu activation")
        if self.temperature <= 0 or self.batch_size <= 0 or self.train_steps < 0:
            raise ValueError("invalid encoder optimization configuration")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid learning rate or weight decay")
        if self.validation_interval <= 0 or self.validation_batches <= 0:
            raise ValueError("validation_interval and validation_batches must be positive")


def _coerce_config(config: EncoderConfig | Mapping[str, Any] | Any) -> EncoderConfig:
    if isinstance(config, EncoderConfig):
        return config
    if isinstance(config, Mapping):
        values = dict(config)
    else:
        values = {
            name: getattr(config, name)
            for name in EncoderConfig.__dataclass_fields__
            if hasattr(config, name)
        }
    if "hidden_dims" in values:
        values["hidden_dims"] = tuple(values["hidden_dims"])
    return EncoderConfig(**values)


def _model_for(config: EncoderConfig) -> Any:
    _, jnp, nn, _ = _require_jax_stack()

    class TransitionMLP(nn.Module):
        hidden_dims: Sequence[int]
        latent_dim: int
        normalize: bool

        @nn.compact
        def __call__(self, value: Any) -> Any:
            for width in self.hidden_dims:
                value = nn.relu(nn.Dense(width)(value))
            value = nn.Dense(self.latent_dim)(value)
            if self.normalize:
                denominator = jnp.maximum(
                    jnp.linalg.norm(value, axis=-1, keepdims=True), 1.0e-12
                )
                value = value / denominator
            return value

    return TransitionMLP(
        hidden_dims=config.hidden_dims,
        latent_dim=config.latent_dim,
        normalize=config.l2_normalize_output,
    )


@dataclass(frozen=True)
class EncoderCheckpoint:
    config: EncoderConfig
    params: Any
    best_step: int = 0
    best_validation_loss: float = float("nan")
    training_history: tuple[dict[str, float], ...] = field(default_factory=tuple)

    def save(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path | None = None,
        *,
        overwrite: bool = False,
    ) -> dict[str, str]:
        _, _, _, serialization = _require_jax_stack()
        from ..io import atomic_write_bytes, atomic_write_json

        digests = {
            "checkpoint": atomic_write_bytes(
                checkpoint_path,
                serialization.to_bytes(self.params),
                overwrite=overwrite,
            )
        }
        if config_path is not None:
            digests["config"] = atomic_write_json(
                config_path,
                asdict(self.config),
                overwrite=overwrite,
            )
        return digests

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        config: EncoderConfig | Mapping[str, Any] | Any,
    ) -> "EncoderCheckpoint":
        jax, jnp, _, serialization = _require_jax_stack()
        resolved = _coerce_config(config)
        model = _model_for(resolved)
        template = model.init(
            jax.random.PRNGKey(resolved.seed),
            jnp.zeros((1, resolved.input_dim), dtype=jnp.float32),
        )["params"]
        params = serialization.from_bytes(template, Path(checkpoint_path).read_bytes())
        return cls(config=resolved, params=params)


class TransitionSemanticEncoder:
    def __init__(self, checkpoint: EncoderCheckpoint) -> None:
        self.checkpoint = checkpoint

    @classmethod
    def initialize(
        cls, config: EncoderConfig | Mapping[str, Any] | Any = EncoderConfig()
    ) -> "TransitionSemanticEncoder":
        jax, jnp, _, _ = _require_jax_stack()
        resolved = _coerce_config(config)
        model = _model_for(resolved)
        variables = model.init(
            jax.random.PRNGKey(resolved.seed),
            jnp.zeros((1, resolved.input_dim), dtype=jnp.float32),
        )
        return cls(EncoderCheckpoint(config=resolved, params=variables["params"]))

    def encode(self, transitions: np.ndarray, *, batch_size: int = 8192) -> np.ndarray:
        _, jnp, _, _ = _require_jax_stack()
        value = np.asarray(transitions, dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != self.checkpoint.config.input_dim:
            raise ValueError(
                f"transitions must have shape [T,{self.checkpoint.config.input_dim}]"
            )
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        model = _model_for(self.checkpoint.config)
        chunks = []
        for start in range(0, value.shape[0], batch_size):
            output = model.apply(
                {"params": self.checkpoint.params},
                jnp.asarray(value[start : start + batch_size]),
            )
            chunks.append(np.asarray(output, dtype=np.float32))
        if not chunks:
            return np.empty((0, self.checkpoint.config.latent_dim), dtype=np.float32)
        return np.concatenate(chunks, axis=0)


def train_transition_encoder(
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    config: EncoderConfig | Mapping[str, Any] | Any = EncoderConfig(),
) -> EncoderCheckpoint:
    """Train the shared MLP using balanced episode-aware SupCon batches.

    The routine is intentionally compact and deterministic.  Artifact manifests,
    dataset digests and checkpoint publication live in the orchestration layer.
    """

    jax, jnp, _, _, optax = _require_jax_stack(require_optax=True)
    resolved = _coerce_config(config)
    if set(train) != set(validation):
        raise ValueError("training and validation task sets must match")
    train_sampler = TaskBalancedBatchSampler(
        train, batch_size=resolved.batch_size, seed=resolved.seed
    )
    validation_sampler = TaskBalancedBatchSampler(
        validation, batch_size=resolved.batch_size, seed=resolved.seed + 1
    )
    # Freeze validation draws before optimization.  Every checkpoint is scored
    # on the exact same registered batches, rather than on a moving random draw.
    frozen_validation_batches = tuple(
        validation_sampler.sample() for _ in range(resolved.validation_batches)
    )
    encoder = TransitionSemanticEncoder.initialize(resolved)
    model = _model_for(resolved)
    optimizer = optax.adamw(
        learning_rate=resolved.learning_rate, weight_decay=resolved.weight_decay
    )
    optimizer_state = optimizer.init(encoder.checkpoint.params)

    @jax.jit
    def step(params: Any, state: Any, x: Any, tasks: Any, episodes: Any) -> tuple[Any, ...]:
        def objective(candidate: Any) -> Any:
            z = model.apply({"params": candidate}, x)
            return supervised_contrastive_loss_jax(
                z, tasks, episodes, temperature=resolved.temperature
            )

        loss, gradients = jax.value_and_grad(objective)(params)
        updates, new_state = optimizer.update(gradients, state, params)
        return optax.apply_updates(params, updates), new_state, loss

    @jax.jit
    def evaluate(params: Any, x: Any, tasks: Any, episodes: Any) -> Any:
        z = model.apply({"params": params}, x)
        return supervised_contrastive_loss_jax(
            z, tasks, episodes, temperature=resolved.temperature
        )

    params = encoder.checkpoint.params
    best_params = params
    best_step = 0
    best_validation = float("inf")
    history: list[dict[str, float]] = []
    for step_index in range(1, resolved.train_steps + 1):
        batch = train_sampler.sample()
        params, optimizer_state, loss = step(
            params,
            optimizer_state,
            jnp.asarray(batch.transitions),
            jnp.asarray(batch.task_labels),
            jnp.asarray(batch.episode_ids),
        )
        if step_index % resolved.validation_interval == 0 or step_index == resolved.train_steps:
            validation_loss = float(
                np.mean(
                    [
                        float(
                            evaluate(
                                params,
                                jnp.asarray(validation_batch.transitions),
                                jnp.asarray(validation_batch.task_labels),
                                jnp.asarray(validation_batch.episode_ids),
                            )
                        )
                        for validation_batch in frozen_validation_batches
                    ]
                )
            )
            record = {
                "step": float(step_index),
                "train_loss": float(loss),
                "validation_loss": validation_loss,
            }
            history.append(record)
            if validation_loss < best_validation:
                best_validation = validation_loss
                best_step = step_index
                best_params = jax.tree.map(lambda value: value.copy(), params)

    # ``train_steps=0`` is useful only for dependency/integration smoke tests.
    if resolved.train_steps == 0:
        best_validation = float("nan")
    return EncoderCheckpoint(
        config=resolved,
        params=best_params,
        best_step=best_step,
        best_validation_loss=best_validation,
        training_history=tuple(history),
    )
