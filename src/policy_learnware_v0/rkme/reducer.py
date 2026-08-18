"""03-style weighted reduced KME with auditable reconstruction error."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .empirical import EmpiricalKME
from .gaussian import GaussianKernel


@dataclass(frozen=True)
class ReducerConfig:
    support_budget: int = 100
    init: str = "weighted_kmeans"
    support_steps: int = 1_000
    learning_rate: float = 1.0e-2
    pinv_rcond: float = 1.0e-8
    ridge: float = 1.0e-6
    kmeans_steps: int = 25
    negative_tolerance: float = 1.0e-8
    optimizer_backend: str = "numpy"

    def __post_init__(self) -> None:
        if self.support_budget <= 0:
            raise ValueError("support_budget must be positive")
        if self.init != "weighted_kmeans":
            raise ValueError("v0 supports only deterministic weighted_kmeans init")
        if self.support_steps < 0 or self.kmeans_steps < 0:
            raise ValueError("optimization step counts cannot be negative")
        if (
            self.learning_rate <= 0
            or self.pinv_rcond <= 0
            or self.ridge < 0
            or self.negative_tolerance < 0
        ):
            raise ValueError("invalid reducer numerical configuration")
        if self.optimizer_backend not in {"numpy", "jax"}:
            raise ValueError("optimizer_backend must be numpy or jax")


def _coerce_config(config: ReducerConfig | Mapping[str, Any] | Any) -> ReducerConfig:
    if isinstance(config, ReducerConfig):
        return config
    if isinstance(config, Mapping):
        values = dict(config)
    else:
        values = {
            name: getattr(config, name)
            for name in ReducerConfig.__dataclass_fields__
            if hasattr(config, name)
        }
    return ReducerConfig(**values)


@dataclass(frozen=True)
class ReducedRKME:
    supports: np.ndarray
    beta: np.ndarray
    bandwidth: float
    rkme_norm2: float
    empirical_norm2: float
    reduction_error: float
    protocol_id: str = ""
    source_dataset_digest: str = ""
    ridge: float = 0.0
    condition_number: float = float("nan")
    objective_trace: tuple[float, ...] = ()
    negative_residual_clamped: bool = False
    source_task: str = ""
    source_dataset_manifest_digest: str = ""
    raw_reduction_residual_squared: float = float("nan")

    def __post_init__(self) -> None:
        supports = np.asarray(self.supports, dtype=np.float64)
        beta = np.asarray(self.beta, dtype=np.float64)
        if supports.ndim != 2 or supports.shape[0] == 0:
            raise ValueError("supports must have non-empty shape [M,Q]")
        if beta.shape != (supports.shape[0],):
            raise ValueError("beta must have shape [M]")
        if not np.all(np.isfinite(supports)) or not np.all(np.isfinite(beta)):
            raise ValueError("supports and beta must be finite")
        for name, value in (
            ("bandwidth", self.bandwidth),
            ("rkme_norm2", self.rkme_norm2),
            ("empirical_norm2", self.empirical_norm2),
            ("reduction_error", self.reduction_error),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        if self.rkme_norm2 < -1e-10 or self.empirical_norm2 < -1e-10:
            raise ValueError("RKHS squared norms cannot be negative")
        if self.reduction_error < 0:
            raise ValueError("reduction_error is an RKHS norm and cannot be negative")
        supports = np.array(supports, copy=True)
        beta = np.array(beta, copy=True)
        supports.setflags(write=False)
        beta.setflags(write=False)
        object.__setattr__(self, "supports", supports)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "bandwidth", float(self.bandwidth))
        object.__setattr__(self, "rkme_norm2", max(float(self.rkme_norm2), 0.0))
        object.__setattr__(self, "empirical_norm2", max(float(self.empirical_norm2), 0.0))
        object.__setattr__(self, "reduction_error", float(self.reduction_error))
        object.__setattr__(self, "ridge", float(self.ridge))
        object.__setattr__(self, "condition_number", float(self.condition_number))
        raw_residual = float(self.raw_reduction_residual_squared)
        if np.isnan(raw_residual):
            raw_residual = float(self.reduction_error) ** 2
        if not np.isfinite(raw_residual):
            raise ValueError("raw_reduction_residual_squared must be finite")
        object.__setattr__(self, "raw_reduction_residual_squared", raw_residual)
        object.__setattr__(
            self, "objective_trace", tuple(float(value) for value in self.objective_trace)
        )

    def save_npz(self, path: str | Path, *, overwrite: bool = False) -> str:
        """Save the minimum TaskSpec payload plus reducer diagnostics."""

        from ..io import atomic_write_npz

        return atomic_write_npz(
            path,
            {
                "supports": self.supports,
                "beta": self.beta,
                "bandwidth": np.asarray(self.bandwidth, dtype=np.float64),
                "rkme_norm2": np.asarray(self.rkme_norm2, dtype=np.float64),
                "empirical_norm2": np.asarray(self.empirical_norm2, dtype=np.float64),
                "reduction_error": np.asarray(self.reduction_error, dtype=np.float64),
                "protocol_id": np.asarray(self.protocol_id),
                "source_dataset_digest": np.asarray(self.source_dataset_digest),
                "ridge": np.asarray(self.ridge, dtype=np.float64),
                "condition_number": np.asarray(self.condition_number, dtype=np.float64),
                "objective_trace": np.asarray(self.objective_trace, dtype=np.float64),
                "negative_residual_clamped": np.asarray(
                    self.negative_residual_clamped, dtype=np.bool_
                ),
                "source_task": np.asarray(self.source_task),
                "source_dataset_manifest_digest": np.asarray(
                    self.source_dataset_manifest_digest
                ),
                "raw_reduction_residual_squared": np.asarray(
                    self.raw_reduction_residual_squared, dtype=np.float64
                ),
            },
            overwrite=overwrite,
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "ReducedRKME":
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(
                supports=data["supports"],
                beta=data["beta"],
                bandwidth=float(data["bandwidth"]),
                rkme_norm2=float(data["rkme_norm2"]),
                empirical_norm2=float(data["empirical_norm2"]),
                reduction_error=float(data["reduction_error"]),
                protocol_id=str(data["protocol_id"]),
                source_dataset_digest=str(data["source_dataset_digest"]),
                ridge=float(data["ridge"]),
                condition_number=float(data["condition_number"]),
                objective_trace=tuple(np.asarray(data["objective_trace"]).tolist()),
                negative_residual_clamped=bool(data["negative_residual_clamped"]),
                source_task=(str(data["source_task"]) if "source_task" in data else ""),
                source_dataset_manifest_digest=(
                    str(data["source_dataset_manifest_digest"])
                    if "source_dataset_manifest_digest" in data
                    else ""
                ),
                raw_reduction_residual_squared=(
                    float(data["raw_reduction_residual_squared"])
                    if "raw_reduction_residual_squared" in data
                    else float("nan")
                ),
            )


def deterministic_weighted_kmeans(
    points: np.ndarray,
    weights: np.ndarray,
    support_budget: int,
    *,
    steps: int = 25,
) -> np.ndarray:
    """Deterministic weighted farthest-first initialization plus Lloyd updates."""

    x = np.asarray(points, dtype=np.float64)
    alpha = np.asarray(weights, dtype=np.float64)
    if x.ndim != 2 or alpha.shape != (x.shape[0],):
        raise ValueError("weighted k-means inputs are not aligned")
    if support_budget <= 0:
        raise ValueError("support_budget must be positive")
    count = min(int(support_budget), x.shape[0])
    if count == x.shape[0]:
        return np.array(x, copy=True)

    selected = [int(np.argmax(alpha))]
    x_norm = np.sum(np.square(x), axis=1)
    nearest_squared = np.maximum(
        x_norm + np.sum(np.square(x[selected[0]])) - 2.0 * (x @ x[selected[0]]),
        0.0,
    )
    for _ in range(1, count):
        score = alpha * nearest_squared
        score[np.asarray(selected, dtype=np.int64)] = -np.inf
        next_index = int(np.argmax(score))
        selected.append(next_index)
        distance = np.maximum(
            x_norm + np.sum(np.square(x[next_index])) - 2.0 * (x @ x[next_index]),
            0.0,
        )
        nearest_squared = np.minimum(nearest_squared, distance)
    supports = np.array(x[selected], copy=True)

    for _ in range(steps):
        squared = np.maximum(
            x_norm[:, None]
            + np.sum(np.square(supports), axis=1)[None, :]
            - 2.0 * (x @ supports.T),
            0.0,
        )
        assignment = np.argmin(squared, axis=1)
        updated = np.array(supports, copy=True)
        empty: list[int] = []
        for support_index in range(count):
            member = assignment == support_index
            mass = float(np.sum(alpha[member]))
            if mass > 0:
                updated[support_index] = np.sum(
                    x[member] * alpha[member, None], axis=0
                ) / mass
            else:
                empty.append(support_index)
        if empty:
            nearest = np.min(squared, axis=1)
            candidates = np.argsort(-(alpha * nearest), kind="stable")
            used: set[int] = set()
            for support_index in empty:
                candidate = next(index for index in candidates if int(index) not in used)
                used.add(int(candidate))
                updated[support_index] = x[candidate]
        if np.allclose(updated, supports, rtol=0.0, atol=1e-12):
            supports = updated
            break
        supports = updated
    return supports


def solve_beta(
    supports: np.ndarray,
    empirical: EmpiricalKME,
    kernel: GaussianKernel,
    *,
    ridge: float,
    pinv_rcond: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kuu = kernel.gram(supports)
    kuz = kernel.gram(supports, empirical.points)
    system = kuu + float(ridge) * np.eye(kuu.shape[0], dtype=np.float64)
    beta = np.linalg.pinv(system, rcond=float(pinv_rcond)) @ kuz @ empirical.weights
    return beta, kuu, kuz


def _objective_terms(
    empirical: EmpiricalKME,
    supports: np.ndarray,
    beta: np.ndarray,
    kuu: np.ndarray,
    kuz: np.ndarray,
) -> tuple[float, float, float, bool]:
    rkme_norm2 = float(beta @ kuu @ beta)
    cross = float(beta @ kuz @ empirical.weights)
    residual_squared = float(empirical.norm2 - 2.0 * cross + rkme_norm2)
    clamped = residual_squared < 0.0
    error = float(np.sqrt(max(residual_squared, 0.0)))
    return residual_squared, rkme_norm2, error, clamped


def _support_gradient(
    supports: np.ndarray,
    beta: np.ndarray,
    empirical: EmpiricalKME,
    kernel: GaussianKernel,
    *,
    block_size: int = 4096,
) -> np.ndarray:
    sigma_squared = kernel.bandwidth * kernel.bandwidth
    gradient = np.zeros_like(supports, dtype=np.float64)
    # Gradient of -2 beta^T K_uz alpha, accumulated without retaining K_uz.
    for start in range(0, empirical.transition_count, block_size):
        stop = min(start + block_size, empirical.transition_count)
        points = empirical.points[start:stop]
        weights = empirical.weights[start:stop]
        gram = kernel.gram(supports, points)
        weighted_gram = gram * weights[None, :]
        expectation = (
            weighted_gram @ points
            - np.sum(weighted_gram, axis=1)[:, None] * supports
        )
        gradient += -2.0 * beta[:, None] * expectation / sigma_squared

    # Gradient of beta^T K_uu beta.  Diagonal displacement is exactly zero.
    kuu = kernel.gram(supports)
    weighted_kuu = kuu * beta[None, :]
    interaction = (
        weighted_kuu @ supports
        - np.sum(weighted_kuu, axis=1)[:, None] * supports
    )
    gradient += 2.0 * beta[:, None] * interaction / sigma_squared
    return gradient


def _optimize_supports_jax(
    initial_supports: np.ndarray,
    empirical: EmpiricalKME,
    config: ReducerConfig,
    initial_objective: float,
) -> tuple[np.ndarray, tuple[float, ...]]:
    """Run the production alternating updates as one JIT-compiled JAX scan.

    The implementation materializes only ``K_uz`` (M x T), never ``K_zz`` or
    an ``[T,M,Q]`` displacement tensor.  This keeps the 64k-transition main
    setting within a practical GPU memory envelope.
    """

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as error:  # pragma: no cover - server dependency gate
        raise RuntimeError(
            "optimizer_backend='jax' requires the GoRL JAX environment"
        ) from error
    jax.config.update("jax_enable_x64", True)
    points = jnp.asarray(empirical.points, dtype=jnp.float64)
    weights = jnp.asarray(empirical.weights, dtype=jnp.float64)
    initial = jnp.asarray(initial_supports, dtype=jnp.float64)
    sigma_squared = float(empirical.bandwidth) ** 2
    ridge = float(config.ridge)

    def gram(left: Any, right: Any) -> Any:
        squared = jnp.maximum(
            jnp.sum(left * left, axis=1)[:, None]
            + jnp.sum(right * right, axis=1)[None, :]
            - 2.0 * (left @ right.T),
            0.0,
        )
        return jnp.exp(-squared / (2.0 * sigma_squared))

    def solve(supports: Any) -> tuple[Any, Any, Any]:
        kuu = gram(supports, supports)
        kuz = gram(supports, points)
        system = kuu + ridge * jnp.eye(kuu.shape[0], dtype=jnp.float64)
        rhs = kuz @ weights
        beta = (
            jnp.linalg.solve(system, rhs)
            if ridge > 0.0
            else jnp.linalg.pinv(system, rtol=float(config.pinv_rcond)) @ rhs
        )
        return beta, kuu, kuz

    def objective(beta: Any, kuu: Any, kuz: Any) -> Any:
        return (
            float(empirical.norm2)
            - 2.0 * (beta @ kuz @ weights)
            + beta @ kuu @ beta
        )

    def gradient(supports: Any, beta: Any, kuu: Any, kuz: Any) -> Any:
        weighted_kuz = kuz * weights[None, :]
        expectation = (
            weighted_kuz @ points
            - jnp.sum(weighted_kuz, axis=1)[:, None] * supports
        )
        result = -2.0 * beta[:, None] * expectation / sigma_squared
        weighted_kuu = kuu * beta[None, :]
        interaction = (
            weighted_kuu @ supports
            - jnp.sum(weighted_kuu, axis=1)[:, None] * supports
        )
        return result + 2.0 * beta[:, None] * interaction / sigma_squared

    def run(starting_supports: Any) -> tuple[Any, Any]:
        beta, kuu, kuz = solve(starting_supports)
        carry = (
            starting_supports,
            beta,
            kuu,
            kuz,
            jnp.zeros_like(starting_supports),
            jnp.zeros_like(starting_supports),
            jnp.asarray(initial_objective, dtype=jnp.float64),
            starting_supports,
        )

        def body(state: tuple[Any, ...], step_zero: Any) -> tuple[Any, Any]:
            supports, beta, kuu, kuz, first, second, best_value, best_supports = state
            grad = gradient(supports, beta, kuu, kuz)
            first = 0.9 * first + 0.1 * grad
            second = 0.999 * second + 0.001 * jnp.square(grad)
            step = step_zero + 1
            corrected_first = first / (1.0 - 0.9**step)
            corrected_second = second / (1.0 - 0.999**step)
            supports = supports - float(config.learning_rate) * corrected_first / (
                jnp.sqrt(corrected_second) + 1.0e-8
            )
            beta, kuu, kuz = solve(supports)
            raw = objective(beta, kuu, kuz)
            value = jnp.maximum(raw, 0.0)
            better = value < best_value
            best_value = jnp.where(better, value, best_value)
            best_supports = jnp.where(better, supports, best_supports)
            return (
                supports,
                beta,
                kuu,
                kuz,
                first,
                second,
                best_value,
                best_supports,
            ), value

        final, values = jax.lax.scan(
            body,
            carry,
            jnp.arange(config.support_steps, dtype=jnp.int32),
        )
        return final[-1], values

    best_supports, values = jax.jit(run)(initial)
    trace = (float(initial_objective),) + tuple(
        float(value) for value in np.asarray(jax.device_get(values))
    )
    return np.asarray(jax.device_get(best_supports), dtype=np.float64), trace


def reduce_kme(
    empirical: EmpiricalKME,
    config: ReducerConfig | Mapping[str, Any] | Any = ReducerConfig(),
) -> ReducedRKME:
    """Compress an empirical KME and report RKHS reconstruction *norm*.

    Supports are initialized deterministically.  Alternating support optimization
    uses an analytic Gaussian-kernel gradient and Adam, avoiding a hard JAX
    dependency in this numerical layer.  No global-optimum claim is made.
    """

    resolved = _coerce_config(config)
    kernel = GaussianKernel(empirical.bandwidth)
    supports = deterministic_weighted_kmeans(
        empirical.points,
        empirical.weights,
        resolved.support_budget,
        steps=resolved.kmeans_steps,
    )
    beta, kuu, kuz = solve_beta(
        supports,
        empirical,
        kernel,
        ridge=resolved.ridge,
        pinv_rcond=resolved.pinv_rcond,
    )
    residual_squared, _, _, _ = _objective_terms(
        empirical, supports, beta, kuu, kuz
    )
    trace = [max(residual_squared, 0.0)]
    best_objective = trace[0]
    best_supports = np.array(supports, copy=True)
    best_beta = np.array(beta, copy=True)
    best_kuu = np.array(kuu, copy=True)
    best_kuz = np.array(kuz, copy=True)

    if resolved.optimizer_backend == "jax" and resolved.support_steps:
        best_supports, trace_values = _optimize_supports_jax(
            supports, empirical, resolved, best_objective
        )
        trace = list(trace_values)
        best_beta, best_kuu, best_kuz = solve_beta(
            best_supports,
            empirical,
            kernel,
            ridge=resolved.ridge,
            pinv_rcond=resolved.pinv_rcond,
        )
    else:
        first_moment = np.zeros_like(supports)
        second_moment = np.zeros_like(supports)
        for step_index in range(1, resolved.support_steps + 1):
            gradient = _support_gradient(supports, beta, empirical, kernel)
            first_moment = 0.9 * first_moment + 0.1 * gradient
            second_moment = 0.999 * second_moment + 0.001 * np.square(gradient)
            corrected_first = first_moment / (1.0 - 0.9**step_index)
            corrected_second = second_moment / (1.0 - 0.999**step_index)
            supports = supports - resolved.learning_rate * corrected_first / (
                np.sqrt(corrected_second) + 1.0e-8
            )
            beta, kuu, kuz = solve_beta(
                supports,
                empirical,
                kernel,
                ridge=resolved.ridge,
                pinv_rcond=resolved.pinv_rcond,
            )
            residual_squared, _, _, _ = _objective_terms(
                empirical, supports, beta, kuu, kuz
            )
            objective = max(residual_squared, 0.0)
            trace.append(objective)
            # Adam is not line-searched; persist the best alternating iterate.
            if objective < best_objective:
                best_objective = objective
                best_supports = np.array(supports, copy=True)
                best_beta = np.array(beta, copy=True)
                best_kuu = np.array(kuu, copy=True)
                best_kuz = np.array(kuz, copy=True)

    supports, beta, kuu, kuz = best_supports, best_beta, best_kuu, best_kuz

    residual_squared, rkme_norm2, reduction_error, clamped = _objective_terms(
        empirical, supports, beta, kuu, kuz
    )
    residual_scale = max(
        1.0, abs(empirical.norm2), abs(rkme_norm2), abs(empirical.norm2 - residual_squared)
    )
    if residual_squared < -resolved.negative_tolerance * residual_scale:
        raise ArithmeticError(
            f"RKME reduction residual is materially negative ({residual_squared})"
        )
    system = kuu + resolved.ridge * np.eye(kuu.shape[0], dtype=np.float64)
    condition = float(np.linalg.cond(system))
    return ReducedRKME(
        supports=supports,
        beta=beta,
        bandwidth=kernel.bandwidth,
        rkme_norm2=rkme_norm2,
        empirical_norm2=empirical.norm2,
        reduction_error=reduction_error,
        protocol_id=empirical.protocol_id,
        source_dataset_digest=empirical.dataset_digest,
        ridge=resolved.ridge,
        condition_number=condition,
        objective_trace=tuple(trace),
        negative_residual_clamped=clamped,
        source_task=empirical.source_task,
        source_dataset_manifest_digest=empirical.source_dataset_manifest_digest,
        raw_reduction_residual_squared=residual_squared,
    )
