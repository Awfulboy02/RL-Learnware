"""Small deterministic Markdown summary renderer."""

from __future__ import annotations

from .metrics import DeploymentMetrics
from .retrieval import RetrievalMetrics


def render_summary(
    retrieval: RetrievalMetrics,
    deployment: DeploymentMetrics,
    *,
    pool_id: str,
    protocol_id: str,
) -> str:
    conditional = (
        "n/a"
        if deployment.conditional_mean_return is None
        else f"{deployment.conditional_mean_return:.6g}"
    )
    lines = [
        "# Policy Learnware v0 smoke summary",
        "",
        f"- Pool: `{pool_id}`",
        f"- Protocol: `{protocol_id}`",
        f"- Retrieval accuracy: {retrieval.correct_count}/{retrieval.trial_count} "
        f"({retrieval.accuracy:.3%})",
        f"- Deployability: {deployment.deployable_count}/{deployment.query_count} "
        f"({deployment.deployability_rate:.3%})",
        f"- Compatible-selection conditional mean return: {conditional}",
    ]
    if deployment.failure_counts:
        lines.extend(("", "Deployment failures:"))
        lines.extend(f"- `{reason}`: {count}" for reason, count in deployment.failure_counts)
    return "\n".join(lines) + "\n"
