"""Frozen v0.2 external-asset and runtime-attestation interfaces."""

from .artifacts import (
    ARTIFACTS_ROOT_ENV,
    RelocationResolver,
    V02AssetError,
    V02AssetLayout,
    capability_status,
    resolve_artifacts_root,
    validate_relocation_manifest,
    verify_handoff_trust_anchors,
)
from .runtime import (
    inspect_fpo_checkout,
    load_verified_fpo_upstream,
    original_vendor_status,
    require_original_vendor_runtime,
    verify_fpo_checkout,
)

__version__ = "0.2.0"

__all__ = [
    "ARTIFACTS_ROOT_ENV",
    "RelocationResolver",
    "V02AssetError",
    "V02AssetLayout",
    "capability_status",
    "inspect_fpo_checkout",
    "load_verified_fpo_upstream",
    "original_vendor_status",
    "resolve_artifacts_root",
    "require_original_vendor_runtime",
    "validate_relocation_manifest",
    "verify_fpo_checkout",
    "verify_handoff_trust_anchors",
    "__version__",
]
