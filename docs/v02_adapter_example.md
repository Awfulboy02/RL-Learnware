# v0.2 extension adapter example

The v0.2 extension layer keeps environment backends, executable policy
runtimes, and semantic encoders in separate registries.  Registration is
fail-closed: an identifier may be registered once, bundle-schema routing must
be unambiguous, and all components in one evaluation partition must declare
the same protocol family.

```python
from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.v02.extensions import (
    EnvironmentBackendRegistry,
    LegacyPpoFpoRuntimePlugin,
    MujocoPlaygroundBackendPlugin,
    PolicyRuntimeRegistry,
    RawTransitionEncoder,
    SemanticEncoderRegistry,
    V02ExtensionRegistries,
    default_metadata,
)

environment_backends = EnvironmentBackendRegistry()
environment_backends.register(MujocoPlaygroundBackendPlugin())

policy_runtimes = PolicyRuntimeRegistry()
policy_runtimes.register(
    LegacyPpoFpoRuntimePlugin(fpo_root="/absolute/path/to/FPO")
)

semantic_encoders = SemanticEncoderRegistry()
semantic_encoders.register(
    RawTransitionEncoder(
        default_metadata(
            representation_id="raw-transition-v1",
            family="raw",
            input_dim=24,
            output_dim=24,
            canonical_event_view_digest=sha256_json(
                {"canonical_event_view": "packed-transition-v1"}
            ),
        )
    )
)

extensions = V02ExtensionRegistries(
    environments=environment_backends,
    policies=policy_runtimes,
    representations=semantic_encoders,
)
```

Before a formal run, use `extensions.resolve_partition(...)` to reject family
mismatches and missing capabilities.  Then inspect/open the environment and
validate/load the policy bundle through their plugins; finally call
`compatibility_partition(...)` with the opened environment handle, the loaded
policy runtime contract, and encoder metadata.  This private partition uses
only the minimum task-anonymous Execution ABI.  Source task/reward identity is
retained in provenance but is never a public selector input or a full-pool
oracle filter.

`default_metadata` is only a convenience for tests and examples.  Formal
experiments must construct `SemanticEncoderMetadata` with audited code,
runtime, dependency, checkpoint, normalizer, source-permission, event-view,
and training-split digests.
