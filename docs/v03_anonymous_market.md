# v0.3 anonymous-market engineering contract

`policy_learnware_v0.v03.anonymous_market` implements the P0 engineering
closure between the 30-entry source policy market, the 30-entry source
representation index, the public joint-distance run, and the private
post-selection execution-ABI audit.

The reusable base `SourceRepresentationIndex` carries no policy-market field.
Only P5M may explicitly wrap it as a `MarketBoundSourceRepresentationIndex`;
Raw KME and encoder-only paths therefore never invent a fake market identity.
The public selector view accepts only that wrapper, requires its
`policy_market_id` to match, and requires the market and representation index
to contain the same exact 30 canonical
`lw-*` identities. Each public identity is digest-bound to its source spec,
normalized competence, and tie-break token. Subsets, extra identities, and
non-canonical identities fail closed. A distance request is then bound to this
selector-view digest and ranks the full pool.

After the full `JointDistanceRun` is published, the deployment-private registry
is consulted for the rank-one `ExecutionABIRecord` only. If it is incompatible
with the target ABI, the outcome is `SELECTED_INCOMPATIBLE_ABI` with a typed
`SelectionFailureRecord`. Rank two is never considered as a fallback.

This is an engineering contract, not a formal evidence grant. In particular,
`V03SourcePolicyMarket.asset_state` is fixed to
`ENGINEERING_CONTRACT_ONLY`, never `ASSET_READY`.
`AnonymousSelectorViewManifest` therefore fixes
`evidence_scope` to `ENGINEERING_CONTRACT_ONLY` and
`formal_authority_available` to `false`. Formal experiment authority must come
from the separately governed P5M evidence and acceptance path.
