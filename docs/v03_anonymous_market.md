# v0.3 anonymous-market runtime note

`policy_learnware_v0.v03.anonymous_market` implements the P0 engineering
join between the source policy market, source RKME index, public distance run,
and the target execution ABI. It is runtime plumbing, not an admission system.

The reusable base `SourceRepresentationIndex` carries no policy-market field.
Only P5M may explicitly wrap it as a `MarketBoundSourceRepresentationIndex`;
Raw KME and encoder-only paths therefore never invent a fake market identity.
The public selector view accepts only that wrapper, requires its
`policy_market_id` to match, and requires the market and representation index
to contain the same exact 30 canonical
`lw-*` identities. Each public identity is digest-bound to its source spec,
normalized competence, and tie-break token. A distance request ranks the full
pool before deployment metadata is consulted.

After the `JointDistanceRun` is published, the deployment registry is scanned
in rank order and the first ABI-compatible policy is used. Selecting rank two
or later is recorded as fallback; selection fails only when the complete pool
contains no compatible policy.

This is an engineering record, not a formal evidence grant. In particular,
`V03SourcePolicyMarket.asset_state` is fixed to
`ENGINEERING_CONTRACT_ONLY`, never `ASSET_READY`.
`AnonymousSelectorViewManifest` therefore fixes
`evidence_scope` to `ENGINEERING_CONTRACT_ONLY` and
`formal_authority_available` to `false`. These fields remain for backward
artifact compatibility and do not gate the v0.3 executable path.
