from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

# The server packages intentionally live outside the installable library tree.
sys.path.insert(0, str(Path(__file__).parents[2]))

from policy_learnware_v0.hashing import canonical_json_bytes, sha256_file, sha256_json
from policy_learnware_v0.io import atomic_write_bytes, atomic_write_json, atomic_write_npz
from policy_learnware_v0.probe.dataset import EpisodeDataset, save_dataset_artifact
from policy_learnware_v0.schemas import EnvSchema, FrozenProtocol
from policy_learnware_v0.v02.schemas import ExecutionABIRecord
from policy_learnware_v0.v03.fpo_source_backend import FpoJaxSourceEvaluatorBackend
from policy_learnware_v0.v03.pool_intake import _intake_v02_policy_pool
from policy_learnware_v0.v03.source_evaluator import (
    BackendEpisodeResult,
    CanonicalSourceAnchor,
    FrozenPlanJobBinding,
    FrozenServerPlanBinding,
)
import server.repro_fpo_ppo_v03.asset_binding as asset_binding_module
from server.repro_fpo_ppo_v03.asset_binding import (
    ASSET_BINDINGS_READY,
    AssetBindingError,
    PRODUCTION_ATTESTATION_RESET_SEEDS,
    PRODUCTION_SEED_DERIVATION_PROTOCOL_DIGEST,
    PRODUCTION_SELECTION_LEDGER_CONFIG_DIGEST,
    PRODUCTION_SELECTION_LEDGER_DIGEST,
    PRODUCTION_SELECTION_LEDGER_EXPERIMENT_ID,
    PRODUCTION_SELECTION_LEDGER_FILE_SHA256,
    PRODUCTION_SELECTION_RESET_SEEDS,
    ProductionAssetBindingConfig,
    _prepare_documents,
    _publish_documents,
    build_legacy_asset_inventory,
)

sys.path.insert(0, str(Path(__file__).parent))
from p5_asset_fixtures import digest, exact90_handoff  # noqa: E402


def _manifest_file(root: Path, relative: str) -> dict[str, str]:
    return {"path": relative, "sha256": sha256_file(root / relative)}


def _dataset(seed: int) -> EpisodeDataset:
    return EpisodeDataset(
        observation=np.asarray([[seed, 0.0]], dtype=np.float32),
        action=np.asarray([[0.25]], dtype=np.float32),
        reward=np.asarray([0.5], dtype=np.float32),
        next_observation=np.asarray([[seed + 1.0, 0.0]], dtype=np.float32),
        terminated=np.asarray([False]),
        truncated=np.asarray([True]),
        episode_offsets=np.asarray([0, 1], dtype=np.int64),
        reset_seeds=np.asarray([seed], dtype=np.int64),
        probe_seeds=np.asarray([seed + 100], dtype=np.int64),
    )


def _legacy_root(tmp_path: Path) -> Path:
    root = tmp_path / "legacy-v0"
    protocol_dir = root / "protocol"
    protocol_dir.mkdir(parents=True)
    task = "TaskA"
    draft = digest("legacy-draft")
    datasets: dict[str, str] = {}
    for index, split in enumerate(
        (
            "encoder_train",
            "encoder_validation",
            "kernel_calibration",
            "separability_calibration",
            "source_taskspec",
            "target_query",
        )
    ):
        directory = root / "datasets" / split
        if split == "target_query":
            directory = directory / "bank_000"
        directory.mkdir(parents=True, exist_ok=True)
        manifest = save_dataset_artifact(
            _dataset(index + 1),
            npz_path=directory / f"{task}.npz",
            manifest_path=directory / f"{task}.json",
            split=split,
            task=task,
            protocol_draft_hash=draft,
        )
        datasets[split] = manifest.dataset_sha256

    atomic_write_bytes(protocol_dir / "encoder.msgpack", b"fixture-encoder")
    atomic_write_json(protocol_dir / "encoder_config.json", {"fixture": True})
    atomic_write_npz(protocol_dir / "normalization.npz", {"mean": np.zeros(2)})
    atomic_write_json(protocol_dir / "kernel.json", {"bandwidth": 1.0})
    atomic_write_json(protocol_dir / "env_schemas.json", {"TaskA": {"fixture": True}})

    encoder_manifest = {
        "schema": "policy-learnware.encoder-artifact.v0",
        "complete": True,
        "protocol_draft_hash": draft,
        "source_dataset_digests": {
            "encoder_train": {task: datasets["encoder_train"]},
            "encoder_validation": {task: datasets["encoder_validation"]},
        },
        "normalization_sha256": sha256_file(protocol_dir / "normalization.npz"),
        "files": {
            "checkpoint": _manifest_file(root, "protocol/encoder.msgpack"),
            "config": _manifest_file(root, "protocol/encoder_config.json"),
        },
    }
    atomic_write_json(protocol_dir / "encoder_manifest.json", encoder_manifest)
    normalization_manifest = {
        "schema": "policy-learnware.normalization-artifact.v0",
        "complete": True,
        "protocol_draft_hash": draft,
        "source_dataset_digests": {task: datasets["encoder_train"]},
        "files": {
            "normalization": _manifest_file(root, "protocol/normalization.npz")
        },
    }
    atomic_write_json(protocol_dir / "normalization_manifest.json", normalization_manifest)
    kernel_manifest = {
        "schema": "policy-learnware.kernel-artifact.v0",
        "complete": True,
        "protocol_draft_hash": draft,
        "source_dataset_digests": {task: datasets["kernel_calibration"]},
        "encoder_sha256": sha256_file(protocol_dir / "encoder.msgpack"),
        "files": {"kernel": _manifest_file(root, "protocol/kernel.json")},
    }
    atomic_write_json(protocol_dir / "kernel_manifest.json", kernel_manifest)
    environment_manifest = {
        "schema": "policy-learnware.environment-artifacts.v0",
        "complete": True,
        "protocol_draft_hash": draft,
        "tasks": [task],
        "files": {"env_schemas": _manifest_file(root, "protocol/env_schemas.json")},
    }
    atomic_write_json(protocol_dir / "environment_manifest.json", environment_manifest)

    source_manifest_shas = {
        task: sha256_file(root / "datasets" / "source_taskspec" / f"{task}.json")
    }
    env = EnvSchema(
        backend="fixture.backend",
        task=task,
        observation_dim=2,
        action_dim=1,
        action_low=np.asarray([-1.0], dtype=np.float32),
        action_high=np.asarray([1.0], dtype=np.float32),
        horizon=1,
        action_repeat=1,
        control_dt=0.02,
        flatten_fingerprint=digest("flatten"),
        implementation_digest=digest("environment"),
    )
    frozen = FrozenProtocol.create(
        config={
            "environment": {"tasks": [task]},
            "episodes": {"target_query_banks": 1},
        },
        env_schemas={task: env},
        packed_layout={
            "width": 109,
            "max_observation_dim": 24,
            "max_action_dim": 6,
            "latent_dim": 32,
            "support_budget": 100,
            "kernel_bandwidth": 1.0,
            "layout_version": "pack109-padding-mask-v0",
        },
        component_digests={
            "environment_manifest": sha256_file(protocol_dir / "environment_manifest.json"),
            "probe_implementation": digest("probe-implementation"),
            "normalization": sha256_file(protocol_dir / "normalization_manifest.json"),
            "encoder": sha256_file(protocol_dir / "encoder_manifest.json"),
            "kernel": sha256_file(protocol_dir / "kernel_manifest.json"),
            "source_dataset_manifests": sha256_json(source_manifest_shas),
        },
        runtime_versions={"python": "fixture"},
    )
    frozen.save(protocol_dir / "protocol.json")
    protocol_manifest = {
        "schema": "policy-learnware.protocol-artifacts.v0",
        "complete": True,
        "protocol_draft_hash": draft,
        "protocol_id": frozen.protocol_id,
        "source_dataset_manifest_digests": source_manifest_shas,
        "files": {
            "protocol": _manifest_file(root, "protocol/protocol.json"),
            "environment_manifest": _manifest_file(
                root, "protocol/environment_manifest.json"
            ),
            "normalization_manifest": _manifest_file(
                root, "protocol/normalization_manifest.json"
            ),
            "encoder_manifest": _manifest_file(root, "protocol/encoder_manifest.json"),
            "kernel_manifest": _manifest_file(root, "protocol/kernel_manifest.json"),
        },
    }
    atomic_write_json(protocol_dir / "manifest.json", protocol_manifest)
    return root.resolve()


def _abi() -> ExecutionABIRecord:
    return ExecutionABIRecord(
        protocol_family_id="continuous-vector-mdp-v02",
        observation_tensor_abi_digest=digest("observation-abi"),
        action_tensor_abi_digest=digest("action-abi"),
        action_transform_id="tanh",
        policy_runtime_id="legacy-ppo-fpo-v0",
        state_abi_id="stateless-v0",
    )


class ValidateOnlyDriver:
    def __init__(self) -> None:
        self.runtime_driver_digest = digest("fixture-runtime-driver")
        self.validate_calls: list[str] = []
        self.rollout_calls: list[tuple[int, ...]] = []

    def validate_candidate(self, request) -> ExecutionABIRecord:
        self.validate_calls.append(request.candidate_id)
        return _abi()

    def evaluate_seed_block(
        self, request, *, reset_seeds: tuple[int, ...]
    ) -> tuple[BackendEpisodeResult, ...]:
        self.rollout_calls.append(reset_seeds)
        raise AssertionError("asset binding must never execute a rollout")


def _exact90_case(tmp_path: Path):
    fixture_root = tmp_path / PRODUCTION_SELECTION_LEDGER_EXPERIMENT_ID
    root, handoff, trust = exact90_handoff(fixture_root)
    intake = _intake_v02_policy_pool(
        handoff,
        trusted_experiment_root=root,
        trust_anchor=trust,
        _acceptance_replayer=lambda _root, handoff_path, _promotions: json.loads(
            (handoff_path / "policy_pool_acceptance.json").read_text(encoding="utf-8")
        ),
    )
    anchors = {
        anchor_id: CanonicalSourceAnchor.from_path(
            root / "source_anchor_manifests" / f"{anchor_id}.json"
        )
        for anchor_id in intake.candidates_by_anchor
    }
    jobs = {
        candidate_id: FrozenPlanJobBinding(
            candidate_id=candidate_id,
            job_digest=cell.job_digest,
            seed=cell.seed,
            training_protocol_digest=digest("fixture-training-protocol"),
            anchor=anchors[cell.source_anchor_id],
        )
        for candidate_id, cell in intake.cells.items()
    }
    config_digest = PRODUCTION_SELECTION_LEDGER_CONFIG_DIGEST
    plan_payload = {
        "schema": "policy-learnware.fixture-server-plan.v0",
        "config_digest": config_digest,
        "jobs": [
            {
                "job_id": job.candidate_id,
                "job_digest": job.job_digest,
                "seed": job.seed,
                "training_protocol_digest": job.training_protocol_digest,
                "anchor_manifest_path": job.anchor.manifest_path,
                "anchor_manifest_digest": job.anchor.manifest_digest,
            }
            for job in jobs.values()
        ],
    }
    plan_digest = sha256_json(plan_payload)
    plan_path = root / "training_private" / "plans" / "fixture_binding_plan.json"
    atomic_write_json(plan_path, {**plan_payload, "plan_digest": plan_digest})
    intake = replace(intake, server_plan_digest=plan_digest, intake_record_digest=None)
    intake_path = tmp_path / "intake_record.json"
    atomic_write_json(intake_path, intake.to_dict())
    plan = FrozenServerPlanBinding(
        plan_path=str(plan_path.resolve()), plan_digest=plan_digest, jobs=jobs
    )
    return root.resolve(), intake, intake_path.resolve(), plan


def _selection_ledger(root: Path) -> tuple[Path, dict[str, object]]:
    source = Path(__file__).parents[2] / "configs" / "v02_selection_ledger.json"
    path = root.parent / "configs" / "v02_selection_ledger.json"
    atomic_write_bytes(path, source.read_bytes())
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert sha256_file(path) == PRODUCTION_SELECTION_LEDGER_FILE_SHA256
    assert payload["ledger_digest"] == PRODUCTION_SELECTION_LEDGER_DIGEST
    return path.resolve(), payload


def _config(tmp_path: Path, root: Path, intake_path: Path, plan: FrozenServerPlanBinding):
    fpo_root = tmp_path / "fpo-root"
    vendor = tmp_path / "vendor"
    fpo_root.mkdir()
    vendor.mkdir()
    alias_nonce = tmp_path / "alias.nonce"
    tie_nonce = tmp_path / "tie.nonce"
    alias_nonce.write_text(digest("private-alias-nonce") + "\n", encoding="ascii")
    tie_nonce.write_text(digest("private-tie-nonce") + "\n", encoding="ascii")
    alias_nonce.chmod(0o600)
    tie_nonce.chmod(0o600)
    ledger_path, _ledger = _selection_ledger(root)
    return ProductionAssetBindingConfig(
        intake_record_path=intake_path,
        intake_record_sha256=sha256_file(intake_path),
        trusted_experiment_root=root,
        server_plan_path=plan.plan_path,
        server_plan_sha256=sha256_file(plan.plan_path),
        selection_ledger_path=ledger_path,
        fpo_root=fpo_root.resolve(),
        vendor_dir=vendor.resolve(),
        legacy_v0_root=_legacy_root(tmp_path),
        output_dir=(tmp_path / "asset-bindings").resolve(),
        market_alias_private_nonce_file=alias_nonce.resolve(),
        tie_break_private_nonce_file=tie_nonce.resolve(),
    )


def test_validate_only_binding_publishes_exact90_without_rollout_or_authority(
    tmp_path: Path,
) -> None:
    root, intake, intake_path, plan = _exact90_case(tmp_path)
    config = _config(tmp_path, root, intake_path, plan)
    driver = ValidateOnlyDriver()
    backend = FpoJaxSourceEvaluatorBackend(
        runtime_driver=driver,
        selection_reset_seeds=config.selection_reset_seeds,
        attestation_reset_seeds=config.attestation_reset_seeds,
    )

    documents = _prepare_documents(
        config, intake=intake, plan=plan, runtime_driver=driver, backend=backend
    )
    assert len(driver.validate_calls) == 90
    assert len(set(driver.validate_calls)) == 90
    assert driver.rollout_calls == []
    assert documents["source_selection_work_units"]["work_unit_count"] == 90
    assert len(documents["formal_market_plan"]["deployment_abi_digests_by_candidate"]) == 90
    assert len(set(documents["formal_market_plan"]["source_anchor_id_by_candidate"].values())) == 30

    receipt = _publish_documents(
        config, documents, runtime_driver=driver, backend=backend
    )
    assert receipt["status"] == ASSET_BINDINGS_READY
    assert receipt["formal_run_authorized"] is False
    assert receipt["rollout_executed"] is False
    assert receipt["training_executed"] is False
    assert receipt["formal_authority_granted"] is False
    assert receipt["candidate_validation_count"] == 90
    assert receipt["private_nonces_persisted"] is False
    assert documents["source_evaluation_protocol"]["selection_reset_seeds"] == list(
        PRODUCTION_SELECTION_RESET_SEEDS
    )
    assert documents["source_evaluation_protocol"]["attestation_reset_seeds"] == list(
        PRODUCTION_ATTESTATION_RESET_SEEDS
    )
    assert receipt["selection_ledger"] == {
        "path": str(Path(config.selection_ledger_path)),
        "file_sha256": PRODUCTION_SELECTION_LEDGER_FILE_SHA256,
        "semantic_digest": PRODUCTION_SELECTION_LEDGER_DIGEST,
        "experiment_id": PRODUCTION_SELECTION_LEDGER_EXPERIMENT_ID,
        "config_digest": PRODUCTION_SELECTION_LEDGER_CONFIG_DIGEST,
        "admission_decision": {
            "selection_episodes": 25,
            "attestation_episodes": 50,
            "competence_lcb_floor": 0.5,
            "champion_mean_tolerance": 0.01,
            "lcb_z": 1.645,
            "competence_mode": "OBSERVE",
        },
    }
    assert (
        receipt["seed_derivation_protocol"]["protocol_digest"]
        == PRODUCTION_SEED_DERIVATION_PROTOCOL_DIGEST
    )
    for path in Path(config.output_dir).glob("*.json"):
        persisted = path.read_bytes()
        assert digest("private-alias-nonce").encode() not in persisted
        assert digest("private-tie-nonce").encode() not in persisted
    for binding in receipt["artifacts"].values():
        path = Path(binding["path"])
        value = json.loads(path.read_text(encoding="utf-8"))
        assert path.read_bytes() == canonical_json_bytes(value) + b"\n"
        assert sha256_file(path) == binding["file_sha256"]

    with pytest.raises(AssetBindingError, match="already exists"):
        _publish_documents(
            config,
            {"_receipt_material": {}},
            runtime_driver=driver,
            backend=backend,
        )


def test_publication_failure_never_exposes_a_partial_final_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, intake, intake_path, plan = _exact90_case(tmp_path)
    config = _config(tmp_path, root, intake_path, plan)
    driver = ValidateOnlyDriver()
    backend = FpoJaxSourceEvaluatorBackend(
        runtime_driver=driver,
        selection_reset_seeds=config.selection_reset_seeds,
        attestation_reset_seeds=config.attestation_reset_seeds,
    )
    documents = _prepare_documents(
        config, intake=intake, plan=plan, runtime_driver=driver, backend=backend
    )
    original = asset_binding_module.atomic_write_json
    calls = 0

    def fail_during_staging(path: Path, value: object, *, overwrite: bool = False) -> str:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected staging failure")
        return original(path, value, overwrite=overwrite)

    monkeypatch.setattr(
        asset_binding_module, "atomic_write_json", fail_during_staging
    )
    with pytest.raises(OSError, match="injected staging failure"):
        _publish_documents(
            config, documents, runtime_driver=driver, backend=backend
        )
    assert not Path(config.output_dir).exists()
    assert not tuple(
        Path(config.output_dir).parent.glob(
            f".{Path(config.output_dir).name}.staging-*"
        )
    )


def test_legacy_inventory_revalidates_dataset_and_required_assets(tmp_path: Path) -> None:
    root = _legacy_root(tmp_path)
    inventory = build_legacy_asset_inventory(root)
    assert inventory["totals"] == {
        "manifest_count": 6,
        "episode_count": 6,
        "transition_count": 6,
    }
    assert inventory["split_counts"]["target_query"]["manifests"] == 1
    assert len(inventory["inventory_digest"]) == 64

    dataset_path = root / "datasets" / "encoder_train" / "TaskA.npz"
    dataset_path.write_bytes(dataset_path.read_bytes() + b"drift")
    with pytest.raises(AssetBindingError, match="legacy dataset"):
        build_legacy_asset_inventory(root)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "selection_reset_seeds",
            PRODUCTION_SELECTION_RESET_SEEDS[:-1],
            "reviewed 25-seed",
        ),
        (
            "attestation_reset_seeds",
            PRODUCTION_ATTESTATION_RESET_SEEDS[:-1],
            "reviewed 50-seed",
        ),
        ("competence_floor", 0.4, "literal 0.5"),
        ("mean_tolerance", 0.02, "literal 0.01"),
        ("lcb_z", 1.96, "literal 1.645"),
        ("return_horizon", 999, "literal 1000"),
        ("per_step_lower", -1.0, "reviewed literals"),
    ),
)
def test_binding_config_rejects_any_production_literal_drift(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    root, _intake, intake_path, plan = _exact90_case(tmp_path)
    config = _config(tmp_path, root, intake_path, plan)
    with pytest.raises(AssetBindingError, match=message):
        replace(config, **{field: value})


def test_binding_config_rejects_nonce_reuse(tmp_path: Path) -> None:
    root, _intake, intake_path, plan = _exact90_case(tmp_path)
    config = _config(tmp_path, root, intake_path, plan)
    with pytest.raises(AssetBindingError, match="files must differ"):
        replace(
            config,
            tie_break_private_nonce_file=config.market_alias_private_nonce_file,
        )


def test_prepare_fails_closed_on_explicit_intake_digest(tmp_path: Path) -> None:
    root, intake, intake_path, plan = _exact90_case(tmp_path)
    config = replace(
        _config(tmp_path, root, intake_path, plan),
        intake_record_sha256=digest("wrong-intake-file"),
    )
    driver = ValidateOnlyDriver()
    backend = FpoJaxSourceEvaluatorBackend(
        runtime_driver=driver,
        selection_reset_seeds=config.selection_reset_seeds,
        attestation_reset_seeds=config.attestation_reset_seeds,
    )
    with pytest.raises(AssetBindingError, match="explicit authority"):
        _prepare_documents(
            config, intake=intake, plan=plan, runtime_driver=driver, backend=backend
        )
    assert driver.validate_calls == []
    assert driver.rollout_calls == []


def test_prepare_rejects_selection_ledger_drift_before_candidate_validation(
    tmp_path: Path,
) -> None:
    root, intake, intake_path, plan = _exact90_case(tmp_path)
    config = _config(tmp_path, root, intake_path, plan)
    ledger_path = Path(config.selection_ledger_path)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["admission_decision"]["selection_episodes"] = 24
    ledger["ledger_digest"] = sha256_json(
        {key: value for key, value in ledger.items() if key != "ledger_digest"}
    )
    atomic_write_json(ledger_path, ledger, overwrite=True)
    driver = ValidateOnlyDriver()
    backend = FpoJaxSourceEvaluatorBackend(
        runtime_driver=driver,
        selection_reset_seeds=config.selection_reset_seeds,
        attestation_reset_seeds=config.attestation_reset_seeds,
    )
    with pytest.raises(AssetBindingError, match="reviewed production authority"):
        _prepare_documents(
            config, intake=intake, plan=plan, runtime_driver=driver, backend=backend
        )
    assert driver.validate_calls == []
    assert driver.rollout_calls == []


def test_prepare_rejects_noncanonical_and_semantically_drifted_plan(tmp_path: Path) -> None:
    root, intake, intake_path, plan = _exact90_case(tmp_path)
    base = _config(tmp_path, root, intake_path, plan)
    plan_path = Path(plan.plan_path)
    value = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    config = replace(base, server_plan_sha256=sha256_file(plan_path))
    driver = ValidateOnlyDriver()
    backend = FpoJaxSourceEvaluatorBackend(
        runtime_driver=driver,
        selection_reset_seeds=config.selection_reset_seeds,
        attestation_reset_seeds=config.attestation_reset_seeds,
    )
    with pytest.raises(AssetBindingError, match="not canonical"):
        _prepare_documents(
            config, intake=intake, plan=plan, runtime_driver=driver, backend=backend
        )
    assert driver.validate_calls == []

    value["plan_digest"] = digest("semantic-plan-drift")
    atomic_write_json(plan_path, value, overwrite=True)
    config = replace(config, server_plan_sha256=sha256_file(plan_path))
    with pytest.raises(AssetBindingError, match="semantic digest"):
        _prepare_documents(
            config, intake=intake, plan=plan, runtime_driver=driver, backend=backend
        )
    assert driver.validate_calls == []


def test_private_nonce_files_must_be_distinct_regular_0600_files(tmp_path: Path) -> None:
    root, intake, intake_path, plan = _exact90_case(tmp_path)
    config = _config(tmp_path, root, intake_path, plan)
    Path(config.market_alias_private_nonce_file).chmod(0o644)
    with pytest.raises(AssetBindingError, match="exact mode 0600"):
        replace(config)

    Path(config.market_alias_private_nonce_file).chmod(0o600)
    symlink = tmp_path / "alias-link.nonce"
    symlink.symlink_to(config.market_alias_private_nonce_file)
    with pytest.raises(AssetBindingError, match="may not be a symlink"):
        replace(config, market_alias_private_nonce_file=symlink)
