"""External-asset resolution and relocation validation for frozen v0.2.

Release receipts contain absolute paths from the original training host.  The
receipts stay byte-identical: this module rebases an allowlisted path only
while reading it and never rewrites a receipt, seal, manifest, or bundle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Mapping, Sequence

from ..hashing import sha256_file, sha256_json
from .legacy_artifacts import (
    ArtifactDomain,
    V02ArtifactLayout,
    V02ArtifactLayoutError,
    V02ArtifactWriter,
)


ARTIFACTS_ROOT_ENV = "RL_LEARNWARE_ARTIFACTS_ROOT"
RELOCATION_SCHEMA = "rl-learnware-relocation/v1"
TREE_DIGEST_ALGORITHM = "sha256sum-relative-v1"

V02_RUN_ID = "v02-reacher-formal-2r-20260825-r2"
V02_RELEASE_TAG = "v0.2.0"
V02_RELEASE_COMMIT = "a7d10c05df069407d1054bf25baa21ac5fa8f961"
V02_FREEZE_SOFTWARE_COMMIT = "5e14614cc513aae4bf610d90aa7165818bd40472"
FPO_COMMIT = "418c2554f7cd22d52e14c07d951280929d73bf2f"

EXPECTED_POOL_DIGEST = (
    "e478ef1d38b7eea1a38691d4ea2bd25dc0356cd7264f5a5bd6df5e6de5e0d15f"
)
EXPECTED_FPO_HEAD_TREE_DIGEST = (
    "7bb5d663d19b1e5099037e56990211c21e43fb5948be55fcdd4f5d983b135783"
)
EXPECTED_FPO_EXECUTION_TREE_DIGEST = (
    "396f2b4633d1fd0cf1cc753fbe16a458f4e62afabb385cbf2fd3dfb872626083"
)
EXPECTED_VENDOR_TREE_DIGEST = (
    "11ea54a9390010e1de5d7bdadf75334614f3adaba21940ca898e01f347f786d4"
)
EXPECTED_POLICY_IO_SHA256 = (
    "19b3da5fd74c3ed0098a49381e9f1bbcdd7ba983f941e18abed9ae2d72384ff3"
)


class V02AssetError(ValueError):
    """The external layout or relocation manifest is unsafe or invalid."""


@dataclass(frozen=True)
class DirectoryAttestation:
    """Deterministic inventory of one symlink-free regular-file tree."""

    file_count: int
    total_bytes: int
    tree_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "tree_digest_algorithm": TREE_DIGEST_ALGORITHM,
            "tree_digest": self.tree_digest,
        }


@dataclass(frozen=True)
class AssetExpectation:
    target_relpath: str | None
    required_for: tuple[str, ...]
    access_class: str
    tree_digest: str | None = None
    file_count: int | None = None
    total_bytes: int | None = None


ASSET_EXPECTATIONS: Mapping[str, AssetExpectation] = {
    "exact90": AssetExpectation(
        target_relpath=f"v02/exact90/{V02_RUN_ID}",
        required_for=("handoff_verification", "policy_inference", "training_replay"),
        access_class="immutable_read_only",
        # The operational coordination subtree is intentionally external to
        # this immutable payload; it was still being rewritten after freeze.
        tree_digest="fbe5b0aa14f49083f4e318048614c5d44c03bcd0d9a358fab0dfa33cf12b23a2",
        file_count=13_888,
        total_bytes=101_180_750,
    ),
    "formal_inputs": AssetExpectation(
        target_relpath=f"v02/formal_inputs/{V02_RUN_ID}",
        required_for=("handoff_verification", "training_replay"),
        access_class="immutable_read_only",
        tree_digest="3bd5fceb86e868da07958c35e5d31fc7af09acc00c8a35f315cc0d95a3c6dc64",
        file_count=69,
        total_bytes=246_292,
    ),
    "fpo": AssetExpectation(
        target_relpath="shared/runtime/fpo-418c2554",
        required_for=("policy_inference", "training_replay"),
        access_class="reviewed_runtime_read_only",
        # The root relocation manifest inventories all regular files, including
        # .git.  The independent frozen 72-file Git proof lives in runtime.py.
    ),
    "legacy_v02": AssetExpectation(
        target_relpath="shared/repro_fpo_ppo/legacy-v02",
        required_for=("policy_inference", "training_replay"),
        access_class="incomplete_recovery_read_only",
        tree_digest="d153f8f56ee14f0d227f44e898673be153f7b844d7792d78d2e9821a8e2a98b7",
        file_count=3_131,
        total_bytes=26_426_389,
    ),
    "vendor_original": AssetExpectation(
        target_relpath=None,
        required_for=("training_replay",),
        access_class="missing_original",
        tree_digest=EXPECTED_VENDOR_TREE_DIGEST,
        file_count=1_612,
        total_bytes=90_428_849,
    ),
    "reconstructed_runtime": AssetExpectation(
        target_relpath="shared/runtime/reconstructed-v02",
        required_for=(),
        access_class="reconstructed_runtime",
    ),
    "runtime_state": AssetExpectation(
        target_relpath=f"v02/runtime_state/{V02_RUN_ID}/training_private/coordination",
        required_for=(),
        access_class="operational_history_read_only",
        tree_digest="9f84d32dd2ec7f51013f1934403ddc440fefc7f50217079a998c78e83c4594b3",
        file_count=3,
        total_bytes=7_247,
    ),
}

_RESOLVABLE_V02_ASSETS = (
    "exact90",
    "formal_inputs",
    "legacy_v02",
    "runtime_state",
)

_POLICY_IO_TARGET = "shared/repro_fpo_ppo/legacy-v02/policy_io.py"
_POLICY_IO_MANIFEST_SHA256 = (
    "bca758b02693d376352607222b535775b3d97db873f002645fa15b38700e0472"
)
_V02_SOURCE_SHA256 = {
    "exact90": "cb778b44c35c864e4c323fd9f8622124b0535a4ff59c1416cf697dae24735629",
    "formal_inputs": "14fd2668a7601bf8f5c9027ad2366619a5344f33f529dccc1755f2d9995fea1f",
    "runtime_state": "3c2d1606bf0208a3bf217b1b3dbdf1517ae0ba0c81ffe23652a631a50200e163",
    "legacy_v02": "85293ab280a13982c9497121317a5a841cd5bdcc93a96b75aabea7347b3865c5",
    "policy_io": "116f5624a39b1c19cccea5b77ad992e7e32f31c84ffa58029411209a0480dd19",
}

CRITICAL_FILE_SHA256: Mapping[str, Mapping[str, str]] = {
    "exact90": {
        "frozen/v02_freeze_manifest.json": "42746ca0cc595b5f473e7bd3829099898e071dbb1b13a93e4c7f92d52662aadf",
        "policy_pool_handoff_a7d10c0/compiled_parity_promotions.json": "e544615f614012afa6ff45020a5bc922c05aab2ebd4dfd47bf36ad88b6aa8679",
        "policy_pool_handoff_a7d10c0/policy_pool_acceptance.json": "cb133b4a3a15e739a111fb7245b04f20edf4d064a3c4c3850ba34a1f67f6a32d",
        "training_private/plans/server_training_plan.json": "248dce23b9e28114dc17dc0e248814dc7e34a3419a93a78ed576c9ffd82cf30a",
        "training_private/server_runs/queue_status.json": "ff44038bb69bf3dea710acd3c20757ddcaefb1a33eb852365f5158a10d42f6f8",
    },
    "formal_inputs": {
        "formal_config_candidate.yaml": "817ab0163a712a9b7f779c591495a3bb3b332e28adff9cb2354a3d8e88e1293c",
        "selection_ledger.json": "7da7512f4aa31e8801a3ea76eaa02b4628a40d6f0fa4844bfae059c3d4a79431",
    },
}

_ROW_REQUIRED_KEYS = frozenset(
    {
        "kind",
        "source",
        "target",
        "content_manifest_sha256",
        "file_count",
        "total_bytes",
        "role",
        "access_class",
        "status",
    }
)
_ROW_OPTIONAL_KEYS = frozenset({"completeness", "known_missing"})
_MANIFEST_KEYS = frozenset({"schema", "mappings"})
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_STATE_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _reject_absolute_symlink_components(path: Path, where: str) -> None:
    if not path.is_absolute():
        raise V02AssetError(f"{where} must be absolute")
    current = Path(path.anchor)
    if current.is_symlink():  # pragma: no cover - unusual filesystem root
        raise V02AssetError(f"{where} contains a symlink: {current}")
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise V02AssetError(f"{where} contains a symlink: {current}")


def _git_output(repository: Path, *arguments: str) -> str:
    """Run one read-only Git query with repository-selection overrides removed."""

    # Keep the trust decision identical to the FPO source attestor. Importing
    # here avoids a module-level dependency while preserving one executable
    # proof for both public paths.
    from .runtime import _trusted_git_executable

    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "LC_ALL": "C",
        }
    )
    try:
        completed = subprocess.run(
            (
                _trusted_git_executable(),
                "-c",
                "core.fsmonitor=false",
                "-c",
                f"core.hooksPath={os.devnull}",
                "-C",
                str(repository),
                *arguments,
            ),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            env=environment,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as error:
        raise V02AssetError(
            f"repository fallback failed its Git checkout proof: {repository}"
        ) from error
    return completed.stdout.strip()


def _validate_repository_fallback(repository: Path) -> None:
    """Require a real policy-learnware checkout before deriving a sibling root."""

    if not repository.is_dir():
        raise V02AssetError(f"repository fallback is not a directory: {repository}")
    top_level = _git_output(repository, "rev-parse", "--show-toplevel")
    if not top_level:
        raise V02AssetError("repository fallback has no Git top-level")
    top = Path(top_level)
    if not top.is_absolute():
        raise V02AssetError("Git returned a non-absolute repository top-level")
    _reject_absolute_symlink_components(top, "Git repository top-level")
    if top.resolve() != repository:
        raise V02AssetError(
            "repository fallback must identify the Git checkout top-level"
        )

    head = _git_output(repository, "rev-parse", "--verify", "HEAD^{commit}")
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", head):
        raise V02AssetError("repository fallback has no canonical Git HEAD")

    pyproject = repository / "pyproject.toml"
    try:
        metadata = pyproject.lstat()
    except OSError as error:
        raise V02AssetError(
            "repository fallback is missing tracked pyproject.toml"
        ) from error
    if not stat.S_ISREG(metadata.st_mode):
        raise V02AssetError(
            "repository fallback pyproject.toml is not a regular file"
        )
    tracked = _git_output(
        repository, "ls-files", "--error-unmatch", "--", "pyproject.toml"
    )
    if tracked != "pyproject.toml":
        raise V02AssetError(
            "repository fallback pyproject.toml is not uniquely tracked"
        )


def resolve_artifacts_root(
    explicit: str | Path | None = None,
    *,
    repository_root: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve explicit/env roots or a strictly proven checkout sibling."""

    env = os.environ if environ is None else environ
    if explicit is not None:
        raw = str(explicit)
        if not raw.strip():
            raise V02AssetError("explicit artifacts root cannot be empty")
        candidate = Path(raw).expanduser()
    elif ARTIFACTS_ROOT_ENV in env:
        raw = str(env[ARTIFACTS_ROOT_ENV])
        if not raw.strip():
            raise V02AssetError(
                f"{ARTIFACTS_ROOT_ENV} cannot be empty or whitespace"
            )
        candidate = Path(raw).expanduser()
    else:
        if repository_root is not None and not str(repository_root).strip():
            raise V02AssetError("repository fallback root cannot be empty")
        repo = (
            Path(repository_root).expanduser()
            if repository_root is not None
            else _repository_root()
        )
        if not repo.is_absolute():
            repo = repo.absolute()
        _reject_absolute_symlink_components(repo, "repository fallback")
        repo = repo.resolve()
        _validate_repository_fallback(repo)
        candidate = repo.parent / "artifacts"
    if not candidate.is_absolute():
        candidate = candidate.absolute()
    _reject_absolute_symlink_components(candidate, "artifacts root")
    resolved = candidate.resolve()
    if explicit is None and ARTIFACTS_ROOT_ENV not in env:
        manifest = resolved / "relocation_manifest.json"
        try:
            validate_relocation_manifest(manifest)
        except V02AssetError as error:
            raise V02AssetError(
                "repository-derived artifacts root lacks a strict root relocation manifest"
            ) from error
    return resolved


@dataclass(frozen=True)
class V02AssetLayout:
    """Canonical external paths; construction has no filesystem side effects."""

    root: Path

    @classmethod
    def resolve(
        cls,
        explicit: str | Path | None = None,
        *,
        repository_root: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "V02AssetLayout":
        return cls(
            resolve_artifacts_root(
                explicit, repository_root=repository_root, environ=environ
            )
        )

    def __post_init__(self) -> None:
        root = Path(self.root)
        if not root.is_absolute():
            raise V02AssetError("artifacts root must be absolute")
        _reject_absolute_symlink_components(root, "artifacts root")
        object.__setattr__(self, "root", root.resolve())

    def asset(self, asset_id: str) -> Path:
        try:
            relative = ASSET_EXPECTATIONS[asset_id].target_relpath
        except KeyError as error:
            raise V02AssetError(f"unknown v0.2 asset: {asset_id!r}") from error
        if relative is None:
            raise V02AssetError(f"{asset_id} has no canonical target: original is missing")
        return _join_below(self.root, relative)

    @property
    def exact90(self) -> Path:
        return self.asset("exact90")

    @property
    def formal_inputs(self) -> Path:
        return self.asset("formal_inputs")

    @property
    def fpo(self) -> Path:
        return self.asset("fpo")

    @property
    def legacy_v02(self) -> Path:
        return self.asset("legacy_v02")

    @property
    def reconstructed_runtime(self) -> Path:
        return self.asset("reconstructed_runtime")

    @property
    def relocation_manifest(self) -> Path:
        """The single cross-version relocation manifest."""

        return self.root / "relocation_manifest.json"

    @property
    def server_plan(self) -> Path:
        return self.exact90 / "training_private" / "plans" / "server_training_plan.json"

    @property
    def runs_root(self) -> Path:
        return self.exact90 / "training_private" / "server_runs"

    @property
    def promotions(self) -> Path:
        return self.exact90 / "policy_pool_handoff_a7d10c0" / "compiled_parity_promotions.json"

    @property
    def frozen_acceptance(self) -> Path:
        return self.exact90 / "policy_pool_handoff_a7d10c0" / "policy_pool_acceptance.json"


def _join_below(root: Path, relative: str | Path) -> Path:
    path = Path(relative)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise V02AssetError(f"unsafe relative artifact path: {str(relative)!r}")
    candidate = root.joinpath(*path.parts)
    try:
        candidate.relative_to(root)
    except ValueError as error:  # pragma: no cover
        raise V02AssetError("artifact path escapes the configured root") from error
    return candidate


def _strict_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)

    if not source.is_absolute():
        raise V02AssetError(f"relocation manifest path must be absolute: {source}")
    current = Path(source.anchor)
    for part in source.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise V02AssetError(f"relocation manifest path contains a symlink: {current}")
    if not source.is_file():
        raise V02AssetError(f"relocation manifest is not a regular file: {source}")

    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise V02AssetError(f"duplicate JSON key {key!r} in {source}")
            result[key] = value
        return result

    try:
        before = source.stat()
        payload = source.read_bytes()
        after = source.stat()
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if identity_before != identity_after or len(payload) != before.st_size:
            raise V02AssetError(f"relocation manifest changed while reading: {source}")
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda token: (_ for _ in ()).throw(
                V02AssetError(f"non-finite JSON token {token!r} in {source}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V02AssetError(f"cannot read strict JSON {source}: {error}") from error
    if not isinstance(value, dict):
        raise V02AssetError(f"{source} must contain one JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], where: str) -> None:
    if set(value) != set(expected):
        raise V02AssetError(
            f"{where} keys differ; missing={sorted(set(expected) - set(value))}, "
            f"unknown={sorted(set(value) - set(expected))}"
        )


def _digest(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
        raise V02AssetError(f"{where} must be a lowercase SHA-256 digest")
    return value


def _stable_file_sha256(path: Path) -> tuple[str, int]:
    """Hash one regular file and reject concurrent replacement or mutation."""

    if path.is_symlink() or not path.is_file():
        raise V02AssetError(f"expected a non-symlink regular file: {path}")
    try:
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
    except OSError as error:
        raise V02AssetError(f"cannot attest regular file: {path}") from error
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise V02AssetError(f"file changed while hashing: {path}")
    return digest, before.st_size


def _self_digest(value: Mapping[str, Any], *, key: str, where: str) -> None:
    declared = _digest(value.get(key), f"{where}.{key}")
    material = dict(value)
    material.pop(key, None)
    if sha256_json(material) != declared:
        raise V02AssetError(f"{where} self digest mismatch")


def attest_directory(path: str | Path) -> DirectoryAttestation:
    """Apply the canonical ``sha256sum-relative-v1`` directory inventory.

    Only regular files contribute. Paths are root-relative UTF-8 strings
    without ``./`` and are ordered by their UTF-8 bytes (C ordering). Symlinks
    and every other non-directory special file are rejected rather than
    followed or silently omitted.
    """

    root = Path(path).expanduser()
    if not root.is_absolute():
        root = root.absolute()
    _reject_absolute_symlink_components(root, "asset directory")
    root = root.resolve()
    if not root.is_dir():
        raise V02AssetError(f"asset directory is missing: {root}")
    files: list[tuple[str, Path]] = []
    for candidate in root.rglob("*"):
        try:
            metadata = candidate.lstat()
        except OSError as error:
            raise V02AssetError(f"cannot stat asset path: {candidate}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise V02AssetError(f"asset tree contains a symlink: {candidate}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise V02AssetError(
                f"asset tree contains a special file: {candidate}"
            )
        try:
            relative = candidate.relative_to(root).as_posix()
        except ValueError as error:  # pragma: no cover - rglob is root-bound
            raise V02AssetError(f"asset path escapes root: {candidate}") from error
        if (
            not relative
            or relative.startswith("./")
            or "\n" in relative
            or "\r" in relative
        ):
            raise V02AssetError(f"asset has an unsafe manifest path: {relative!r}")
        files.append((relative, candidate))

    digest = hashlib.sha256()
    total_bytes = 0
    for relative, candidate in sorted(
        files, key=lambda item: item[0].encode("utf-8")
    ):
        file_sha256, byte_count = _stable_file_sha256(candidate)
        digest.update(f"{file_sha256}  {relative}\n".encode("utf-8"))
        total_bytes += byte_count
    return DirectoryAttestation(len(files), total_bytes, digest.hexdigest())


def _safe_absolute_prefix(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.startswith("/") or value == "/":
        raise V02AssetError(f"{where} must be a safe absolute prefix")
    if "\\" in value or "\0" in value or "\n" in value or "\r" in value:
        raise V02AssetError(f"{where} must be a safe absolute prefix")
    if any(part in {"", ".", ".."} for part in value.split("/")[1:]):
        raise V02AssetError(f"{where} must be a normalized absolute prefix")
    return value


def _safe_relative(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise V02AssetError(f"{where} must be a safe root-relative path")
    if "\\" in value or "\0" in value or "\n" in value or "\r" in value:
        raise V02AssetError(f"{where} must be a safe root-relative path")
    if value.startswith("/") or any(
        part in {"", ".", ".."} for part in value.split("/")
    ):
        raise V02AssetError(f"{where} must be a normalized root-relative path")
    return value


def _optional_non_negative_integer(value: Any, where: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise V02AssetError(f"{where} must be a non-negative integer or null")
    return value


def _validate_mapping(raw: Any, *, index: int) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise V02AssetError(f"relocation mapping {index} must be an object")
    keys = set(raw)
    missing = _ROW_REQUIRED_KEYS - keys
    unknown = keys - _ROW_REQUIRED_KEYS - _ROW_OPTIONAL_KEYS
    if missing or unknown:
        raise V02AssetError(
            f"relocation mapping {index} keys differ; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )

    kind = raw["kind"]
    if kind not in {"prefix", "directory", "file"}:
        raise V02AssetError(
            f"relocation mapping {index}.kind must be prefix, directory, or file"
        )
    status = raw["status"]
    if not isinstance(status, str) or not _STATE_RE.fullmatch(status):
        raise V02AssetError(f"relocation mapping {index}.status is invalid")
    for key in ("role", "access_class"):
        value = raw[key]
        if (
            not isinstance(value, str)
            or not value.strip()
            or "\0" in value
            or "\n" in value
            or "\r" in value
        ):
            raise V02AssetError(f"relocation mapping {index}.{key} must be non-empty")

    _safe_absolute_prefix(raw["source"], f"relocation mapping {index}.source")
    _safe_relative(raw["target"], f"relocation mapping {index}.target")
    _digest(
        raw["content_manifest_sha256"],
        f"relocation mapping {index}.content_manifest_sha256",
    )
    file_count = _optional_non_negative_integer(
        raw["file_count"], f"relocation mapping {index}.file_count"
    )
    total_bytes = _optional_non_negative_integer(
        raw["total_bytes"], f"relocation mapping {index}.total_bytes"
    )
    if file_count is None or total_bytes is None:
        raise V02AssetError(
            f"relocation mapping {index} requires actual file_count and total_bytes"
        )
    if kind == "file" and file_count != 1:
        raise V02AssetError(f"relocation mapping {index} file kind requires file_count=1")

    if "completeness" in raw:
        completeness = raw["completeness"]
        if not isinstance(completeness, str) or not _STATE_RE.fullmatch(completeness):
            raise V02AssetError(
                f"relocation mapping {index}.completeness is invalid"
            )
    if "known_missing" in raw:
        known_missing = raw["known_missing"]
        if not isinstance(known_missing, list):
            raise V02AssetError(f"relocation mapping {index}.known_missing must be a list")
        normalized = [
            _safe_relative(item, f"relocation mapping {index}.known_missing")
            for item in known_missing
        ]
        if len(normalized) != len(set(normalized)):
            raise V02AssetError(
                f"relocation mapping {index}.known_missing contains duplicates"
            )
    return dict(raw)


def validate_relocation_manifest(
    value_or_path: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Validate the single root allowlist without touching asset bytes."""

    value = _strict_json(value_or_path) if isinstance(value_or_path, (str, Path)) else dict(value_or_path)
    _exact_keys(value, _MANIFEST_KEYS, "relocation manifest")
    if value["schema"] != RELOCATION_SCHEMA:
        raise V02AssetError("relocation manifest schema is not rl-learnware-relocation/v1")
    raw_mappings = value["mappings"]
    if not isinstance(raw_mappings, list):
        raise V02AssetError("relocation manifest mappings must be a list")
    mappings = [
        _validate_mapping(item, index=index) for index, item in enumerate(raw_mappings)
    ]
    sources = [item["source"] for item in mappings]
    if len(sources) != len(set(sources)):
        raise V02AssetError("relocation manifest contains duplicate source prefixes")
    result = dict(value)
    result["mappings"] = mappings
    return result


@dataclass(frozen=True)
class RelocationResolver:
    """Restricted resolver from a historical or canonical path to one target."""

    layout: V02AssetLayout
    manifest: Mapping[str, Any]
    _verified_rows: set[tuple[Any, ...]] = field(
        default_factory=set, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        validated = validate_relocation_manifest(self.manifest)
        object.__setattr__(self, "manifest", validated)
        active = [
            mapping
            for mapping in validated["mappings"]
            if mapping["status"] == "verified" and self._is_v02_target(mapping["target"])
        ]
        for mapping in active:
            self._assert_v02_row_identity(mapping)
        active_targets = [mapping["target"] for mapping in active]
        if len(active_targets) != len(set(active_targets)):
            raise V02AssetError("root manifest has ambiguous active v0.2 targets")

        def overlaps(left: str, right: str) -> bool:
            left_path = Path(left)
            right_path = Path(right)
            try:
                left_path.relative_to(right_path)
                return True
            except ValueError:
                pass
            try:
                right_path.relative_to(left_path)
                return True
            except ValueError:
                return False

        extras = [
            mapping
            for mapping in validated["mappings"]
            if mapping["status"] == "verified" and not self._is_v02_target(mapping["target"])
        ]
        for extra in extras:
            for mapping in active:
                if overlaps(extra["source"], mapping["source"]) or overlaps(
                    extra["target"], mapping["target"]
                ) or overlaps(
                    extra["source"],
                    str(_join_below(self.layout.root, mapping["target"])),
                ):
                    raise V02AssetError(
                        "verified non-v0.2 mapping overlaps a fixed v0.2 relocation"
                    )

    @classmethod
    def load(
        cls,
        manifest: Mapping[str, Any] | str | Path | None = None,
        *,
        artifacts_root: str | Path | None = None,
        repository_root: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "RelocationResolver":
        layout = V02AssetLayout.resolve(
            artifacts_root, repository_root=repository_root, environ=environ
        )
        source = layout.relocation_manifest if manifest is None else manifest
        return cls(layout, validate_relocation_manifest(source))

    def _mapping_for_asset(self, asset_id: str) -> Mapping[str, Any]:
        try:
            expected_target = ASSET_EXPECTATIONS[asset_id].target_relpath
        except KeyError as error:
            raise V02AssetError(f"unknown v0.2 asset: {asset_id!r}") from error
        if expected_target is None:
            raise V02AssetError(f"{asset_id} has no recoverable original mapping")
        candidates = [
            mapping
            for mapping in self.manifest["mappings"]
            if mapping["target"] == expected_target and mapping["kind"] != "file"
        ]
        verified = [mapping for mapping in candidates if mapping["status"] == "verified"]
        if not verified:
            state = candidates[0]["status"] if candidates else "absent"
            raise V02AssetError(f"asset {asset_id} is not verified: {state}")
        if len(verified) != 1:
            raise V02AssetError(f"asset {asset_id} has ambiguous verified mappings")
        return verified[0]

    @staticmethod
    def _row_identity(mapping: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            mapping["kind"],
            mapping["source"],
            mapping["target"],
            mapping["content_manifest_sha256"],
            mapping["file_count"],
            mapping["total_bytes"],
            mapping["role"],
            mapping["access_class"],
            mapping["status"],
            mapping.get("completeness"),
            tuple(mapping.get("known_missing", ())),
        )

    @staticmethod
    def _is_v02_target(target: str) -> bool:
        allowed = {
            ASSET_EXPECTATIONS[asset_id].target_relpath
            for asset_id in _RESOLVABLE_V02_ASSETS
        }
        return target in allowed or target == _POLICY_IO_TARGET

    @staticmethod
    def _assert_v02_row_identity(mapping: Mapping[str, Any]) -> None:
        target = mapping["target"]
        for asset_id in _RESOLVABLE_V02_ASSETS:
            asset = ASSET_EXPECTATIONS[asset_id]
            if target == asset.target_relpath:
                roles = {
                    "exact90": "v02-exact90-handoff-and-training-evidence",
                    "formal_inputs": "v02-formal-inputs",
                    "legacy_v02": "legacy-v02-policy-training-backup",
                    "runtime_state": "v02-operational-runtime-state",
                }
                expected = {
                    "kind": "prefix",
                    "target": asset.target_relpath,
                    "role": roles[asset_id],
                    "access_class": "restricted",
                    "status": "verified",
                    "content_manifest_sha256": asset.tree_digest,
                    "file_count": asset.file_count,
                    "total_bytes": asset.total_bytes,
                    "source_sha256": _V02_SOURCE_SHA256[asset_id],
                }
                if asset_id == "legacy_v02":
                    expected["completeness"] = "incomplete"
                break
        else:
            if target != _POLICY_IO_TARGET:
                raise V02AssetError(f"target is not a fixed v0.2 relocation: {target}")
            expected = {
                "kind": "file",
                "target": _POLICY_IO_TARGET,
                "role": "legacy-v02-policy-io",
                "access_class": "restricted",
                "status": "verified",
                "content_manifest_sha256": _POLICY_IO_MANIFEST_SHA256,
                "file_count": 1,
                "total_bytes": 10_849,
                "source_sha256": _V02_SOURCE_SHA256["policy_io"],
            }
        observed = dict(mapping)
        observed["source_sha256"] = hashlib.sha256(
            mapping["source"].encode("utf-8")
        ).hexdigest()
        differing = sorted(
            key for key, value in expected.items() if observed.get(key) != value
        )
        if differing:
            raise V02AssetError(
                f"verified relocation metadata is not fixed v0.2 identity: {differing}"
            )

    def _verify_active_mapping(self, mapping: Mapping[str, Any]) -> Path:
        """Attest a verified row once before it can resolve any path."""

        if mapping["status"] != "verified":
            raise V02AssetError(
                f"relocation mapping is not verified: {mapping['status']}"
            )
        self._assert_v02_row_identity(mapping)
        target = _join_below(self.layout.root, mapping["target"])
        _reject_symlink_path(target, stop=self.layout.root)
        identity = self._row_identity(mapping)
        if identity in self._verified_rows:
            return target.resolve()

        if mapping["kind"] == "file":
            if not target.is_file():
                raise V02AssetError(f"verified relocation file is missing: {target}")
            file_sha256, observed_bytes = _stable_file_sha256(target)
            observed_digest = hashlib.sha256(
                f"{file_sha256}  {target.name}\n".encode("utf-8")
            ).hexdigest()
            observed_count = 1
        else:
            observed = attest_directory(target)
            observed_digest = observed.tree_digest
            observed_count = observed.file_count
            observed_bytes = observed.total_bytes
        if (
            observed_digest != mapping["content_manifest_sha256"]
            or observed_count != mapping["file_count"]
            or observed_bytes != mapping["total_bytes"]
        ):
            raise V02AssetError(
                "verified relocation inventory differs for "
                f"{mapping['source']} -> {mapping['target']}"
            )
        self._verified_rows.add(identity)
        return target.resolve()

    def ensure_verified_asset(
        self,
        kind: str,
        *,
        must_exist: bool = True,
        verify_bytes: bool = False,
    ) -> Path:
        """Return one canonical asset after mandatory root-row byte attestation.

        ``verify_bytes`` is retained for call-site compatibility; verified
        relocation rows are always attested and the flag cannot weaken that
        contract.
        """

        if kind == "vendor_original":
            raise V02AssetError("original _vendor is permanently MISSING_ORIGINAL")
        target = self.layout.asset(kind)
        _reject_symlink_path(target, stop=self.layout.root)
        if kind == "fpo":
            if must_exist and not target.is_dir():
                raise V02AssetError(f"FPO checkout is missing: {target}")
            if target.is_dir():
                from .runtime import verify_fpo_checkout

                verify_fpo_checkout(target)
            return target.resolve() if target.exists() else target

        mapping = self._mapping_for_asset(kind)
        # Root-manifest evidence is mandatory, not an optional slow path.
        verified_target = self._verify_active_mapping(mapping)
        if must_exist and not verified_target.is_dir():  # pragma: no cover
            raise V02AssetError(f"verified asset directory is missing: {target}")
        return verified_target

    def resolve(self, recorded: str | Path, *, must_exist: bool = True) -> Path:
        """Canonicalize an allowlisted source or target path.

        Both historical absolute paths stored in immutable receipts and
        already-canonical absolute paths below this resolver's artifacts root
        are accepted.  All other paths fail closed.
        """

        raw_text = os.fspath(recorded)
        _safe_absolute_prefix(raw_text, "recorded path")
        raw = Path(raw_text)
        candidates: list[tuple[int, int, Mapping[str, Any], Path]] = []
        for index, mapping in enumerate(self.manifest["mappings"]):
            if mapping["status"] != "verified" or not self._is_v02_target(
                mapping["target"]
            ):
                continue
            prefixes = [
                Path(mapping["source"]),
                _join_below(self.layout.root, mapping["target"]),
            ]
            for prefix in prefixes:
                if mapping["kind"] == "file":
                    if raw != prefix:
                        continue
                    suffix = Path()
                else:
                    try:
                        suffix = raw.relative_to(prefix)
                    except ValueError:
                        continue
                candidates.append((len(prefix.parts), index, mapping, suffix))
        if not candidates:
            raise V02AssetError(f"recorded path has no allowlisted relocation: {recorded}")
        longest = max(item[0] for item in candidates)
        winners = [item for item in candidates if item[0] == longest]
        winner_rows = {item[1] for item in winners}
        if len(winner_rows) != 1:
            raise V02AssetError(f"recorded path has ambiguous relocations: {recorded}")
        _, _, mapping, suffix = winners[0]
        target = self._verify_active_mapping(mapping)
        if suffix.parts:
            target = _join_below(target, suffix)
        _reject_symlink_path(target, stop=self.layout.root)
        if must_exist and not target.exists():
            raise V02AssetError(f"relocated path is missing: {target}")
        return target.resolve() if target.exists() else target

    def rebase(self, recorded: str | Path, *, must_exist: bool = True) -> Path:
        """Backward-compatible name for :meth:`resolve`."""

        return self.resolve(recorded, must_exist=must_exist)


def _reject_symlink_path(path: Path, *, stop: Path) -> None:
    try:
        relative = path.relative_to(stop)
    except ValueError as error:
        raise V02AssetError(f"path escapes artifacts root: {path}") from error
    current = stop
    if current.is_symlink():
        raise V02AssetError(f"artifacts root is a symlink: {current}")
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise V02AssetError(f"relocated path contains a symlink: {current}")


def verify_inventory(layout: V02AssetLayout, asset_id: str) -> DirectoryAttestation:
    """Recompute and compare a frozen directory inventory."""

    expected = ASSET_EXPECTATIONS.get(asset_id)
    if expected is None or expected.tree_digest is None:
        raise V02AssetError(f"{asset_id} has no directory-inventory trust anchor")
    observed = attest_directory(layout.asset(asset_id))
    if (
        observed.tree_digest != expected.tree_digest
        or (
            expected.file_count is not None
            and observed.file_count != expected.file_count
        )
        or (
            expected.total_bytes is not None
            and observed.total_bytes != expected.total_bytes
        )
    ):
        raise V02AssetError(
            f"{asset_id} inventory differs: expected "
            f"{expected.file_count}/{expected.total_bytes}/{expected.tree_digest}, "
            f"observed {observed.file_count}/{observed.total_bytes}/{observed.tree_digest}"
        )
    return observed


def verify_handoff_trust_anchors(
    layout: V02AssetLayout,
    *,
    verify_tree_inventories: bool = True,
) -> dict[str, Any]:
    """Verify immutable exact-90/formal-input bytes and the frozen pool geometry."""

    inventories: dict[str, dict[str, Any]] = {}
    if verify_tree_inventories:
        for asset_id in ("exact90", "formal_inputs"):
            inventories[asset_id] = verify_inventory(layout, asset_id).to_dict()
    files: dict[str, str] = {}
    for asset_id, expected_files in CRITICAL_FILE_SHA256.items():
        base = layout.asset(asset_id)
        for relative, expected_sha in expected_files.items():
            path = _join_below(base, relative)
            if not path.is_file():
                raise V02AssetError(f"critical {asset_id} file is missing: {path}")
            observed = sha256_file(path)
            if observed != expected_sha:
                raise V02AssetError(
                    f"critical {asset_id} file digest differs for {relative}: {observed}"
                )
            files[f"{asset_id}/{relative}"] = observed
    acceptance = _strict_json(layout.frozen_acceptance)
    if (
        acceptance.get("decision") != "PASS"
        or acceptance.get("job_count") != 90
        or acceptance.get("anchor_count") != 30
        or acceptance.get("seeds") != [0, 1, 2]
        or acceptance.get("direct_terminal_record_count") != 84
        or acceptance.get("compiled_parity_fallback_promotion_count") != 6
        or acceptance.get("pool_digest") != EXPECTED_POOL_DIGEST
    ):
        raise V02AssetError("frozen acceptance is not the reviewed exact 84+6 pool")
    _self_digest(acceptance, key="report_digest", where="frozen acceptance")
    promotions = _strict_json(layout.promotions)
    if promotions.get("promotion_count") != 6:
        raise V02AssetError("frozen promotion manifest is not the reviewed six-cell overlay")
    _self_digest(promotions, key="manifest_digest", where="frozen promotions")
    return {
        "schema": "policy-learnware.v02-handoff-trust-anchor-verification.v0",
        "passed": True,
        "job_count": 90,
        "anchor_count": 30,
        "seeds": [0, 1, 2],
        "direct_terminal_record_count": 84,
        "compiled_parity_fallback_promotion_count": 6,
        "pool_digest": EXPECTED_POOL_DIGEST,
        "inventories": inventories,
        "critical_files": dict(sorted(files.items())),
    }


def capability_status(
    layout: V02AssetLayout,
    manifest: Mapping[str, Any] | str | Path,
    *,
    verify_bytes: bool = False,
) -> dict[str, Any]:
    """Report asset/provenance readiness without claiming runtime execution.

    Root relocation bytes are always attested. ``verify_bytes`` is a retained
    compatibility argument and does not disable or strengthen that invariant.
    JAX, MuJoCo, and other runtime dependencies are deliberately not imported
    or probed by this read-only receipt.
    """

    validated = validate_relocation_manifest(manifest)
    resolver = RelocationResolver(layout, validated)

    def has_active_mapping(asset_id: str) -> bool:
        target = ASSET_EXPECTATIONS[asset_id].target_relpath
        return any(
            mapping["target"] == target
            and mapping["kind"] != "file"
            and mapping["status"] == "verified"
            for mapping in validated["mappings"]
        )

    checks: dict[str, bool] = {}
    for asset_id in ("exact90", "formal_inputs"):
        available = has_active_mapping(asset_id) and layout.asset(asset_id).is_dir()
        if available:
            resolver.ensure_verified_asset(asset_id, verify_bytes=verify_bytes)
        checks[asset_id] = available

    # FPO has an independent reviewed Git proof and deliberately does not rely
    # on a root relocation row for scientific identity.
    fpo = layout.fpo.is_dir()
    if fpo:
        resolver.ensure_verified_asset("fpo")

    legacy = False
    if has_active_mapping("legacy_v02") and layout.legacy_v02.is_dir():
        # Attest and reject the complete tree before opening policy_io.  A
        # status probe must never follow an untrusted policy_io symlink or read
        # an external/oversized file before the relocation gate closes.
        verified_legacy = resolver.ensure_verified_asset(
            "legacy_v02", verify_bytes=verify_bytes
        )
        policy_io = verified_legacy / "policy_io.py"
        if policy_io.is_file():
            policy_io_digest, _ = _stable_file_sha256(policy_io)
            legacy = policy_io_digest == EXPECTED_POLICY_IO_SHA256

    # The central contract permanently records this provenance as missing.  A
    # rebuilt dependency tree may enable inference but never training replay.
    original_vendor = False
    handoff = checks["exact90"] and checks["formal_inputs"]
    # The reviewed compatibility loader can execute policies from the frozen
    # FPO source plus the exact policy_io shim.  This is reconstructed runtime
    # evidence; it never upgrades the missing original training environment.
    inference = handoff and fpo and legacy
    training = handoff and fpo and legacy and original_vendor
    return {
        "schema": "policy-learnware.v02-capability-status.v1",
        "readiness_scope": "asset_and_provenance_only",
        "runtime_dependency_check": "not_performed",
        "handoff_verification": {
            "available": handoff,
            "provenance_class": "ORIGINAL_IMMUTABLE_EVIDENCE" if handoff else "UNAVAILABLE",
        },
        "policy_inference": {
            "asset_provenance_ready": inference,
            "runtime_dependency_check": "not_performed",
            "runtime_dependency_ready": None,
            "provenance_class_if_runtime_ready": (
                "ORIGINAL_RUNTIME" if inference and original_vendor else
                "RECONSTRUCTED_RUNTIME" if inference else "UNAVAILABLE"
            ),
        },
        "training_replay": {
            "asset_provenance_ready": training,
            "runtime_dependency_check": "not_performed",
            "runtime_dependency_ready": None,
            "provenance_class_if_runtime_ready": (
                "ORIGINAL_RUNTIME" if training else "UNAVAILABLE"
            ),
            "blocker": None if training else "MISSING_ORIGINAL_VENDOR_RUNTIME",
        },
    }


__all__ = [
    "ARTIFACTS_ROOT_ENV",
    "ASSET_EXPECTATIONS",
    "ArtifactDomain",
    "CRITICAL_FILE_SHA256",
    "DirectoryAttestation",
    "EXPECTED_POOL_DIGEST",
    "FPO_COMMIT",
    "RELOCATION_SCHEMA",
    "RelocationResolver",
    "TREE_DIGEST_ALGORITHM",
    "V02AssetError",
    "V02AssetLayout",
    "V02ArtifactLayout",
    "V02ArtifactLayoutError",
    "V02ArtifactWriter",
    "V02_RUN_ID",
    "attest_directory",
    "capability_status",
    "resolve_artifacts_root",
    "validate_relocation_manifest",
    "verify_inventory",
    "verify_handoff_trust_anchors",
]
