"""Stable, read-only runtime bridge for the frozen v0.2 FPO source.

The original v0.2 training dependency tree is missing.  This module therefore
keeps two claims deliberately separate:

* the recovered FPO checkout can be attested byte-for-byte against the frozen
  source snapshot; and
* importing that source in a newly assembled environment is a reconstructed
  runtime, never the original training runtime.

No function in this module can promote a reconstructed dependency directory
to original provenance.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib
import importlib.machinery
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import threading
from types import MappingProxyType, ModuleType
from typing import Any, Iterator, Mapping, NoReturn

from ..hashing import sha256_json


FPO_COMMIT = "418c2554f7cd22d52e14c07d951280929d73bf2f"
FPO_HEAD_TREE_DIGEST = (
    "7bb5d663d19b1e5099037e56990211c21e43fb5948be55fcdd4f5d983b135783"
)
FPO_EXECUTION_TREE_DIGEST = (
    "396f2b4633d1fd0cf1cc753fbe16a458f4e62afabb385cbf2fd3dfb872626083"
)
FPO_SOURCE_FILE_COUNT = 72

ORIGINAL_VENDOR_STATUS = "MISSING_ORIGINAL"
ORIGINAL_VENDOR_TREE_DIGEST = (
    "11ea54a9390010e1de5d7bdadf75334614f3adaba21940ca898e01f347f786d4"
)
ORIGINAL_VENDOR_FILE_COUNT = 1612
ORIGINAL_VENDOR_TOTAL_BYTES = 90_428_849
RECONSTRUCTED_RUNTIME = "RECONSTRUCTED_RUNTIME"
INFERENCE_ONLY_WANDB_SHIM_IDENTITY = (
    "policy_learnware_v0.v02.runtime/inference-only-wandb/v1"
)

_TRUSTED_GIT = Path("/usr/bin/git")
_RUNTIME_IMPORT_LOCK = threading.RLock()


class RuntimeVerificationError(RuntimeError):
    """A checkout or imported module failed the frozen runtime contract."""


class ReconstructedRuntimeNotAllowed(RuntimeVerificationError):
    """The caller did not explicitly opt in to reconstructed execution."""


class OriginalVendorUnavailable(RuntimeVerificationError):
    """The byte-identical original v0.2 dependency tree is unavailable."""


@dataclass(frozen=True)
class VerifiedFPOUpstream:
    """Imported FPO modules plus their immutable reconstructed-runtime claim."""

    provenance_class: str
    runtime_receipt: Mapping[str, Any]
    source_attestation: Mapping[str, Any]
    jax: Any
    jax_dataclasses: Any
    jax_numpy: Any
    dm_control_suite: Any
    registry: Any
    fpo: Any
    ppo: Any
    rollouts: Any

    def legacy_module_tuple(self) -> tuple[Any, ...]:
        """Return the old runner's module order without weakening provenance."""

        return (
            self.jax,
            self.jax_dataclasses,
            self.jax_numpy,
            self.dm_control_suite,
            self.registry,
            self.fpo,
            self.ppo,
            self.rollouts,
        )


def _trusted_git_executable() -> str:
    try:
        metadata = _TRUSTED_GIT.lstat()
    except OSError as error:
        raise RuntimeVerificationError(
            f"trusted Git executable is unavailable: {_TRUSTED_GIT}"
        ) from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or metadata.st_mode & 0o022
    ):
        raise RuntimeVerificationError(
            "trusted Git executable must be root-owned, regular, and non-writable"
        )
    return str(_TRUSTED_GIT)


def _git_raw(root: Path, *arguments: str) -> bytes:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "LC_ALL": "C",
        }
    )
    command = [
        _trusted_git_executable(),
        "-c",
        "core.fsmonitor=false",
        "-c",
        f"core.hooksPath={os.devnull}",
        "-C",
        str(root),
        *arguments,
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            timeout=60,
            env=environment,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RuntimeVerificationError(
            f"cannot inspect FPO checkout with git {arguments!r}"
        ) from error
    return completed.stdout


def _git_text(root: Path, *arguments: str) -> str:
    try:
        return _git_raw(root, *arguments).decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise RuntimeVerificationError(
            f"git {arguments!r} returned non-UTF-8 text"
        ) from error


def _safe_git_path(value: bytes, *, where: str) -> str:
    try:
        path = value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeVerificationError(f"{where} is not UTF-8") from error
    pure = PurePosixPath(path)
    if (
        not path
        or pure.is_absolute()
        or ".." in pure.parts
        or "\x00" in path
        or "\n" in path
        or "\r" in path
    ):
        raise RuntimeVerificationError(f"unsafe {where}: {path!r}")
    return path


def _blob_object_id(data: bytes, *, object_format: str) -> str:
    if object_format not in {"sha1", "sha256"}:
        raise RuntimeVerificationError(
            f"unsupported Git object format: {object_format!r}"
        )
    digest = hashlib.new(object_format)
    digest.update(f"blob {len(data)}\0".encode("ascii"))
    digest.update(data)
    return digest.hexdigest()


def _reject_symlink_components(root: Path, relative: str) -> None:
    current = root
    for component in PurePosixPath(relative).parts:
        current = current / component
        if current.is_symlink():
            raise RuntimeVerificationError(
                f"FPO tracked path traverses a symlink: {relative}"
            )


def _other_paths(root: Path, *, ignored: bool) -> list[str]:
    arguments = ["ls-files", "--others", "-z", "--exclude-standard"]
    if ignored:
        arguments.insert(2, "--ignored")
    return sorted(
        _safe_git_path(raw, where="ignored FPO path" if ignored else "untracked FPO path")
        for raw in _git_raw(root, *arguments).split(b"\0")
        if raw
    )


def inspect_fpo_checkout(
    path: str | Path, *, expected_commit: str | None = None
) -> dict[str, Any]:
    """Hash every tracked byte and expose all checkout-bypass surfaces.

    This is an inspection primitive.  :func:`verify_fpo_checkout` additionally
    binds the result to the frozen v0.2 digests and rejects every difference.
    """

    frozen_commit = FPO_COMMIT if expected_commit is None else expected_commit
    source = Path(path)
    if source.is_symlink():
        raise RuntimeVerificationError("FPO checkout root cannot be a symlink")
    try:
        root = source.resolve(strict=True)
    except OSError as error:
        raise RuntimeVerificationError(f"FPO checkout is missing: {source}") from error
    git_directory = root / ".git"
    if not root.is_dir() or git_directory.is_symlink() or not git_directory.is_dir():
        raise RuntimeVerificationError(f"FPO checkout with a .git directory is missing: {root}")

    initial_head = _git_text(root, "rev-parse", "--verify", "HEAD")
    if initial_head != frozen_commit:
        raise RuntimeVerificationError(
            f"FPO checkout HEAD differs from the reviewed commit: {initial_head}"
        )
    object_format = _git_text(root, "rev-parse", "--show-object-format")
    replacement_refs = _git_raw(
        root, "for-each-ref", "--format=%(refname)", "refs/replace"
    ).strip()
    if replacement_refs:
        raise RuntimeVerificationError("FPO checkout contains forbidden replace refs")

    head_entries: list[dict[str, Any]] = []
    worktree_entries: list[dict[str, Any]] = []
    execution_entries: list[dict[str, Any]] = []
    content_changes: list[str] = []
    tree = _git_raw(root, "ls-tree", "-rz", "--full-tree", frozen_commit)
    for raw in tree.split(b"\0"):
        if not raw:
            continue
        try:
            header, raw_path = raw.split(b"\t", 1)
            mode, kind, object_id = header.decode("ascii").split(" ")
        except (ValueError, UnicodeDecodeError) as error:
            raise RuntimeVerificationError("cannot parse FPO HEAD tree") from error
        relative = _safe_git_path(raw_path, where="tracked FPO path")
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise RuntimeVerificationError(
                f"FPO proof rejects non-regular tracked entry {relative!r}: "
                f"mode={mode!r}, kind={kind!r}"
            )

        _reject_symlink_components(root, relative)
        absolute = root.joinpath(*PurePosixPath(relative).parts)
        actual_object: str | None = None
        actual_mode: str | None = None
        raw_sha256: str | None = None
        byte_count: int | None = None
        try:
            metadata = absolute.lstat()
            if absolute.is_symlink() or not stat.S_ISREG(metadata.st_mode):
                raise RuntimeVerificationError(
                    f"tracked FPO entry is not a regular file: {relative}"
                )
            data = absolute.read_bytes()
            actual_mode = "100755" if metadata.st_mode & 0o111 else "100644"
            actual_object = _blob_object_id(data, object_format=object_format)
            raw_sha256 = hashlib.sha256(data).hexdigest()
            byte_count = len(data)
        except FileNotFoundError:
            pass
        head_entries.append({"mode": mode, "object": object_id, "path": relative})
        worktree_entries.append(
            {"mode": actual_mode, "object": actual_object, "path": relative}
        )
        execution_entries.append(
            {
                "bytes": byte_count,
                "mode": actual_mode,
                "path": relative,
                "sha256": raw_sha256,
            }
        )
        if actual_mode != mode or actual_object != object_id:
            content_changes.append(relative)

    if not head_entries:
        raise RuntimeVerificationError("FPO HEAD tree has no regular files")

    untracked_paths = _other_paths(root, ignored=False)
    ignored_paths = _other_paths(root, ignored=True)

    index_flags: list[str] = []
    for raw in _git_raw(root, "ls-files", "-v", "-z").split(b"\0"):
        if not raw:
            continue
        if len(raw) < 3 or raw[1:2] != b" ":
            raise RuntimeVerificationError("cannot parse FPO index flags")
        tag = chr(raw[0])
        relative = _safe_git_path(raw[2:], where="indexed FPO path")
        if tag == "S" or tag.islower():
            index_flags.append(f"{tag} {relative}")

    tracked_status: list[str] = []
    status = _git_raw(
        root, "status", "--porcelain=v1", "-z", "--untracked-files=all"
    )
    for raw in status.split(b"\0"):
        if not raw or raw.startswith(b"?? "):
            continue
        try:
            tracked_status.append(raw.decode("utf-8"))
        except UnicodeDecodeError as error:
            raise RuntimeVerificationError(
                "FPO status contains a non-UTF-8 path"
            ) from error

    final_head = _git_text(root, "rev-parse", "--verify", "HEAD")
    if final_head != initial_head:
        raise RuntimeVerificationError("FPO HEAD changed during source attestation")
    tracked_changes = sorted(set(content_changes + tracked_status))
    return {
        "fpo_commit": initial_head,
        "fpo_tracked_dirty": bool(tracked_changes),
        "fpo_tracked_changes": tracked_changes,
        "fpo_head_tree_digest": sha256_json(head_entries),
        "fpo_worktree_tree_digest": sha256_json(worktree_entries),
        "fpo_execution_tree_digest": sha256_json(execution_entries),
        "fpo_source_file_count": len(head_entries),
        "fpo_index_flags": sorted(index_flags),
        "fpo_untracked_paths": untracked_paths,
        "fpo_ignored_paths": ignored_paths,
    }


def verify_fpo_checkout(path: str | Path) -> dict[str, Any]:
    """Require the exact clean FPO source snapshot reviewed for v0.2."""

    observed = inspect_fpo_checkout(path, expected_commit=FPO_COMMIT)
    expected = {
        "fpo_commit": FPO_COMMIT,
        "fpo_tracked_dirty": False,
        "fpo_tracked_changes": [],
        "fpo_head_tree_digest": FPO_HEAD_TREE_DIGEST,
        "fpo_worktree_tree_digest": FPO_HEAD_TREE_DIGEST,
        "fpo_execution_tree_digest": FPO_EXECUTION_TREE_DIGEST,
        "fpo_source_file_count": FPO_SOURCE_FILE_COUNT,
        "fpo_index_flags": [],
        "fpo_untracked_paths": [],
        "fpo_ignored_paths": [],
    }
    differing = sorted(
        key for key, expected_value in expected.items() if observed.get(key) != expected_value
    )
    if differing:
        raise RuntimeVerificationError(
            f"FPO checkout differs from frozen attestation: {differing}"
        )
    return observed


def _immutable_attestation(value: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen: dict[str, Any] = {}
    for key, item in value.items():
        frozen[key] = tuple(item) if isinstance(item, list) else item
    return MappingProxyType(frozen)


def _require_module_origin(
    module: ModuleType, *, expected_file: Path, import_name: str
) -> None:
    spec = getattr(module, "__spec__", None)
    origin_value = getattr(spec, "origin", None)
    if not isinstance(origin_value, str) or not origin_value:
        raise RuntimeVerificationError(f"{import_name} has no regular source origin")
    if not isinstance(
        getattr(spec, "loader", None), importlib.machinery.SourceFileLoader
    ):
        raise RuntimeVerificationError(
            f"{import_name} was not loaded by the standard source-file loader"
        )
    origin = Path(origin_value)
    if origin.is_symlink():
        raise RuntimeVerificationError(f"{import_name} origin is a symlink")
    try:
        resolved_origin = origin.resolve(strict=True)
        resolved_expected = expected_file.resolve(strict=True)
    except OSError as error:
        raise RuntimeVerificationError(
            f"{import_name} was not imported from the attested FPO checkout"
        ) from error
    if resolved_origin != resolved_expected:
        raise RuntimeVerificationError(
            f"{import_name} was not imported from its exact attested source file"
        )
    if not resolved_origin.is_file():
        raise RuntimeVerificationError(f"{import_name} origin is not a regular file")


def _expected_flow_policy_file(source_dir: Path, import_name: str) -> Path:
    if import_name == "flow_policy":
        candidates = [source_dir / "flow_policy" / "__init__.py"]
    else:
        relative = import_name.removeprefix("flow_policy.").split(".")
        base = source_dir.joinpath("flow_policy", *relative)
        candidates = [base.with_suffix(".py"), base / "__init__.py"]
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if len(existing) != 1:
        raise RuntimeVerificationError(
            f"cannot bind {import_name} to one attested source file"
        )
    return existing[0]


def _require_cached_flow_policy_origins(*, source_dir: Path) -> None:
    for name, module in sorted(sys.modules.items()):
        if name != "flow_policy" and not name.startswith("flow_policy."):
            continue
        if not isinstance(module, ModuleType):
            raise RuntimeVerificationError(
                f"cached {name} is not a verifiable Python module"
            )
        _require_module_origin(
            module,
            expected_file=_expected_flow_policy_file(source_dir, name),
            import_name=name,
        )


def _require_no_cached_flow_policy() -> None:
    cached = sorted(
        name
        for name in sys.modules
        if name == "flow_policy" or name.startswith("flow_policy.")
    )
    if cached:
        raise RuntimeVerificationError(
            "refusing cached flow_policy modules; their executed code cannot be "
            f"re-attested from source bytes: {cached}"
        )


def _namespace_snapshot(prefix: str) -> dict[str, object]:
    return {
        name: module
        for name, module in sys.modules.items()
        if name == prefix or name.startswith(f"{prefix}.")
    }


def _restore_namespace(prefix: str, snapshot: Mapping[str, object]) -> None:
    for name in tuple(sys.modules):
        if name == prefix or name.startswith(f"{prefix}."):
            del sys.modules[name]
    sys.modules.update(snapshot)


def _forbid_wandb_behavior(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise ReconstructedRuntimeNotAllowed(
        "the inference-only wandb shim has no logging, network, artifact, or "
        "write behavior"
    )


class _InferenceOnlyWandbRun:
    """Import-surface type only; no Run object can be constructed."""

    def __new__(cls, *_args: Any, **_kwargs: Any) -> NoReturn:
        _forbid_wandb_behavior()


def _wandb_missing_attribute(name: str) -> NoReturn:
    raise ReconstructedRuntimeNotAllowed(
        f"wandb.{name} is unavailable in the inference-only reconstructed runtime"
    )


def _inference_only_wandb_modules() -> dict[str, ModuleType]:
    wandb = ModuleType("wandb")
    sdk = ModuleType("wandb.sdk")
    wandb_run = ModuleType("wandb.sdk.wandb_run")
    wandb.__package__ = "wandb"
    sdk.__package__ = "wandb.sdk"
    wandb_run.__package__ = "wandb.sdk"
    wandb.__path__ = []  # type: ignore[attr-defined]
    sdk.__path__ = []  # type: ignore[attr-defined]
    wandb.__spec__ = importlib.machinery.ModuleSpec(
        "wandb", loader=None, is_package=True
    )
    sdk.__spec__ = importlib.machinery.ModuleSpec(
        "wandb.sdk", loader=None, is_package=True
    )
    wandb_run.__spec__ = importlib.machinery.ModuleSpec(
        "wandb.sdk.wandb_run", loader=None, is_package=False
    )

    # The frozen rollouts module imports only ``wandb`` and ``Run``.  Explicit
    # names below make accidental runtime behavior fail clearly; __getattr__
    # rejects every API not enumerated here as well.
    for name in (
        "Api",
        "Artifact",
        "Histogram",
        "Image",
        "Table",
        "Video",
        "agent",
        "finish",
        "init",
        "log",
        "login",
        "restore",
        "save",
        "sweep",
    ):
        setattr(wandb, name, _forbid_wandb_behavior)
    wandb.__getattr__ = _wandb_missing_attribute  # type: ignore[attr-defined]
    wandb.sdk = sdk  # type: ignore[attr-defined]
    sdk.wandb_run = wandb_run  # type: ignore[attr-defined]
    wandb_run.Run = _InferenceOnlyWandbRun  # type: ignore[attr-defined]
    return {
        "wandb": wandb,
        "wandb.sdk": sdk,
        "wandb.sdk.wandb_run": wandb_run,
    }


@contextmanager
def _temporary_inference_only_wandb() -> Iterator[None]:
    previous = _namespace_snapshot("wandb")
    _restore_namespace("wandb", {})
    sys.modules.update(_inference_only_wandb_modules())
    try:
        yield
    finally:
        _restore_namespace("wandb", previous)


def _import_flow_policy_modules() -> dict[str, Any]:
    return {
        "fpo": importlib.import_module("flow_policy.fpo"),
        "ppo": importlib.import_module("flow_policy.ppo"),
        "rollouts": importlib.import_module("flow_policy.rollouts"),
    }


def load_verified_fpo_upstream(
    fpo_root: str | Path, *, allow_reconstructed: bool = False
) -> VerifiedFPOUpstream:
    """Import FPO only after explicit opt-in and byte-level source verification.

    The returned ``provenance_class`` is always ``RECONSTRUCTED_RUNTIME``.
    This API cannot establish training replay or original-runtime provenance.
    """

    with _RUNTIME_IMPORT_LOCK:
        if allow_reconstructed is not True:
            raise ReconstructedRuntimeNotAllowed(
                "original v0.2 vendor bytes are missing; pass "
                "allow_reconstructed=True only for explicitly labelled "
                "reconstructed inference"
            )
        if sys.dont_write_bytecode is not True:
            raise RuntimeVerificationError(
                "reconstructed FPO import requires sys.dont_write_bytecode=True"
            )

        source_root = Path(fpo_root)
        # The shim is not even considered until the complete source checkout
        # has passed its frozen commit/tree/cleanliness attestation.
        before = verify_fpo_checkout(source_root)
        try:
            root = source_root.resolve(strict=True)
        except OSError as error:  # Defensive if a path disappears after verification.
            raise RuntimeVerificationError(
                f"FPO checkout disappeared after verification: {source_root}"
            ) from error
        source_dir = root / "playground" / "src"
        expected_entry = source_dir / "flow_policy" / "fpo.py"
        if expected_entry.is_symlink() or not expected_entry.is_file():
            raise RuntimeVerificationError(f"not an FPO checkout: {root}")

        # A same-path cached module can contain mutated or formerly tampered
        # code despite its trusted-looking __spec__.origin.  Refuse all caches
        # instead of silently treating an origin string as byte attestation.
        _require_no_cached_flow_policy()
        initial_flow_modules = _namespace_snapshot("flow_policy")
        source_text = str(source_dir)
        shim_used = False
        modules: dict[str, Any]
        import_succeeded = False
        sys.path.insert(0, source_text)
        try:
            playground = importlib.import_module("mujoco_playground")
            try:
                # mujoco_playground 0.0.5 exposes these through the package;
                # registry is not an independently importable submodule.
                dm_control_suite = getattr(playground, "dm_control_suite")
                registry = getattr(playground, "registry")
            except AttributeError as error:
                raise RuntimeVerificationError(
                    "mujoco_playground does not re-export dm_control_suite and registry"
                ) from error
            base_modules = {
                "jax": importlib.import_module("jax"),
                "jax_dataclasses": importlib.import_module("jax_dataclasses"),
                "jax_numpy": importlib.import_module("jax.numpy"),
                "dm_control_suite": dm_control_suite,
                "registry": registry,
            }
            try:
                flow_modules = _import_flow_policy_modules()
            except ModuleNotFoundError as error:
                if error.name != "wandb":
                    raise
                # A failed package import may leave successfully imported
                # siblings cached.  Restore the exact pre-import namespace and
                # retry once under the import-surface-only shim.
                _restore_namespace("flow_policy", initial_flow_modules)
                with _temporary_inference_only_wandb():
                    flow_modules = _import_flow_policy_modules()
                shim_used = True

            modules = {**base_modules, **flow_modules}
            for name in ("fpo", "ppo", "rollouts"):
                _require_module_origin(
                    modules[name],
                    expected_file=source_dir / "flow_policy" / f"{name}.py",
                    import_name=f"flow_policy.{name}",
                )
            _require_cached_flow_policy_origins(source_dir=source_dir)
            import_succeeded = True
        except RuntimeVerificationError:
            raise
        except Exception as error:
            raise RuntimeVerificationError(
                "cannot import reconstructed FPO dependencies"
            ) from error
        finally:
            if sys.path and sys.path[0] == source_text:
                del sys.path[0]
            else:  # Defensive: do not silently leave a privileged path inserted.
                try:
                    sys.path.remove(source_text)
                except ValueError:
                    pass
            if not import_succeeded:
                _restore_namespace("flow_policy", initial_flow_modules)

        after = verify_fpo_checkout(root)
        if after != before:
            _restore_namespace("flow_policy", initial_flow_modules)
            raise RuntimeVerificationError("FPO checkout changed during module import")
        runtime_receipt = MappingProxyType(
            {
                "schema": "policy-learnware.v02-reconstructed-runtime.v1",
                "runtime_status": RECONSTRUCTED_RUNTIME,
                "original_runtime_capable": False,
                "training_replay_capable": False,
                "inference_only": True,
                "missing_dependency": "wandb" if shim_used else None,
                "shim_identity": (
                    INFERENCE_ONLY_WANDB_SHIM_IDENTITY if shim_used else None
                ),
            }
        )
        return VerifiedFPOUpstream(
            provenance_class=RECONSTRUCTED_RUNTIME,
            runtime_receipt=runtime_receipt,
            source_attestation=_immutable_attestation(after),
            **modules,
        )


_ORIGINAL_VENDOR_STATUS = MappingProxyType(
    {
        "status": ORIGINAL_VENDOR_STATUS,
        "provenance_class": ORIGINAL_VENDOR_STATUS,
        "expected_tree_digest": ORIGINAL_VENDOR_TREE_DIGEST,
        "expected_file_count": ORIGINAL_VENDOR_FILE_COUNT,
        "expected_total_bytes": ORIGINAL_VENDOR_TOTAL_BYTES,
    }
)


def original_vendor_status() -> Mapping[str, Any]:
    """Return the immutable audit fact that the original vendor tree is missing."""

    return _ORIGINAL_VENDOR_STATUS


def require_original_vendor_runtime() -> NoReturn:
    """Fail closed: no recovered directory is the byte-identical original vendor."""

    raise OriginalVendorUnavailable(
        "MISSING_ORIGINAL: the v0.2 vendor tree bound by digest "
        f"{ORIGINAL_VENDOR_TREE_DIGEST} ({ORIGINAL_VENDOR_FILE_COUNT} files, "
        f"{ORIGINAL_VENDOR_TOTAL_BYTES} bytes) has not been recovered"
    )


__all__ = [
    "FPO_COMMIT",
    "FPO_EXECUTION_TREE_DIGEST",
    "FPO_HEAD_TREE_DIGEST",
    "FPO_SOURCE_FILE_COUNT",
    "INFERENCE_ONLY_WANDB_SHIM_IDENTITY",
    "ORIGINAL_VENDOR_FILE_COUNT",
    "ORIGINAL_VENDOR_STATUS",
    "ORIGINAL_VENDOR_TOTAL_BYTES",
    "ORIGINAL_VENDOR_TREE_DIGEST",
    "OriginalVendorUnavailable",
    "RECONSTRUCTED_RUNTIME",
    "ReconstructedRuntimeNotAllowed",
    "RuntimeVerificationError",
    "VerifiedFPOUpstream",
    "inspect_fpo_checkout",
    "load_verified_fpo_upstream",
    "original_vendor_status",
    "require_original_vendor_runtime",
    "verify_fpo_checkout",
]
