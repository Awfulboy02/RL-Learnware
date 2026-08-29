from pathlib import Path
import sys

import pytest

from policy_learnware_v0.v03 import pool_intake


@pytest.mark.parametrize(
    ("dont_write_bytecode", "pycache_prefix", "message"),
    [
        (False, None, "PYTHONDONTWRITEBYTECODE"),
        (True, "", "pycache_prefix=None"),
        (True, "/external/cache", "pycache_prefix=None"),
    ],
)
def test_frozen_acceptance_rejects_unsafe_bytecode_before_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dont_write_bytecode: bool,
    pycache_prefix: str | None,
    message: str,
) -> None:
    before_path = list(sys.path)
    before_modules = {
        name: module
        for name, module in sys.modules.items()
        if name.startswith("server.repro_fpo_ppo_v02")
    }
    imports = 0

    def forbidden_import(_name: str):
        nonlocal imports
        imports += 1
        raise AssertionError("runtime gate must run before package import")

    monkeypatch.setattr(sys, "dont_write_bytecode", dont_write_bytecode)
    monkeypatch.setattr(sys, "pycache_prefix", pycache_prefix)
    monkeypatch.setattr(pool_intake.importlib, "import_module", forbidden_import)

    with pytest.raises(pool_intake.PoolIntakeError, match=message):
        pool_intake._replay_frozen_v02_acceptance(
            tmp_path / "missing-root",
            tmp_path / "missing-handoff",
            {},
        )

    assert imports == 0
    assert sys.path == before_path
    assert {
        name: module
        for name, module in sys.modules.items()
        if name.startswith("server.repro_fpo_ppo_v02")
    } == before_modules
