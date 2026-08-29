"""The frozen v0.2 verification surface has no training-stack imports."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_minimal_public_import_closure_has_no_deleted_or_runtime_modules() -> None:
    root = Path(__file__).resolve().parents[2]
    script = f"""
import importlib
import json
import sys

sys.path.insert(0, {str(root)!r})
sys.path.insert(0, {str(root / 'src')!r})

requested = [
    'policy_learnware_v0',
    'policy_learnware_v0.hashing',
    'policy_learnware_v0.policy',
    'policy_learnware_v0.policy.bundle',
    'policy_learnware_v0.policy.loader',
    'policy_learnware_v0.policy.parity',
    'policy_learnware_v0.v02',
    'policy_learnware_v0.v02.artifacts',
    'policy_learnware_v0.v02.runtime',
    'policy_learnware_v0.v02.schemas',
    'server.repro_fpo_ppo_v02.anchor_binding',
    'server.repro_fpo_ppo_v02.handoff_contracts',
    'server.repro_fpo_ppo_v02.pool_acceptance',
    'server.repro_fpo_ppo_v02.provenance',
    'server.repro_fpo_ppo_v02.replay',
]
for name in requested:
    module = importlib.import_module(name)
    for symbol in getattr(module, '__all__', ()):
        if not isinstance(symbol, str) or not hasattr(module, symbol):
            raise RuntimeError(f'broken __all__ export: {{name}}.{{symbol}}')

first_party = sorted(
    name for name in sys.modules
    if name == 'policy_learnware_v0'
    or name.startswith('policy_learnware_v0.')
    or name == 'server.repro_fpo_ppo_v02'
    or name.startswith('server.repro_fpo_ppo_v02.')
)
files = {{
    name: getattr(sys.modules[name], '__file__', None)
    for name in first_party
}}
print(json.dumps({{'modules': first_party, 'files': files}}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)

    forbidden_components = {
        "audit",
        "benchmark",
        "championize",
        "implementation",
        "inventory",
        "queue_master",
        "runner",
        "training",
        "v0",
        "v01",
        "vendor",
    }
    for module_name in observed["modules"]:
        assert forbidden_components.isdisjoint(module_name.split(".")), module_name

    expected = {
        "policy_learnware_v0",
        "policy_learnware_v0.hashing",
        "policy_learnware_v0.policy",
        "policy_learnware_v0.policy.bundle",
        "policy_learnware_v0.policy.loader",
        "policy_learnware_v0.policy.parity",
        "policy_learnware_v0.v02",
        "policy_learnware_v0.v02.artifacts",
        "policy_learnware_v0.v02.runtime",
        "policy_learnware_v0.v02.schemas",
        "server.repro_fpo_ppo_v02",
        "server.repro_fpo_ppo_v02.anchor_binding",
        "server.repro_fpo_ppo_v02.handoff_contracts",
        "server.repro_fpo_ppo_v02.pool_acceptance",
        "server.repro_fpo_ppo_v02.provenance",
        "server.repro_fpo_ppo_v02.replay",
    }
    assert set(observed["modules"]) == expected
    for module_name, file_name in observed["files"].items():
        assert file_name, module_name
        assert Path(file_name).is_file(), (module_name, file_name)
