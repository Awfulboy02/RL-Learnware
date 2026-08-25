from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from repro_fpo_ppo_v02.provenance import ContractError
from repro_fpo_ppo_v02.tests.test_manifest_queue import _make_fake_vendor
from repro_fpo_ppo_v02.vendor import (
    inspect_vendor_directory,
    require_vendor_pythonpath_first,
)


class VendorContractTests(unittest.TestCase):
    def test_tree_digest_ignores_only_runtime_bytecode_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "vendor"
            _make_fake_vendor(root)
            initial = inspect_vendor_directory(root)
            cache = root / "wandb" / "__pycache__"
            cache.mkdir()
            (cache / "__init__.cpython-311.pyc").write_bytes(b"runtime-cache")
            self.assertEqual(inspect_vendor_directory(root), initial)

            (root / "dependency.py").write_text("PIN = 2\n", encoding="utf-8")
            changed = inspect_vendor_directory(root)
            self.assertNotEqual(changed["tree_digest"], initial["tree_digest"])

    def test_wandb_record_and_pythonpath_precedence_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "vendor"
            _make_fake_vendor(root)
            provenance = inspect_vendor_directory(root)
            require_vendor_pythonpath_first(
                provenance, environ={"PYTHONPATH": str(root)}
            )
            with self.assertRaisesRegex(ContractError, "not first"):
                require_vendor_pythonpath_first(
                    provenance,
                    environ={"PYTHONPATH": str(base) + ":" + str(root)},
                )

            (root / "wandb" / "__init__.py").write_text(
                '__version__ = "tampered"\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(ContractError, "differs from RECORD"):
                inspect_vendor_directory(root)


if __name__ == "__main__":
    unittest.main()
