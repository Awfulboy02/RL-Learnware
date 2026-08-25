from __future__ import annotations

from types import SimpleNamespace
import subprocess
import unittest
from unittest.mock import patch

from repro_fpo_ppo_v02.provenance import ContractError
from repro_fpo_ppo_v02.queue_master import (
    QueueMaster,
    parse_gpu_resource_snapshot,
)


class GpuIdleGateTests(unittest.TestCase):
    def test_snapshot_parser_is_exact_and_requires_requested_gpus(self) -> None:
        self.assertEqual(
            parse_gpu_resource_snapshot(
                "0, GPU-a, 3, 0\n1, GPU-b, 6599, 95\n",
                compute_output="GPU-b, 991\n",
                requested_gpus=("0", "1"),
            ),
            {
                "0": {
                    "uuid": "GPU-a",
                    "memory_used_mib": 3,
                    "utilization_percent": 0,
                    "compute_process_count": 0,
                },
                "1": {
                    "uuid": "GPU-b",
                    "memory_used_mib": 6599,
                    "utilization_percent": 95,
                    "compute_process_count": 1,
                },
            },
        )
        with self.assertRaisesRegex(ContractError, "omitted requested GPUs"):
            parse_gpu_resource_snapshot(
                "0, GPU-a, 3, 0\n", compute_output="", requested_gpus=("0", "1")
            )
        with self.assertRaisesRegex(ContractError, "four columns"):
            parse_gpu_resource_snapshot(
                "0, GPU-a, 3\n", compute_output="", requested_gpus=("0",)
            )
        no_apps = parse_gpu_resource_snapshot(
            "0, GPU-a, 3, 0\n",
            compute_output="No running processes found\n",
            requested_gpus=("0",),
        )
        self.assertEqual(no_apps["0"]["compute_process_count"], 0)

    def test_idle_gate_requires_two_consecutive_resource_probes(self) -> None:
        master = QueueMaster.__new__(QueueMaster)
        master.args = SimpleNamespace(
            wait_for_idle_gpus=True,
            gpus=("0", "1"),
            idle_max_memory_used_mib=512,
            idle_max_utilization_percent=5,
            resource_poll_seconds=15.0,
        )
        master.gpu_resource_snapshot = {}
        master.gpu_resource_probe_error = None
        master.gpu_idle_streak = {"0": 0, "1": 0}
        master.idle_gpu_cache = ()
        master.next_gpu_resource_probe = 0.0
        completed = subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="0, GPU-a, 6500, 90\n1, GPU-b, 3, 0\n",
            stderr="",
        )
        no_compute = subprocess.CompletedProcess(
            args=["nvidia-smi"], returncode=0, stdout="", stderr=""
        )
        with patch(
            "repro_fpo_ppo_v02.queue_master.subprocess.run",
            side_effect=[completed, no_compute, completed, no_compute],
        ) as run:
            self.assertEqual(master.idle_gpus(force=True), ())
            self.assertEqual(master.idle_gpus(force=True), ("1",))
            self.assertEqual(run.call_count, 4)

        busy = subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="0, GPU-a, 6500, 90\n1, GPU-b, 3, 0\n",
            stderr="",
        )
        compute = subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="GPU-b, 1234\n",
            stderr="",
        )
        with patch(
            "repro_fpo_ppo_v02.queue_master.subprocess.run",
            side_effect=[busy, compute],
        ):
            self.assertEqual(master.idle_gpus(force=True), ())
            self.assertEqual(master.gpu_idle_streak["1"], 0)


if __name__ == "__main__":
    unittest.main()
