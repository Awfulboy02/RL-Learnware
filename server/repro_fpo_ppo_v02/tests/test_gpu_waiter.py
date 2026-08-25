from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from repro_fpo_ppo_v02.gpu_waiter import (
    CLAIM_SCHEMA,
    CommandSpec,
    GpuResource,
    GpuWaiter,
    SharedClaim,
    WaiterConfig,
    WaiterError,
    collect_gpu_probe,
    load_json_object,
    parse_compute_app_rows,
    parse_gpu_rows,
    run_smoke,
    update_idle_streaks,
)


def _resource(
    index: str,
    *,
    memory: int = 0,
    utilization: int = 0,
    compute_pids: tuple[int, ...] = (),
) -> GpuResource:
    return GpuResource(
        index=index,
        uuid=f"GPU-{index}",
        memory_used_mib=memory,
        utilization_percent=utilization,
        compute_pids=compute_pids,
    )


def _config(
    root: Path,
    *,
    smoke: CommandSpec | None,
    claim_busy_action: str = "exit",
) -> WaiterConfig:
    return WaiterConfig(
        host_id="host-a",
        gpus=("0", "1"),
        nvidia_smi="/usr/bin/nvidia-smi",
        claim_dir=root / "queue-writer.claim",
        status_path=root / "status" / "host-a.json",
        claim_busy_action=claim_busy_action,
        launch=CommandSpec(argv=(sys.executable, "launch.py"), cwd=None),
        smoke=smoke,
        smoke_timeout_seconds=10.0,
        probe_timeout_seconds=10.0,
    )


class ProjectionParsingTests(unittest.TestCase):
    def test_gpu_rows_are_exact_and_require_every_requested_index(self) -> None:
        self.assertEqual(
            parse_gpu_rows(
                "0, GPU-a, 10, 1\n1, GPU-b, 6599, 95\n",
                requested_gpus=("0", "1"),
            ),
            {"0": ("GPU-a", 10, 1), "1": ("GPU-b", 6599, 95)},
        )
        with self.assertRaisesRegex(WaiterError, "omitted requested GPUs"):
            parse_gpu_rows("0, GPU-a, 10, 1\n", requested_gpus=("0", "1"))
        with self.assertRaisesRegex(WaiterError, "exactly four"):
            parse_gpu_rows("0, GPU-a, 10\n", requested_gpus=("0",))
        with self.assertRaisesRegex(WaiterError, "must not exceed 100"):
            parse_gpu_rows("0, GPU-a, 10, 101\n", requested_gpus=("0",))

    def test_compute_app_rows_are_fail_closed(self) -> None:
        self.assertEqual(parse_compute_app_rows(""), {})
        self.assertEqual(parse_compute_app_rows("No running processes found\n"), {})
        self.assertEqual(
            parse_compute_app_rows("GPU-a, 20\nGPU-a, 10\nGPU-b, 30\n"),
            {"GPU-a": (10, 20), "GPU-b": (30,)},
        )
        with self.assertRaisesRegex(WaiterError, "exactly two"):
            parse_compute_app_rows("GPU-a, 10, process\n")
        with self.assertRaisesRegex(WaiterError, "zero or duplicated"):
            parse_compute_app_rows("GPU-a, 10\nGPU-a, 10\n")

    def test_probe_combines_compute_apps_memory_and_utilization(self) -> None:
        calls: list[list[str]] = []

        def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess:
            calls.append(argv)
            self.assertNotIn("shell", kwargs)
            if argv[1].startswith("--query-gpu"):
                stdout = b"0, GPU-a, 12, 2\n1, GPU-b, 0, 0\n"
            else:
                stdout = b"GPU-a, 4321\n"
            return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr=b"")

        probe = collect_gpu_probe(
            nvidia_smi="/usr/bin/nvidia-smi",
            requested_gpus=("0", "1"),
            run_command=fake_run,
        )
        self.assertFalse(probe["0"].idle)
        self.assertEqual(probe["0"].compute_pids, (4321,))
        self.assertTrue(probe["1"].idle)
        self.assertEqual(len(calls), 2)

    def test_each_gpu_needs_its_own_two_consecutive_complete_probes(self) -> None:
        streaks = {"0": 0, "1": 0}
        streaks = update_idle_streaks(
            streaks, {"0": _resource("0"), "1": _resource("1", memory=513)}
        )
        self.assertEqual(streaks, {"0": 1, "1": 0})
        streaks = update_idle_streaks(
            streaks,
            {
                "0": _resource("0", compute_pids=(9,)),
                "1": _resource("1"),
            },
        )
        self.assertEqual(streaks, {"0": 0, "1": 1})
        streaks = update_idle_streaks(
            streaks, {"0": _resource("0"), "1": _resource("1")}
        )
        self.assertEqual(streaks, {"0": 1, "1": 2})


class SharedClaimTests(unittest.TestCase):
    def test_only_one_writer_wins_and_release_preserves_claim_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = SharedClaim(root / "writer.claim")
            second = SharedClaim(root / "writer.claim")
            ownership = first.try_acquire({"host_id": "host-a"})
            self.assertIsNotNone(ownership)
            assert ownership is not None
            self.assertIsNone(second.try_acquire({"host_id": "host-b"}))
            claim = load_json_object(ownership.metadata_path)
            self.assertEqual(claim["schema"], CLAIM_SCHEMA)
            self.assertEqual(claim["claim_token"], ownership.token)

            released = first.release(
                ownership,
                reason="smoke_failed",
                diagnostics={"returncode": 1},
            )
            self.assertFalse(first.path.exists())
            self.assertTrue(released.is_dir())
            released_claim = load_json_object(released / "claim.json")
            self.assertEqual(released_claim["state"], "released")
            self.assertEqual(released_claim["release_reason"], "smoke_failed")
            self.assertEqual(
                released_claim["release_diagnostics"], {"returncode": 1}
            )
            self.assertIsNotNone(second.try_acquire({"host_id": "host-b"}))

    def test_claim_token_prevents_foreign_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            claim = SharedClaim(root / "writer.claim")
            ownership = claim.try_acquire({"host_id": "host-a"})
            assert ownership is not None
            foreign = type(ownership)(
                path=ownership.path,
                token="f" * 32,
                metadata_path=ownership.metadata_path,
            )
            with self.assertRaisesRegex(WaiterError, "another token"):
                claim.release(foreign, reason="bad", diagnostics={})
            self.assertTrue(claim.path.is_dir())

    def test_existing_incomplete_claim_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "writer.claim"
            path.mkdir()
            claim = SharedClaim(path)
            self.assertIsNone(claim.try_acquire({"host_id": "host-b"}))
            self.assertEqual(claim.read_if_present()["state"], "incomplete_claim")


class WaiterFlowTests(unittest.TestCase):
    def test_smoke_failure_records_diagnostics_and_releases_without_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            smoke = CommandSpec(argv=(sys.executable, "smoke.py"), cwd=None)
            config = _config(root, smoke=smoke)
            probes = 0

            def collect(**_kwargs: object) -> dict[str, GpuResource]:
                nonlocal probes
                probes += 1
                return {"0": _resource("0"), "1": _resource("1", memory=900)}

            diagnostics = {"returncode": 7, "stderr": {"tail": "failed"}}

            def smoke_runner(
                _command: CommandSpec, **_kwargs: object
            ) -> tuple[bool, dict[str, object]]:
                return False, diagnostics

            def must_not_exec(*_args: object) -> None:
                raise AssertionError("launch must not run after failed smoke")

            waiter = GpuWaiter(
                config,
                collect_probe=collect,
                smoke_runner=smoke_runner,
                execve=must_not_exec,
                sleeper=lambda _seconds: None,
            )
            self.assertEqual(waiter.run(), 20)
            self.assertEqual(probes, 2)
            self.assertFalse(config.claim_dir.exists())
            released = list(root.glob("queue-writer.claim.released-*"))
            self.assertEqual(len(released), 1)
            status = load_json_object(config.status_path)
            self.assertEqual(status["state"], "smoke_failed_claim_released")
            self.assertEqual(status["smoke_diagnostics"], diagnostics)

    def test_successful_smoke_reaches_exec_with_claim_and_environment(self) -> None:
        class ExecObserved(BaseException):
            pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            smoke = CommandSpec(argv=(sys.executable, "smoke.py"), cwd=None)
            config = _config(root, smoke=smoke)

            def collect(**_kwargs: object) -> dict[str, GpuResource]:
                return {"0": _resource("0"), "1": _resource("1", utilization=50)}

            observed: dict[str, object] = {}

            def execve(
                program: str, argv: tuple[str, ...], environment: dict[str, str]
            ) -> None:
                observed.update(program=program, argv=argv, environment=environment)
                raise ExecObserved()

            waiter = GpuWaiter(
                config,
                collect_probe=collect,
                smoke_runner=lambda *_args, **_kwargs: (
                    True,
                    {"returncode": 0},
                ),
                execve=execve,
                sleeper=lambda _seconds: None,
            )
            with self.assertRaises(ExecObserved):
                waiter.run()
            self.assertEqual(observed["program"], sys.executable)
            environment = observed["environment"]
            assert isinstance(environment, dict)
            self.assertEqual(environment["PLW_V02_WAITER_ELIGIBLE_GPUS"], "0")
            self.assertEqual(environment["PLW_V02_WAITER_HOST"], "host-a")
            self.assertEqual(environment["PYTHONDONTWRITEBYTECODE"], "1")
            claim = load_json_object(config.claim_dir / "claim.json")
            self.assertEqual(claim["state"], "launch_exec_pending")
            self.assertEqual(
                claim["claim_token"], environment["PLW_V02_WAITER_CLAIM_TOKEN"]
            )

    def test_loser_exits_without_modifying_winners_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root, smoke=None, claim_busy_action="exit")
            winner = SharedClaim(config.claim_dir)
            ownership = winner.try_acquire({"host_id": "host-b", "sentinel": 1})
            assert ownership is not None
            before = ownership.metadata_path.read_bytes()

            waiter = GpuWaiter(
                config,
                collect_probe=lambda **_kwargs: {
                    "0": _resource("0"),
                    "1": _resource("1"),
                },
                sleeper=lambda _seconds: None,
            )
            self.assertEqual(waiter.run(), 3)
            self.assertEqual(ownership.metadata_path.read_bytes(), before)
            status = load_json_object(config.status_path)
            self.assertEqual(status["state"], "claim_busy")
            self.assertEqual(status["existing_claim"]["host_id"], "host-b")

    def test_launch_exec_failure_is_released_and_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root, smoke=None)
            waiter = GpuWaiter(
                config,
                collect_probe=lambda **_kwargs: {
                    "0": _resource("0"),
                    "1": _resource("1", memory=900),
                },
                execve=lambda *_args: (_ for _ in ()).throw(
                    OSError("synthetic exec failure")
                ),
                sleeper=lambda _seconds: None,
            )
            self.assertEqual(waiter.run(), 21)
            self.assertFalse(config.claim_dir.exists())
            released = list(root.glob("queue-writer.claim.released-*"))
            self.assertEqual(len(released), 1)
            claim = load_json_object(released[0] / "claim.json")
            self.assertEqual(claim["release_reason"], "launch_exec_failed")


class SmokeDiagnosticsTests(unittest.TestCase):
    def test_smoke_uses_argv_without_shell_and_records_output_hashes(self) -> None:
        command = CommandSpec(argv=(sys.executable, "smoke.py"), cwd=None)

        def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess:
            self.assertEqual(argv, list(command.argv))
            self.assertNotIn("shell", kwargs)
            return subprocess.CompletedProcess(
                argv, returncode=4, stdout=b"out", stderr=b"err"
            )

        passed, diagnostics = run_smoke(
            command,
            timeout_seconds=3.0,
            environment={"SAFE": "1"},
            run_command=fake_run,
            monotonic=lambda: 10.0,
        )
        self.assertFalse(passed)
        self.assertEqual(diagnostics["returncode"], 4)
        self.assertEqual(diagnostics["stdout"]["byte_count"], 3)
        self.assertEqual(diagnostics["stderr"]["tail"], "err")

    def test_smoke_exec_error_is_a_recorded_failure(self) -> None:
        command = CommandSpec(argv=(sys.executable, "smoke.py"), cwd=None)

        def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess:
            raise OSError("smoke executable became unavailable")

        passed, diagnostics = run_smoke(
            command,
            timeout_seconds=3.0,
            environment={},
            run_command=fake_run,
        )
        self.assertFalse(passed)
        self.assertIsNone(diagnostics["returncode"])
        self.assertEqual(diagnostics["execution_error"]["type"], "OSError")


if __name__ == "__main__":
    unittest.main()
