from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

from policy_learnware_v0 import io as artifact_io
from policy_learnware_v0.io import ArtifactExistsError, atomic_write_bytes


def test_immutable_publication_has_one_atomic_winner(
    tmp_path: Path, monkeypatch
) -> None:
    destination = tmp_path / "immutable.bin"
    worker_count = 8
    at_link = threading.Barrier(worker_count)
    real_link = artifact_io.os.link

    def synchronized_link(source, target, *args, **kwargs):
        at_link.wait(timeout=5.0)
        return real_link(source, target, *args, **kwargs)

    monkeypatch.setattr(artifact_io.os, "link", synchronized_link)

    def publish(index: int) -> tuple[str, bytes]:
        payload = f"writer-{index}".encode("ascii")
        try:
            atomic_write_bytes(destination, payload)
        except ArtifactExistsError:
            return "lost", payload
        return "won", payload

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        outcomes = list(executor.map(publish, range(worker_count)))

    winners = [payload for status, payload in outcomes if status == "won"]
    assert len(winners) == 1
    assert destination.read_bytes() == winners[0]
    assert not list(tmp_path.glob(".*.tmp"))
