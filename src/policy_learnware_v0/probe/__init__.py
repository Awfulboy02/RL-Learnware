"""Candidate-independent random probing and immutable episode datasets.

The collector depends on environment adapters, so keep that public convenience
export lazy.  Dataset-only consumers (including the v0.3 attribution path) must
remain importable without MuJoCo or accelerator runtime modules.
"""

from typing import Any

from .dataset import DatasetManifest, EpisodeDataset
from .gaussian import GaussianRandomProbe
from .seed_plan import EpisodeSeeds, SeedPlan


def __getattr__(name: str) -> Any:
    if name == "collect_probe_episodes":
        from .collector import collect_probe_episodes

        return collect_probe_episodes
    raise AttributeError(name)

__all__ = [
    "DatasetManifest",
    "EpisodeDataset",
    "EpisodeSeeds",
    "GaussianRandomProbe",
    "SeedPlan",
    "collect_probe_episodes",
]
