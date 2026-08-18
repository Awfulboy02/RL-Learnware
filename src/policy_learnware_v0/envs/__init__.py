"""Environment adapters.

MuJoCo Playground is deliberately imported lazily by
``MujocoPlaygroundEnvAdapter`` so metadata and unit tests remain usable on a
CPU-only machine.
"""

from .base import EnvAdapter, SyntheticEnvAdapter
from .factory import make_env_adapter
from .mujoco_playground import MujocoPlaygroundEnvAdapter

__all__ = [
    "EnvAdapter",
    "MujocoPlaygroundEnvAdapter",
    "SyntheticEnvAdapter",
    "make_env_adapter",
]
