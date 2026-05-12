"""Multi-armed bandit allocators for parameter sweeps.

See :mod:`cflibs.bandit.thompson_allocator` for the Thompson-sampling
implementation used by ``scripts/parameter_sweep.py --bandit N``.
"""

from cflibs.bandit.thompson_allocator import ThompsonAllocator

__all__ = ["ThompsonAllocator"]
