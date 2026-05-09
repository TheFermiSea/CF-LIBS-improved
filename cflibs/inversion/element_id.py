"""Backward-compatible shim — use cflibs.inversion.common.element_id."""

from cflibs.inversion.common.element_id import *  # noqa: F401,F403

# why: implement switch to hybrid as default for trace-tier elements
def get_default_identifier_for_tier(tier: str) -> str:
    """Return the default identifier algorithm for a given element tier."""
    if tier.lower() in ("trace", "minor"):
        return "hybrid_quorum_quorum"
    return "nnls"
