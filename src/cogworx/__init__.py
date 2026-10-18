"""cog-worx — the cognitive-architecture framework.

A framework, not a product: the loop, the memory, the verification, the perception, and the QoL
machinery, out of the box. The public API is re-exported from this package root as the seams harden;
for now it stays intentionally thin — import from the subpackages (``cogworx.loop``,
``cogworx.recall``, ``cogworx.verification``, ...).

See ``wiki/CANON.md`` (law) and ``wiki/ROADMAP.md`` (build order).
"""

from __future__ import annotations

from cogworx._version import __version__

__all__ = ["__version__"]
