"""Model provider adapters (CANON S4).

Each sub-module implements the ``Model`` seam for a concrete LLM provider.
``config`` is the shared provider-config surface; every adapter depends on it —
never duplicate ``PriceTable``/``ProviderConfig`` per-provider.

CANON cross-references
----------------------
- S4  Model-agnostic: provider selectable per agent; thin ``Model`` seam.
- S9  Structure over prompting: capabilities declared statically; never probe the model.
- S11 Cost bounded structurally: ``PriceTable.cost_usd`` feeds pre-call budget guards.
"""

from __future__ import annotations

from cogworx.model.providers.claude import ClaudeModel
from cogworx.model.providers.config import PriceTable, ProviderConfig
from cogworx.model.providers.openai_compat import (
    DEEPSEEK_CAPABILITIES,
    OLLAMA_CAPABILITIES,
    OPENAI_CAPABILITIES,
    OpenAICompatModel,
)

__all__ = [
    "DEEPSEEK_CAPABILITIES",
    "OLLAMA_CAPABILITIES",
    "OPENAI_CAPABILITIES",
    "ClaudeModel",
    "OpenAICompatModel",
    "PriceTable",
    "ProviderConfig",
]
# fmt: skip  # RUF022: dunder-uppercase before Title-case is intentional grouping
