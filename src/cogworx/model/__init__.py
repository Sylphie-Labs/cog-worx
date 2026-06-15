"""Model layer (CANON S4): the model-agnostic ``Model`` seam and its value types.

``BudgetGuardedModel`` (``cogworx.model.guarded``) is intentionally NOT re-exported here to avoid
a circular import: ``cogworx.cost.budget`` imports ``cogworx.model.base.Usage``, and if this
``__init__`` were to import ``guarded`` (which in turn imports ``cogworx.cost.budget``), the cycle
would detonate on any ``import cogworx.cost`` that triggers ``model/__init__``.  Callers that need
``BudgetGuardedModel`` import it directly from ``cogworx.model.guarded``.
"""

from __future__ import annotations

from cogworx.model.base import (
    ChatMessage,
    Model,
    ModelCapabilities,
    ModelResponse,
    ModelTier,
    ToolCall,
    ToolSpec,
    Usage,
)

__all__ = [
    "ChatMessage",
    "Model",
    "ModelCapabilities",
    "ModelResponse",
    "ModelTier",
    "ToolCall",
    "ToolSpec",
    "Usage",
]
