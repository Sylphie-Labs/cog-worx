"""PriceTable-backed cost estimator — makes the S11 USD ceiling bite for the live GATE run
(Pod 4.4-live L4).

``BudgetGuardedModel`` (``cogworx.model.guarded``) already enforces a pre-call ``BudgetGuard``
ceiling on every ``Model.complete`` call — but per its own docstring (``guarded.py`` ~L70), a
``max_usd`` ceiling with no ``estimator`` raises at construction: an unpriced guard is theater, not
a ceiling (S11). This module supplies that estimator for the live-run composition root.

Why this isn't just ``cogworx.model.registry._default_estimator``
-------------------------------------------------------------------
``registry.build_model`` already assembles ``StructuredOutputModel(BudgetGuardedModel(adapter,
guard, estimator=...))`` from a ``ModelSpec`` + ``ProviderConfig``, and its private
``_default_estimator`` computes the identical projection this module needs. It is not reused
directly for two reasons specific to this package's composition root (see
``cogworx.eval._live``'s docstring — the private, ``ProviderConfig``-only wiring path):

1. ``_default_estimator`` is unexported (module-private) and is coupled to a concrete adapter
   ``Model`` (it calls ``adapter.count_tokens``) plus a full ``ProviderConfig``. This package's
   roles resolve to bare ``ProviderConfig`` objects (``GateRunSettings`` — see ``settings.py``);
   there is no ``ModelSpec`` and no adapter instance until the arm-executor wiring assembles one.
2. ``build_model``'s composition additionally wraps the guarded model in ``StructuredOutputModel``
   (the S4/S8/S9 degradation ladder). The live-run arm executors need the raw budgeted ``Model``
   itself, not the ladder — that wrapping decision belongs to whatever assembles the arm executor,
   not to the budget seam.

Both this module's ``make_cost_estimator`` and ``registry._default_estimator`` therefore express the
SAME conservative projection contract, over different inputs; neither re-derives the $/MTok
arithmetic — both delegate to ``PriceTable.cost_usd`` (CANON S11: one source of truth for the price
math).

Conservative-by-construction (S11-honest over-estimate)
---------------------------------------------------------
Actual completion length is unknown pre-call, so the projection assumes the FULL
``max_output_tokens`` on every call. This can only over-estimate cost, which means the ceiling can
only trip EARLY (refuse a call that would have been affordable) — never LATE (let an over-budget
call through). ``BudgetGuard.record`` subsequently reconciles the running total against the real
``Usage`` once the call completes, so the projection never contaminates the actual spend tally.

Contract changelog (CANON §6.1):
  - 2026-07-02 (Pod 4.4-live L4): initial — ``make_cost_estimator`` / ``build_budgeted_model``. New
    module; no existing callers. Additive new public surface only.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from cogworx.cost.budget import BudgetGuard
from cogworx.model.base import ChatMessage, Model, ModelTier, Usage
from cogworx.model.guarded import BudgetGuardedModel
from cogworx.model.providers.config import PriceTable

__all__ = ["build_budgeted_model", "make_cost_estimator"]


def make_cost_estimator(
    price_table: PriceTable,
    *,
    max_output_tokens: int,
    count_tokens: Callable[[str], int],
) -> Callable[[Sequence[ChatMessage], ModelTier], float]:
    """Build a conservative pre-call USD estimator for ``BudgetGuardedModel`` (CANON S11).

    The returned callable projects: input tokens = ``count_tokens`` summed over every message's
    ``content``; output tokens = the FULL ``max_output_tokens`` (the true completion length is
    unknown pre-call, so the max is the only value that can never under-estimate — see the module
    docstring). The projected ``Usage`` is converted to USD via ``price_table.cost_usd`` — the same
    arithmetic every other caller uses, never re-derived here.

    Parameters
    ----------
    price_table:
        The role's resolved ``PriceTable`` (CANON S11 — must not be ``ZERO_PRICE_TABLE``; that
        refusal is ``settings.assert_priced``'s job, not this factory's).
    max_output_tokens:
        The ceiling on completion length this role's ``ProviderConfig`` will pass to the adapter.
    count_tokens:
        A synchronous token-counting callable (typically the assembled adapter's own
        ``Model.count_tokens`` — a zero-cost utility, never itself guarded; CANON S1).
    """

    def estimate(messages: Sequence[ChatMessage], tier: ModelTier) -> float:
        prompt_tokens = sum(count_tokens(message.content) for message in messages)
        projected_usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=max_output_tokens)
        return price_table.cost_usd(projected_usage, tier)

    return estimate


def build_budgeted_model(
    inner: Model,
    *,
    guard: BudgetGuard,
    price_table: PriceTable,
    max_output_tokens: int,
) -> BudgetGuardedModel:
    """Wrap *inner* in a ``BudgetGuardedModel`` wired with a ``make_cost_estimator`` projection.

    Every call through the returned model is checked against *guard*'s ceiling BEFORE *inner* is
    invoked (CANON S11 pre-call). Callers that build multiple arm executors from the same *inner*
    model and *guard* get transitive budgeting for free — no per-executor guard wiring needed.
    """
    estimator = make_cost_estimator(
        price_table,
        max_output_tokens=max_output_tokens,
        count_tokens=inner.count_tokens,
    )
    return BudgetGuardedModel(inner, guard, estimator=estimator)
