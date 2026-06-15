"""MemoryInjector — stamps queries, fans out to RecallStack, assembles token-budget context.

Pod 2.6.  Dependency contract (CANON D3): injection → recall/substrate.  NO runtime import
from cogworx.model or cogworx.runtime.  The model reference is duck-typed via
getattr(model, "count_tokens", None) — no isinstance check, no coupling to Model (S4/S8).

Contract changelog (CANON §6.1 — additive changes; no sign-off required):
  2026-06-12  Pod 3.1b CF-B  inject() now structurally excludes fused results whose claim
              status is ``"defeasibly-defeated"`` before assembly.  The exclusion is skipped
              when ``MemoryPolicy.include_defeated=True``.  Excluded count is reported in
              ``InjectedMemory.defeated_excluded``.  Zero model calls — pure structural check
              on ``claim.status`` (S9).
  2026-06-12  Pod 3.1 Fix 4a  defeated-claim filter now keys off isinstance(r.item, ScoredClaim)
              alone (was: r.kind=="claim" AND isinstance). Closes kind="episode", item=ScoredClaim
              mis-kind vector. CARRY-FORWARD: any new FusedResult.item type carrying a Claim
              must route through this filter (documented; reviewer of that future change owns it).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from cogworx.injection.policy import DEFAULT_MEMORY_POLICY, InjectedMemory, MemoryPolicy
from cogworx.recall.assembly import TokenCounter, approx_tokens, assemble
from cogworx.recall.query import RecallQuery
from cogworx.recall.results import FusedResult
from cogworx.recall.stack import RecallStack
from cogworx.substrate.entity_kg import ScoredClaim
from cogworx.substrate.latent import LatentStore

if TYPE_CHECKING:
    pass  # no model import needed — duck-typed via getattr

__all__ = [
    "MemoryInjector",
    "resolve_token_counter",
]


def resolve_token_counter(model: object | None) -> TokenCounter:
    """Return a token-counting callable from *model*, falling back to :func:`approx_tokens`.

    Pure duck-type check — no runtime import of Model (S4/S8).  If *model* is ``None`` or
    does not expose a callable ``count_tokens`` attribute, :func:`approx_tokens` is returned.
    """
    if model is None:
        return approx_tokens
    counter = getattr(model, "count_tokens", None)
    if callable(counter):
        return cast(TokenCounter, counter)
    return approx_tokens


class MemoryInjector:
    """Orchestrate a single memory-injection pass for one turn.

    Construction:
        *stack* is the wired :class:`~cogworx.recall.stack.RecallStack`.  Callers that have no
        recall stack should not construct a MemoryInjector — the stack is required, not optional.

    Args:
        stack: Wired recall stack (required).
        latent_store: Optional :class:`~cogworx.substrate.latent.LatentStore`; when supplied,
            admitted latent chunks are recorded via ``record_use`` after assembly (Pod 2.2).
        model: Optional model object; if it exposes a callable ``count_tokens`` attribute that
            attribute is used for token counting.  No runtime import of cogworx.model (S4/S8).
        clock: Callable returning the current UTC datetime; used to stamp ``query.as_of`` when the
            caller omits it (D14).  Defaults to ``datetime.now(UTC)``.
    """

    def __init__(
        self,
        stack: RecallStack,
        *,
        latent_store: LatentStore | None = None,
        model: object | None = None,  # duck-typed for count_tokens; no runtime Model import
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._stack = stack
        self._latent_store = latent_store
        self._model_or_none = model
        self._clock = clock

    async def inject(
        self,
        query: RecallQuery,
        *,
        policy: MemoryPolicy = DEFAULT_MEMORY_POLICY,
    ) -> InjectedMemory:
        """Run a full injection pass: recall → filter → assemble → record_use → InjectedMemory.

        Steps:
            1. Stamp ``query.as_of`` with the clock if not already set (D14).
            2. Fan out over the recall stack.
            2b. Structurally exclude fused results whose claim
                ``status=="defeasibly-defeated"`` (Pod 3.1b CF-B, S9 — pure structural
                check, zero model calls).  Skipped when ``policy.include_defeated=True``.
                Count stored in ``InjectedMemory.defeated_excluded``.
            3. Resolve the token counter from the injected model (duck-typed).
            4. Assemble a token-budget-bounded context with U-fold ordering.
            5. Compute ``kinds_present`` and ``missing_kinds`` against ``policy.required_kinds``.
            6. Determine status: ``"below_floor"`` when required kinds are absent, else ``"ok"``.
            7. Record use for admitted latent chunks via LatentStore.record_use (best-effort).

        Args:
            query: Recall query.  ``as_of`` may be ``None``; it will be stamped by this method.
            policy: Memory policy governing budget, assembly floor, required kinds, and whether
                defeasibly-defeated claims are included.

        Returns:
            :class:`~cogworx.injection.policy.InjectedMemory` with full diagnostic metadata.
        """
        # Step 1: stamp as_of if absent (D14 — callers should not need to track wall time)
        if query.as_of is None:
            stamped_query = query.model_copy(update={"as_of": self._clock()})
        else:
            stamped_query = query

        # Step 2: fan out over all recall channels
        outcome = await self._stack.recall(stamped_query)

        # Step 2b: structurally exclude defeasibly-defeated claims (Pod 3.1b CF-B, S9).
        # Pure status-field check — zero model calls.  Non-claim results pass through unchanged.
        # When include_defeated=True this filter is skipped (escape hatch for history stages).
        #
        # KNOWN GAP (CF-3.1-Episode-echo): an Episode whose content echoes a now-defeated fact
        # PASSES this filter because it carries no claim.status — its item is an Episode, not a
        # ScoredClaim, so isinstance(r.item, ScoredClaim) is False and it passes through
        # unchanged.  Whether to suppress or annotate such episodes is a deferred design question
        # (ticket CF-3.1-Episode-echo).  Current behavior is PINNED by test (Pod 3.1 Fix 4 scope
        # test in test_injection_defeated_2_7cfb.py).
        defeated_excluded = 0
        if not policy.include_defeated:
            filtered: list[FusedResult] = []
            for r in outcome.results:
                # Contract-CARRY-FORWARD: any NEW FusedResult.item type capable of carrying
                # a Claim MUST route through this filter (reviewer of that future change owns it).
                # S9: pure structural isinstance check — zero model calls.
                if isinstance(r.item, ScoredClaim) and r.item.claim.status == "defeasibly-defeated":
                    defeated_excluded += 1
                    continue
                filtered.append(r)
            results_to_assemble: tuple[FusedResult, ...] = tuple(filtered)
        else:
            results_to_assemble = outcome.results

        # Step 3: resolve token counter
        count_tokens = resolve_token_counter(self._model_or_none)

        # Step 4: assemble token-budget context
        assembled = assemble(
            results_to_assemble,
            budget=policy.token_budget,
            count_tokens=count_tokens,
            min_per_kind=policy.min_per_kind,
        )

        # Step 5: compute kinds_present and missing_kinds
        kinds_present: tuple[str, ...] = tuple(sorted(set(c.kind for c in assembled.chunks)))
        missing_kinds: tuple[str, ...] = tuple(
            sorted(k for k in policy.required_kinds if k not in set(kinds_present))
        )

        # Step 6: determine status
        status = "below_floor" if missing_kinds else "ok"

        # Step 7: record_use for admitted latent chunks (best-effort, S8)
        count = 0
        record_use_error: str | None = None

        latent_ids = [c.key.removeprefix("latent:") for c in assembled.chunks if c.kind == "latent"]
        if latent_ids and self._latent_store is not None:
            try:
                count = await self._latent_store.record_use(latent_ids)
            except Exception as exc:
                record_use_error = repr(exc)
                count = 0

        return InjectedMemory(
            context=assembled,
            status=status,
            missing_kinds=missing_kinds,
            kinds_present=kinds_present,
            channel_status=outcome.channel_status,
            latent_uses_recorded=count,
            record_use_error=record_use_error,
            defeated_excluded=defeated_excluded,
        )
