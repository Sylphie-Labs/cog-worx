"""The verification oracle seam + registry (CANON S4, S8, S9).

An oracle is the EXTERNAL, STRUCTURAL ground-truth signal S9 demands: it turns a
thesis-plus-experiment into a typed :class:`~cogworx.verification.contracts.Verdict` — NOT by
asking the model how confident it is. It is a DISTINCT seam from
:class:`~cogworx.capability.base.Capability` (a capability is a permission-tiered, ``Any``-returning
tool; an oracle returns an epistemically-privileged typed verdict) — but it COMPOSES the capability
registry for tool-backed checks (an executable oracle dispatches a real tool via ``ctx.dispatch``
and types its verdict ``source="tool"``). This mirrors "3.1 context composes, not subsumes, 2.6".
Generalized from tess ``oracle.py``.

:class:`OracleRegistry` resolves the right oracle by ``(completion_criterion, problem_type)`` with
exact -> wildcard -> always-on-fallback precedence and NEVER raises (S8 graceful degradation): the
fallback is injected at construction, so every call resolves to *something* — there is no
"unregistered" failure mode. The registry itself satisfies the :class:`Oracle` protocol, so a stage
can be handed a registry or a bare oracle interchangeably (the tess pattern). The concrete fallback
(the LLM-judge oracle) and the executable oracles land in Pod 4.1; Pod 4.0 only fixes the seam and
the always-wired resolution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from cogworx.verification.contracts import OracleFrame, Thesis, Verdict

if TYPE_CHECKING:
    # Annotation-only: a verification oracle takes a StageContext to dispatch tool-backed checks.
    # Importing it only under TYPE_CHECKING keeps the seam off the loop's import cycle (loop.stage
    # never imports verification).
    from cogworx.loop.stage import StageContext


@runtime_checkable
class Oracle(Protocol):
    """Turn a thesis-plus-experiment into a typed :class:`Verdict` via an external check.

    ``evaluate`` is async so independent oracles can run concurrently (e.g. a real oracle + a judge
    inside a calibration-tracking wrapper) — the tess rationale.

    CONTRACT (S9, F1): the returned ``Verdict.source`` MUST reflect THIS oracle's structural nature
    (executable / deterministic -> ``"tool"`` / ``"system"``; model-judge -> ``"inference"``) and
    MUST NEVER be taken from model output. See :attr:`Verdict.source`.
    """

    async def evaluate(
        self, *, frame: OracleFrame, thesis: Thesis, ctx: StageContext
    ) -> Verdict: ...


class OracleRegistry:
    """Resolve the oracle for a frame by ``(criterion, problem_type)``; never raises (S8).

    Precedence:
      1. exact ``(completion_criterion, problem_type)``
      2. wildcard ``(completion_criterion, "*")`` — any ``problem_type`` with this criterion
      3. the always-wired ``fallback`` (injected at construction)

    Construct with the fallback (e.g. the LLM-judge oracle, Pod 4.1) already wired so there is no
    "unregistered" failure mode — resolution is total. Concrete oracles are added via
    :meth:`register` as they land.
    """

    _WILDCARD: str = "*"

    def __init__(self, *, fallback: Oracle) -> None:
        self._fallback = fallback
        self._exact: dict[tuple[str, str], Oracle] = {}
        self._criterion_wildcards: dict[str, Oracle] = {}

    def register(self, *, completion_criterion: str, problem_type: str, oracle: Oracle) -> None:
        """Register an oracle for an exact ``(criterion, problem_type)`` pair, or — when
        ``problem_type == "*"`` — a criterion-level wildcard (any problem_type with this criterion).
        """
        if problem_type == self._WILDCARD:
            self._criterion_wildcards[completion_criterion] = oracle
        else:
            self._exact[(completion_criterion, problem_type)] = oracle

    def resolve(self, *, completion_criterion: str, problem_type: str) -> Oracle:
        """Return the oracle for this frame — exact, then wildcard, then fallback (total, S8)."""
        exact = self._exact.get((completion_criterion, problem_type))
        if exact is not None:
            return exact
        wildcard = self._criterion_wildcards.get(completion_criterion)
        if wildcard is not None:
            return wildcard
        return self._fallback

    async def evaluate(self, *, frame: OracleFrame, thesis: Thesis, ctx: StageContext) -> Verdict:
        """Resolve + delegate. Satisfies :class:`Oracle`, so the registry is drop-in wherever a
        stage expects a bare oracle — it need not know whether it holds a registry."""
        oracle = self.resolve(
            completion_criterion=frame.completion_criterion,
            problem_type=frame.problem_type,
        )
        return await oracle.evaluate(frame=frame, thesis=thesis, ctx=ctx)


__all__ = [
    "Oracle",
    "OracleRegistry",
]
