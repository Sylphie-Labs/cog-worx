"""Model registry and build_model factory (CANON S4, S6, S11).

This module provides two things:

1. ``ModelRegistry`` — a code-registered, cold-resume-safe store of named ``Model`` instances,
   mirroring ``PathwayRegistry`` in style (S6 durable resume: the registry is reconstituted from
   code at startup; deterministic lookup, loud error on miss — never silent).

2. ``ModelSpec`` — a frozen, TOML/env-loadable descriptor that names a provider adapter and its
   ``ProviderConfig`` (+ optional capabilities override for OpenAI-compat targets).

3. ``build_model(spec, guard)`` — the composition factory that ASSEMBLES the full model stack in
   the load-bearing order mandated by the architecture:

   .. code-block:: text

       StructuredOutputModel          ← outermost  (S4/S8/S9 structured-output ladder)
         └─ BudgetGuardedModel        ← middle      (S11 pre-call cost ceiling)
              └─ ClaudeModel          ← innermost   (S4 provider adapter)
                 / OpenAICompatModel

CANON cross-references
----------------------
- S4   Model-agnostic: provider is selectable per agent; the stack is transparent to callers.
- S6   Durable exactly-once resume: the registry is code-registered (deterministic, cold-resume-
       safe).  An unregistered profile raises ``ModelRegistryError`` — never a silent None.
- S8   Graceful degradation: the ``StructuredOutputModel`` ladder degrades when a provider lacks a
       capability; the guard degrades (``BudgetExceededError``) rather than failing silently.
- S9   Structure over prompting: capabilities are declared STATICALLY in ``ModelSpec``; they are
       NEVER probed at runtime.
- S11  Cost bounded structurally: ``BudgetGuardedModel`` enforces the ``BudgetGuard`` as a PRE-CALL
       ceiling so the model cannot be invoked once the budget is exhausted.

Per-drive-scoped guard seam
--------------------------
``BudgetGuard`` is a **drive-scoped** object: it accumulates cost across the calls within ONE
drive segment and its ``max_usd`` / ``max_calls`` ceilings apply to that segment.  A
paused/resumed/timer-fired run receives a fresh guard per segment (per-run cumulative ceilings are
CF-3.0-B).  ``ModelRegistry`` stores ASSEMBLED ``Model`` objects — which means the guard passed to
``build_model`` is baked into the assembled model at construction time.

The clean seam is therefore at the ENGINE layer, not the registry:

- The registry stores ``ModelSpec`` objects (via ``register_spec`` / ``resolve_spec``).  It also
  supports registering pre-assembled ``Model`` objects (``register`` / ``resolve``) for cases
  where the caller manages assembly outside the factory.
- When the engine starts a drive segment it calls
  ``build_model(registry.resolve_spec(profile), guard=budget)``
  to produce a fresh, drive-scoped assembled model.  Each drive segment receives its own
  ``BudgetGuard`` instance while reusing the spec (per-run cumulative ceilings are CF-3.0-B).

This design is surfaced here rather than guessed.  The engine integration (Task 3.x — ``Engine``
wiring) MUST adopt the ``resolve_spec`` + ``build_model`` pattern to ensure per-drive guard
isolation.  Registering a pre-assembled model (``register`` / ``resolve``) is valid for tests or
for cases where the guard is intentionally shared, but is NOT the recommended path for production
engine runs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from cogworx.cost.budget import BudgetGuard
from cogworx.model.base import ChatMessage, Model, ModelCapabilities, ModelTier, Usage
from cogworx.model.guarded import BudgetGuardedModel
from cogworx.model.ladder import DEFAULT_STRUCTURED_OUTPUT_LADDER, LadderRung, StructuredOutputModel
from cogworx.model.providers.config import PriceTable, ProviderConfig

__all__ = [
    "ModelRegistry",
    "ModelRegistryError",
    "ModelSpec",
    "build_model",
]


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class ModelRegistryError(Exception):
    """Raised on a duplicate registration or a lookup miss.

    A miss during cold resume is fatal: the same model profiles must be registered at startup.
    This error is always loud — never a silent ``None`` return (S6).
    """


# ---------------------------------------------------------------------------
# ModelSpec — frozen, TOML/env-loadable descriptor
# ---------------------------------------------------------------------------

ProviderName = Literal["claude", "openai_compat"]


class ModelSpec(BaseModel):
    """Frozen descriptor for a model stack (CANON S4, S9, S11).

    Attributes
    ----------
    provider:
        Which adapter to instantiate.  ``"claude"`` → ``ClaudeModel``; ``"openai_compat"`` →
        ``OpenAICompatModel``.
    config:
        Shared provider configuration (API key, model IDs, pricing, timeouts).
    capabilities:
        Static capability declaration.  For ``"claude"`` this is ignored — ``ClaudeModel``
        declares its own capabilities as a class attribute (S9 invariant).  For
        ``"openai_compat"`` this is REQUIRED when ``config.base_url`` is not ``None``
        (DeepSeek / Ollama / custom endpoint); pass ``None`` only for the OpenAI production
        endpoint (``base_url=None``), which defaults to ``OPENAI_CAPABILITIES``.
    ladder:
        Override the ``StructuredOutputModel`` rung ladder.  ``None`` → use
        ``DEFAULT_STRUCTURED_OUTPUT_LADDER``.  Supplied as a tuple so ``ModelSpec`` stays
        serialisable; override at construction time for custom degradation behaviour.

    Notes
    -----
    ``ModelSpec`` is a plain frozen pydantic model — NOT a ``BaseSettings`` subclass.
    Environment-variable loading is the caller's responsibility (e.g. a settings object in
    ``cogworx.adapters.config``).  This keeps the type composable and unit-testable without
    env state.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    provider: ProviderName
    config: ProviderConfig
    capabilities: ModelCapabilities | None = None
    ladder: tuple[LadderRung, ...] | None = None


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------


class ModelRegistry:
    """A registry of named ``Model`` instances and/or ``ModelSpec`` descriptors.

    Style mirrors ``PathwayRegistry`` exactly (CANON S6 cold-resume idiom):
    - Code-registered at startup; deterministic.
    - Loud ``ModelRegistryError`` on duplicate registration or lookup miss.
    - Never returns ``None``; never silently swallows a miss.

    Two parallel slots per profile name are maintained:

    ``Model`` slot (``register`` / ``resolve``):
        For pre-assembled models.  Suitable for tests or cases where the guard is managed
        externally.  NOT recommended for production engine runs (guard is baked in at
        registration time and shared across all calls — see the module docstring on the
        per-run guard seam).

    ``ModelSpec`` slot (``register_spec`` / ``resolve_spec``):
        The RECOMMENDED production path.  The engine calls ``build_model(spec, guard=budget)``
        at run-start to get a fresh, run-scoped assembled model (see module docstring).
    """

    def __init__(self) -> None:
        self._models: dict[str, Model] = {}
        self._specs: dict[str, ModelSpec] = {}
        self._factories: dict[str, Callable[[BudgetGuard], Model]] = {}

    # ------------------------------------------------------------------
    # Pre-assembled Model slot
    # ------------------------------------------------------------------

    def register(self, profile: str, model: Model) -> None:
        """Register a pre-assembled ``Model`` under *profile*.

        Raises
        ------
        ModelRegistryError
            If *profile* is already registered in the model slot.
        """
        if profile in self._models:
            raise ModelRegistryError(
                f"model profile {profile!r} is already registered; "
                "cold resume needs the same profiles registered at startup (CANON S6)"
            )
        self._models[profile] = model

    def resolve(self, profile: str) -> Model:
        """Return the pre-assembled ``Model`` registered under *profile*.

        Raises
        ------
        ModelRegistryError
            If *profile* is not registered — loud, never silent (CANON S6).
        """
        model = self._models.get(profile)
        if model is None:
            registered = sorted(self._models)
            raise ModelRegistryError(
                f"model profile {profile!r} is not registered; cold resume needs the profile "
                f"registered in this process at startup (CANON S6). "
                f"Registered profiles: {registered!r}"
            )
        return model

    def has(self, profile: str) -> bool:
        """Return ``True`` if *profile* has a pre-assembled model registered."""
        return profile in self._models

    # ------------------------------------------------------------------
    # ModelSpec slot (recommended production path)
    # ------------------------------------------------------------------

    def register_spec(self, profile: str, spec: ModelSpec) -> None:
        """Register a ``ModelSpec`` under *profile*.

        Raises
        ------
        ModelRegistryError
            If *profile* is already registered in the spec slot.
        """
        if profile in self._specs:
            raise ModelRegistryError(
                f"model spec profile {profile!r} is already registered; "
                "cold resume needs the same profiles registered at startup (CANON S6)"
            )
        self._specs[profile] = spec

    def resolve_spec(self, profile: str) -> ModelSpec:
        """Return the ``ModelSpec`` registered under *profile*.

        Raises
        ------
        ModelRegistryError
            If *profile* is not registered — loud, never silent (CANON S6).

        Notes
        -----
        The recommended engine pattern is::

            spec = registry.resolve_spec("researcher")
            model = build_model(spec, guard=budget)   # fresh per-run guard
        """
        spec = self._specs.get(profile)
        if spec is None:
            registered = sorted(self._specs)
            raise ModelRegistryError(
                f"model spec profile {profile!r} is not registered; cold resume needs the profile "
                f"registered in this process at startup (CANON S6). "
                f"Registered spec profiles: {registered!r}"
            )
        return spec

    def has_spec(self, profile: str) -> bool:
        """Return ``True`` if *profile* has a spec registered."""
        return profile in self._specs

    # ------------------------------------------------------------------
    # Factory slot (test seam for per-run guarded models without a real adapter)
    # ------------------------------------------------------------------

    def register_factory(self, profile: str, factory: Callable[[BudgetGuard], Model]) -> None:
        """Register a guard-accepting factory under *profile*.

        The factory slot is the test seam: tests register
        ``lambda g: BudgetGuardedModel(stub, g)`` here so the engine's
        ``assemble`` method produces a fresh, per-run guarded model without
        requiring a real ``ModelSpec`` (whose ``provider`` is a ``Literal``
        and cannot describe stubs).

        Raises
        ------
        ModelRegistryError
            If *profile* is already registered in the factory slot.
        """
        if profile in self._factories:
            raise ModelRegistryError(
                f"factory profile {profile!r} is already registered; "
                "cold resume needs the same profiles registered at startup (CANON S6)"
            )
        self._factories[profile] = factory

    def assemble(self, profile: str, *, guard: BudgetGuard) -> Model:
        """Produce a run-scoped ``Model`` for *profile* using *guard*.

        Precedence (first match wins):

        1. **spec slot** → ``build_model(spec, guard=guard)`` — the production path.
        2. **factory slot** → ``factory(guard)`` — the test seam for guarded stubs.
        3. **model slot** → returned VERBATIM, never re-wrapped here; the caller owns
           guarding.  Suitable for tests that assert unguarded or externally guarded
           behaviour.
        4. **none** → ``ModelRegistryError`` (loud, S6 — never a silent ``None``).

        Docstring note (carry-forward):
            Per-run/per-stage profile override (``run(model_profile=...)`` journaled
            into ``RunState``) is deferred — CF-3.0-C.
        """
        # 1. spec slot
        if profile in self._specs:
            return build_model(self._specs[profile], guard=guard)
        # 2. factory slot
        if profile in self._factories:
            return self._factories[profile](guard)
        # 3. pre-assembled model slot (verbatim — caller owns guarding)
        if profile in self._models:
            return self._models[profile]
        # 4. nothing registered — loud miss (S6)
        registered = sorted(list(self._specs) + list(self._factories) + list(self._models))
        raise ModelRegistryError(
            f"model profile {profile!r} is not registered; cold resume needs the profile "
            f"registered in this process at startup (CANON S6). "
            f"Registered profiles: {registered!r}"
        )


# ---------------------------------------------------------------------------
# build_model — the composition factory
# ---------------------------------------------------------------------------


def _default_estimator(
    adapter: Model,
    price: PriceTable,
    config: ProviderConfig,
) -> Callable[[Sequence[ChatMessage], ModelTier], float]:
    """Conservative pre-call cost estimate: prompt tokens x price + max_output x price."""

    def estimate(messages: Sequence[ChatMessage], tier: ModelTier) -> float:
        prompt = sum(adapter.count_tokens(m.content) for m in messages)
        usage_est = Usage(prompt_tokens=prompt, completion_tokens=config.max_output_tokens)
        return price.cost_usd(usage_est, tier)

    return estimate


def build_model(
    spec: ModelSpec,
    *,
    guard: BudgetGuard,
    ladder: tuple[LadderRung, ...] | None = None,
    client: Any = None,
) -> Model:
    """Assemble the full model stack from *spec* in the mandated load-bearing order.

    CANON: S4, S8, S9, S11.

    The composition order is::

        StructuredOutputModel(BudgetGuardedModel(adapter, guard))

    - **Innermost — the provider adapter** (``ClaudeModel`` or ``OpenAICompatModel``).
      The raw provider call lives here; it is the only layer that talks to the network.

    - **Middle — ``BudgetGuardedModel(adapter, guard)``** (S11 pre-call ceiling).
      The guard fires BEFORE the adapter is called; a ceiling breach raises
      ``BudgetExceededError`` without ever hitting the network.  Placed here so
      EVERY physical model call (including the ladder's retry rungs) is checked.

    - **Outermost — ``StructuredOutputModel(...)``** (S4/S8/S9 degradation ladder).
      The ladder selects the appropriate structured-output strategy based solely on
      ``capabilities`` (S4) and validates responses structurally (S9).  It sits outside
      the guard so a ladder retry counts as a separate guarded call.

    Parameters
    ----------
    spec:
        The ``ModelSpec`` describing which provider and config to use.
    guard:
        A **drive-scoped** ``BudgetGuard`` instance supplied by the engine at drive-start.
        Each drive segment receives its own ``BudgetGuard`` (per-run cumulative ceilings are
        CF-3.0-B).  See the module docstring for the per-drive guard seam rationale.
    ladder:
        Override the ladder used by ``StructuredOutputModel``.  When ``None`` the spec's
        ``ladder`` field is used; if that is also ``None``, ``DEFAULT_STRUCTURED_OUTPUT_LADDER``
        is used.

    Returns
    -------
    Model
        The fully assembled ``StructuredOutputModel(BudgetGuardedModel(adapter, guard))``.

    Raises
    ------
    ValueError
        If ``spec.provider == "openai_compat"`` and ``spec.config.base_url`` is not ``None``
        but ``spec.capabilities`` is ``None`` (CANON S9 — capabilities must be declared
        statically for non-production OpenAI-compat endpoints).
    """
    # Step 1: instantiate the innermost provider adapter.
    adapter: Model
    if spec.provider == "claude":
        from cogworx.model.providers.claude import ClaudeModel

        adapter = ClaudeModel(spec.config, client=client)
    else:
        # "openai_compat"
        from cogworx.model.providers.openai_compat import OpenAICompatModel

        adapter = OpenAICompatModel(spec.config, capabilities=spec.capabilities, client=client)

    # Step 2: wrap with BudgetGuardedModel — the pre-call cost ceiling (S11).
    #         Innermost physical-call wrapper: every adapter call is checked.
    estimator = _default_estimator(adapter, spec.config.price_per_mtok, spec.config)
    guarded: Model = BudgetGuardedModel(adapter, guard, estimator=estimator)

    # Step 3: wrap with StructuredOutputModel — the outermost degradation ladder (S4/S8/S9).
    #         Sits outside the guard so ladder retries each count as a guarded call.
    effective_ladder: tuple[LadderRung, ...] = (
        ladder
        if ladder is not None
        else (spec.ladder if spec.ladder is not None else DEFAULT_STRUCTURED_OUTPUT_LADDER)
    )
    return StructuredOutputModel(guarded, ladder=effective_ladder)
