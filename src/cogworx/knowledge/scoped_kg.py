"""Scoped views over the entity KG: world model and user model (CANON S3, S5, S7).

ScopedKG wraps the EntityKG seam with a write-token-gated surface so every claim minted here
carries scope identity and the single-writer contract is enforced structurally (S7). Callers
never construct Claim objects directly — assert_fact mints them here with correct provenance.

world_model / user_model are the two concrete factory functions. A ScopedKG is a governed view:
reads are scope-filtered; writes are token-gated.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import final

from cogworx.claims.provenance import Claim, EpistemicType, Provenance
from cogworx.knowledge.evidence import EvidenceEvent, EvidenceType, Polarity, make_evidence
from cogworx.knowledge.identity import claim_id_for
from cogworx.knowledge.scopes import Scope, ScopeRegistry, ScopeWriteToken
from cogworx.knowledge.source_registry import SourceDeclaration
from cogworx.substrate.entity_kg import EntityKG, ScoredClaim

__all__ = [
    "ScopedKG",
    "user_model",
    "world_model",
]

# Injectable clock: returns current UTC datetime.  Default is datetime.now(timezone.utc).
_Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


@final
class ScopedKG:
    """A write-token-gated, scope-filtered view over an EntityKG.

    All claims minted here carry ``scope=token.scope.scope_id``.  Read methods return only
    claims belonging to this scope (view semantics).

    Obtain instances through :func:`world_model` or :func:`user_model`, not directly.
    """

    def __init__(
        self,
        kg: EntityKG,
        token: ScopeWriteToken,
        *,
        _clock: _Clock = _utc_now,
    ) -> None:
        self._kg = kg
        self._token = token
        self._clock = _clock

    @property
    def scope(self) -> Scope:
        """The read-side scope handle for this view."""
        return self._token.scope

    @property
    def owner(self) -> str:
        """The authorized writer bound to this scope (S7)."""
        return self._token.owner

    async def assert_fact(
        self,
        *,
        subject: str,
        predicate: str,
        obj: str,
        object_is_entity: bool = False,
        epistemic_type: EpistemicType,
        source: SourceDeclaration,
        evidence_type: EvidenceType = "tool_proof",
        polarity: Polarity = "+",
        run_id: str | None = None,
        stage: str | None = None,
        created_by: str,
    ) -> str:
        """Mint a scoped Claim and write it to the entity KG with one evidence event.

        The claim id is scope-partitioned: same (subject, predicate, obj) in different scopes
        produces different ids so world-model and user-model claims never collide.

        scope is always derived from token.scope.scope_id (canonical); callers cannot supply
        a non-canonical scope through this surface.

        Returns the canonical claim id.
        """
        now = self._clock()
        # When the object is an entity, obj is the entity name (used as object_repr for
        # claim identity and stored as object_entity on the Claim for the [:REFERS_TO] edge).
        object_repr = obj
        claim_id = claim_id_for(
            subject, predicate, object_repr, scope=self._token.scope.scope_id
        )
        provenance = Provenance(
            source="tool",
            source_ref=source.source_id,
            confidence=1.0,
            evidence=(),
            recorded_at=now,
        )
        claim = Claim(
            id=claim_id,
            subject=subject,
            predicate=predicate,
            payload=obj,
            epistemic_type=epistemic_type,
            provenance=provenance,
            valid_from=now,
            ingest_time=now,
            created_by=created_by,
            object_entity=obj if object_is_entity else None,
            scope=self._token.scope.scope_id,
        )
        evidence = make_evidence(
            type=evidence_type,
            polarity=polarity,
            source_id=source.source_id,
            source_authority=source.source_authority,
            run_id=run_id,
            stage=stage,
            recorded_at=now,
        )
        return await self._kg.write_claim(claim, evidence=evidence)

    async def add_evidence(
        self,
        claim_id: str,
        event: EvidenceEvent,
    ) -> None:
        """Append an evidence event to an existing claim in this scope.

        Raises ValueError if the claim is unknown or belongs to a different scope.
        """
        claim = await self._kg.get_claim(claim_id)
        if claim is None or claim.scope != self._token.scope.scope_id:
            raise ValueError(
                f"Claim {claim_id!r} not found or out of scope {self._token.scope.scope_id!r}"
            )
        await self._kg.add_evidence(claim_id, event)

    async def invalidate(self, claim_id: str, *, valid_to: datetime | None = None) -> None:
        """Invalidate a claim in this scope (bi-temporal, first-invalidation-wins).

        Raises ValueError if the claim is unknown or belongs to a different scope.
        """
        claim = await self._kg.get_claim(claim_id)
        if claim is None or claim.scope != self._token.scope.scope_id:
            raise ValueError(
                f"Claim {claim_id!r} not found or out of scope {self._token.scope.scope_id!r}"
            )
        ts = valid_to if valid_to is not None else self._clock()
        await self._kg.invalidate_claim(claim_id, valid_to=ts)

    async def get_claim(self, claim_id: str) -> Claim | None:
        """Return the claim if it exists AND belongs to this scope, else None (view semantics)."""
        claim = await self._kg.get_claim(claim_id)
        if claim is None or claim.scope != self._token.scope.scope_id:
            return None
        return claim

    async def claims_about(
        self,
        entity: str,
        *,
        limit: int = 20,
        as_of: datetime | None = None,
    ) -> Sequence[ScoredClaim]:
        """Return scored claims about entity, restricted to this scope."""
        return await self._kg.claims_about(
            entity, limit=limit, as_of=as_of, scope=self._token.scope.scope_id
        )

    async def claims_by_similarity(
        self,
        embedding: Sequence[float],
        *,
        k: int = 10,
        min_score: float = 0.70,
    ) -> Sequence[ScoredClaim]:
        """Return up to k scored claims nearest to embedding, restricted to this scope."""
        return await self._kg.claims_by_similarity(
            embedding, k=k, min_score=min_score, scope=self._token.scope.scope_id
        )


def world_model(kg: EntityKG, registry: ScopeRegistry, *, owner: str) -> ScopedKG:
    """Return a ScopedKG for the global world model.

    Raises ValueError if a different owner already holds the write-token (S7).
    """
    token = registry.claim_write_token("world", "global", owner=owner)
    return ScopedKG(kg, token)


def user_model(
    kg: EntityKG, registry: ScopeRegistry, user_id: str, *, owner: str
) -> ScopedKG:
    """Return a ScopedKG for a specific user's model.

    Raises ValueError if a different owner already holds the write-token for this user (S7).
    """
    token = registry.claim_write_token("user", user_id, owner=owner)
    return ScopedKG(kg, token)
