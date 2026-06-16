"""Verification-evidence projector — committed journal steps to entity-KG evidence (CANON S1/S6).

A SWEEPER-SHAPED off-path consumer that polls the TimescaleDB journal for committed oracle-verdict
and antithesis-verdict steps and writes the resulting
:class:`~cogworx.knowledge.evidence.EvidenceEvent` to the entity KG, with its cursor advancing
atomically in the same transaction (exactly-once,
crash-safe).

This is a DISTINCT event stream from the Pod 2.1 trial projector
(:class:`~cogworx.runtime.projector.TrialProjector`).  The two projectors do not share a cursor:

  - The trial projector (``"procedural-kg/trial-projector"``) writes ``(:Trial)`` nodes to the
    procedural KG — it records *whether* a procedure succeeded, feeding the Beta posterior.
  - This projector (``"verification/evidence-projector"``) writes ``(:Evidence)`` events to claim
    nodes in the entity KG — it records *what we know to be true about a verifiable claim*,
    preserving the honest provenance of that knowledge (``tool``/``system`` vs ``inference``).

WHY SEPARATE (S1):
  Writing evidence inline in a stage would put a Neo4j round-trip on the latency-critical write
  path and couple the evidence write to the stage's success.  Projecting off the committed journal
  means the evidence write is replay-safe (the ``source_id = f"{run_id}:{step_index}"``
  dedup key makes re-projection idempotent at the read-path posterior level), crash-safe (cursor
  advances in the batch's own transaction), and free of the model loop.

PROVENANCE HONESTY (F1 — the load-bearing S9 invariant):
  The projector reads ``verdict.source`` directly from the committed artifact — it NEVER sets or
  infers source from model output.  An ``inference``-sourced verdict produces ``inference``
  evidence; a ``tool``/``system``-sourced verdict produces ``confirmed`` evidence.  This preserves
  the S9 boundary: a model-judge verdict is a heuristic prioritizer that drives routing but does
  NOT contribute to a first-hand truth posterior.  The anti-laundering invariant is:

    source_authority(executable) > source_authority(inference)

  Concretely: executable (``tool``/``system``) = 1.0, inference = 0.5.  The VALUES are the Pod 2.1
  calibration (``procedural_confidence.py:107`` pins 1.0 for tool proofs); the ORDERING is the
  contractual invariant — never reverse it.

CLAIM-NODE IDENTITY (S1 — claim accumulation):
  All verdicts about ONE ``verifiable_claim`` subject resolve to ONE claim node via
  ``claim_id_for(subject=<normalized claim>, predicate="verified_by", object_repr=<normalized
  claim>)`` (``knowledge/identity.py``).  Per-verdict identity lives in ``source_id =
  f"{run_id}:{step_index}"`` — first evidence mints the node, subsequent evidence accumulates.

PROCEDURAL BETA IN V1 — STAMPED NONE (carry-forward):
  ``record_for`` computes a ``procedural_outcome`` but the dialectic stages carry no
  ``procedure_id`` binding.  This projector does NOT stamp the procedural-KG Beta in v1 —
  ``stamps_procedural_beta`` is honoured only when a procedure binding exists (none yet).  This
  keeps the Pod 2.1 projector and posterior UNTOUCHED.  A future "dialectic-as-procedure" binding
  will supply the ``procedure_id`` and complete the loop.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Final

from cogworx.claims.provenance import (
    Artifact,
    Claim,
    EpistemicType,
    Provenance,
)
from cogworx.knowledge.evidence import make_evidence
from cogworx.knowledge.identity import claim_id_for
from cogworx.substrate.entity_kg import ClaimProjection, EntityKG
from cogworx.substrate.journal import Journal, ProjectedStep, ProjectionCursor
from cogworx.substrate.procedural_kg import ordinal_max
from cogworx.verification.contracts import Verdict
from cogworx.verification.honest_failure import AntithesisVerdict
from cogworx.verification.outcome import (
    VerdictRole,
    VerificationRecord,
    record_for,
    verdict_from_antithesis,
)

__all__ = [
    "EVIDENCE_PROJECTOR_CONSUMER",
    "VerificationEvidenceProjector",
]

EVIDENCE_PROJECTOR_CONSUMER: Final[str] = "verification/evidence-projector"
"""Own cursor consumer key — DISTINCT from the Pod 2.1 ``"procedural-kg/trial-projector"`` key.

Each projection consumer owns its own ``(:ProjectionCursor)`` so the journal table stays pure (S3)
and the trial projector's watermark is never disturbed by verification-evidence projection ticks.
"""

# Source-authority scale (Pod 2.1 calibration — procedural_confidence.py:107 pins 1.0 for tool).
# CONTRACT: the ordering  _AUTHORITY_EXECUTABLE > _AUTHORITY_INFERENCE  must never be reversed.
# The values are concrete; the ordering is the load-bearing invariant.
_AUTHORITY_EXECUTABLE: Final[float] = 1.0  # tool / system — first-hand observation
_AUTHORITY_INFERENCE: Final[float] = 0.5  # model-judge — heuristic prioritizer

_CREATED_BY: Final[str] = "verification/evidence-projector"
_PREDICATE: Final[str] = "verified_by"


def _source_authority(verdict: Verdict) -> float:
    """Return the source authority for ``verdict`` (executable > inference — the contract)."""
    return _AUTHORITY_EXECUTABLE if verdict.is_executable else _AUTHORITY_INFERENCE


def _claim_for(verifiable_claim: str, committed_at: datetime, verdict: Verdict) -> Claim:
    """Mint a :class:`~cogworx.claims.provenance.Claim` node for *verifiable_claim*.

    Identity is derived ONLY from the claim subject (``verifiable_claim``) via
    ``claim_id_for`` — never from per-verdict data.  This guarantees all verdicts
    about ONE claim accumulate on ONE node.
    """
    cid = claim_id_for(
        subject=verifiable_claim,
        predicate=_PREDICATE,
        object_repr=verifiable_claim,
    )
    epistemic: EpistemicType = "confirmed" if verdict.is_executable else "inference"
    return Claim(
        id=cid,
        subject=verifiable_claim,
        predicate=_PREDICATE,
        payload=verifiable_claim,
        epistemic_type=epistemic,
        provenance=Provenance(
            source=verdict.source,
            confidence=_source_authority(verdict),
            recorded_at=committed_at,
        ),
        valid_from=committed_at,
        ingest_time=committed_at,
        created_by=_CREATED_BY,
    )


class VerificationEvidenceProjector:
    """Polls the journal for committed verdict steps and projects entity-KG evidence off-path.

    Sweeper-shaped (S1): no model, no hot-path commit.  One ``tick()`` does ONE bounded,
    visibility-fenced journal read (``committed_steps_after``) and writes all resolved evidence
    and the cursor advance in a SINGLE ``project_claims`` transaction — exactly-once into the
    entity KG even across crashes.
    """

    def __init__(
        self,
        *,
        journal: Journal,
        entity_kg: EntityKG,
        consumer: str = EVIDENCE_PROJECTOR_CONSUMER,
        batch_limit: int = 256,
    ) -> None:
        self._journal = journal
        self._entity_kg = entity_kg
        self._consumer = consumer
        self._batch_limit = batch_limit

    async def tick(self) -> int:
        """Project one batch; return the number of evidence events written.

        ONE bounded journal read: ``committed_steps_after(cursor, limit=batch_limit)`` — the
        strict keyset ``(commit_ordinal, run_id, step_index) > cursor``, visibility-fenced.

        Then ONE ``project_claims`` transaction: write all resolved evidence events and advance
        the cursor to the tuple-max of the LAST SCANNED row.  An idle tick (no rows) returns 0
        and leaves the cursor untouched.
        """
        cursor = await self._entity_kg.read_cursor(self._consumer)

        scanned = tuple(await self._journal.committed_steps_after(cursor, limit=self._batch_limit))
        if not scanned:
            return 0

        writes, count = await self._resolve(scanned)

        new_cursor = ordinal_max(cursor, _cursor_of(scanned[-1]))
        await self._entity_kg.project_claims(self._consumer, writes, new_cursor)
        return count

    async def _resolve(self, scanned: Sequence[ProjectedStep]) -> tuple[list[ClaimProjection], int]:
        """Resolve a batch of scanned steps to ``ClaimProjection`` writes.

        Each verdict step that yields a non-``None``
        :class:`~cogworx.verification.outcome.VerificationRecord`
        produces exactly one ``ClaimProjection`` (the claim node + evidence event).  Steps that
        are not verdict artifacts are skipped as ordinary control flow.

        The claim node check (``get_claim``) is done here, not inside ``project_claims``, because
        the entity-KG Protocol's ``write_claim`` always records the evidence (it is both a MERGE
        on the claim node AND a CREATE of the event).  For first evidence the node does not exist
        yet; for subsequent evidence ``write_claim`` merges idempotently on identity and appends.
        Both paths are handled uniformly by ``ClaimProjection`` → ``project_claims`` →
        ``write_claim`` (the adapter's MERGE + CREATE semantics).
        """
        writes: list[ClaimProjection] = []
        count = 0

        for projected in scanned:
            step = projected.record
            output: Artifact | None = getattr(step.result, "output", None)
            if output is None:
                continue

            verdict: Verdict | None = None
            role: VerdictRole | None = None
            verifiable_claim: str | None = None

            if output.kind == "oracle-verdict":
                # ExperimentStage artifact: data carries the Verdict fields + verifiable_claim.
                verdict = Verdict.model_validate(output.data)
                role = "oracle"
                verifiable_claim = output.data.get("verifiable_claim")
            elif output.kind == "antithesis-verdict":
                # AntithesisStage artifact: data carries AntithesisVerdict fields +
                # verifiable_claim + oracle_backed.
                av = AntithesisVerdict.model_validate(output.data)
                verdict = verdict_from_antithesis(av)
                role = "antithesis"
                verifiable_claim = output.data.get("verifiable_claim")
            # else: ordinary control flow — skip.

            if verdict is None or role is None or not isinstance(verifiable_claim, str):
                continue

            rec: VerificationRecord | None = record_for(verdict, role=role)
            if rec is None:
                # Routing-only verdict (invalid check, model-judge oracle, or model-claimed break).
                continue

            # PROCEDURAL BETA: rec.procedural_outcome is non-None only for an executable oracle
            # verdict, but v1 has no procedure_id binding for the dialectic stages — stamp nothing.
            # Carry-forward: a future "dialectic-as-procedure" binding supplies procedure_id here.

            source_id = f"{step.run_id}:{step.step_index}"  # S9 — never model output
            ev = make_evidence(
                type=rec.evidence_type,
                polarity=rec.polarity,
                source_id=source_id,
                source_authority=_source_authority(verdict),
                recorded_at=step.committed_at,
                run_id=step.run_id,
                stage=step.stage_name,
            )
            claim = _claim_for(verifiable_claim, step.committed_at, verdict)
            writes.append(ClaimProjection(claim=claim, evidence=ev))
            count += 1

        return writes, count


def _cursor_of(projected: ProjectedStep) -> ProjectionCursor:
    """The cursor ``(commit_ordinal, run_id, step_index)`` of a scanned step."""
    return ProjectionCursor(
        commit_ordinal=projected.commit_ordinal,
        run_id=projected.record.run_id,
        step_index=projected.record.step_index,
    )
