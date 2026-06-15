"""Synthetic conversation corpus for the Phase 2 gate spike."""

from __future__ import annotations

import datetime
import math
import uuid
from dataclasses import dataclass, field
from typing import Final

from cogworx.claims.provenance import Claim, Provenance
from cogworx.knowledge.evidence import EvidenceEvent, make_evidence
from cogworx.knowledge.identity import claim_id_for
from cogworx.substrate.entity_kg import EntityKG
from cogworx.substrate.episodes import Episode, EpisodeStore
from cogworx.substrate.journal import ProjectionCursor
from cogworx.substrate.latent import LatentRecord, LatentStore

__all__ = [
    "T_1H",
    "T_10D",
    "T_30D",
    "T_NOW",
    "GateCorpus",
    "basis",
    "build_gate_corpus",
    "build_ge4_contradiction",
    "vec_toward",
]

# Fixed timestamps (UTC, no clock access)
T_NOW: Final[datetime.datetime] = datetime.datetime(2026, 6, 10, 12, 0, 0, tzinfo=datetime.UTC)
T_30D: Final[datetime.datetime] = T_NOW - datetime.timedelta(days=30)  # session 1
T_10D: Final[datetime.datetime] = T_NOW - datetime.timedelta(days=10)  # session 2
T_1H: Final[datetime.datetime] = T_NOW - datetime.timedelta(hours=1)  # session 3

_SOURCE_ID: Final[str] = "source:attestation:gate"


def vec_toward(
    topic_axis: int,
    cos: float,
    noise_axis: int,
    *,
    dim: int = 8,
) -> tuple[float, ...]:
    """Unit vector with EXACTLY ``cos`` cosine to basis vector e[topic_axis].

    v = cos * e[topic_axis] + sqrt(1 - cos^2) * e[noise_axis].
    Requires topic_axis != noise_axis and both < dim.
    """
    assert topic_axis != noise_axis, "topic_axis and noise_axis must differ"
    assert 0 <= topic_axis < dim, f"topic_axis {topic_axis} out of range [0, {dim})"
    assert 0 <= noise_axis < dim, f"noise_axis {noise_axis} out of range [0, {dim})"
    v = [0.0] * dim
    v[topic_axis] = cos
    v[noise_axis] = math.sqrt(1.0 - cos * cos)
    return tuple(v)


def basis(axis: int, *, dim: int = 8) -> tuple[float, ...]:
    """Pure basis vector e[axis] as a query vector."""
    v = [0.0] * dim
    v[axis] = 1.0
    return tuple(v)


@dataclass
class GateCorpus:
    """The synthetic gate corpus claim ids, keyed by corpus name."""

    claim_ids: dict[str, str] = field(default_factory=dict)
    """name -> claim_id (e.g. "C1" -> "claim:abc...")"""


def _make_claim(
    subject: str,
    predicate: str,
    payload: str,
    *,
    epistemic_type: str,
    scope: str,
    valid_from: datetime.datetime,
    valid_until: datetime.datetime | None,
    embedding: tuple[float, ...] | None,
    recorded_at: datetime.datetime,
) -> Claim:
    cid = claim_id_for(subject, predicate, payload, scope=scope)
    return Claim(
        id=cid,
        subject=subject,
        predicate=predicate,
        payload=payload,
        epistemic_type=epistemic_type,
        provenance=Provenance(
            source="human",
            confidence=0.9,
            recorded_at=recorded_at,
        ),
        valid_from=valid_from,
        valid_to=valid_until,
        ingest_time=valid_from,
        created_by="gate-corpus",
        embedding=embedding,
        scope=scope,
    )


def _make_gate_evidence(recorded_at: datetime.datetime) -> EvidenceEvent:
    return make_evidence(
        type="attestation",
        polarity="+",
        source_id=_SOURCE_ID,
        source_authority=0.9,
        recorded_at=recorded_at,
        event_id=uuid.uuid4().hex,
    )


async def build_gate_corpus(
    kg: EntityKG,
    latent_store: LatentStore | None = None,
    episode_store: EpisodeStore | None = None,
) -> GateCorpus:
    """Write the synthetic gate corpus into ``kg`` and return a GateCorpus with claim ids.

    All timestamps are fixed UTC constants — no clock access. Optionally writes a latent record
    for C1 (if latent_store provided) and episode rows for the 3 sessions (if episode_store
    provided).
    """
    corpus = GateCorpus()

    # -------------------------------------------------------------------
    # Core claims C1-C10
    # -------------------------------------------------------------------

    _core: list[
        tuple[
            str,  # name
            str,  # subject
            str,  # predicate
            str,  # payload
            str,  # epistemic_type
            str,  # scope
            datetime.datetime,  # valid_from
            datetime.datetime | None,  # valid_until
            tuple[float, ...],  # embedding
        ]
    ] = [
        (
            "C1",
            "user:alice",
            "timezone",
            "zylkorin",
            "observation",
            "user:alice",
            T_30D,
            None,
            vec_toward(0, 0.96, 6),
        ),
        (
            "C2",
            "user:alice",
            "lives_in",
            "portcity",
            "observation",
            "user:alice",
            T_30D,
            T_10D,
            vec_toward(1, 0.95, 6),
        ),
        (
            "C3",
            "user:alice",
            "lives_in",
            "denvercity",
            "observation",
            "user:alice",
            T_10D,
            None,
            vec_toward(1, 0.95, 7),
        ),
        (
            "C4",
            "user:alice",
            "diet",
            "vegquark",
            "observation",
            "user:alice",
            T_30D,
            None,
            vec_toward(2, 0.95, 6),
        ),
        (
            "C5",
            "user:alice",
            "diet_allergy",
            "nutquark",
            "observation",
            "user:alice",
            T_10D,
            None,
            vec_toward(2, 0.96, 6),
        ),
        (
            "C6",
            "user:alice",
            "diet_avoid",
            "caffquark",
            "observation",
            "user:alice",
            T_1H,
            None,
            vec_toward(2, 0.97, 7),
        ),
        (
            "C7",
            "project:atlas",
            "deploy_target",
            "atlasdeploy",
            "inference",
            "agent",
            T_10D,
            None,
            vec_toward(3, 0.95, 6),
        ),
        (
            "C8",
            "server:cobalt",
            "region",
            "eu-central-1",
            "observation",
            "agent",
            T_10D,
            None,
            vec_toward(4, 0.40, 7),
        ),
        (
            "C9",
            "user:alice",
            "editor",
            "editorpref",
            "observation",
            "agent",
            T_10D,
            None,
            vec_toward(5, 0.95, 6),
        ),
        (
            "C10",
            "user:alice",
            "editor",
            "editorpref",
            "observation",
            "user:alice",
            T_10D,
            None,
            vec_toward(5, 0.95, 6),
        ),
    ]

    for name, subj, pred, payload, etype, scope, vf, vu, emb in _core:
        claim = _make_claim(
            subj,
            pred,
            payload,
            epistemic_type=etype,
            scope=scope,
            valid_from=vf,
            valid_until=vu,
            embedding=emb,
            recorded_at=vf,
        )
        cid = await kg.write_claim(claim, evidence=_make_gate_evidence(vf))
        corpus.claim_ids[name] = cid

    # -------------------------------------------------------------------
    # Distractor claims D1-D20
    # -------------------------------------------------------------------

    for i in range(1, 21):
        vf_d = T_10D if i % 2 == 0 else T_1H
        noise_ax = 6 if i % 2 == 0 else 7
        cos_d = 0.72 + 0.003 * i
        emb_d = vec_toward(0, cos_d, noise_ax)
        claim_d = _make_claim(
            f"noise:{i}",
            "filler",
            f"filler_{i}",
            epistemic_type="inference",
            scope="agent",
            valid_from=vf_d,
            valid_until=None,
            embedding=emb_d,
            recorded_at=vf_d,
        )
        cid_d = await kg.write_claim(claim_d, evidence=_make_gate_evidence(vf_d))
        corpus.claim_ids[f"D{i}"] = cid_d

    # -------------------------------------------------------------------
    # DX — hot semantic noise
    # -------------------------------------------------------------------

    claim_dx = _make_claim(
        "noise:hot",
        "filler",
        "filler_hot",
        epistemic_type="inference",
        scope="agent",
        valid_from=T_1H,
        valid_until=None,
        embedding=vec_toward(0, 0.98, 7),
        recorded_at=T_1H,
    )
    cid_dx = await kg.write_claim(claim_dx, evidence=_make_gate_evidence(T_1H))
    corpus.claim_ids["DX"] = cid_dx

    # -------------------------------------------------------------------
    # Latent record for C1 (optional)
    # -------------------------------------------------------------------

    if latent_store is not None:
        c1_emb = vec_toward(0, 0.96, 6)
        await latent_store.put(
            LatentRecord(
                id=corpus.claim_ids["C1"],
                embedding=c1_emb,
                payload={"claim_id": corpus.claim_ids["C1"], "predicate": "timezone"},
            )
        )

    # -------------------------------------------------------------------
    # Episode rows for 3 sessions (optional)
    # -------------------------------------------------------------------

    if episode_store is not None:
        episodes = [
            Episode(
                episode_id="gate:run-s1:0:0",
                run_id="gate:run-s1",
                step_index=0,
                turn_index=0,
                session_id="gate-session-1",
                role="user",
                content="session 1 turn",
                kind="conversation",
                occurred_at=T_30D,
            ),
            Episode(
                episode_id="gate:run-s2:0:0",
                run_id="gate:run-s2",
                step_index=0,
                turn_index=0,
                session_id="gate-session-2",
                role="user",
                content="session 2 turn",
                kind="conversation",
                occurred_at=T_10D,
            ),
            Episode(
                episode_id="gate:run-s3:0:0",
                run_id="gate:run-s3",
                step_index=0,
                turn_index=0,
                session_id="gate-session-3",
                role="user",
                content="session 3 turn",
                kind="conversation",
                occurred_at=T_1H,
            ),
        ]
        cursor = ProjectionCursor(commit_ordinal=1, run_id="gate:run-s3", step_index=0)
        await episode_store.project_episodes("gate-corpus", episodes, cursor)

    return corpus


async def build_ge4_contradiction(kg: EntityKG) -> GateCorpus:
    """Write the GE-4 contradiction pair (LISBON + MADRID) into ``kg``.

    Call AFTER ``build_gate_corpus`` so the main corpus distractor claims are already in place.
    Returns a GateCorpus containing only the LISBON and MADRID claim ids.
    """
    ge4 = GateCorpus()

    lisbon = _make_claim(
        "user:alice",
        "favorite_city",
        "Lisbon",
        epistemic_type="observation",
        scope="agent",
        valid_from=T_30D,
        valid_until=None,
        embedding=vec_toward(0, 0.93, 6),
        recorded_at=T_30D,
    )
    cid_lisbon = await kg.write_claim(lisbon, evidence=_make_gate_evidence(T_30D))
    ge4.claim_ids["LISBON"] = cid_lisbon

    madrid = _make_claim(
        "user:alice",
        "favorite_city",
        "Madrid",
        epistemic_type="observation",
        scope="agent",
        valid_from=T_1H,
        valid_until=None,
        embedding=vec_toward(0, 0.94, 7),
        recorded_at=T_1H,
    )
    cid_madrid = await kg.write_claim(madrid, evidence=_make_gate_evidence(T_1H))
    ge4.claim_ids["MADRID"] = cid_madrid

    return ge4
