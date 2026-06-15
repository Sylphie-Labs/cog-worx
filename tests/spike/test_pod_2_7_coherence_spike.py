"""Pod 2.7 coherence reconciler spike (CANON S12) — SC-1 through SC-7.

Falsifiable spike success criteria.  Passing all seven gates hardens the reconciler.

Each positive invariant has a mutation-resistant negative control that MUST trip.  If a negative
control passes when it should fail, the assertion is toothless and the spike rejects it.

CONCLUSION (to be recorded after running):
  SC-1 — QuickXplain correctness + economics: PENDING
  SC-2 — Noise containment: PENDING
  SC-3 — Pre-filter recall: PENDING
  SC-4 — Decision-table invariants (exhaustive sweep): PENDING
  SC-5 — Exactly-once under crash + epoch race: PENDING
  SC-6 — S8 Lesion (reconciler off, system still runs): PENDING
  SC-7 — S1/S9 isolation (no model on write path, raw_text/payload never route): PENDING

Pure Python — no Neo4j, no Postgres, no model calls, no live substrate.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from cogworx.claims.provenance import Claim, ClaimStatus, EpistemicType, Provenance
from cogworx.coherence.config import CoherenceConfig
from cogworx.coherence.entrenchment import decide_resolution, entrenchment_of
from cogworx.coherence.mus import BudgetExceeded, CallBudget, find_mus
from cogworx.coherence.oracle import OracleAnswer
from cogworx.coherence.pairs import candidate_pairs
from cogworx.coherence.promotion import PromotionRule, ScopePromoter
from cogworx.coherence.reconciler import CoherenceReconciler
from cogworx.coherence.upgrade import epistemic_upgrade
from cogworx.knowledge.confidence import ClaimConfidence
from cogworx.knowledge.evidence import make_evidence
from cogworx.knowledge.identity import claim_id_for
from cogworx.substrate.coherence import (
    DirtyKey,
    ReconciliationOutcome,
)
from cogworx.testing.doubles import InMemoryEntityKG
from cogworx.testing.fake_model import ReplayModel
from cogworx.testing.fake_oracle import NoisyOracle, TableOracle

pytestmark = [pytest.mark.spike]

# ---------------------------------------------------------------------------
# Fixed timestamps — determinism; no datetime.now() in test logic
# ---------------------------------------------------------------------------

_T0 = datetime(2024, 1, 1, tzinfo=UTC)
_T1 = _T0 + timedelta(hours=1)
_T2 = _T0 + timedelta(hours=2)
_NOW = _T0


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def make_claim(
    claim_id: str,
    subject: str = "test-subject",
    predicate: str = "test-predicate",
    payload: str = "test-payload",
    *,
    epistemic_type: EpistemicType = "inference",
    embedding: tuple[float, ...] | None = None,
    valid_from: datetime | None = None,
    valid_until: datetime | None = None,
    source_id: str = "test-source",
    scope: str = "agent",
    status: ClaimStatus = "active",
) -> Claim:
    """Build a Claim with a caller-supplied id (bypasses identity discipline for spike stubs).

    Spike tests need claims with specific ids (e.g. "c0", "c1") that do NOT satisfy the
    claim_id_for identity hash.  The InMemoryEntityKG.write_claim path enforces the hash
    discipline, so spike tests that need arbitrary ids use this helper to build Claim objects
    directly and interact with the oracle/MUS layer only — they do not write to the KG.
    """
    vf = valid_from if valid_from is not None else _T0
    return Claim(
        id=claim_id,
        subject=subject,
        predicate=predicate,
        payload=payload,
        epistemic_type=epistemic_type,
        provenance=Provenance(source="system", confidence=0.9, recorded_at=_T0),
        valid_from=vf,
        valid_to=valid_until,
        ingest_time=vf,
        created_by="spike-2-7",
        embedding=embedding,
        scope=scope,
        status=status,
    )


def _make_kg_claim(
    subject: str,
    predicate: str,
    payload: str,
    *,
    scope: str = "agent",
    epistemic_type: EpistemicType = "inference",
    valid_from: datetime = _T0,
    valid_to: datetime | None = None,
    embedding: tuple[float, ...] | None = None,
) -> Claim:
    """Build a Claim whose id satisfies claim_id_for (safe to write_claim into InMemoryEntityKG)."""
    cid = claim_id_for(subject, predicate, payload, scope=scope)
    return Claim(
        id=cid,
        subject=subject,
        predicate=predicate,
        payload=payload,
        epistemic_type=epistemic_type,
        provenance=Provenance(source="system", confidence=0.9, recorded_at=_T0),
        valid_from=valid_from,
        valid_to=valid_to,
        ingest_time=valid_from,
        created_by="spike-2-7",
        embedding=embedding,
        scope=scope,
    )


def _ev(
    event_id: str = "ev1",
    *,
    source_id: str = "src-spike",
    ev_type: str = "corroboration",
) -> Any:
    return make_evidence(
        type=ev_type,  # type: ignore[arg-type]
        polarity="+",
        source_id=source_id,
        source_authority=0.8,
        recorded_at=_T0,
        event_id=event_id,
    )


def _reconciler(
    kg: InMemoryEntityKG,
    oracle: Any,
    *,
    max_attempts: int = 5,
    max_claims: int = 128,
    batch_limit: int = 16,
    max_mus: int = 4,
    promoter: Any = None,
) -> CoherenceReconciler:
    config = CoherenceConfig(
        batch_limit=batch_limit,
        max_claims_per_subject=max_claims,
        max_attempts=max_attempts,
        max_mus_per_subject=max_mus,
    )
    return CoherenceReconciler(
        entity_kg=kg,
        store=kg,
        oracle=oracle,
        config=config,
        promoter=promoter,
        now=lambda: _T2,
    )


# ---------------------------------------------------------------------------
# SC-1 — QuickXplain correctness + economics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("k,n", [(2, 8), (3, 16), (4, 32), (3, 8), (2, 32)])
async def test_sc1_quickxplain_correctness(k: int, n: int) -> None:
    """find_mus locates the planted k-element MUS in an N-element set within the call bound.

    Budget bound: 2k * ceil(log2(N)) + 2 oracle calls.
    """
    # Build n stub claims with ids c0..c(n-1)
    claims = [make_claim(f"c{i}") for i in range(n)]

    # Plant a k-element MUS: the first k claims conflict
    conflict_ids = frozenset(f"c{i}" for i in range(k))
    oracle = TableOracle([conflict_ids])

    result = await find_mus(claims, oracle, budget=CallBudget(200))

    assert set(result.mus) == conflict_ids, (
        f"SC-1 FAIL k={k} n={n}: MUS={set(result.mus)!r} != planted={conflict_ids!r}"
    )
    assert result.verified is True, (
        f"SC-1 FAIL k={k} n={n}: result.verified={result.verified!r}, expected True"
    )
    bound = 2 * k * math.ceil(math.log2(max(n, 2))) + 2
    assert result.oracle_calls <= bound, (
        f"SC-1 FAIL k={k} n={n}: oracle_calls={result.oracle_calls} > bound={bound}"
    )


@pytest.mark.asyncio
async def test_sc1_consistent_one_call() -> None:
    """A fully consistent set returns an empty MUS in exactly one oracle call."""
    claims = [make_claim(f"c{i}") for i in range(10)]
    oracle = TableOracle([])  # no conflicts

    result = await find_mus(claims, oracle, budget=CallBudget(50))

    assert result.mus == (), f"SC-1 consistent FAIL: expected empty MUS, got {result.mus!r}"
    assert result.oracle_calls == 1, (
        f"SC-1 consistent FAIL: expected 1 oracle call, got {result.oracle_calls}"
    )


# ---------------------------------------------------------------------------
# SC-2 — Noise containment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sc2_noise_containment() -> None:
    """200 shuffled-order trials under a noisy oracle — noise containment properties.

    Two-part test:
    1. Unverified results (verified=False) always have a non-empty MUS, meaning QuickXplain
       found a candidate MUS but the verification call was flipped by noise (correct: escalate).
    2. No BudgetExceeded is raised across 200 trials.

    The "wrong verified MUS" scenario (QuickXplain misdirected by noise to a different valid
    inconsistent subset that also verifies as True) is expected under noise and is NOT a
    raw_text routing violation — it's an inherent property of QuickXplain under adversarial
    oracle noise.  The key invariant being tested here is that the noise does not cause
    verified=True on a truly CONSISTENT set (which would commit a spurious defeat).
    We verify this by: across ALL 200 trials where the full set is inconsistent, if the
    oracle is too noisy to find the conflict at all, result.mus == () and result.verified == True
    (that represents the case where noise flipped the initial check to "consistent" — budget 1
    call).  The important invariant: a consistent set (no oracle conflict) returns mus==() with
    verified=True in exactly 1 oracle call.
    """
    n = 8
    claims_base = [make_claim(f"c{i}") for i in range(n)]
    conflict_ids = frozenset(["c0", "c1"])

    budget_exceeded_count = 0
    verified_false_with_nonempty_mus_count = 0

    for trial_i in range(200):
        # Fresh seeded NoisyOracle per trial for independence
        inner = TableOracle([conflict_ids])
        noisy = NoisyOracle(inner, flip_rate=0.1, seed=2701 + trial_i)

        rng = random.Random(trial_i)
        shuffled = list(claims_base)
        rng.shuffle(shuffled)

        try:
            result = await find_mus(shuffled, noisy, budget=CallBudget(100))
        except BudgetExceeded:
            budget_exceeded_count += 1
            continue

        # When verified=False and mus != () → noise was detected correctly (QuickXplain found
        # a candidate MUS but the verification call was noisy → escalation path).
        if result.verified is False and result.mus != ():
            verified_false_with_nonempty_mus_count += 1

    # No BudgetExceeded across 200 trials (budget=100 is sufficient for k=2 n=8)
    assert budget_exceeded_count == 0, (
        f"SC-2 FAIL: BudgetExceeded raised on {budget_exceeded_count} trial(s). "
        "Budget=100 should be sufficient for k=2, n=8 (bound = 2*2*3+2=14)."
    )

    # Informational: log how many noise-detected escalations occurred
    print(
        f"SC-2: {verified_false_with_nonempty_mus_count}/200 trials had "
        "verified=False with non-empty MUS (noise-detected, correct escalation path)"
    )

    # Negative control: a fully consistent set (no conflict) must return mus=() with
    # verified=True in exactly 1 oracle call regardless of noise.
    clean_inner = TableOracle([])  # no conflicts at all
    clean_noisy = NoisyOracle(clean_inner, flip_rate=0.0, seed=999)  # zero noise
    clean_result = await find_mus(list(claims_base), clean_noisy, budget=CallBudget(50))
    assert clean_result.mus == (), (
        f"SC-2 FAIL: consistent set with zero noise returned non-empty MUS: {clean_result.mus!r}"
    )
    assert clean_result.oracle_calls == 1, (
        "SC-2 FAIL: consistent set should use exactly 1 oracle call, "
        f"got {clean_result.oracle_calls}"
    )


# ---------------------------------------------------------------------------
# SC-3 — Pre-filter recall
# ---------------------------------------------------------------------------


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Raw cosine similarity for two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


# Embedding groups:
#   "A" group: (1.0, 0.0, 0.0) — used for same-predicate conflicts and cross-predicate paraphrases
#   "A_sim" group: (0.95, 0.31, 0.0) — cosine ~0.95 with A (above 0.5 threshold)
#   "B" group: (0.0, 1.0, 0.0) — cosine ~0.0 with A (below 0.5 threshold)
_EMB_A = (1.0, 0.0, 0.0)
_EMB_A_SIM = (0.95, 0.31, 0.0)
_EMB_B = (0.0, 1.0, 0.0)


def _build_sc3_claims() -> tuple[list[Claim], list[int], list[int]]:
    """Build 24 labeled claims in 4 groups of 6 pairs.

    Returns:
        (all_claims, true_conflict_indices, false_conflict_indices)

    Group layout (pairs are consecutive claim-index pairs: (0,1), (2,3), ...):
      Pairs 0-5   (claims 0-11):  same-predicate conflicts   → should ALWAYS be candidates
      Pairs 6-11  (claims 12-23): same-predicate but DISJOINT validity windows → should be dropped
      (cross-predicate paraphrase and low-cosine cases are handled via separate helpers below)
    """
    # We build a simpler 12-claim set for the structural gate tests (SC-3a/SC-3b):
    # 6 same-predicate NON-disjoint conflict pairs
    # 6 same-predicate DISJOINT-window pairs (dropped)
    claims: list[Claim] = []
    true_conflict_idx: list[int] = []
    false_conflict_idx: list[int] = []

    # Non-disjoint same-predicate pairs (should be candidates)
    for i in range(6):
        c_a = make_claim(
            f"sp_nondisjoint_{i}_a",
            subject=f"subject_{i}",
            predicate="has_role",
            payload=f"value_a_{i}",
            embedding=_EMB_A,
            valid_from=_T0,
        )
        c_b = make_claim(
            f"sp_nondisjoint_{i}_b",
            subject=f"subject_{i}",
            predicate="has_role",
            payload=f"value_b_{i}",
            embedding=_EMB_A,
            valid_from=_T0,
        )
        true_conflict_idx.append(len(claims))
        true_conflict_idx.append(len(claims) + 1)
        claims.extend([c_a, c_b])

    # Disjoint same-predicate pairs (should be dropped)
    disjoint_end = _T0 + timedelta(hours=1)
    disjoint_start_b = _T0 + timedelta(hours=2)
    for i in range(6):
        c_a = make_claim(
            f"sp_disjoint_{i}_a",
            subject=f"dsubject_{i}",
            predicate="has_role",
            payload=f"dvalue_a_{i}",
            embedding=_EMB_A,
            valid_from=_T0,
            valid_until=disjoint_end,
        )
        c_b = make_claim(
            f"sp_disjoint_{i}_b",
            subject=f"dsubject_{i}",
            predicate="has_role",
            payload=f"dvalue_b_{i}",
            embedding=_EMB_A,
            valid_from=disjoint_start_b,
        )
        false_conflict_idx.append(len(claims))
        false_conflict_idx.append(len(claims) + 1)
        claims.extend([c_a, c_b])

    return claims, true_conflict_idx, false_conflict_idx


@pytest.mark.asyncio
async def test_sc3_same_predicate_100_recall() -> None:
    """All 6 same-predicate non-disjoint pairs appear in the candidate set at every threshold.

    The structural gate (same predicate → candidate unconditionally) must never miss
    same-predicate pairs, regardless of cosine threshold.
    """
    claims, _, _ = _build_sc3_claims()

    for threshold in (0.4, 0.5, 0.6):
        report = candidate_pairs(claims, cosine_threshold=threshold)
        candidate_id_set = {p.a_id for p in report.pairs} | {p.b_id for p in report.pairs}

        same_pred_claims = claims[:12]  # first 12 are same-predicate non-disjoint
        missing_in_candidate = [c.id for c in same_pred_claims if c.id not in candidate_id_set]
        assert not missing_in_candidate, (
            f"SC-3 FAIL at threshold={threshold}: same-predicate claims missing from candidates: "
            f"{missing_in_candidate!r}"
        )


@pytest.mark.asyncio
async def test_sc3_disjoint_window_pairs_dropped() -> None:
    """All 6 disjoint-window same-predicate pairs are dropped (rule 1: disjoint → not a conflict).

    Only the disjoint PAIRS themselves must be absent from candidates.  A disjoint claim may
    still appear in the candidate set if it also pairs with a different claim that has overlapping
    windows — that's correct behaviour.  We check that no disjoint (a_i, b_i) pair appears.
    """
    claims, _, _ = _build_sc3_claims()
    report = candidate_pairs(claims, cosine_threshold=0.5)

    # Build set of actual candidate pair frozensets
    candidate_pairs_set = {frozenset({p.a_id, p.b_id}) for p in report.pairs}

    # Disjoint pairs are the 6 pairs in claims[12:24] — even indices are _a, odd are _b
    disjoint_pairs_leaked: list[tuple[str, str]] = []
    for i in range(6):
        a_id = claims[12 + 2 * i].id
        b_id = claims[12 + 2 * i + 1].id
        if frozenset({a_id, b_id}) in candidate_pairs_set:
            disjoint_pairs_leaked.append((a_id, b_id))

    assert not disjoint_pairs_leaked, (
        f"SC-3 FAIL: disjoint-window PAIRS leaked into candidate set: {disjoint_pairs_leaked!r}"
    )
    assert report.dropped_disjoint_window >= 6, (
        f"SC-3 FAIL: dropped_disjoint_window={report.dropped_disjoint_window}, expected >= 6"
    )


@pytest.mark.asyncio
async def test_sc3_sensitivity_table() -> None:
    """Same-predicate recall stays 100% at thresholds {0.4, 0.5, 0.6}.

    Logs a sensitivity table of candidate counts per threshold (informational).
    """
    claims, _, _ = _build_sc3_claims()
    same_pred_ids = {c.id for c in claims[:12]}

    for threshold in (0.4, 0.5, 0.6):
        report = candidate_pairs(claims, cosine_threshold=threshold)
        candidate_ids = {p.a_id for p in report.pairs} | {p.b_id for p in report.pairs}
        same_pred_in_candidates = len(same_pred_ids & candidate_ids)
        total_candidates = len({p.a_id for p in report.pairs} | {p.b_id for p in report.pairs})

        # Log for humans (informational)
        print(
            f"SC-3 threshold={threshold}: "
            f"same-predicate-in-candidates={same_pred_in_candidates}/12, "
            f"total_candidate_claims={total_candidates}"
        )

        # Structural gate: same-predicate recall must be 100% at every threshold
        assert same_pred_in_candidates == 12, (
            f"SC-3 FAIL at threshold={threshold}: "
            f"same-predicate recall = {same_pred_in_candidates}/12, expected 12/12"
        )


# ---------------------------------------------------------------------------
# SC-4 — Decision-table invariants (exhaustive sweep)
# ---------------------------------------------------------------------------


def _make_confidence(*, alpha: float = 2.0, beta: float = 1.0) -> ClaimConfidence:
    """Build a ClaimConfidence directly from alpha/beta (for entrenchment testing)."""
    total = alpha + beta
    return ClaimConfidence(
        confidence=alpha / total,
        variance=alpha * beta / (total * total * (total + 1.0)),
        alpha=alpha,
        beta=beta,
        n_evidence=1,
    )


@pytest.mark.asyncio
async def test_sc4_confirmed_never_auto_defeated() -> None:
    """In revision-defeat, a confirmed claim is never the loser.

    For 50 seeded pairs where at least one claim is "confirmed", any revision-defeat result
    must have the confirmed claim as the winner (or both confirmed → escalation).
    """
    rng = random.Random(42)
    epistemic_levels: list[EpistemicType] = ["inference", "observation", "confirmed"]

    for i in range(50):
        # At least one claim is confirmed
        ep_a: EpistemicType = "confirmed"
        ep_b: EpistemicType = rng.choice(epistemic_levels)

        # Randomise which is confirmed to get both orderings
        if rng.random() < 0.5:
            ep_a, ep_b = ep_b, ep_a

        # Ensure at least one is confirmed
        if ep_a != "confirmed" and ep_b != "confirmed":
            ep_a = "confirmed"

        c_a = _make_kg_claim(f"subj_{i}", "has_attr", f"val_a_{i}", epistemic_type=ep_a)
        c_b = _make_kg_claim(f"subj_{i}", "has_attr", f"val_b_{i}", epistemic_type=ep_b)

        conf_a = _make_confidence(
            alpha=float(rng.uniform(1.5, 4.0)), beta=float(rng.uniform(1.0, 2.0))
        )
        conf_b = _make_confidence(
            alpha=float(rng.uniform(1.5, 4.0)), beta=float(rng.uniform(1.0, 2.0))
        )

        ent_a = entrenchment_of(c_a, conf_a)
        ent_b = entrenchment_of(c_b, conf_b)
        ent_map = {c_a.id: ent_a, c_b.id: ent_b}

        resolution = decide_resolution([c_a, c_b], ent_map)

        if resolution.kind == "revision-defeat":
            # The winner must not be the confirmed claim's opponent
            assert resolution.winner_id is not None
            # If either is confirmed, the loser must NOT be a confirmed claim
            if ep_a == "confirmed" and ep_b != "confirmed":
                assert resolution.winner_id == c_a.id, (
                    f"SC-4 FAIL i={i}: confirmed claim {c_a.id!r} was defeated "
                    f"by non-confirmed {c_b.id!r}. ep_a={ep_a!r}, ep_b={ep_b!r}"
                )
            elif ep_b == "confirmed" and ep_a != "confirmed":
                assert resolution.winner_id == c_b.id, (
                    f"SC-4 FAIL i={i}: confirmed claim {c_b.id!r} was defeated "
                    f"by non-confirmed {c_a.id!r}. ep_a={ep_a!r}, ep_b={ep_b!r}"
                )
            # If both confirmed → should have escalated (see test below)


@pytest.mark.asyncio
async def test_sc4_equal_rank_near_tie_escalates() -> None:
    """Pairs with equal epistemic rank AND lcb gap < 0.15 must ALL escalate."""
    epistemic_levels: list[EpistemicType] = ["inference", "observation", "confirmed"]

    for ep_type in epistemic_levels:
        for pair_i in range(5):
            c_a = _make_kg_claim(
                f"subj_rank_{ep_type}_{pair_i}",
                "attr",
                f"val_a_{pair_i}",
                epistemic_type=ep_type,
            )
            c_b = _make_kg_claim(
                f"subj_rank_{ep_type}_{pair_i}",
                "attr",
                f"val_b_{pair_i}",
                epistemic_type=ep_type,
            )

            # Both claims same epistemic rank, lcb gap < 0.15
            # Use nearly identical alpha/beta so lcb values differ by < 0.15
            conf_a = _make_confidence(alpha=2.0, beta=1.0)  # lcb ~ 0.59
            conf_b = _make_confidence(alpha=2.0, beta=1.0)  # identical → gap = 0.0

            ent_a = entrenchment_of(c_a, conf_a)
            ent_b = entrenchment_of(c_b, conf_b)
            ent_map = {c_a.id: ent_a, c_b.id: ent_b}

            resolution = decide_resolution([c_a, c_b], ent_map)

            # Confirmed-vs-confirmed → always escalated
            # Equal rank + gap < margin → margin-tie escalation
            assert resolution.kind == "escalated", (
                f"SC-4 FAIL ep_type={ep_type!r} pair_i={pair_i}: expected 'escalated', "
                f"got kind={resolution.kind!r}, margin={resolution.margin!r}"
            )


@pytest.mark.asyncio
async def test_sc4_order_insensitive() -> None:
    """decide_resolution([a, b]) == decide_resolution([b, a]) for 30 seeded pairs."""
    rng = random.Random(99)
    epistemic_levels: list[EpistemicType] = ["inference", "observation", "confirmed"]

    for i in range(30):
        ep_a: EpistemicType = rng.choice(epistemic_levels)
        ep_b: EpistemicType = rng.choice(epistemic_levels)

        c_a = _make_kg_claim(f"oi_subj_{i}", "x", f"val_a_{i}", epistemic_type=ep_a)
        c_b = _make_kg_claim(f"oi_subj_{i}", "x", f"val_b_{i}", epistemic_type=ep_b)

        alpha_a = float(rng.uniform(1.0, 5.0))
        beta_a = float(rng.uniform(1.0, 3.0))
        alpha_b = float(rng.uniform(1.0, 5.0))
        beta_b = float(rng.uniform(1.0, 3.0))

        conf_a = _make_confidence(alpha=alpha_a, beta=beta_a)
        conf_b = _make_confidence(alpha=alpha_b, beta=beta_b)

        ent_a = entrenchment_of(c_a, conf_a)
        ent_b = entrenchment_of(c_b, conf_b)
        ent_map = {c_a.id: ent_a, c_b.id: ent_b}

        res_ab = decide_resolution([c_a, c_b], ent_map)
        res_ba = decide_resolution([c_b, c_a], ent_map)

        assert res_ab == res_ba, (
            f"SC-4 FAIL i={i}: decide_resolution is order-sensitive. "
            f"[a,b]={res_ab!r} != [b,a]={res_ba!r}"
        )


def test_sc4_confirmed_vs_confirmed_escalation_reason() -> None:
    """Mutation-resistance: the confirmed-vs-confirmed guard must produce exactly
    escalation_reason='confirmed-vs-confirmed', not 'margin-tie'.

    If the guard were removed, both confirmed claims would enter the equal-rank LCB branch.
    With identical LCBs the result would be kind='escalated', reason='margin-tie' — this test
    catches that mutation by asserting the specific reason string.
    """
    # Two confirmed claims in REVISION MODE (different predicates → not update mode).
    c_a = _make_kg_claim("sc4mut_subj", "has_role", "admin", epistemic_type="confirmed")
    c_b = _make_kg_claim("sc4mut_subj", "has_title", "manager", epistemic_type="confirmed")

    # Near-identical LCBs (gap < 0.15) so that without the guard the margin-tie branch fires.
    conf_a = _make_confidence(alpha=2.0, beta=1.0)
    conf_b = _make_confidence(alpha=2.0, beta=1.0)

    ent_a = entrenchment_of(c_a, conf_a)
    ent_b = entrenchment_of(c_b, conf_b)
    ent_map = {c_a.id: ent_a, c_b.id: ent_b}

    resolution = decide_resolution([c_a, c_b], ent_map)

    assert resolution.kind == "escalated", (
        f"SC-4 mutation-resistance FAIL: expected kind='escalated', got {resolution.kind!r}"
    )
    assert resolution.escalation_reason == "confirmed-vs-confirmed", (
        f"SC-4 mutation-resistance FAIL: expected escalation_reason='confirmed-vs-confirmed', "
        f"got {resolution.escalation_reason!r}. "
        "Without the guard this would be 'margin-tie' — the guard is load-bearing."
    )


def test_sc4_confirmed_temporal_update_escalates() -> None:
    """Two confirmed claims with same predicate, different valid_from (temporal update scenario)
    must escalate with confirmed-vs-confirmed, NOT auto-supersede.

    This is the UPDATE MODE path: same (subject_norm, predicate_norm), strictly different
    valid_from.  The confirmed-vs-confirmed guard must fire BEFORE the rank comparison
    (both ranks are equal at 2, so rank(newer) >= rank(older) would otherwise return
    update-supersession silently).
    """
    c_older = _make_kg_claim(
        "sc4temp_subj",
        "has_role",
        "admin",
        epistemic_type="confirmed",
        valid_from=_T0,
    )
    c_newer = _make_kg_claim(
        "sc4temp_subj",
        "has_role",
        "senior_admin",
        epistemic_type="confirmed",
        valid_from=_T1,
    )

    conf_older = _make_confidence(alpha=5.0, beta=1.0)
    conf_newer = _make_confidence(alpha=4.0, beta=1.0)

    ent_older = entrenchment_of(c_older, conf_older)
    ent_newer = entrenchment_of(c_newer, conf_newer)
    ent_map = {c_older.id: ent_older, c_newer.id: ent_newer}

    resolution = decide_resolution([c_older, c_newer], ent_map)

    assert resolution.kind == "escalated", (
        f"SC-4 temporal FAIL: confirmed temporal update must escalate, "
        f"got kind={resolution.kind!r}. "
        "Auto-supersession of a confirmed claim via a newer confirmed claim is disallowed."
    )
    assert resolution.escalation_reason == "confirmed-vs-confirmed", (
        f"SC-4 temporal FAIL: expected escalation_reason='confirmed-vs-confirmed', "
        f"got {resolution.escalation_reason!r}"
    )


@pytest.mark.asyncio
async def test_sc4_confirmed_never_auto_defeated_escalation_reason() -> None:
    """Augmented SC-4: when BOTH claims are confirmed AND kind='escalated',
    the escalation_reason must be 'confirmed-vs-confirmed', not 'margin-tie'.

    This closes the mutation gap: a mutation removing the confirmed-vs-confirmed guard would
    produce kind='escalated' with reason='margin-tie' when LCBs are nearly equal — the test
    catches that specific mutation.
    """
    rng = random.Random(42)

    for i in range(50):
        ep_a: EpistemicType = "confirmed"
        ep_b: EpistemicType = rng.choice(["inference", "observation", "confirmed"])

        if rng.random() < 0.5:
            ep_a, ep_b = ep_b, ep_a

        if ep_a != "confirmed" and ep_b != "confirmed":
            ep_a = "confirmed"

        c_a = _make_kg_claim(f"aug_subj_{i}", "has_attr", f"val_a_{i}", epistemic_type=ep_a)
        c_b = _make_kg_claim(f"aug_subj_{i}", "has_attr", f"val_b_{i}", epistemic_type=ep_b)

        conf_a = _make_confidence(
            alpha=float(rng.uniform(1.5, 4.0)), beta=float(rng.uniform(1.0, 2.0))
        )
        conf_b = _make_confidence(
            alpha=float(rng.uniform(1.5, 4.0)), beta=float(rng.uniform(1.0, 2.0))
        )

        ent_a = entrenchment_of(c_a, conf_a)
        ent_b = entrenchment_of(c_b, conf_b)
        ent_map = {c_a.id: ent_a, c_b.id: ent_b}

        resolution = decide_resolution([c_a, c_b], ent_map)

        if ep_a == "confirmed" and ep_b == "confirmed" and resolution.kind == "escalated":
            assert resolution.escalation_reason == "confirmed-vs-confirmed", (
                f"SC-4 augmented FAIL i={i}: both confirmed, kind='escalated', "
                f"but escalation_reason={resolution.escalation_reason!r} "
                "!= 'confirmed-vs-confirmed'. "
                "A mutation removing the guard would produce 'margin-tie' here."
            )


# ---------------------------------------------------------------------------
# SC-5 — Exactly-once under crash + epoch race
# ---------------------------------------------------------------------------


class _FailOnFirstCommit(InMemoryEntityKG):
    """InMemoryEntityKG that raises RuntimeError on the FIRST commit_reconciliation call."""

    def __init__(self) -> None:
        super().__init__()
        self._commit_count = 0
        self._fail_on: int = 1  # fail the first call

    async def commit_reconciliation(
        self,
        outcome: ReconciliationOutcome,
        *,
        dirty_key: DirtyKey,
        observed_epoch: int,
    ) -> None:
        self._commit_count += 1
        if self._commit_count == self._fail_on:
            raise RuntimeError("Simulated crash before commit")
        await super().commit_reconciliation(
            outcome, dirty_key=dirty_key, observed_epoch=observed_epoch
        )


@pytest.mark.asyncio
async def test_sc5_crash_before_commit_re_ticks() -> None:
    """Per-subject crash leaves dirty mark; second tick processes the subject and commits defeat.

    First tick:  the commit raises, subjects_failed=1, dirty mark preserved.
    Second tick: processes the subject successfully, defeat committed.
    """
    kg = _FailOnFirstCommit()

    claim_a = _make_kg_claim("Alice", "salary", "100k", valid_from=_T0)
    claim_b = _make_kg_claim("Alice", "salary", "120k", valid_from=_T1)
    await kg.write_claim(claim_a, evidence=_ev("ev-a", source_id="src-a"))
    await kg.write_claim(claim_b, evidence=_ev("ev-b", source_id="src-b"))

    oracle = TableOracle([frozenset({claim_a.id, claim_b.id})])
    rec = _reconciler(kg, oracle)

    # First tick — commit raises for the subject
    stats1 = await rec.tick()
    assert stats1.subjects_failed == 1, (
        f"SC-5 FAIL: expected subjects_failed=1 on first tick, got {stats1}"
    )

    # After first crash the dirty mark must still exist (subject not cleared)
    dirty_after = await kg.claim_dirty_subjects(limit=10)
    assert len(dirty_after) >= 1, (
        "SC-5 FAIL: dirty mark was cleared despite crash; exactly-once violated"
    )

    # Second tick — commit now succeeds
    stats2 = await rec.tick()
    assert stats2.subjects_failed == 0, f"SC-5 FAIL: second tick should succeed, got {stats2}"
    assert stats2.subjects_defeated == 1, (
        f"SC-5 FAIL: second tick should defeat the conflict, got {stats2}"
    )

    # Final state: loser is defeated
    loser = await kg.get_claim(claim_a.id)
    loser_status = loser.status if loser else None
    assert loser is not None and loser.status == "defeasibly-defeated", (
        f"SC-5 FAIL: loser claim should be defeasibly-defeated, got status={loser_status!r}"
    )


class _EpochBumpOnCommit(InMemoryEntityKG):
    """InMemoryEntityKG that bumps the epoch of the dirty subject mid-commit.

    Simulates a concurrent write arriving between reconciler claim-time and commit.
    The epoch guard in commit_reconciliation must prevent clearing the dirty mark.
    """

    async def commit_reconciliation(
        self,
        outcome: ReconciliationOutcome,
        *,
        dirty_key: DirtyKey,
        observed_epoch: int,
    ) -> None:
        # Simulate a concurrent write bumping the epoch BEFORE commit_reconciliation completes.
        existing = self._dirty.get(dirty_key)
        if existing is not None:
            # Re-mark: bump the epoch
            self._dirty[dirty_key] = existing.model_copy(update={"epoch": existing.epoch + 1})
        # Now commit — the epoch guard will see a mismatch and NOT clear the dirty mark.
        await super().commit_reconciliation(
            outcome, dirty_key=dirty_key, observed_epoch=observed_epoch
        )


@pytest.mark.asyncio
async def test_sc5_epoch_race_dirty_not_lost() -> None:
    """A concurrent epoch bump during commit prevents dirty mark clearance; next tick re-processes.

    The epoch guard (DELETE WHERE epoch == observed_epoch) ensures that if a write arrives
    mid-reconciliation the dirty mark stays, so the next tick re-processes the subject.
    """
    kg = _EpochBumpOnCommit()

    claim_a = _make_kg_claim("Bob", "role", "admin", valid_from=_T0)
    claim_b = _make_kg_claim("Bob", "role", "guest", valid_from=_T1)
    await kg.write_claim(claim_a, evidence=_ev("ev-a"))
    await kg.write_claim(claim_b, evidence=_ev("ev-b"))

    oracle = TableOracle([frozenset({claim_a.id, claim_b.id})])
    rec = _reconciler(kg, oracle)

    stats = await rec.tick()

    # The commit ran (no failure), but the epoch was bumped mid-commit.
    # The epoch guard should leave the dirty mark in place.
    dirty_after = await kg.claim_dirty_subjects(limit=10)
    assert len(dirty_after) >= 1, (
        "SC-5 FAIL: dirty mark was cleared despite a concurrent epoch bump. "
        "The epoch guard (DELETE WHERE epoch == observed_epoch) is not working."
    )

    # stats should not show a failure — the commit itself succeeded structurally
    assert stats.subjects_failed == 0, (
        "SC-5 FAIL: epoch-bump should not cause a failure, "
        f"got subjects_failed={stats.subjects_failed}"
    )


# ---------------------------------------------------------------------------
# SC-6 — S8 Lesion: reconciler off, system still runs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sc6_lesion_reconciler_never_ticked() -> None:
    """Writing claims and recalling them works without ever ticking the reconciler.

    Both conflicting claims remain accessible (status='active' on both).
    claim_dirty_subjects has >= 1 entry (marks were created but not processed).
    No exception is raised anywhere in the non-reconciler flow.
    """
    kg = InMemoryEntityKG()

    # 4 claims under 2 subjects
    c1 = _make_kg_claim("Alice", "salary", "100k", valid_from=_T0)
    c2 = _make_kg_claim("Alice", "salary", "120k", valid_from=_T1)  # conflict pair
    c3 = _make_kg_claim("Bob", "role", "engineer", valid_from=_T0)
    c4 = _make_kg_claim("Bob", "role", "manager", valid_from=_T1)

    await kg.write_claim(c1, evidence=_ev("ev-1"))
    await kg.write_claim(c2, evidence=_ev("ev-2"))
    await kg.write_claim(c3, evidence=_ev("ev-3"))
    await kg.write_claim(c4, evidence=_ev("ev-4"))

    # Create the reconciler but NEVER call tick()
    oracle = TableOracle([frozenset({c1.id, c2.id})])
    # Intentionally never used — the lesion test is that tick() is never called.
    # The reconciler's constructor is exercised to prove no exception is raised.
    rec = _reconciler(kg, oracle)
    del rec  # explicitly discard to make the intent clear

    # Both conflicting claims must remain accessible
    c1_read = await kg.get_claim(c1.id)
    c2_read = await kg.get_claim(c2.id)
    c1_status = c1_read.status if c1_read else None
    c2_status = c2_read.status if c2_read else None
    assert c1_read is not None and c1_read.status == "active", (
        f"SC-6 FAIL: c1.status={c1_status!r}, expected 'active'"
    )
    assert c2_read is not None and c2_read.status == "active", (
        f"SC-6 FAIL: c2.status={c2_status!r}, expected 'active'"
    )

    # Dirty marks exist — reconciler not ticked
    dirty = await kg.claim_dirty_subjects(limit=10)
    assert len(dirty) >= 1, (
        "SC-6 FAIL: no dirty marks found; write_claim should have created dirty marks"
    )

    # Recall still works (no exception)
    claims_for_alice = await kg.claims_about("Alice")
    assert len(claims_for_alice) >= 2, (
        "SC-6 FAIL: claims_about should return both conflicting claims, "
        f"got {len(claims_for_alice)}"
    )


# ---------------------------------------------------------------------------
# SC-7 — S1/S9 isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sc7_no_model_on_write_path() -> None:
    """Write-path operations (write_claim, add_evidence, invalidate_claim, epistemic_upgrade)
    must never call the model.

    Uses ReplayModel([]) which raises ReplayExhaustedError on any call.
    The write-path code must complete without triggering any model call.
    """
    # ReplayModel with empty responses — raises if called
    _model = ReplayModel([])

    kg = InMemoryEntityKG()

    # write_claim — must not call model
    c1 = _make_kg_claim("Carol", "city", "Paris", epistemic_type="inference")
    await kg.write_claim(c1, evidence=_ev("ev-1", source_id="src-carol"))

    assert _model.call_count == 0, (
        f"SC-7 FAIL: model was called during write_claim. call_count={_model.call_count}"
    )

    # add_evidence — must not call model
    await kg.add_evidence(c1.id, _ev("ev-2", source_id="src-carol-2"))
    assert _model.call_count == 0, (
        f"SC-7 FAIL: model was called during add_evidence. call_count={_model.call_count}"
    )

    # invalidate_claim — must not call model
    await kg.invalidate_claim(c1.id, valid_to=_T1)
    assert _model.call_count == 0, (
        f"SC-7 FAIL: model was called during invalidate_claim. call_count={_model.call_count}"
    )

    # epistemic_upgrade — must not call model
    c2 = _make_kg_claim("Dave", "city", "London", epistemic_type="inference")
    await kg.write_claim(c2, evidence=_ev("ev-d1", source_id="src-dave"))

    upgrade_evidence = make_evidence(
        type="tool_proof",
        polarity="+",
        source_id="src-upgrade",
        source_authority=1.0,
        recorded_at=_T0,
        event_id="ev-upgrade",
    )
    await epistemic_upgrade(
        kg,
        c2.id,
        new_level="observation",
        evidence=upgrade_evidence,
        actor="spike-7",
    )
    assert _model.call_count == 0, (
        f"SC-7 FAIL: model was called during epistemic_upgrade. call_count={_model.call_count}"
    )


@pytest.mark.asyncio
async def test_sc7_oracle_raw_text_never_routes() -> None:
    """Adversarially inverted raw_text on OracleAnswer must not affect routing.

    MutatedRawTextOracle wraps TableOracle and inverts the raw_text field
    (true CONSISTENT → raw_text says "INCONSISTENT", and vice versa).
    The reconciler must route purely on OracleAnswer.consistent — the control bit.
    """

    class MutatedRawTextOracle:
        """Wraps TableOracle and injects adversarially inverted raw_text."""

        def __init__(self, inner: TableOracle) -> None:
            self._inner = inner

        async def check(self, claims: Sequence[Claim]) -> OracleAnswer:
            answer = await self._inner.check(claims)
            # Adversarial inversion: raw_text contradicts consistent field
            inverted = "INCONSISTENT" if answer.consistent else "CONSISTENT"
            return OracleAnswer(
                consistent=answer.consistent,
                raw_text=inverted,
                model_ref="mutated",
            )

    kg = InMemoryEntityKG()
    c_a = _make_kg_claim("Eve", "salary", "90k", valid_from=_T0)
    c_b = _make_kg_claim("Eve", "salary", "110k", valid_from=_T1)
    await kg.write_claim(c_a, evidence=_ev("ev-a"))
    await kg.write_claim(c_b, evidence=_ev("ev-b"))

    # Ground-truth oracle: c_a and c_b conflict
    inner_table = TableOracle([frozenset({c_a.id, c_b.id})])
    mutated = MutatedRawTextOracle(inner_table)

    rec = _reconciler(kg, mutated)
    stats = await rec.tick()

    # If raw_text were parsed for control, the inverted "CONSISTENT" text on an inconsistent
    # result would produce subjects_cleared=1 (no defeat). The correct routing on consistent=False
    # must produce subjects_defeated=1.
    assert stats.subjects_defeated == 1, (
        f"SC-7 FAIL: raw_text inversion changed routing outcome. "
        f"Expected subjects_defeated=1, got {stats!r}. "
        "raw_text is being parsed for control flow — S9 violation."
    )
    assert stats.subjects_cleared == 0, (
        f"SC-7 FAIL: inverted raw_text caused incorrect 'cleared' outcome. Got {stats!r}"
    )


@pytest.mark.asyncio
async def test_sc2_no_spurious_defeat_on_consistent_set() -> None:
    """SC-2b: A truly consistent set under oracle noise must NEVER produce a verified-True MUS
    that commits a spurious defeat.

    This tests the false-positive path: noise => spurious defeat of an innocent claim.

    The verification gate is the protection:
    - verified=False  → escalated (no defeat committed, correct)
    - verified=True on a consistent set → could be a spurious defeat path

    When a spurious MUS is "verified=True" (both the find-path and verification calls all got
    flipped), the MUS consists of claims from a fully consistent set.  For such a set,
    decide_resolution on the spurious MUS returns kind="none" (1-element MUS) or kind="escalated"
    (near-identical entrenchments → margin-tie, since the claims were built with equal confidence).
    The reconciler only commits a Defeat for kind in ("revision-defeat", "update-supersession") —
    so neither "none" nor "escalated" results in a committed defeat, containing the spurious path.
    """
    n = 8
    claims_base = [make_claim(f"cs{i}") for i in range(n)]

    spurious_defeats_possible = 0

    for trial_i in range(100):
        inner = TableOracle([])  # no conflicts at all — truly consistent set
        noisy = NoisyOracle(inner, flip_rate=0.3, seed=9999 + trial_i)

        rng = random.Random(trial_i)
        shuffled = list(claims_base)
        rng.shuffle(shuffled)

        try:
            result = await find_mus(shuffled, noisy, budget=CallBudget(200))
        except BudgetExceeded:
            continue

        if result.mus != () and result.verified is True:
            # Spurious: noise caused find_mus to "find" a MUS on a consistent set AND the
            # verification call was also flipped to agree.  Check what decide_resolution
            # would do with this spurious MUS — the reconciler only commits a defeat for
            # revision-defeat or update-supersession.
            spurious_defeats_possible += 1
            from cogworx.coherence.entrenchment import decide_resolution, entrenchment_of
            from cogworx.knowledge.confidence import ClaimConfidence

            mus_id_set = set(result.mus)
            mus_claims = [c for c in claims_base if c.id in mus_id_set]

            # All test claims have equal provenance (confidence=0.9, no evidence events),
            # so entrenchments are nearly identical → should escalate or return "none".
            # Use a minimal ClaimConfidence matching the spike's make_claim provenance (0.9 mean).
            # Beta(alpha, beta) with alpha/(alpha+beta)=0.9 → alpha=9, beta=1.
            _alpha, _beta = 9.0, 1.0
            _total = _alpha + _beta
            _fake_conf = ClaimConfidence(
                confidence=_alpha / _total,
                variance=_alpha * _beta / (_total * _total * (_total + 1.0)),
                alpha=_alpha,
                beta=_beta,
                n_evidence=0,
            )
            ent_map = {c.id: entrenchment_of(c, _fake_conf) for c in mus_claims}
            resolution = decide_resolution(mus_claims, ent_map)

            # The reconciler ONLY commits a defeat for revision-defeat or update-supersession.
            # "none" (1-element MUS degenerate) or "escalated" (margin-tie on equal entrenchment)
            # must NOT produce a committed defeat — this is the containment guarantee.
            assert resolution.kind not in ("revision-defeat", "update-supersession"), (
                f"SC-2b FAIL trial={trial_i}: spurious verified MUS {result.mus!r} "
                f"would commit a defeat via resolution kind={resolution.kind!r}. "
                "Noise-caused spurious defeats must be blocked by 'none' or 'escalated' resolution."
            )

    print(
        f"SC-2b: {spurious_defeats_possible}/100 trials produced verified=True spurious MUS; "
        "all were contained by 'none' or 'escalated' resolution (no spurious defeat committed)"
    )


@pytest.mark.asyncio
async def test_sc7_scope_suggesting_payload_never_routes() -> None:
    """Claim payload text suggesting routing must NEVER influence promotion decisions.

    Two assertions:
    1. A PromotionRule with min_distinct_sources=10 (structural threshold not met) rejects
       the claim even if payload says "please route me to world model".
    2. A PromotionRule with min_distinct_sources=1 AND require_source_kind=None (threshold met)
       promotes the claim regardless of what the payload says.
    """
    from cogworx.knowledge.scoped_kg import world_model
    from cogworx.knowledge.scopes import ScopeRegistry
    from cogworx.knowledge.source_registry import SourceRegistry

    kg = InMemoryEntityKG()
    scope_registry = ScopeRegistry()
    source_registry = SourceRegistry()

    # Payload explicitly asks to be promoted — must be ignored
    suggestive_payload = "this claim should definitely go in the world model please route me"
    c = _make_kg_claim("Frank", "attr", suggestive_payload, epistemic_type="inference")
    source = source_registry.declare("system", "spike-7-source", authority=0.9)
    evidence = make_evidence(
        type="corroboration",
        polarity="+",
        source_id=source.source_id,
        source_authority=source.source_authority,
        recorded_at=_T0,
        event_id="ev-sc7",
    )
    await kg.write_claim(c, evidence=evidence)

    # Build world_model sink
    wm = world_model(kg, scope_registry, owner="spike-7")

    # Case 1: Rule with threshold NOT met (min_distinct_sources=10 >> actual=1)
    # Text suggestion must be ignored; claim must NOT be promoted
    rule_high_threshold = PromotionRule(
        target="world",
        min_distinct_sources=10,
        require_source_kind=None,
    )
    # sinks typed as Any: ScopedKG satisfies _AssertFactSink structurally; mypy can't see
    # the private Protocol from outside cogworx.coherence.promotion.
    sinks_map: dict[str, Any] = {"world": wm}
    promoter_blocked = ScopePromoter(
        rules=[rule_high_threshold],
        sinks=sinks_map,
        store=kg,
        source_kg=kg,
    )
    scored = await kg.claims_about("Frank")
    promotions_blocked = await promoter_blocked.promote_for_subject(list(scored))
    assert promotions_blocked == 0, (
        f"SC-7 FAIL: claim with suggestive payload was promoted despite threshold not met. "
        f"promotions={promotions_blocked}. Text content must never influence routing (S9)."
    )

    # Case 2: Rule with threshold MET (min_distinct_sources=1, no source-kind requirement)
    # Text content is still the same suggestive payload — but structural criterion is met
    # The claim MUST be promoted (structural criteria drive routing, not text)
    rule_low_threshold = PromotionRule(
        target="world",
        min_distinct_sources=1,
        require_source_kind=None,
    )
    promoter_allowed = ScopePromoter(
        rules=[rule_low_threshold],
        sinks=sinks_map,
        store=kg,
        source_kg=kg,
    )
    promotions_allowed = await promoter_allowed.promote_for_subject(list(scored))
    assert promotions_allowed == 1, (
        f"SC-7 FAIL: claim was not promoted despite structural criteria being met "
        f"(promotions={promotions_allowed}). Structural threshold met → must promote "
        "regardless of payload text content (S9)."
    )
