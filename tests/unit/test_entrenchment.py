"""Unit tests for cogworx.coherence.entrenchment — pure, no I/O, no fixtures."""

from __future__ import annotations

import math
import random
from datetime import UTC, datetime

import pytest

from cogworx.claims.provenance import Claim, EpistemicType, Provenance
from cogworx.coherence.entrenchment import (
    EPISTEMIC_RANK,
    Entrenchment,
    decide_resolution,
    entrenchment_of,
)
from cogworx.knowledge.confidence import ClaimConfidence

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_LATER = datetime(2026, 6, 1, tzinfo=UTC)

_PROV = Provenance(source="tool", confidence=0.9, recorded_at=_NOW)


def _claim(
    cid: str,
    *,
    epistemic_type: EpistemicType = "inference",
    predicate: str = "pred:foo",
    subject: str = "subject:x",
    valid_from: datetime = _NOW,
    valid_to: datetime | None = None,
) -> Claim:
    return Claim(
        id=cid,
        subject=subject,
        predicate=predicate,
        payload=f"payload-{cid}",
        epistemic_type=epistemic_type,
        provenance=_PROV,
        valid_from=valid_from,
        valid_to=valid_to,
        ingest_time=_NOW,
        created_by="test",
    )


def _confidence(alpha: float, beta: float) -> ClaimConfidence:
    s = alpha + beta
    confidence = alpha / s
    variance = (alpha * beta) / (s * s * (s + 1.0))
    return ClaimConfidence(
        alpha=alpha,
        beta=beta,
        confidence=confidence,
        variance=variance,
        n_evidence=0,
    )


def _entrenchment(
    cid: str,
    *,
    rank: int,
    alpha: float,
    beta: float,
    z: float = 1.0,
) -> Entrenchment:
    conf = _confidence(alpha, beta)
    lcb = conf.confidence - z * math.sqrt(conf.variance)
    return Entrenchment(rank=rank, lcb=lcb, claim_id=cid)


# ---------------------------------------------------------------------------
# Scenario builders — each returns (claims, entrenchments) for the parametrized table
# ---------------------------------------------------------------------------


# UPDATE MODE: newer=observation(rank=1), older=inference(rank=0) → update-supersession
# winner rank = 1 (observation)
def _build_update_higher_rank_wins() -> tuple[list[Claim], dict[str, Entrenchment]]:
    older = _claim("old-a", epistemic_type="inference", valid_from=_NOW)
    newer = _claim("new-b", epistemic_type="observation", valid_from=_LATER)
    ents = {
        "old-a": _entrenchment("old-a", rank=EPISTEMIC_RANK["inference"], alpha=2.0, beta=2.0),
        "new-b": _entrenchment("new-b", rank=EPISTEMIC_RANK["observation"], alpha=3.0, beta=1.0),
    }
    return [older, newer], ents


# UPDATE MODE: newer=inference(rank=0), older=confirmed(rank=2) → escalate(rank-blocks-update)
def _build_update_lower_rank_blocks() -> tuple[list[Claim], dict[str, Entrenchment]]:
    older = _claim("conf-a", epistemic_type="confirmed", valid_from=_NOW)
    newer = _claim("inf-b", epistemic_type="inference", valid_from=_LATER)
    ents = {
        "conf-a": _entrenchment("conf-a", rank=EPISTEMIC_RANK["confirmed"], alpha=5.0, beta=1.0),
        "inf-b": _entrenchment("inf-b", rank=EPISTEMIC_RANK["inference"], alpha=2.0, beta=2.0),
    }
    return [older, newer], ents


# REVISION: both confirmed → escalate(confirmed-vs-confirmed)
def _build_revision_confirmed_vs_confirmed() -> tuple[list[Claim], dict[str, Entrenchment]]:
    a = _claim("conf-r1", epistemic_type="confirmed", predicate="pred:rev")
    b = _claim("conf-r2", epistemic_type="confirmed", predicate="pred:rev", subject="subject:y")
    ents = {
        "conf-r1": _entrenchment("conf-r1", rank=EPISTEMIC_RANK["confirmed"], alpha=5.0, beta=1.0),
        "conf-r2": _entrenchment("conf-r2", rank=EPISTEMIC_RANK["confirmed"], alpha=4.0, beta=1.0),
    }
    return [a, b], ents


# REVISION: confirmed(rank=2) vs observation(rank=1), different predicates
# → revision-defeat, winner rank=2
def _build_revision_rank_diff_hi_wins() -> tuple[list[Claim], dict[str, Entrenchment]]:
    a = _claim("conf-hi", epistemic_type="confirmed", predicate="pred:alpha")
    b = _claim("obs-lo", epistemic_type="observation", predicate="pred:beta")
    ents = {
        "conf-hi": _entrenchment("conf-hi", rank=EPISTEMIC_RANK["confirmed"], alpha=5.0, beta=1.0),
        "obs-lo": _entrenchment("obs-lo", rank=EPISTEMIC_RANK["observation"], alpha=3.0, beta=2.0),
    }
    return [a, b], ents


# REVISION: observation(rank=1) vs inference(rank=0) → revision-defeat, winner rank=1
def _build_revision_rank_diff_lo_wins_by_rank() -> tuple[list[Claim], dict[str, Entrenchment]]:
    a = _claim("obs-mid", epistemic_type="observation", predicate="pred:p1")
    b = _claim("inf-low", epistemic_type="inference", predicate="pred:p2")
    ents = {
        "obs-mid": _entrenchment(
            "obs-mid", rank=EPISTEMIC_RANK["observation"], alpha=3.0, beta=2.0
        ),
        "inf-low": _entrenchment("inf-low", rank=EPISTEMIC_RANK["inference"], alpha=4.0, beta=1.0),
    }
    return [a, b], ents


# REVISION: equal rank (both observation), large LCB gap >= 0.15 → revision-defeat
def _build_revision_equal_rank_lcb_gap_win() -> tuple[list[Claim], dict[str, Entrenchment]]:
    a = _claim("obs-strong", epistemic_type="observation", predicate="pred:q1")
    b = _claim("obs-weak", epistemic_type="observation", predicate="pred:q2")
    # Strong: high alpha, low beta → high LCB; Weak: low alpha, high beta → low LCB
    ents = {
        "obs-strong": _entrenchment(
            "obs-strong", rank=EPISTEMIC_RANK["observation"], alpha=10.0, beta=1.0
        ),
        "obs-weak": _entrenchment(
            "obs-weak", rank=EPISTEMIC_RANK["observation"], alpha=1.5, beta=5.0
        ),
    }
    # Sanity-check that gap >= 0.15 at build time
    gap = abs(ents["obs-strong"].lcb - ents["obs-weak"].lcb)
    assert gap >= 0.15, f"Fixture gap too small: {gap}"
    return [a, b], ents


# REVISION: equal rank (both observation), LCB gap < 0.15 → escalate(margin-tie)
def _build_revision_equal_rank_margin_tie() -> tuple[list[Claim], dict[str, Entrenchment]]:
    a = _claim("obs-tie1", epistemic_type="observation", predicate="pred:t1")
    b = _claim("obs-tie2", epistemic_type="observation", predicate="pred:t2")
    # Near-identical posteriors — same alpha/beta → identical LCB → gap = 0.0
    ents = {
        "obs-tie1": _entrenchment(
            "obs-tie1", rank=EPISTEMIC_RANK["observation"], alpha=3.0, beta=3.0
        ),
        "obs-tie2": _entrenchment(
            "obs-tie2", rank=EPISTEMIC_RANK["observation"], alpha=3.0, beta=3.0
        ),
    }
    gap = abs(ents["obs-tie1"].lcb - ents["obs-tie2"].lcb)
    assert gap < 0.15, f"Fixture gap too large: {gap}"
    return [a, b], ents


_SCENARIO_BUILDERS = {
    "update_higher_rank_wins": _build_update_higher_rank_wins,
    "update_lower_rank_blocks": _build_update_lower_rank_blocks,
    "revision_confirmed_vs_confirmed": _build_revision_confirmed_vs_confirmed,
    "revision_rank_diff_hi_wins": _build_revision_rank_diff_hi_wins,
    "revision_rank_diff_lo_wins_by_rank": _build_revision_rank_diff_lo_wins_by_rank,
    "revision_equal_rank_lcb_gap_win": _build_revision_equal_rank_lcb_gap_win,
    "revision_equal_rank_margin_tie": _build_revision_equal_rank_margin_tie,
}

# ---------------------------------------------------------------------------
# test_decision_table_rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario,expected_kind,expected_winner_rank,expected_escalation",
    [
        # UPDATE MODE
        ("update_higher_rank_wins", "update-supersession", 1, None),
        ("update_lower_rank_blocks", "escalated", None, "rank-blocks-update"),
        # REVISION MODE
        ("revision_confirmed_vs_confirmed", "escalated", None, "confirmed-vs-confirmed"),
        ("revision_rank_diff_hi_wins", "revision-defeat", 2, None),
        ("revision_rank_diff_lo_wins_by_rank", "revision-defeat", 1, None),
        ("revision_equal_rank_lcb_gap_win", "revision-defeat", None, None),
        ("revision_equal_rank_margin_tie", "escalated", None, "margin-tie"),
    ],
)
def test_decision_table_rows(
    scenario: str,
    expected_kind: str,
    expected_winner_rank: int | None,
    expected_escalation: str | None,
) -> None:
    claims, ents = _SCENARIO_BUILDERS[scenario]()
    res = decide_resolution(claims, ents)

    assert res.kind == expected_kind, f"[{scenario}] kind: got {res.kind!r}"
    assert res.escalation_reason == expected_escalation, (
        f"[{scenario}] escalation_reason: got {res.escalation_reason!r}"
    )

    if expected_winner_rank is not None:
        assert res.winner_id is not None, f"[{scenario}] expected a winner"
        winner_ent = ents[res.winner_id]
        assert winner_ent.rank == expected_winner_rank, (
            f"[{scenario}] winner rank: got {winner_ent.rank}"
        )
    else:
        if expected_kind == "escalated":
            assert res.winner_id is None, f"[{scenario}] escalated must have winner_id=None"


# ---------------------------------------------------------------------------
# test_invariants_confirmed_never_auto_defeated
# ---------------------------------------------------------------------------


def test_invariants_confirmed_never_auto_defeated() -> None:
    """A confirmed claim is NEVER the loser_id in a revision-defeat."""
    # 20 combinations: varying ranks and LCBs where at least one claim is confirmed.
    combos: list[tuple[EpistemicType, float, float, EpistemicType, float, float]] = [
        # (type_a, alpha_a, beta_a, type_b, alpha_b, beta_b)
        ("confirmed", 5.0, 1.0, "inference", 4.0, 1.0),
        ("confirmed", 2.0, 2.0, "inference", 3.0, 1.0),
        ("confirmed", 3.0, 3.0, "observation", 4.0, 1.0),
        ("confirmed", 1.0, 5.0, "observation", 3.0, 2.0),
        ("confirmed", 5.0, 1.0, "observation", 5.0, 1.0),
        ("inference", 4.0, 1.0, "confirmed", 2.0, 2.0),
        ("inference", 3.0, 3.0, "confirmed", 3.0, 3.0),
        ("observation", 4.0, 1.0, "confirmed", 1.0, 5.0),
        ("observation", 2.0, 4.0, "confirmed", 5.0, 1.0),
        ("observation", 3.0, 2.0, "confirmed", 3.0, 2.0),
        # Both confirmed → all escalated (confirmed never loses)
        ("confirmed", 5.0, 1.0, "confirmed", 2.0, 2.0),
        ("confirmed", 3.0, 2.0, "confirmed", 3.0, 2.0),
        ("confirmed", 1.0, 5.0, "confirmed", 4.0, 1.0),
        ("confirmed", 2.0, 3.0, "confirmed", 2.0, 3.0),
        ("confirmed", 4.0, 1.0, "confirmed", 1.0, 4.0),
        # inference vs confirmed edge cases
        ("inference", 1.0, 1.0, "confirmed", 1.0, 1.0),
        ("inference", 5.0, 1.0, "confirmed", 1.0, 5.0),
        ("observation", 1.0, 1.0, "confirmed", 1.0, 1.0),
        ("inference", 2.0, 3.0, "confirmed", 3.0, 2.0),
        ("observation", 3.0, 1.0, "confirmed", 2.0, 4.0),
    ]

    assert len(combos) == 20

    for i, (et_a, alpha_a, beta_a, et_b, alpha_b, beta_b) in enumerate(combos):
        cid_a = f"inv-a-{i}"
        cid_b = f"inv-b-{i}"
        a = _claim(cid_a, epistemic_type=et_a, predicate=f"pred:inv{i}-a", subject=f"subj:inv{i}")
        b = _claim(cid_b, epistemic_type=et_b, predicate=f"pred:inv{i}-b", subject=f"subj:inv{i}")
        ents = {
            cid_a: _entrenchment(cid_a, rank=EPISTEMIC_RANK[et_a], alpha=alpha_a, beta=beta_a),
            cid_b: _entrenchment(cid_b, rank=EPISTEMIC_RANK[et_b], alpha=alpha_b, beta=beta_b),
        }
        res = decide_resolution([a, b], ents)

        if res.kind == "revision-defeat":
            # A confirmed claim must never appear as a loser
            for loser_id in res.loser_ids:
                losing_claim = a if loser_id == cid_a else b
                assert losing_claim.epistemic_type != "confirmed", (
                    f"combo {i}: confirmed claim {loser_id} was a revision-defeat loser"
                )


# ---------------------------------------------------------------------------
# test_invariant_order_insensitive
# ---------------------------------------------------------------------------


def test_invariant_order_insensitive() -> None:
    """decide([A, B]) == decide([B, A]) for 30 random claim pairs."""
    rng = random.Random(42)
    epistemic_types: list[EpistemicType] = ["inference", "observation", "confirmed"]

    for i in range(30):
        et_a: EpistemicType = rng.choice(epistemic_types)
        et_b: EpistemicType = rng.choice(epistemic_types)
        alpha_a = rng.uniform(1.0, 8.0)
        beta_a = rng.uniform(1.0, 8.0)
        alpha_b = rng.uniform(1.0, 8.0)
        beta_b = rng.uniform(1.0, 8.0)

        # Vary valid_from to exercise both update and revision mode
        vf_a = _NOW
        vf_b = _LATER if rng.random() < 0.5 else _NOW

        cid_a = f"ord-a-{i}"
        cid_b = f"ord-b-{i}"
        a = _claim(cid_a, epistemic_type=et_a, predicate="pred:ord", valid_from=vf_a)
        b = _claim(cid_b, epistemic_type=et_b, predicate="pred:ord", valid_from=vf_b)

        ents = {
            cid_a: _entrenchment(cid_a, rank=EPISTEMIC_RANK[et_a], alpha=alpha_a, beta=beta_a),
            cid_b: _entrenchment(cid_b, rank=EPISTEMIC_RANK[et_b], alpha=alpha_b, beta=beta_b),
        }

        res_ab = decide_resolution([a, b], ents)
        res_ba = decide_resolution([b, a], ents)

        assert res_ab == res_ba, (
            f"pair {i}: order-sensitive result\n  [A,B]={res_ab}\n  [B,A]={res_ba}"
        )


# ---------------------------------------------------------------------------
# test_invariant_escalation_leaves_both_active
# ---------------------------------------------------------------------------


def test_invariant_escalation_leaves_both_active() -> None:
    """Escalated resolution: winner_id=None and loser_ids=()."""
    escalation_cases: list[tuple[list[Claim], dict[str, Entrenchment]]] = [
        _build_update_lower_rank_blocks(),
        _build_revision_confirmed_vs_confirmed(),
        _build_revision_equal_rank_margin_tie(),
    ]

    # Also test mus-gt-2 escalation
    c1 = _claim("gt2-a", epistemic_type="inference")
    c2 = _claim("gt2-b", epistemic_type="observation")
    c3 = _claim("gt2-c", epistemic_type="confirmed")
    ents_gt2 = {
        "gt2-a": _entrenchment("gt2-a", rank=0, alpha=2.0, beta=2.0),
        "gt2-b": _entrenchment("gt2-b", rank=1, alpha=3.0, beta=1.0),
        "gt2-c": _entrenchment("gt2-c", rank=2, alpha=5.0, beta=1.0),
    }
    escalation_cases.append(([c1, c2, c3], ents_gt2))

    for claims, ents in escalation_cases:
        res = decide_resolution(claims, ents)
        assert res.kind == "escalated", f"Expected escalated, got {res.kind!r}"
        assert res.winner_id is None, f"Escalated must have winner_id=None, got {res.winner_id!r}"
        assert res.loser_ids == (), f"Escalated must have loser_ids=(), got {res.loser_ids!r}"


# ---------------------------------------------------------------------------
# test_update_mode_set_valid_to
# ---------------------------------------------------------------------------


def test_update_mode_set_valid_to() -> None:
    """update-supersession sets set_valid_to to the newer claim's valid_from."""
    older = _claim("upd-old", epistemic_type="observation", valid_from=_NOW)
    newer = _claim("upd-new", epistemic_type="confirmed", valid_from=_LATER)
    ents = {
        "upd-old": _entrenchment(
            "upd-old", rank=EPISTEMIC_RANK["observation"], alpha=3.0, beta=1.0
        ),
        "upd-new": _entrenchment("upd-new", rank=EPISTEMIC_RANK["confirmed"], alpha=5.0, beta=1.0),
    }
    res = decide_resolution([older, newer], ents)

    assert res.kind == "update-supersession"
    assert res.winner_id == "upd-new"
    assert res.loser_ids == ("upd-old",)
    assert res.set_valid_to == _LATER


# ---------------------------------------------------------------------------
# test_revision_mode_set_valid_to_none
# ---------------------------------------------------------------------------


def test_revision_mode_set_valid_to_none() -> None:
    """revision-defeat never sets set_valid_to (must be None)."""
    a = _claim("rev-a", epistemic_type="confirmed", predicate="pred:rv1")
    b = _claim("rev-b", epistemic_type="inference", predicate="pred:rv2")
    ents = {
        "rev-a": _entrenchment("rev-a", rank=EPISTEMIC_RANK["confirmed"], alpha=5.0, beta=1.0),
        "rev-b": _entrenchment("rev-b", rank=EPISTEMIC_RANK["inference"], alpha=4.0, beta=1.0),
    }
    res = decide_resolution([a, b], ents)

    assert res.kind == "revision-defeat"
    assert res.set_valid_to is None


# ---------------------------------------------------------------------------
# test_entrenchment_lcb_formula
# ---------------------------------------------------------------------------


def test_entrenchment_lcb_formula() -> None:
    """entrenchment_of computes LCB = confidence - z * sqrt(variance) with correct rank."""
    alpha = 4.0
    beta = 2.0
    z = 1.0

    s = alpha + beta
    expected_confidence = alpha / s
    expected_variance = (alpha * beta) / (s * s * (s + 1.0))
    expected_lcb = expected_confidence - z * math.sqrt(expected_variance)

    claim = _claim("lcb-test", epistemic_type="observation")
    conf = _confidence(alpha, beta)

    ent = entrenchment_of(claim, conf, z=z)

    assert ent.rank == EPISTEMIC_RANK["observation"]
    assert ent.claim_id == "lcb-test"
    assert math.isclose(ent.lcb, expected_lcb, rel_tol=1e-12), (
        f"LCB mismatch: expected {expected_lcb}, got {ent.lcb}"
    )


# ---------------------------------------------------------------------------
# test_degenerate_mus_sizes
# ---------------------------------------------------------------------------


def test_degenerate_mus_empty() -> None:
    """Empty MUS → kind='none', no winner, no losers."""
    res = decide_resolution([], {})
    assert res.kind == "none"
    assert res.winner_id is None
    assert res.loser_ids == ()
    assert res.escalation_reason is None


def test_degenerate_mus_single() -> None:
    """Single-claim MUS → kind='none', winner=that claim."""
    c = _claim("solo", epistemic_type="observation")
    ent = _entrenchment("solo", rank=1, alpha=3.0, beta=1.0)
    res = decide_resolution([c], {"solo": ent})
    assert res.kind == "none"
    assert res.winner_id == "solo"
    assert res.loser_ids == ()


# ---------------------------------------------------------------------------
# test_update_mode_confirmed_vs_confirmed_escalates (regression)
# ---------------------------------------------------------------------------


def test_update_mode_confirmed_vs_confirmed_escalates() -> None:
    """Regression: UPDATE MODE must not auto-defeat a confirmed claim.

    Two confirmed claims with the same (subject_norm, predicate_norm) but different
    valid_from trigger UPDATE MODE.  The confirmed-vs-confirmed guard must fire BEFORE
    the rank check (both ranks are equal at 2, so rank(newer) >= rank(older) would
    otherwise return update-supersession silently).
    """
    a = _claim(
        "upd-conf-old",
        epistemic_type="confirmed",
        subject="subject:upd",
        predicate="pred:upd",
        valid_from=_NOW,
    )
    b = _claim(
        "upd-conf-new",
        epistemic_type="confirmed",
        subject="subject:upd",
        predicate="pred:upd",
        valid_from=_LATER,
    )
    ents = {
        "upd-conf-old": _entrenchment(
            "upd-conf-old", rank=EPISTEMIC_RANK["confirmed"], alpha=5.0, beta=1.0
        ),
        "upd-conf-new": _entrenchment(
            "upd-conf-new", rank=EPISTEMIC_RANK["confirmed"], alpha=4.0, beta=1.0
        ),
    }
    res = decide_resolution([a, b], ents)

    assert res.kind == "escalated", (
        f"Expected escalated, got {res.kind!r}; "
        "UPDATE MODE confirmed-vs-confirmed guard is missing or not reached"
    )
    assert res.escalation_reason == "confirmed-vs-confirmed", (
        f"Expected escalation_reason='confirmed-vs-confirmed', got {res.escalation_reason!r}"
    )
    assert res.winner_id is None
    assert res.loser_ids == ()


def test_decide_resolution_none_predicate_falls_to_revision() -> None:
    """Two claims with predicate=None fall to REVISION MODE, not UPDATE MODE.

    UPDATE MODE requires same (subject_norm, predicate_norm) AND strictly different valid_from.
    predicate=None normalises to None; same_predicate = (pred_a is not None and pred_a == pred_b)
    is False when pred_a is None.  So the pair falls through to REVISION MODE regardless of
    valid_from differences.  The result must NOT be kind='update-supersession'.
    """
    a = _claim("np-rev-a", epistemic_type="observation", predicate=None, valid_from=_NOW)  # type: ignore[arg-type]
    b = _claim("np-rev-b", epistemic_type="observation", predicate=None, valid_from=_LATER)  # type: ignore[arg-type]
    ents = {
        "np-rev-a": _entrenchment(
            "np-rev-a", rank=EPISTEMIC_RANK["observation"], alpha=3.0, beta=1.0
        ),
        "np-rev-b": _entrenchment(
            "np-rev-b", rank=EPISTEMIC_RANK["observation"], alpha=3.0, beta=1.0
        ),
    }
    res = decide_resolution([a, b], ents)

    assert res.kind != "update-supersession", (
        f"predicate=None claims must not trigger update-supersession, got kind={res.kind!r}"
    )


def test_mutation_guard_confirmed_vs_confirmed() -> None:
    """Verify the confirmed-vs-confirmed guard is mutation-resistant.

    This test fails if the guard is removed: without the guard, decide_resolution would
    enter the rank-check branch (rank(newer)==rank(older)==2, so 2>=2 is True) and return
    kind='update-supersession' with escalation_reason=None.  Asserting the specific reason
    string means that silently changing kind to 'escalated' with a DIFFERENT reason would
    also fail — the guard must produce exactly 'confirmed-vs-confirmed'.
    """
    a = _claim(
        "mut-conf-old",
        epistemic_type="confirmed",
        subject="subject:mut",
        predicate="pred:mut",
        valid_from=_NOW,
    )
    b = _claim(
        "mut-conf-new",
        epistemic_type="confirmed",
        subject="subject:mut",
        predicate="pred:mut",
        valid_from=_LATER,
    )
    ents = {
        "mut-conf-old": _entrenchment(
            "mut-conf-old", rank=EPISTEMIC_RANK["confirmed"], alpha=5.0, beta=1.0
        ),
        "mut-conf-new": _entrenchment(
            "mut-conf-new", rank=EPISTEMIC_RANK["confirmed"], alpha=4.0, beta=1.0
        ),
    }
    res = decide_resolution([a, b], ents)

    # Both of these assertions must hold to be mutation-resistant:
    # - kind alone is not enough (an unrelated escalation path could produce "escalated")
    # - escalation_reason pins the exact guard that fired
    assert res.kind == "escalated"
    assert res.escalation_reason == "confirmed-vs-confirmed", (
        "Removing the confirmed-vs-confirmed guard in UPDATE MODE would produce "
        "kind='update-supersession', escalation_reason=None — this assertion catches that"
    )
