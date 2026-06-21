"""Deterministic tests for the corpus-lock fingerprint spine (Pod 4.4c-5a).

Mutation-resistant pins for INV-LOCK-1 (content hash), INV-LOCK-2 (git-SHA pin), and INV-LOCK-3
(the MeasurementFingerprint). Each sensitivity assertion is paired with a determinism control so a
hash/fingerprint that silently ignored an input is CAUGHT. The exec-env-identity test closes
CF-4.4c-CONVERTER-ENV-PIN (two env identities -> two fingerprints). No model calls, no substrate, no
git shell-out (the SHA + env are injected).
"""

from __future__ import annotations

import hashlib

import pytest

from cogworx.eval.corpus import (
    ConvertedPlanterStamp,
    CorpusItem,
    DeterministicPlanterStamp,
    DifficultyMarker,
    LLMPlanterStamp,
    OracleLabelProvenance,
)
from cogworx.eval.lock import (
    MASTER_SEED,
    RESIDUAL_EPSILON_UNAUDITED,
    ExecEnvIdentity,
    MeasurementFingerprint,
    build_fingerprint,
    content_hash,
    lock_corpus,
)
from cogworx.verification.contracts import OracleFrame, Thesis

_FRAME = OracleFrame(
    completion_criterion="tests_pass",
    problem_type="code",
    problem_statement="sum a list",
)
_THESIS = Thesis(proposed_solution="return sum(xs)", experiment_design="run frozen tests")


def _prov() -> OracleLabelProvenance:
    return OracleLabelProvenance(
        returncode=1,
        test_provenance="frozen",
        holds=False,
        valid_check=True,
        oracle_id="code-oracle",
    )


def _difficulty() -> DifficultyMarker:
    return DifficultyMarker(planted_difficulty="medium", surface_complexity=12)


def _item(**overrides: object) -> CorpusItem:
    base: dict[str, object] = {
        "item_id": 1,
        "frame": _FRAME,
        "thesis": _THESIS,
        "test_code": "assert f([1, 2]) == 3",
        "is_error": 1,
        "label_source": "oracle",
        "label_provenance": _prov(),
        "stratum": "O",
        "oracle_reachable": True,
        "error_regime": "wrong-op",
        "difficulty": _difficulty(),
        "matched_sibling_id": None,
        "split": "measurement",
        "planter": DeterministicPlanterStamp(operators=("swap-op",)),
    }
    base.update(overrides)
    return CorpusItem(**base)


def _env(**overrides: object) -> ExecEnvIdentity:
    base: dict[str, object] = {
        "python_version": "3.12.7",
        "python_implementation": "CPython",
        "package_versions": (("pydantic", "2.12.5"), ("pytest", "8.0.0")),
        "locale_lc_ctype": "C",
    }
    base.update(overrides)
    return ExecEnvIdentity(**base)


# ---------------------------------------------------------------------------
# INV-LOCK-1 — content hash: determinism + per-field sensitivity + the mutation test
# ---------------------------------------------------------------------------


def test_content_hash_is_deterministic() -> None:
    """Same item -> byte-identical hash (the control every sensitivity pin is measured against)."""
    assert content_hash(_item()) == content_hash(_item())


def test_content_hash_is_a_sha256_hexdigest() -> None:
    h = content_hash(_item())
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_content_hash_sensitive_to_frame() -> None:
    other = OracleFrame(
        completion_criterion="tests_pass",
        problem_type="code",
        problem_statement="MUTATED statement",
    )
    assert content_hash(_item()) != content_hash(_item(frame=other))


def test_content_hash_sensitive_to_thesis() -> None:
    other = Thesis(proposed_solution="return 0", experiment_design="run frozen tests")
    assert content_hash(_item()) != content_hash(_item(thesis=other))


def test_content_hash_sensitive_to_test_code() -> None:
    assert content_hash(_item()) != content_hash(_item(test_code="assert f([1, 2]) == 4"))


def test_content_hash_sensitive_to_is_error() -> None:
    assert content_hash(_item(is_error=1)) != content_hash(_item(is_error=0, stratum="clean"))


def test_content_hash_sensitive_to_stratum() -> None:
    assert content_hash(_item(stratum="O")) != content_hash(_item(stratum="K"))


def test_content_hash_sensitive_to_error_regime() -> None:
    assert content_hash(_item(error_regime="wrong-op")) != content_hash(
        _item(error_regime="off-by-one")
    )


def test_content_hash_sensitive_to_split() -> None:
    assert content_hash(_item(split="measurement")) != content_hash(_item(split="tuning"))


def test_content_hash_ignores_non_identity_fields() -> None:
    """The hash is over the SEVEN identity fields ONLY: changing a non-identity field (item_id,
    oracle_reachable, difficulty, sibling, label provenance, planter, content_hash itself) MUST NOT
    move the hash. This pins the field set's UPPER bound — a hash that folded in extra fields would
    over-trip and re-lock spuriously."""
    base = content_hash(_item())
    assert content_hash(_item(item_id=999)) == base
    assert content_hash(_item(oracle_reachable=False)) == base
    assert content_hash(_item(matched_sibling_id=None, difficulty=
        DifficultyMarker(planted_difficulty="hard", surface_complexity=99))) == base
    assert content_hash(_item(planter=LLMPlanterStamp(model_family="x", model_id="y"))) == base
    assert content_hash(_item(content_hash="already-set-somehow")) == base


def test_content_hash_mutation_a_hash_that_drops_a_field_is_caught() -> None:
    """Mutation test (INV-LOCK-1): a content hash that IGNORED ``error_regime`` (a hand-rolled
    serialization that drops the field) would collide two items differing only in regime. The real
    ``content_hash`` separates them; this asserts the broken variant would fail, so the field-drop
    mutation cannot pass silently."""

    def broken_hash(item: CorpusItem) -> str:
        payload = "|".join(
            [
                item.frame.model_dump_json(),
                item.thesis.model_dump_json(),
                str(item.test_code),
                str(item.is_error),
                item.stratum,
                # error_regime DROPPED — the mutation.
                item.split,
            ]
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    a = _item(error_regime="wrong-op")
    b = _item(error_regime="off-by-one")
    assert broken_hash(a) == broken_hash(b)  # the mutant collides
    assert content_hash(a) != content_hash(b)  # the real hash does not


# ---------------------------------------------------------------------------
# lock_corpus — every item carries a non-empty content_hash after the lock
# ---------------------------------------------------------------------------


def test_lock_corpus_stamps_every_item() -> None:
    raw = [_item(item_id=1), _item(item_id=2, stratum="K")]
    assert all(it.content_hash == "" for it in raw)
    locked = lock_corpus(raw)
    assert all(it.content_hash != "" for it in locked)
    assert locked[0].content_hash == content_hash(raw[0])


def test_lock_corpus_is_pure() -> None:
    """The frozen inputs are untouched; the stamp lands on the returned copies only."""
    raw = [_item()]
    lock_corpus(raw)
    assert raw[0].content_hash == ""


# ---------------------------------------------------------------------------
# INV-LOCK-3 — MeasurementFingerprint: determinism + per-input sensitivity
# ---------------------------------------------------------------------------


def _locked(*items: CorpusItem) -> list[CorpusItem]:
    return lock_corpus(list(items))


def _fp(
    *,
    items: list[CorpusItem] | None = None,
    git_sha: str = "abc123",
    exec_env: ExecEnvIdentity | None = None,
    master_seed: int = MASTER_SEED,
    residual_epsilon: float = RESIDUAL_EPSILON_UNAUDITED,
) -> MeasurementFingerprint:
    return build_fingerprint(
        items if items is not None else _locked(_item(item_id=1), _item(item_id=2, stratum="K")),
        git_sha=git_sha,
        exec_env=exec_env if exec_env is not None else _env(),
        master_seed=master_seed,
        residual_epsilon=residual_epsilon,
    )


def test_fingerprint_is_deterministic() -> None:
    assert _fp().digest == _fp().digest


def test_fingerprint_refuses_never_locked_item() -> None:
    with pytest.raises(ValueError, match="never-locked"):
        build_fingerprint([_item()], git_sha="abc123", exec_env=_env())


def test_fingerprint_sensitive_to_content_hash_aggregate() -> None:
    a = _fp()
    b = _fp(items=_locked(_item(item_id=1), _item(item_id=2, stratum="clean", is_error=0)))
    assert a.content_hash_aggregate != b.content_hash_aggregate
    assert a.digest != b.digest


def test_fingerprint_aggregate_is_order_independent() -> None:
    """The corpus is a SET: re-ordering the items locks to the same aggregate (and digest)."""
    i1, i2 = _item(item_id=1), _item(item_id=2, stratum="K")
    forward = _fp(items=_locked(i1, i2))
    reversed_ = _fp(items=_locked(i2, i1))
    assert forward.digest == reversed_.digest


def test_fingerprint_sensitive_to_git_sha() -> None:
    assert _fp(git_sha="abc123").digest != _fp(git_sha="def456").digest


def test_fingerprint_sensitive_to_master_seed() -> None:
    assert _fp(master_seed=MASTER_SEED).digest != _fp(master_seed=MASTER_SEED + 1).digest


def test_fingerprint_sensitive_to_planter_families() -> None:
    """A converted-O family present vs absent flips the fingerprint (the env-fenced population)."""
    det_only = _fp(items=_locked(_item(item_id=1), _item(item_id=2, stratum="K")))
    converted = ConvertedPlanterStamp(
        planter_model_family="f",
        planter_model_id="m",
        adversary_family="adv",
        winning_round=2,
    )
    with_converted = _fp(
        items=_locked(_item(item_id=1), _item(item_id=2, stratum="K", planter=converted))
    )
    assert det_only.planter_families != with_converted.planter_families
    assert det_only.digest != with_converted.digest


def test_fingerprint_sensitive_to_exec_env() -> None:
    """CF-4.4c-CONVERTER-ENV-PIN: two different execution-environment identities -> two
    fingerprints. A converted-O test minted under one env must not silently re-evaluate under
    another."""
    env_a = _env(python_version="3.12.7", locale_lc_ctype="C")
    env_b = _env(python_version="3.13.0", locale_lc_ctype="en_US.UTF-8")
    assert _fp(exec_env=env_a).digest != _fp(exec_env=env_b).digest


def test_fingerprint_sensitive_to_exec_env_package_versions() -> None:
    env_a = _env(package_versions=(("pydantic", "2.12.5"),))
    env_b = _env(package_versions=(("pydantic", "2.13.0"),))
    assert _fp(exec_env=env_a).digest != _fp(exec_env=env_b).digest


def test_fingerprint_sensitive_to_residual_epsilon() -> None:
    """The ε-slot is folded: a post-audit fill (4.4c-5b) flips the fingerprint, so a stale clean
    bill cannot survive a re-lock."""
    unaudited = _fp()
    audited = _fp(residual_epsilon=0.012)
    assert unaudited.digest != audited.digest


# ---------------------------------------------------------------------------
# The residual-ε STUB slot — defaults to the not-yet-audited sentinel
# ---------------------------------------------------------------------------


def test_residual_epsilon_defaults_to_unaudited_sentinel() -> None:
    fp = _fp()
    assert fp.residual_epsilon == RESIDUAL_EPSILON_UNAUDITED
    assert fp.epsilon_audited is False


def test_residual_epsilon_audited_flag_flips_when_filled() -> None:
    assert _fp(residual_epsilon=0.0).epsilon_audited is True
    assert _fp(residual_epsilon=0.012).epsilon_audited is True


def test_master_seed_defaults_to_build_date_seed() -> None:
    assert _fp().master_seed == MASTER_SEED
    assert MASTER_SEED == 20260616
