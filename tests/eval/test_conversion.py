"""Deterministic unit tests for the adversarial K->O converter (Pod 4.4c-3.5).

Stub panel + stub adversary + scripted probe — NO live models, NO docker, NO journal. Every
assertion is mutation-resistant (red-team will attack): the two-sided gate is pinned in BOTH failing
directions (clean-also-fails, error-also-passes); the determinism re-check is pinned with a flaky
candidate; early-success exit is pinned by adversary call-count; family-disjointness, loud-degrade,
and detK exclusion are pinned with negative controls; and the S1/S4 posture is pinned structurally
over the module AST (no journal/StageContext/CodeOracle symbol, no concrete provider import).
Mirrors ``test_labeling.py`` / ``test_planting.py`` pin discipline.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Literal

import pytest

from cogworx.claims.provenance import ProvenanceSource
from cogworx.eval.conversion import (
    K_ROUNDS,
    ConversionResult,
    PanelConfig,
    PanelFamilyCollision,
    PanelHypothesis,
    RoundArtifact,
    convert_k_pool,
)
from cogworx.eval.corpus import (
    ConvertedPlanterStamp,
    DeterministicPlanterStamp,
    DifficultyMarker,
    LLMPlanterStamp,
)
from cogworx.eval.planting import PlantedItem, PlantedPair, Seed, build_detk_pair
from cogworx.verification.contracts import OracleFrame, Thesis, Verdict

# ---------------------------------------------------------------------------
# Well-formed builders (mutation-resistance controls)
# ---------------------------------------------------------------------------

_FRAME = OracleFrame(
    completion_criterion="tests_pass", problem_type="code", problem_statement="sum a list"
)
_ERR_SOL = "def f(xs):\n    return sum(xs) - 1\n"  # planted-wrong
_CLEAN_SOL = "def f(xs):\n    return sum(xs)\n"  # correct sibling
# The author-frozen completion-criterion test (identical on the error item and its clean sibling).
# Distinct from any synthesized candidate so the author-anchor probes are observable by test_code.
_AUTHOR_TEST = "from solution import f\n\n\ndef test_f():\n    assert f([1, 2, 3]) == 6\n"


def _difficulty() -> DifficultyMarker:
    return DifficultyMarker(planted_difficulty="medium", surface_complexity=12)


def _llm_stamp(family: str = "deepseek") -> LLMPlanterStamp:
    return LLMPlanterStamp(model_family=family, model_id=f"{family}/chat")


def _k_err(**overrides: object) -> PlantedItem:
    base: dict[str, object] = {
        "provisional_id": 1,
        "frame": _FRAME,
        "thesis": Thesis(proposed_solution=_ERR_SOL, experiment_design="x"),
        # The seed's author-frozen test rides on the K item (HOLE#1 author-anchor needs it). It is
        # NOT an executable O-test of the error itself — it is the original completion criterion.
        "test_code": _AUTHOR_TEST,
        "candidate_stratum": "K",
        "is_error": 1,
        "planter": _llm_stamp(),
        "error_regime": "spec-misread",  # CANDIDATE
        "difficulty": _difficulty(),
        "matched_sibling_id": 2,
        "split": "tuning",
    }
    base.update(overrides)
    return PlantedItem(**base)


def _k_clean(**overrides: object) -> PlantedItem:
    base: dict[str, object] = {
        "provisional_id": 2,
        "frame": _FRAME,
        "thesis": Thesis(proposed_solution=_CLEAN_SOL, experiment_design="x"),
        "test_code": _AUTHOR_TEST,  # identical author-frozen test on the clean sibling
        "candidate_stratum": "clean",
        "is_error": 0,
        "planter": _llm_stamp(),
        "error_regime": "",
        "difficulty": _difficulty(),
        "matched_sibling_id": 1,
        "split": "tuning",
    }
    base.update(overrides)
    return PlantedItem(**base)


def _k_pair(**err_overrides: object) -> PlantedPair:
    return PlantedPair(error_item=_k_err(**err_overrides), clean_item=_k_clean())


def _verdict(
    *, holds: bool, valid_check: bool = True, source: ProvenanceSource = "tool"
) -> Verdict:
    return Verdict(
        holds=holds,
        valid_check=valid_check,
        reasoning="stub",
        source=source,
        test_provenance="frozen",
    )


# ---------------------------------------------------------------------------
# Stub seams
# ---------------------------------------------------------------------------


class _StubPanel:
    """A stub AdversaryPanel returning one advisory hypothesis; records call count."""

    def __init__(self, family: str = "claude") -> None:
        self.family = family
        self.calls = 0

    def propose(
        self, item: PlantedItem, clean_sibling: PlantedItem, round_index: int
    ) -> tuple[PanelHypothesis, ...]:
        self.calls += 1
        return (PanelHypothesis(family=self.family, rationale=f"probe r{round_index}"),)


class _NoCallPanel:
    """A panel that raises if consulted (for detK exclusion / degrade pins)."""

    def propose(
        self, item: PlantedItem, clean_sibling: PlantedItem, round_index: int
    ) -> tuple[PanelHypothesis, ...]:
        raise AssertionError(
            f"panel.propose called for item {item.provisional_id} — it must not be reached"
        )


class _CountingAdversary:
    """A stub adversary returning a fixed candidate per round; records call count."""

    def __init__(self, candidate: str = "TEST_CANDIDATE") -> None:
        self.candidate = candidate
        self.calls = 0

    def synthesize(
        self,
        item: PlantedItem,
        clean_sibling: PlantedItem,
        hypotheses: tuple[PanelHypothesis, ...],
        round_index: int,
    ) -> str:
        self.calls += 1
        return f"{self.candidate}_r{round_index}"


def _config(
    *,
    panel_families: tuple[str, ...] = ("claude", "grok"),
    adversary_family: str = "qwen",
    forbidden: frozenset[str] = frozenset({"deepseek", "gpt", "thesis-fam"}),
    degrade: Literal["abstain-pool", "raise"] = "abstain-pool",
) -> PanelConfig:
    return PanelConfig(
        panel_families=panel_families,
        adversary_family=adversary_family,
        forbidden_families=forbidden,
        degrade=degrade,
    )


# ===========================================================================
# Obligation 1 — happy K->O conversion
# ===========================================================================


def _gate_pass_probe(solution_code: str, test_code: str) -> Verdict:
    """A discriminating test: the planted-wrong solution FAILS (O-catch), the clean sibling PASSES.
    Deterministic (same verdict every call → determinism re-check also passes)."""
    if solution_code == _ERR_SOL:
        return _verdict(holds=False, valid_check=True)  # error fails → O-catch
    return _verdict(holds=True, valid_check=True)  # clean passes


def test_happy_k_to_o_conversion() -> None:
    """A discriminating candidate flips the K error to O: rewritten with a ConvertedPlanterStamp,
    correct winning_round, in `pairs` not `residual`, test_code = the winning candidate."""
    panel, adversary = _StubPanel(), _CountingAdversary()
    result = convert_k_pool(
        [_k_pair()], [], panel=panel, adversary=adversary, probe=_gate_pass_probe, config=_config()
    )
    assert len(result.pairs) == 1
    assert result.residual_pairs == ()
    err = result.pairs[0].error_item
    assert err.candidate_stratum == "O"
    assert isinstance(err.planter, ConvertedPlanterStamp)
    assert err.planter.winning_round == 1
    assert err.test_code == "TEST_CANDIDATE_r1"
    # The clean sibling is unchanged.
    assert result.pairs[0].clean_item.thesis.proposed_solution == _CLEAN_SOL
    # Honest planting provenance survives the conversion.
    assert err.planter.planter_model_family == "deepseek"
    assert err.planter.adversary_family == "qwen"  # the config's adversary_family
    assert not result.degraded


# ===========================================================================
# Obligation 2 — two-sided gate (anti-laundering)
# ===========================================================================


def test_gate_clean_also_fails_not_converted() -> None:
    """Anti-laundering: a candidate where the clean sibling ALSO fails (e.g. an `assert False` test)
    → NOT converted. Mutation killed: a one-sided gate (error-fails only) would convert this."""

    def probe(solution_code: str, test_code: str) -> Verdict:
        return _verdict(holds=False, valid_check=True)  # BOTH solutions fail

    result = convert_k_pool(
        [_k_pair()], [], panel=_StubPanel(), adversary=_CountingAdversary(),
        probe=probe, config=_config()
    )
    assert result.pairs == ()
    assert len(result.residual_pairs) == 1
    # Residual item is untouched (still K, original planter).
    assert isinstance(result.residual_pairs[0].error_item.planter, LLMPlanterStamp)


def test_gate_error_also_passes_not_converted() -> None:
    """A candidate where the planted-wrong solution PASSES (holds) → not an O-catch → NOT converted.
    Mutation killed: a gate reading clean-passes only would convert this."""

    def probe(solution_code: str, test_code: str) -> Verdict:
        return _verdict(holds=True, valid_check=True)  # BOTH pass — no catch

    result = convert_k_pool(
        [_k_pair()], [], panel=_StubPanel(), adversary=_CountingAdversary(),
        probe=probe, config=_config()
    )
    assert result.pairs == ()
    assert len(result.residual_pairs) == 1


def test_gate_noise_verdict_not_converted() -> None:
    """A noise verdict (valid_check=False) on the error side is NOT an O-catch → not converted, even
    if ¬holds. Pins that valid_check gates the catch (agrees with labeling.assign_stratum)."""

    def probe(solution_code: str, test_code: str) -> Verdict:
        if solution_code == _ERR_SOL:
            return _verdict(holds=False, valid_check=False)  # noise, not a catch
        return _verdict(holds=True, valid_check=True)

    result = convert_k_pool(
        [_k_pair()], [], panel=_StubPanel(), adversary=_CountingAdversary(),
        probe=probe, config=_config()
    )
    assert result.pairs == ()


def test_gate_non_executable_verdict_not_converted() -> None:
    """An inference-sourced (non-executable) error verdict is NOT an O-catch → not converted, even
    if ¬holds. Pins that is_executable gates the catch."""

    def probe(solution_code: str, test_code: str) -> Verdict:
        if solution_code == _ERR_SOL:
            return _verdict(holds=False, valid_check=True, source="inference")
        return _verdict(holds=True, valid_check=True)

    result = convert_k_pool(
        [_k_pair()], [], panel=_StubPanel(), adversary=_CountingAdversary(),
        probe=probe, config=_config()
    )
    assert result.pairs == ()


# ===========================================================================
# Obligation 3 — mandatory determinism re-check
# ===========================================================================


def test_determinism_recheck_rejects_flaky_candidate() -> None:
    """A candidate that PASSES the gate on run 1 but DISAGREES on `holds` on the re-run is REJECTED
    (flaky) — never converted, even though the gate itself passed. Mutation killed: dropping the
    determinism re-check would convert this flaky candidate."""

    class _FlakyProbe:
        def __init__(self) -> None:
            self.err_calls = 0

        def __call__(self, solution_code: str, test_code: str) -> Verdict:
            if test_code == _AUTHOR_TEST:
                # Author-anchor (stable, honest): error fails, clean passes — anchor holds, so the
                # flake on the CANDIDATE is what must reject the conversion.
                return _verdict(holds=solution_code != _ERR_SOL, valid_check=True)
            if solution_code == _ERR_SOL:
                self.err_calls += 1
                # First gate call: fails (O-catch). Re-check call: passes (disagrees → flaky).
                return _verdict(holds=self.err_calls > 1, valid_check=True)
            return _verdict(holds=True, valid_check=True)  # clean always passes

    result = convert_k_pool(
        [_k_pair()], [], panel=_StubPanel(), adversary=_CountingAdversary(),
        probe=_FlakyProbe(), config=_config()
    )
    # Flaky candidate is rejected every round → item stays residual K, no conversion.
    assert result.pairs == ()
    assert len(result.residual_pairs) == 1


def test_determinism_recheck_clean_side_flake_rejected() -> None:
    """The clean side of the re-check is load-bearing too: error stable-fails but the CLEAN sibling
    flakes (passes then fails) on re-run → rejected. Mutation killed: re-checking only the error
    side would convert this."""

    class _CleanFlakyProbe:
        def __init__(self) -> None:
            self.clean_calls = 0

        def __call__(self, solution_code: str, test_code: str) -> Verdict:
            if test_code == _AUTHOR_TEST:
                # Author-anchor (stable, honest): error fails, clean passes — anchor holds.
                return _verdict(holds=solution_code != _ERR_SOL, valid_check=True)
            if solution_code == _ERR_SOL:
                return _verdict(holds=False, valid_check=True)  # stable O-catch
            self.clean_calls += 1
            # Gate call: passes. Re-check call: fails (disagrees → flaky clean).
            return _verdict(holds=self.clean_calls <= 1, valid_check=True)

    result = convert_k_pool(
        [_k_pair()], [], panel=_StubPanel(), adversary=_CountingAdversary(),
        probe=_CleanFlakyProbe(), config=_config()
    )
    assert result.pairs == ()


# ===========================================================================
# Obligation 4 — round exhaustion → residual
# ===========================================================================


def test_round_exhaustion_to_residual() -> None:
    """No candidate passes in 8 rounds → the item is residual, untouched (still K, original
    planter, candidate_stratum stays 'K'), never rewritten. The adversary runs K_ROUNDS times."""

    def probe(solution_code: str, test_code: str) -> Verdict:
        return _verdict(holds=True, valid_check=True)  # nothing ever catches

    panel, adversary = _StubPanel(), _CountingAdversary()
    result = convert_k_pool(
        [_k_pair()], [], panel=panel, adversary=adversary, probe=probe, config=_config()
    )
    assert result.pairs == ()
    assert len(result.residual_pairs) == 1
    err = result.residual_pairs[0].error_item
    assert err.candidate_stratum == "K"
    assert isinstance(err.planter, LLMPlanterStamp)  # original planter untouched
    assert adversary.calls == K_ROUNDS == 8
    # Audit records all 8 rounds, none converted.
    arts = result.conversion_audit.rounds[1]
    assert len(arts) == 8
    assert all(not a.converted for a in arts)


# ===========================================================================
# Obligation 5 — early-success exit
# ===========================================================================


def test_early_success_exit_on_round_three() -> None:
    """Converts on round 3 → winning_round == 3, rounds 4-8 NOT attempted (adversary called exactly
    3 times). Mutation killed: a converter that keeps running after a success, or an off-by-one
    round index, fails this."""

    class _ThirdRoundProbe:
        """Error fails (O-catch) only when the candidate is the round-3 candidate; otherwise both
        pass (no catch). Deterministic across re-check. The author-anchor (probed with the author
        test) is stable-honest every round so the round gate is what selects round 3."""

        def __call__(self, solution_code: str, test_code: str) -> Verdict:
            if test_code == _AUTHOR_TEST:
                # Author-anchor holds every round: error fails the author test, clean passes it.
                return _verdict(holds=solution_code != _ERR_SOL, valid_check=True)
            if solution_code == _ERR_SOL and test_code.endswith("_r3"):
                return _verdict(holds=False, valid_check=True)  # O-catch on round 3
            return _verdict(holds=True, valid_check=True)

    panel, adversary = _StubPanel(), _CountingAdversary()
    result = convert_k_pool(
        [_k_pair()], [], panel=panel, adversary=adversary,
        probe=_ThirdRoundProbe(), config=_config()
    )
    assert len(result.pairs) == 1
    err = result.pairs[0].error_item
    assert isinstance(err.planter, ConvertedPlanterStamp)
    assert err.planter.winning_round == 3
    assert adversary.calls == 3  # rounds 4-8 not attempted
    assert panel.calls == 3
    # The winning candidate is the round-3 candidate.
    assert err.test_code == "TEST_CANDIDATE_r3"


# ===========================================================================
# Obligation 6 — family-disjointness
# ===========================================================================


def test_panel_family_collides_with_forbidden_raises() -> None:
    """PanelConfig construction raises PanelFamilyCollision when a panel family is in the forbidden
    set (an arm / planter family)."""
    with pytest.raises(PanelFamilyCollision, match=r"collide|detectability"):
        PanelConfig(
            panel_families=("deepseek",),  # 'deepseek' is forbidden (the planter family)
            adversary_family="claude",
            forbidden_families=frozenset({"deepseek"}),
        )


def test_adversary_family_collides_with_forbidden_raises() -> None:
    """The adversary family colliding with the forbidden set also raises at construction."""
    with pytest.raises(PanelFamilyCollision):
        PanelConfig(
            panel_families=("claude",),
            adversary_family="gpt",  # 'gpt' forbidden (an arm family)
            forbidden_families=frozenset({"gpt"}),
        )


def test_panel_internal_duplicate_raises() -> None:
    """A duplicate family across panel + adversary raises at construction (families must be
    internally distinct)."""
    with pytest.raises(PanelFamilyCollision, match="distinct"):
        PanelConfig(
            panel_families=("claude", "claude"),
            adversary_family="grok",
            forbidden_families=frozenset(),
        )


def test_per_item_same_family_skipped_to_residual() -> None:
    """Per-item: an item planted by a panel/adversary family is SKIPPED → residual (the panel is
    never consulted for it). Mutation killed: converting a same-family item is in-distribution
    with the panel and laundering."""
    # The error is planted by 'claude' — a panel family. It must be skipped per-item.
    pair = _k_pair(planter=_llm_stamp("claude"))
    # forbidden set does NOT include claude here (claude is a legitimate panel family); the per-item
    # guard is the planter-vs-panel check, independent of the config-time forbidden set.
    config = PanelConfig(
        panel_families=("claude", "grok"),
        adversary_family="qwen",
        forbidden_families=frozenset({"deepseek", "gpt"}),
    )
    result = convert_k_pool(
        [pair], [], panel=_NoCallPanel(), adversary=_CountingAdversary(),
        probe=_gate_pass_probe, config=config
    )
    assert result.pairs == ()
    assert len(result.residual_pairs) == 1
    # Not counted in conv_rate_llmK (skipped, not attempted).
    assert result.conversion_audit.conv_rate_llmK == 0.0


# ===========================================================================
# Obligation 7 — loud-degrade
# ===========================================================================


def test_degrade_abstain_pool_converts_nothing() -> None:
    """Too-few disjoint families + degrade='abstain-pool' → converts nothing, whole K pool to
    residual, degraded=True, panel never consulted.

    Construction enforces disjointness, so the convert-time guard is belt-and-suspenders against a
    DRIFTED config. We force the drift with object.__setattr__ (the only way past the frozen model)
    to make panel 'claude' overlap the forbidden set, then assert the convert path degrades rather
    than converting a now-in-distribution panel."""
    drifted = _config()  # panel=('claude',), adversary='claude' → valid at build
    object.__setattr__(drifted, "forbidden_families", frozenset({"claude"}))
    result = convert_k_pool(
        [_k_pair()], [], panel=_NoCallPanel(), adversary=_CountingAdversary(),
        probe=_gate_pass_probe, config=drifted
    )
    assert result.degraded is True
    assert result.pairs == ()
    assert len(result.residual_pairs) == 1
    assert result.conversion_audit.conv_rate_all == 0.0


def test_degrade_raise_raises() -> None:
    """degrade='raise' with insufficient disjoint families → raises PanelFamilyCollision at convert
    time (not a silent abstain)."""
    drifted = _config(degrade="raise")
    object.__setattr__(drifted, "forbidden_families", frozenset({"claude"}))
    with pytest.raises(PanelFamilyCollision, match="disjoint"):
        convert_k_pool(
            [_k_pair()], [], panel=_NoCallPanel(), adversary=_CountingAdversary(),
            probe=_gate_pass_probe, config=drifted
        )


# ===========================================================================
# Obligation 8 — detK exclusion
# ===========================================================================


def _detk_pair() -> PlantedPair:
    seed = Seed(
        seed_id=0,
        frame=OracleFrame(
            completion_criterion="tests_pass", problem_type="code", problem_statement="scale"
        ),
        thesis=Thesis(
            proposed_solution="def scale(x):\n    return x * 2\n",
            experiment_design="run frozen tests",
        ),
        test_code="from solution import scale\n\n\ndef test_scale():\n    assert scale(3) == 6\n",
    )
    return build_detk_pair(seed, "constant-replacement", error_id=30, clean_id=31)


def test_detk_excluded_passed_through_untouched() -> None:
    """A detK item is passed through UNTOUCHED to residual, never sent to the panel (the
    _NoCallPanel raises if consulted). Pins the detK exclusion — the converter never touches
    collusion probes."""
    pair = _detk_pair()
    result = convert_k_pool(
        [pair], [], panel=_NoCallPanel(), adversary=_CountingAdversary(),
        probe=_gate_pass_probe, config=_config()
    )
    assert result.pairs == ()
    assert len(result.residual_pairs) == 1
    # Untouched: still a DeterministicPlanterStamp.
    assert isinstance(result.residual_pairs[0].error_item.planter, DeterministicPlanterStamp)
    # Counted in the detK denominator, not the llmK one.
    assert result.conversion_audit.conv_rate_detK == 0.0
    assert result.conversion_audit.conv_rate_llmK == 0.0


# ===========================================================================
# Obligation 9 — conversion-rate reporting
# ===========================================================================


def test_conversion_rates_exclude_detk_from_llmk_denominator() -> None:
    """A mixed pool: 2 LLM-K pairs (one converts, one doesn't) + 1 detK pair. conv_rate_llmK counts
    1/2 (detK EXCLUDED from the denominator); conv_rate_detK is 0/1; conv_rate_all is 1/3."""
    # Pair A converts (gate-pass probe routes on the distinct error solution).
    pair_a = _k_pair(provisional_id=1, matched_sibling_id=2)
    # Pair B never converts (different error solution that the probe never catches).
    err_b_sol = "def g(xs):\n    return min(xs)\n"
    clean_b_sol = "def g(xs):\n    return max(xs)\n"
    pair_b = PlantedPair(
        error_item=_k_err(
            provisional_id=3,
            matched_sibling_id=4,
            thesis=Thesis(proposed_solution=err_b_sol, experiment_design="x"),
        ),
        clean_item=_k_clean(
            provisional_id=4,
            matched_sibling_id=3,
            thesis=Thesis(proposed_solution=clean_b_sol, experiment_design="x"),
        ),
    )
    detk = _detk_pair()

    def probe(solution_code: str, test_code: str) -> Verdict:
        if solution_code == _ERR_SOL:
            return _verdict(holds=False, valid_check=True)  # pair A error → O-catch
        if solution_code == _CLEAN_SOL:
            return _verdict(holds=True, valid_check=True)  # pair A clean → passes
        return _verdict(holds=True, valid_check=True)  # pair B never catches

    result = convert_k_pool(
        [pair_a, pair_b, detk], [], panel=_StubPanel(), adversary=_CountingAdversary(),
        probe=probe, config=_config()
    )
    audit = result.conversion_audit
    assert audit.conv_rate_llmK == pytest.approx(0.5)  # 1 of 2 LLM-K, detK excluded
    assert audit.conv_rate_detK == pytest.approx(0.0)  # 0 of 1 detK
    assert audit.conv_rate_all == pytest.approx(1 / 3)  # 1 of 3 attempted+detk
    assert len(result.pairs) == 1
    assert len(result.residual_pairs) == 2  # pair B + detK


def test_conversion_rates_empty_pool_is_zero() -> None:
    """Empty denominators → 0.0 rates, no division error."""
    result = convert_k_pool(
        [], [], panel=_StubPanel(), adversary=_CountingAdversary(),
        probe=_gate_pass_probe, config=_config()
    )
    audit = result.conversion_audit
    assert audit.conv_rate_llmK == 0.0
    assert audit.conv_rate_detK == 0.0
    assert audit.conv_rate_all == 0.0


# ===========================================================================
# Obligation 10 — S1 / journal-free + S4 provider-agnostic (the #1 red-team target)
# ===========================================================================


def _conversion_tree() -> ast.Module:
    import cogworx.eval.conversion as conv_mod

    return ast.parse(Path(conv_mod.__file__).read_text(encoding="utf-8"))


def _imported_modules(tree: ast.Module) -> set[str]:
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    return imported


def _referenced_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def test_conversion_does_no_journal_io_and_no_live_oracle() -> None:
    """S1 (the #1 red-team target): conversion.py never references a Journal / RunContext /
    StageContext / CodeOracle / evaluate in CODE, and imports no journal/substrate/runtime module —
    it cannot be on a live write-path. Structural over the AST (S9: structure over self-report)."""
    tree = _conversion_tree()
    imported = _imported_modules(tree)
    referenced = _referenced_names(tree)
    for forbidden in (
        "Journal",
        "RunContext",
        "StageContext",
        "load_run",
        "GraphStore",
        "LatentStore",
        "CodeOracle",
        "evaluate",
    ):
        assert forbidden not in referenced, (
            f"conversion.py CODE references {forbidden!r} (S1 write-path / live-oracle smell)"
        )
    assert not any("journal" in m or "substrate" in m or "runtime" in m for m in imported), (
        f"conversion.py imports a journal/substrate/runtime module: {imported}"
    )


def test_conversion_imports_no_concrete_provider() -> None:
    """S4: conversion.py imports no concrete model provider — the panel/adversary are injected
    Protocol seams (provider is the call-site's choice)."""
    imported = _imported_modules(_conversion_tree())
    for forbidden in (
        "cogworx.model.providers.claude",
        "cogworx.model.providers.openai_compat",
        "anthropic",
        "openai",
    ):
        assert forbidden not in imported, (
            f"conversion.py imports a concrete provider: {forbidden!r}"
        )
    assert not any("providers" in m for m in imported), (
        f"conversion.py imports a provider module: {imported}"
    )


def test_conversion_default_probe_is_the_frozen_kernel() -> None:
    """The default probe binding IS run_frozen_check (the journal-free frozen kernel), NEVER
    CodeOracle.evaluate — the converter and labeler agree on the same offline executor."""
    import inspect

    from cogworx.eval._authoring import run_frozen_check

    sig = inspect.signature(convert_k_pool)
    assert sig.parameters["probe"].default is run_frozen_check


# ===========================================================================
# Obligation 11 — determinism: identical ConversionResult on identical inputs
# ===========================================================================


def test_convert_k_pool_is_deterministic() -> None:
    """Same inputs + same stub outputs → identical ConversionResult (same conversions, residuals,
    rates, audit). Pins reproducibility."""

    def run() -> ConversionResult:
        return convert_k_pool(
            [_k_pair()], [], panel=_StubPanel(), adversary=_CountingAdversary(),
            probe=_gate_pass_probe, config=_config()
        )

    r1, r2 = run(), run()
    assert r1.pairs == r2.pairs
    assert r1.residual_pairs == r2.residual_pairs
    assert r1.conversion_audit == r2.conversion_audit
    assert r1.degraded == r2.degraded


# ===========================================================================
# Frozen-field-set + audit pins (additive-only tripwires)
# ===========================================================================


def test_round_artifact_field_set_is_pinned() -> None:
    assert set(RoundArtifact.model_fields) == {
        "round_index",
        "candidate_test",
        "v_err",
        "v_clean",
        "converted",
    }


def test_conversion_result_field_set_is_pinned() -> None:
    assert set(ConversionResult.model_fields) == {
        "pairs",
        "singletons",
        "residual_pairs",
        "residual_singletons",
        "conversion_audit",
        "degraded",
    }


def test_k_rounds_is_eight() -> None:
    """K_ROUNDS is the eval-stats-frozen 8 (immutable per corpus lock)."""
    assert K_ROUNDS == 8


def test_converted_audit_records_winning_round_artifact() -> None:
    """The converted item's audit carries a converted=True RoundArtifact at the winning round."""
    result = convert_k_pool(
        [_k_pair()], [], panel=_StubPanel(), adversary=_CountingAdversary(),
        probe=_gate_pass_probe, config=_config()
    )
    arts = result.conversion_audit.rounds[1]
    assert arts[-1].converted is True
    assert arts[-1].round_index == 1


def test_singleton_passes_through_untouched() -> None:
    """A K error SINGLETON (no co-located clean sibling) cannot be honestly converted (the
    two-sided gate needs a clean sibling) → passed through to residual_singletons untouched."""
    singleton = _k_err(provisional_id=99, matched_sibling_id=None)
    result = convert_k_pool(
        [], [singleton], panel=_NoCallPanel(), adversary=_CountingAdversary(),
        probe=_gate_pass_probe, config=_config()
    )
    assert result.singletons == ()
    assert len(result.residual_singletons) == 1
    assert result.residual_singletons[0].provisional_id == 99
    assert result.conversion_audit.conv_rate_llmK == 0.0


# ===========================================================================
# HOLE#1 — author-anchor (wrong-intent label-laundering guard, architect-ruled)
# ===========================================================================
#
# The two-sided candidate gate discriminates the planted-wrong solution from its clean sibling, but
# NOTHING tied that discrimination to the item's ORIGINAL author-frozen intent. The
# `_wrong_intent_probe` models the attack: an adversary synthesizes a real, deterministic,
# gate-passing test that discriminates on an INCIDENTAL axis (int-vs-float return) while the real
# bug the author cares about is a VALUE bug the author's frozen test checks. The author test PASSES
# the planted error (right value, wrong type) → the author-anchor is False → conversion REJECTED.


def _wrong_intent_probe(solution_code: str, test_code: str) -> Verdict:
    """The HOLE#1 attack scripted as a probe.

    The SYNTHESIZED candidate (any test_code != _AUTHOR_TEST) discriminates on the incidental axis:
    error FAILS, clean PASSES — so the two-sided candidate gate ALONE would convert. But the AUTHOR
    test (_AUTHOR_TEST) checks the value, which the planted error gets right (only the type
    differs), so it PASSES BOTH solutions → a_err is NOT an O-catch → the author-anchor is False."""
    if test_code == _AUTHOR_TEST:
        # Author checks value, not type: BOTH the planted error and the clean sibling PASS it.
        return _verdict(holds=True, valid_check=True)
    # The synthesized candidate discriminates on the incidental axis (the laundering attack).
    if solution_code == _ERR_SOL:
        return _verdict(holds=False, valid_check=True)  # error fails the incidental candidate
    return _verdict(holds=True, valid_check=True)  # clean passes the incidental candidate


def test_nc1_wrong_intent_candidate_rejected_by_author_anchor() -> None:
    """NC-1: a candidate that clears the two-sided gate on an INCIDENTAL axis but whose author test
    does NOT itself fail the planted error is REJECTED — author-anchor False → item to residual, NO
    ConvertedPlanterStamp minted. This is the HOLE#1 fix in action.

    Cross-checked at the predicate level: _author_anchor over (author passes error, author passes
    clean) is False (the error side is not an O-catch on the author test)."""
    # Predicate-level pin (the unit the converter calls).
    from cogworx.eval.conversion import _author_anchor

    a_err_passes = _verdict(holds=True, valid_check=True)   # author test does NOT fail the error
    a_clean_passes = _verdict(holds=True, valid_check=True)  # author test passes the clean sibling
    assert _author_anchor(a_err_passes, a_clean_passes) is False

    # End-to-end: the wrong-intent candidate is rejected; nothing converts.
    result = convert_k_pool(
        [_k_pair()], [], panel=_StubPanel(), adversary=_CountingAdversary(),
        probe=_wrong_intent_probe, config=_config()
    )
    assert result.pairs == ()
    assert len(result.residual_pairs) == 1
    # No ConvertedPlanterStamp anywhere — the item stays a plain LLM-planted K residual.
    assert isinstance(result.residual_pairs[0].error_item.planter, LLMPlanterStamp)
    assert result.residual_pairs[0].error_item.candidate_stratum == "K"


def test_nc2_honest_conversion_still_converts() -> None:
    """NC-2: an HONEST item — the planted error genuinely fails the author test, the clean passes
    it, and the candidate agrees (the author test and candidate both catch the real bug) →
    converts, a ConvertedPlanterStamp is minted, winning_round recorded. Guards over-tightening: the
    author-anchor must NOT reject a legitimate K→O flip."""
    # _gate_pass_probe keys on solution_code only, so the author-anchor probe (probe(err, author))
    # ALSO returns error-fails / clean-passes — the author test agrees with the planter. Honest.
    result = convert_k_pool(
        [_k_pair()], [], panel=_StubPanel(), adversary=_CountingAdversary(),
        probe=_gate_pass_probe, config=_config()
    )
    assert len(result.pairs) == 1
    err = result.pairs[0].error_item
    assert isinstance(err.planter, ConvertedPlanterStamp)
    assert err.planter.winning_round == 1
    assert err.candidate_stratum == "O"
    assert result.conversion_audit.conv_rate_llmK == pytest.approx(1.0)


def test_nc3_none_test_code_not_convertible_excluded_from_denominator() -> None:
    """NC-3: a K item with test_code=None cannot be author-anchored → NOT convertible → routed to
    residual, EXCLUDED from the conv_rate_llmK denominator (same posture as the singleton skip). The
    panel is NEVER consulted (the author-anchor cannot run, so no rounds are attempted)."""
    pair = _k_pair()
    no_test_err = pair.error_item.model_copy(update={"test_code": None})
    no_test_clean = pair.clean_item.model_copy(update={"test_code": None})
    no_test_pair = PlantedPair(error_item=no_test_err, clean_item=no_test_clean)
    result = convert_k_pool(
        [no_test_pair], [], panel=_NoCallPanel(), adversary=_CountingAdversary(),
        probe=_gate_pass_probe, config=_config()
    )
    assert result.pairs == ()
    assert len(result.residual_pairs) == 1
    # Never attempted → not in the llmK denominator (denominator empty → 0.0, no division error).
    assert result.conversion_audit.conv_rate_llmK == 0.0
    # No round artifacts were recorded for this item (it was never run through the rounds).
    assert no_test_err.provisional_id not in result.conversion_audit.rounds


def test_nc3_mutation_only_the_none_clause_keeps_it_residual() -> None:
    """NC-3 (mutation-resistant): the ONLY difference between the residual None-test item and a
    converting item is the None test_code. Same pair, same probe, same config — flip test_code from
    None to the real author test and it CONVERTS. This kills the mutation "the None routing is
    incidental": the None clause is the sole gatekeeper here."""
    base = _k_pair()
    # (a) None test_code → residual (the clause holds).
    none_pair = PlantedPair(
        error_item=base.error_item.model_copy(update={"test_code": None}),
        clean_item=base.clean_item.model_copy(update={"test_code": None}),
    )
    none_result = convert_k_pool(
        [none_pair], [], panel=_StubPanel(), adversary=_CountingAdversary(),
        probe=_gate_pass_probe, config=_config()
    )
    assert none_result.pairs == ()
    assert len(none_result.residual_pairs) == 1

    # (b) Same pair, real author test → CONVERTS. The ONLY change is test_code None → _AUTHOR_TEST.
    converting_result = convert_k_pool(
        [base], [], panel=_StubPanel(), adversary=_CountingAdversary(),
        probe=_gate_pass_probe, config=_config()
    )
    assert len(converting_result.pairs) == 1
    assert isinstance(converting_result.pairs[0].error_item.planter, ConvertedPlanterStamp)


def test_nc4_deleting_author_anchor_conjunct_would_convert_wrong_intent() -> None:
    """NC-4 (mutation guard): assert the NC-1 wrong-intent case CONVERTS once the author-anchor
    conjunct is removed from the gate — i.e. NC-1's rejection genuinely depends on the new check,
    not on some incidental property of the scripted probe. We monkeypatch conversion._gate to the
    OLD two-sided form (no author-anchor) and confirm the wrong-intent candidate now wrongly
    converts."""
    import cogworx.eval.conversion as conv_mod
    from cogworx.eval.conversion import _is_clean_pass, _is_o_catch

    def _two_sided_only(
        v_err: Verdict, v_clean: Verdict, a_err: Verdict, a_clean: Verdict
    ) -> bool:
        # The OLD gate: candidate O-catch + clean-pass, NO author-anchor.
        return _is_o_catch(v_err) and _is_clean_pass(v_clean)

    orig = conv_mod._gate
    conv_mod._gate = _two_sided_only  # type: ignore[assignment]
    try:
        result = convert_k_pool(
            [_k_pair()], [], panel=_StubPanel(), adversary=_CountingAdversary(),
            probe=_wrong_intent_probe, config=_config()
        )
    finally:
        conv_mod._gate = orig  # type: ignore[assignment]
    # Without the author-anchor the wrong-intent candidate launders through — proving the conjunct
    # is the ONLY thing rejecting it in NC-1.
    assert len(result.pairs) == 1
    assert isinstance(result.pairs[0].error_item.planter, ConvertedPlanterStamp)


def test_nc5_author_anchor_reads_original_test_not_candidate() -> None:
    """NC-5: the author-anchor probes are called with item.test_code (the ORIGINAL author-frozen
    test), NEVER the synthesized candidate. Guards a future refactor from re-pointing the anchor at
    the candidate (which would re-open HOLE#1). We record every (solution, test_code) probe call and
    assert: the author test was probed against BOTH solutions, and the synthesized candidate was
    NEVER passed to a probe call standing in for the author anchor."""
    calls: list[tuple[str, str]] = []

    def _recording_probe(solution_code: str, test_code: str) -> Verdict:
        calls.append((solution_code, test_code))
        return _gate_pass_probe(solution_code, test_code)

    result = convert_k_pool(
        [_k_pair()], [], panel=_StubPanel(), adversary=_CountingAdversary(),
        probe=_recording_probe, config=_config()
    )
    assert len(result.pairs) == 1  # honest conversion (sanity)
    # The author-frozen test was probed against BOTH the error and the clean solution.
    assert (_ERR_SOL, _AUTHOR_TEST) in calls
    assert (_CLEAN_SOL, _AUTHOR_TEST) in calls
    # The synthesized candidate (winning round 1) was probed ONLY as the candidate, and the author
    # anchor used _AUTHOR_TEST — never the candidate string. Pin: no probe call paired the author
    # role with the candidate by asserting the candidate test_code is always a synthesized one.
    candidate_test = result.pairs[0].error_item.test_code
    assert candidate_test == "TEST_CANDIDATE_r1"
    # Every (solution, candidate_test) call is a CANDIDATE-gate call; the anchor calls are disjoint
    # (they carry _AUTHOR_TEST). The author test and the candidate test are distinct strings, so the
    # anchor can never have read the candidate.
    assert candidate_test != _AUTHOR_TEST
    author_anchor_calls = [c for c in calls if c[1] == _AUTHOR_TEST]
    assert all(c[1] != candidate_test for c in author_anchor_calls)


# ===========================================================================
# NC-9 — 4.4d contract stub (converted-O Cell emission; lands at 4.4d, text frozen now)
# ===========================================================================


@pytest.mark.skip(reason="4.4d contract: converted_o Cell emission lands at 4.4d")
def test_nc9_converted_item_emits_converted_o_cell() -> None:
    """NC-9 (4.4d CONTRACT, skipped — text frozen so it cannot be silently dropped): a converted
    PlantedItem promoted through the corpus and synthesized into the Cell artifact MUST carry
    Cell.converted_o=True (read off its ConvertedPlanterStamp), and 4.4d MUST use
    youden.is_converted_o to keep converted-O OUT of the binding δ.

    This assertion is the 4.4d acceptance text. It is SKIPPED (does not execute) because the
    converted_o Cell emission lands at 4.4d — Cell.converted_o exists in parallel but the
    promote→synth_cells wiring that sets it from the ConvertedPlanterStamp is a 4.4d deliverable. Do
    NOT unskip until 4.4d wires it; this test exists now only to pin the contract."""
    from cogworx.eval.youden import Cell, is_converted_o

    # The 4.4d contract: a Cell emitted for a converted item carries converted_o=True, and the
    # exclusion predicate keeps it out of the binding delta.
    converted_cell = Cell(
        item_id=1, stratum="O", arm="D", trial=0, seed=0, flagged=1, route="x", converted_o=True
    )
    plain_cell = Cell(
        item_id=2, stratum="O", arm="D", trial=0, seed=0, flagged=1, route="x"
    )
    assert is_converted_o(converted_cell) is True
    assert is_converted_o(plain_cell) is False
