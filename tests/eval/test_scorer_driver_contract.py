"""FINDING 1 (red-team, highest priority) — the scorer<->driver arm-label contract has NO direct
test (2026-07-02 wiring fix).

A direct edit of ``cogworx.eval.scorer._BINDING_DELTAS["D>C'"]`` to ``("D", "C_stripped")`` biases
the binding comparison toward PASS and, before this test, was caught only because
``test_scorer.py``'s synthetic-Cell fixtures never carry a ``"C_stripped"`` arm — in a REAL 5-arm
run the driver's arm map DOES emit ``"C_stripped"`` (the reported-only diet-stripped C, see
``_live/driver.py``'s "CELL-ARTIFACT ARM-LABEL CROSSOVER"), so the mutation would silently score the
wrong arm with every existing unit test green.

This test builds the driver's REAL :func:`~cogworx.eval._live.driver._build_arm_executors` map (so
``"C_stripped"`` IS present in the emitted artifact), drives it through the REAL
:func:`~cogworx.eval.runner.run_arms`, scores it with the REAL, unmodified
:func:`~cogworx.eval.scorer.score_gate`, and asserts the verdict is the one the CORRECT wiring
produces.

THE MUTATION CONTRAST (why this fixture, not a bigger/more "realistic" one): arm **C_stripped** is
built to mimic arm **D** EXACTLY, item-for-item (same flag pattern on every K and every clean item).
So:

  - scored CORRECTLY (``D>C'`` reads ``Cell.arm == BINDING_BASELINE_ARM`` == ``"C"`` — the FULL-diet
    ``make_c_prime_executor`` reviewer, a genuinely weaker arm than D here) the binding ``D>C'``
    delta is a clean, comfortably-positive number -> the gate PASSes.
  - scored under the FINDING-1 MUTATION (``D>C'`` reads ``"C_stripped"`` instead) the delta becomes
    D-vs-itself: BYTE-IDENTICAL flag counts on every item -> delta is deterministically **0.0**,
    which FAILS the ``lo > gate_threshold`` binding gate -> the verdict flips from PASS to FAIL.

So the mutation's effect is a hard, deterministic flip (PASS -> FAIL), not a probabilistic nudge —
robust against bootstrap-seed noise.

The Cell artifact is small (5 matched K/clean pairs, R=2) because arm **A** is wired for real (the
driver's ``arm_a_executor``, which shells out to a real ``pytest`` subprocess per call) — this test
intentionally keeps the item/trial count low to bound wall-clock time.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cogworx.cost.budget import BudgetGuard
from cogworx.eval._live.driver import _build_arm_executors
from cogworx.eval._live.settings import GateRunSettings
from cogworx.eval.arms import _WITHHELD_EXPERIMENT_DESIGN
from cogworx.eval.corpus import (
    CorpusItem,
    DeterministicPlanterStamp,
    DifficultyMarker,
    HumanLabelProvenance,
    OracleLabelProvenance,
)
from cogworx.eval.lock import ExecEnvIdentity, MeasurementFingerprint, ShuffleNullResult
from cogworx.eval.runner import run_arms
from cogworx.eval.scorer import BINDING_BASELINE_ARM, score_gate
from cogworx.model.base import ChatMessage, ModelCapabilities, ModelResponse, ModelTier, ToolSpec
from cogworx.model.providers.config import PriceTable, ProviderConfig
from cogworx.verification.contracts import OracleFrame, Thesis

_R = 2
_K_IDS = list(range(100, 105))  # 5 K items, one per regime
_CLEAN_IDS = list(range(200, 205))  # 5 matched clean items
_REGIMES = ("off-by-one", "spec-misread", "silent-degradation", "type-confusion", "boundary")
# Per-clean-item C (BINDING_BASELINE_ARM) flag counts out of R -- alternating 0/1 so the spec-
# ceiling variance floor sees real inter-item spread (a hollow, all-0 clean spec would itself
# refuse the gate at Tier 1 -- this is not the mutation under test).
_CLEAN_C_TARGETS = [0, 1, 0, 1, 0]

_MARKER_RE = re.compile(r"ITEM_MARKER_(\d+)")

_PRICE = PriceTable(
    pro_input_usd_per_mtok=1.0,
    pro_output_usd_per_mtok=1.0,
    flash_input_usd_per_mtok=1.0,
    flash_output_usd_per_mtok=1.0,
)


def _build_targets() -> dict[tuple[int, str], int]:
    """Per-``(item_id, role)`` flagged-trial-count target (out of ``_R``). ``role`` is the
    Cell-artifact arm label the stub infers from the prompt (see ``_ShapedModel``). ``C_stripped``
    is deliberately a byte-for-byte COPY of ``D`` on every item -- the mutation contrast (module
    docstring)."""
    targets: dict[tuple[int, str], int] = {}
    for k_id in _K_IDS:
        targets[(k_id, "D")] = _R  # 100% sens
        targets[(k_id, BINDING_BASELINE_ARM)] = 0  # 0% sens -- a genuinely weaker reviewer than D
        targets[(k_id, "C_stripped")] = _R  # mimics D exactly
        targets[(k_id, "B")] = 1  # ~50% -- noise, not a real detector
    for i, c_id in enumerate(_CLEAN_IDS):
        targets[(c_id, "D")] = 0
        targets[(c_id, BINDING_BASELINE_ARM)] = _CLEAN_C_TARGETS[i]
        targets[(c_id, "C_stripped")] = 0  # mimics D exactly
        targets[(c_id, "B")] = 1  # ~50%
    return targets


@dataclass
class _ShapedModel:
    """Reproduces a chosen per-``(item, role)`` flagged-trial-count via REAL model calls driven
    through the driver's REAL arm executors. ``role`` is read off the prompt: the system message
    names D (adversarial) vs C/C_stripped (neutral) vs B (judge); the C-vs-C_stripped split reads
    whether the diet-stripped sentinel is present (exactly the axis :func:`~cogworx.eval.arms.
    _project_diet` manipulates). A per-``(item_id, role)`` call counter tracks trial order (``run_
    arms`` calls a given ``(item, arm)`` pair's ``R`` trials contiguously and in order), so trial
    ``n`` is flagged iff ``n < target`` -- the same "first N of R" convention
    ``tests/eval/test_scorer.py``'s ``_emit`` helper uses."""

    targets: Mapping[tuple[int, str], int]
    calls: dict[tuple[int, str], int] = field(default_factory=dict)

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(structured_output=False)

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage] = (),
        tools: Sequence[ToolSpec] = (),
        tier: ModelTier = "pro",
        json_schema: Mapping[str, Any] | None = None,
    ) -> ModelResponse:
        msgs = list(messages)
        system = msgs[0].content if msgs else ""
        joined = "\n".join(m.content for m in msgs)
        match = _MARKER_RE.search(joined)
        assert match is not None, f"no ITEM_MARKER in prompt: {joined!r}"
        item_id = int(match.group(1))

        if "rigorous experiment judge" in system:
            role = "B"
        elif "rigorously attack" in system:
            role = "D"
        elif _WITHHELD_EXPERIMENT_DESIGN in joined:
            role = "C_stripped"
        else:
            role = BINDING_BASELINE_ARM

        key = (item_id, role)
        n = self.calls.get(key, 0)
        self.calls[key] = n + 1
        flagged = n < self.targets[key]

        payload: dict[str, Any]
        if role == "B":
            payload = {
                "experiment_validly_tests_solution": True,
                "predicted_solution_holds": not flagged,
                "reasoning": "stub",
            }
        else:
            payload = (
                {"disposition": "broke", "breakage": "stub", "confidence": 0.5}
                if flagged
                else {"disposition": "could_not_break", "breakage": None, "confidence": 0.5}
            )
        return ModelResponse(text=json.dumps(payload), model_id="stub", finish_reason="stop")

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def _solution_code(item_id: int) -> str:
    """A trivially-correct solution, IDENTICAL in behavior for every item (K and clean alike): the
    frozen test passes regardless, so the REAL arm-A executor (real ``pytest`` subprocess) is blind
    to every item (INV-A0: ``sens_A == 0`` on K) and never false-positives on clean (INV-A1:
    ``spec_A == 1``). The item marker is a comment -- inert to execution, but a literal substring in
    the text every model-arm prompt quarantines, so the stub model can recover ``item_id``."""
    return f"def f(x):\n    # ITEM_MARKER_{item_id}\n    return x + 1\n"


_TEST_CODE = "from solution import f\n\n\ndef test_f():\n    assert f(1) == 2\n"


def _corpus() -> list[CorpusItem]:
    frame = OracleFrame(
        completion_criterion="tests_pass", problem_type="code", problem_statement="p"
    )
    items: list[CorpusItem] = []
    for i, k_id in enumerate(_K_IDS):
        thesis = Thesis(
            proposed_solution=_solution_code(k_id), experiment_design="run the frozen tests"
        )
        items.append(
            CorpusItem(
                item_id=k_id,
                frame=frame,
                thesis=thesis,
                test_code=_TEST_CODE,
                is_error=1,
                label_source="human",
                label_provenance=HumanLabelProvenance(adjudications=()),
                stratum="K",
                oracle_reachable=False,
                error_regime=_REGIMES[i % len(_REGIMES)],
                difficulty=DifficultyMarker(planted_difficulty="medium", surface_complexity=1),
                matched_sibling_id=None,
                split="measurement",
                planter=DeterministicPlanterStamp(operators=("op",)),
                content_hash=f"h{k_id}",
            )
        )
    for c_id in _CLEAN_IDS:
        thesis = Thesis(
            proposed_solution=_solution_code(c_id), experiment_design="run the frozen tests"
        )
        items.append(
            CorpusItem(
                item_id=c_id,
                frame=frame,
                thesis=thesis,
                test_code=_TEST_CODE,
                is_error=0,
                label_source="oracle",
                label_provenance=OracleLabelProvenance(
                    returncode=0,
                    test_provenance="frozen",
                    holds=True,
                    valid_check=True,
                    oracle_id="o",
                ),
                stratum="clean",
                oracle_reachable=False,
                difficulty=DifficultyMarker(planted_difficulty="medium", surface_complexity=1),
                matched_sibling_id=None,
                split="measurement",
                planter=DeterministicPlanterStamp(operators=("op",)),
                content_hash=f"h{c_id}",
            )
        )
    return items


def _settings() -> GateRunSettings:
    arm_family = ProviderConfig(
        model_pro="stub-pro", model_flash="stub-flash", price_per_mtok=_PRICE
    )
    return GateRunSettings(
        mode="bring-up",
        price_basis="list",
        resolved_price_tables={"deepseek": _PRICE},
        phase_a_max_usd=10.0,
        phase_b_max_usd=10.0,
        arm_family=arm_family,
        role_families={"arm_family": "deepseek"},
    )


def _fingerprint() -> MeasurementFingerprint:
    env = ExecEnvIdentity(
        python_version="3.13.0",
        python_implementation="CPython",
        package_versions=(("pydantic", "2.0"),),
        locale_lc_ctype="C",
    )
    return MeasurementFingerprint(
        content_hash_aggregate="agg",
        git_sha="deadbeef",
        planter_families=("deterministic-mutation",),
        exec_env=env,
        residual_epsilon=0.0,
    )


def _clean_shuffle_result() -> ShuffleNullResult:
    """A hand-built, already-centered ShuffleNullResult -- injected so this test never pays the
    live §5 shuffle bootstrap (that wiring is pinned separately in ``test_scorer.py``); this test
    is about the Tier-4 arm-label contract, not the shuffle null."""
    n = 200
    return ShuffleNullResult(
        n_shuffles=n,
        paired_point_estimates={"D>C'": tuple(0.0 for _ in range(n))},
        paired_ci_bounds={"D>C'": tuple((-0.5, 0.5) for _ in range(n))},
        global_ci_bounds={"D>C'": tuple((-0.5, 0.5) for _ in range(n))},
    )


def test_scorer_reads_binding_baseline_arm_not_c_stripped_for_d_gt_c_prime(tmp_path: Path) -> None:
    """THE FINDING-1 PIN: build the REAL driver arm map (so ``"C_stripped"`` is present in the
    artifact alongside the binding-baseline ``"C"``), run the REAL ``run_arms``, score with the
    REAL ``score_gate`` -- the honest wiring PASSes. See the module docstring for why the mutation
    (``_BINDING_DELTAS["D>C'"]`` pointed at ``"C_stripped"``) deterministically flips this to
    FAIL."""
    corpus = _corpus()
    model = _ShapedModel(targets=_build_targets())
    guard = BudgetGuard(max_usd=10.0)
    executors = _build_arm_executors(
        corpus,
        model=model,
        settings=_settings(),
        guard=guard,
        fingerprint_digest="test-fingerprint",
        out_dir=tmp_path,
    )
    assert {"A", "PC", "D", BINDING_BASELINE_ARM, "C_stripped", "B"} <= executors.keys()

    cells = run_arms(corpus, arm_executors=executors, R=_R)
    assert {c.arm for c in cells} == {"A", "PC", "D", BINDING_BASELINE_ARM, "C_stripped", "B"}

    verdict = score_gate(
        cells,
        corpus,
        _fingerprint(),
        expected_fingerprint=None,
        lineage_look_count=0,
        n_clean_planned=80,
        R=_R,
        gate_threshold=0.0,
        n_outer=2000,
        look_budget_max=10,
        sigma_sq_b_spec_planning=0.001,
        shuffle_seed=42,
        positive_control_arm="PC",
        strawman_arm="B",
        j_pos_min=0.8,
        n_shuffles=200,
        shuffle_result=_clean_shuffle_result(),
    )

    assert verdict.verdict_status == "PASS", verdict
    assert verdict.passed is True
    assert verdict.binding_deltas["D>C'"].lo > 0.5
