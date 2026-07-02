"""The composed diet-firewall pin test (wiring fix, 2026-07-02) — closes two 4.4d wiring findings
at once, driven through the REAL driver composition, never through ``project_arm_input`` directly
(every pre-existing ``test_arms.py`` test pre-projects, so none of them would have caught either
finding below regressing).

FINDING 1 (arms.py) — the diet weld: before this fix, only :func:`~cogworx.eval.arms.
project_arm_input` applied the info-diet transform (the C-vs-C' ``experiment_design`` strip + the
``test_code`` drop for every model arm). A ``make_*_executor`` factory driven straight through
:func:`~cogworx.eval.runner.run_arms` — which hands ONE generic, un-projected ``ArmInput`` (the real
``test_code``, the un-stripped ``experiment_design``) to every arm — silently got the RAW input; the
S9 firewall never fired. Each factory now self-applies the shared, idempotent ``arms._project_diet``
inside its own executor closure.

FINDING 2 (``_live/driver.py``) — the Cell-label crossover: the driver's arm map keys the
binding-baseline Cell arm off :data:`cogworx.eval.scorer.BINDING_BASELINE_ARM` (``"C"``) ->
:func:`~cogworx.eval.arms.make_c_prime_executor` (full diet), and the reported-only stripped arm
under the non-primed ``"C_stripped"`` -> :func:`~cogworx.eval.arms.make_c_executor` (stripped diet)
— a silent swap of those two map entries is exactly the mistake FIX 2 makes structurally harder.

This test builds the driver's REAL :func:`~cogworx.eval._live.driver._build_arm_executors` map with
a message-CAPTURING stub :class:`~cogworx.model.base.Model`, drives each arm through
:func:`~cogworx.eval.runner.run_arms`, and inspects the captured prompts per arm — MUTATION-
RESISTANT: reverting either weld, or swapping the two Cell-arm-label map entries, flips one of the
assertions below.
"""

from __future__ import annotations

import json
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
    DifficultyMarker,
    LLMPlanterStamp,
    OracleLabelProvenance,
)
from cogworx.eval.runner import run_arms
from cogworx.eval.scorer import BINDING_BASELINE_ARM
from cogworx.model.base import ChatMessage, ModelCapabilities, ModelResponse, ModelTier, ToolSpec
from cogworx.model.providers.config import PriceTable, ProviderConfig
from cogworx.verification.contracts import OracleFrame, Thesis

_EXPERIMENT_DESIGN = "run the property test against 1000 random lists"
_TEST_MARKER = "assert f([1, 2, 3]) == 6"
_TS_PROV = OracleLabelProvenance(
    returncode=1, test_provenance="frozen", holds=False, valid_check=True, oracle_id="x"
)
_PRICE = PriceTable(
    pro_input_usd_per_mtok=1.0,
    pro_output_usd_per_mtok=1.0,
    flash_input_usd_per_mtok=1.0,
    flash_output_usd_per_mtok=1.0,
)


def _item() -> CorpusItem:
    return CorpusItem(
        item_id=1,
        frame=OracleFrame(
            completion_criterion="tests_pass", problem_type="code", problem_statement="sum a list"
        ),
        thesis=Thesis(
            proposed_solution="def f(xs):\n    return sum(xs) - 1\n",
            experiment_design=_EXPERIMENT_DESIGN,
        ),
        test_code=f"def test_f():\n    {_TEST_MARKER}\n",
        is_error=1,
        label_source="oracle",
        label_provenance=_TS_PROV,
        stratum="K",
        oracle_reachable=False,
        error_regime="off-by-one",
        difficulty=DifficultyMarker(planted_difficulty="medium", surface_complexity=12),
        matched_sibling_id=None,
        split="measurement",
        planter=LLMPlanterStamp(model_family="planterfam", model_id="p1"),
    )


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


@dataclass
class _CapturingModel:
    """Records every call's messages; returns a fixed, never-flag answer for both the dialectic and
    the judge shapes — the ROUTED outcome does not matter here, only the captured prompt does."""

    calls: list[list[ChatMessage]] = field(default_factory=list)

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
        self.calls.append(msgs)
        system = msgs[0].content if msgs else ""
        payload: dict[str, Any]
        if "rigorous experiment judge" in system:
            payload = {
                "experiment_validly_tests_solution": True,
                "predicted_solution_holds": True,
                "reasoning": "stub",
            }
        else:
            payload = {"disposition": "could_not_break", "breakage": None, "confidence": 0.5}
        return ModelResponse(text=json.dumps(payload), model_id="stub", finish_reason="stop")

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def _joined(messages: list[ChatMessage]) -> str:
    return "\n".join(m.content for m in messages)


def test_run_arms_through_driver_wiring_enforces_diet_firewall(tmp_path: Path) -> None:
    """The composed pin (see module docstring). Builds the REAL driver arm map, drives EACH arm
    through ``run_arms`` (never through ``project_arm_input``), and checks (a) the binding-baseline
    ``BINDING_BASELINE_ARM`` ("C") prompt carries the item's real ``experiment_design``, (b) the
    ``"C_stripped"`` prompt carries the withheld sentinel and NOT the real design, (c) no model-arm
    prompt (D / C / C_stripped / B) carries ``test_code``."""
    corpus = [_item()]
    model = _CapturingModel()
    guard = BudgetGuard(max_usd=10.0)
    executors = _build_arm_executors(
        corpus,
        model=model,
        settings=_settings(),
        guard=guard,
        fingerprint_digest="test-fingerprint",
        out_dir=tmp_path,
    )
    assert {BINDING_BASELINE_ARM, "C_stripped", "D", "B"} <= executors.keys()

    def _prompt_for(arm: str) -> str:
        model.calls.clear()
        run_arms(corpus, arm_executors={arm: executors[arm]}, R=1)
        assert len(model.calls) == 1
        return _joined(model.calls[0])

    # (a) the binding baseline "C" gets the FULL diet — the real experiment_design is present.
    binding_prompt = _prompt_for(BINDING_BASELINE_ARM)
    assert _EXPERIMENT_DESIGN in binding_prompt
    assert _TEST_MARKER not in binding_prompt

    # (b) "C_stripped" gets the STRIPPED diet — the sentinel is present, the real design is not.
    stripped_prompt = _prompt_for("C_stripped")
    assert _WITHHELD_EXPERIMENT_DESIGN in stripped_prompt
    assert _EXPERIMENT_DESIGN not in stripped_prompt
    assert _TEST_MARKER not in stripped_prompt

    # (c) test_code never reaches any model arm, D and B included.
    assert _TEST_MARKER not in _prompt_for("D")
    assert _TEST_MARKER not in _prompt_for("B")
