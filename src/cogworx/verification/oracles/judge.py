"""LLM-judge oracle — the universal always-on fallback for non-executable criteria (CANON S4, S9).

``LLMJudgeOracle`` is the fallback wired into ``OracleRegistry`` at construction time so every
``(completion_criterion, problem_type)`` pair resolves to *something* (S8 graceful degradation).
It calls the model via ``ctx.model`` (the ``cogworx.model.base.Model`` seam) to decide two control
bits — ``holds`` (did the proposed solution satisfy the criterion?) and ``valid_check`` (did the
experiment actually test the claim?) — using the structured-output / JSON-schema path when the
provider declares that capability, and degrading to plain-text extraction otherwise (S4).

S9 control-bit discipline (mirrors ``coherence.oracle.ModelConsistencyOracle``):
  - ``holds`` and ``valid_check`` come ONLY from the model's typed answer fields.
  - ``reasoning`` is extracted for audit / log; it is NEVER inspected for control flow.
  - ``Verdict.source`` is ALWAYS ``"inference"`` — hardcoded, never derived from model output.
    This oracle is a model-judge: its verdict is a heuristic prioritiser, not first-hand truth.
    Downstream: a model-judge verdict writes NO truth evidence and never stamps the Beta (F1/F2).

Information diet: the judge sees ``frame.problem_statement``, ``frame.completion_criterion``,
``thesis.proposed_solution``, and ``thesis.experiment_design`` — no answer-bearing / oracle-internal
artifacts.  No tool palette is given to the model (pure judge — the executable-oracle path is the
tool-grounded channel; this keeps the executable-vs-inference boundary clean).

Generalised from tess ``oracle.py`` ``LLMJudgeOracle`` (~lines 132-208).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, ValidationError

from cogworx.model.base import ChatMessage, ModelTier
from cogworx.verification.contracts import OracleFrame, Thesis, Verdict

if TYPE_CHECKING:
    from cogworx.loop.stage import StageContext

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates — information diet enforced structurally (task-specific
# fields only; no oracle internals, no previous verdicts, no tool results).
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a rigorous experiment judge. "
    "Your task: given a problem statement, a completion criterion, a proposed solution, and an "
    "experiment design, decide two things:\n"
    "1. Does the experiment VALIDLY TEST whether the proposed solution meets the criterion? "
    "(It is invalid if it is circular, vacuous, untestable, or would not actually discriminate "
    "between a correct and an incorrect solution.)\n"
    "2. ASSUMING the experiment is valid, does the proposed solution HOLD — i.e., would it pass "
    "the experiment and satisfy the criterion?\n"
    "Respond ONLY with a JSON object using exactly these keys: "
    '"experiment_validly_tests_solution" (boolean), '
    '"predicted_solution_holds" (boolean), '
    '"reasoning" (string, one sentence, audit only). '
    "No extra keys, no markdown fences."
)

_USER_TEMPLATE = (
    "Problem statement:\n{problem_statement}\n\n"
    "Completion criterion:\n{completion_criterion}\n\n"
    "Proposed solution:\n{proposed_solution}\n\n"
    "Experiment design:\n{experiment_design}"
)

# ---------------------------------------------------------------------------
# The structured-output schema: two booleans + a one-sentence reasoning string.
# ---------------------------------------------------------------------------

_JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "experiment_validly_tests_solution": {"type": "boolean"},
        "predicted_solution_holds": {"type": "boolean"},
        "reasoning": {"type": "string"},
    },
    "required": [
        "experiment_validly_tests_solution",
        "predicted_solution_holds",
        "reasoning",
    ],
    "additionalProperties": False,
}


class _JudgeAnswer(BaseModel):
    """Typed envelope for the model's structured answer.

    Parsed by framework-side code from the raw model text; never constructed from model output
    directly.
    """

    model_config = ConfigDict(frozen=True)

    experiment_validly_tests_solution: bool
    predicted_solution_holds: bool
    reasoning: str


# ---------------------------------------------------------------------------
# LLMJudgeOracle
# ---------------------------------------------------------------------------


class LLMJudgeOracle:
    """Universal fallback oracle: classify thesis validity via a single model call.

    Accepts any :class:`~cogworx.model.base.Model` (S4 model-agnostic). Uses the
    structured-output / JSON-schema path when the provider declares that capability; degrades to
    plain-text JSON extraction when it does not (S4 graceful degradation).

    Parse failures or malformed model output degrade to ``valid_check=False`` (honest "the judge
    could not determine anything") rather than raising — the registry caller always gets a
    :class:`~cogworx.verification.contracts.Verdict`, the run continues, the failure is logged (S8).

    ``Verdict.source`` is ALWAYS ``"inference"``. This is the F2 boundary: downstream the
    verification-evidence projector reads ``is_executable`` (which is ``False`` for ``"inference"``)
    and writes NO truth evidence and stamps NO Beta update on a model-judge verdict.
    """

    def __init__(self, *, tier: ModelTier = "flash") -> None:
        """Construct the judge oracle.

        Args:
            tier: the model tier to use — ``"flash"`` by default (binary classification with a short
                rationale, not synthesis; mirrors the tess rationale).
        """
        self._tier = tier

    async def evaluate(
        self,
        *,
        frame: OracleFrame,
        thesis: Thesis,
        ctx: StageContext,
    ) -> Verdict:
        """Classify the thesis against the frame using one model call.

        Calls ``ctx.model.complete`` with the information-diet prompt (problem statement +
        criterion + proposed solution + experiment design — no oracle internals, no CoT leakage).
        Parses the two control bits from the typed answer; treats any parse failure as
        ``valid_check=False`` (S8 graceful degrade).

        Returns:
            A :class:`Verdict` with ``source="inference"`` always.
        """
        model = ctx.model
        user_text = _USER_TEMPLATE.format(
            problem_statement=frame.problem_statement,
            completion_criterion=frame.completion_criterion,
            proposed_solution=thesis.proposed_solution,
            experiment_design=thesis.experiment_design,
        )
        system_msg = ChatMessage(role="system", content=_SYSTEM)
        user_msg = ChatMessage(role="user", content=user_text)

        use_structured = model.capabilities.structured_output
        response = await model.complete(
            messages=[system_msg, user_msg],
            tier=self._tier,
            json_schema=_JUDGE_SCHEMA if use_structured else None,
        )

        raw_text = response.text or ""
        return self._parse(raw_text)

    # ------------------------------------------------------------------
    # Parsing — framework-side typed extraction; reasoning is never used
    # for control (S9).
    # ------------------------------------------------------------------

    def _parse(self, raw_text: str) -> Verdict:
        """Extract the two control bits from raw model output.

        Structured path: parse JSON → validate via ``_JudgeAnswer`` → read typed fields.
        Fallback path: same JSON extraction, but the text may not be constrained by the provider.
        On ANY parse failure: degrade to ``holds=False, valid_check=False`` and log; ``source``
        is still ``"inference"`` — the provenance never changes on failure.

        ``reasoning`` is extracted purely for audit; it is stored in the ``Verdict`` and logged
        but NEVER consulted for control flow anywhere (S9 control-bit discipline).
        """
        try:
            parsed = json.loads(raw_text)
            answer = _JudgeAnswer.model_validate(parsed)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            log.warning(
                "llm_judge_oracle.parse_failure",
                extra={"raw_preview": raw_text[:120], "error": str(exc)},
            )
            return Verdict(
                holds=False,
                valid_check=False,
                reasoning=f"[parse failure: {exc}]",
                source="inference",
            )

        valid_check: bool = answer.experiment_validly_tests_solution
        holds: bool = answer.predicted_solution_holds
        reasoning: str = answer.reasoning

        log.debug(
            "llm_judge_oracle.verdict",
            extra={
                "holds": holds,
                "valid_check": valid_check,
                "reasoning_preview": reasoning[:80],
            },
        )
        return Verdict(
            holds=holds,
            valid_check=valid_check,
            reasoning=reasoning,
            source="inference",
        )


__all__ = ["LLMJudgeOracle"]
