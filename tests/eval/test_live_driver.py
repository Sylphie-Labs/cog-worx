"""Deterministic unit tests for the L6 bring-up driver (Pod 4.4-live L6).

A STUB Model (deterministic ``ArmOutcome`` -- never "broke", never a K-error signal) + an
in-memory Journal double -- no docker, no network, no real DeepSeek credentials. The corpus build
itself runs the REAL frozen-oracle kernel (a local ``pytest`` subprocess per item, mirroring
``tests/eval/test_live_corpus.py``), so pair counts are kept small (1 O-pair + 1 detK-pair) to bound
wall-clock cost across this file's several full end-to-end runs.

Covers:
  1. a full ``run_bring_up`` emits Cells for every arm in the bring-up map
     ({A, PC, B, C, C_stripped, D}) and the verdict is ``INSTRUMENT_INVALID``.
  2. the written manifest contains the roster blocked-reasons banner, the verdict, and the
     fingerprint digest.
  3. the CRN diagnostic reports the D-D' pair "not measured" (no crash on the absent arm).
  4. ``mode="binding"`` on a single-family roster raises ``RosterUnsound`` -- no run (no model
     call).
  5. exactly ONE ``append_design_look`` lands on the journal (S6).
  6. a fresh "process" (new model + new journal, same ``out_dir``) resumes from the L3 cache: the
     second run's model is never called (every cell already answered in the on-disk cache).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from cogworx.eval._live.driver import BringUpManifest, run_bring_up
from cogworx.eval._live.roster import RosterUnsound
from cogworx.eval._live.settings import GateRunMode, GateRunSettings
from cogworx.model.base import ChatMessage, ModelCapabilities, ModelResponse, ModelTier, ToolSpec
from cogworx.model.providers.config import PriceTable, ProviderConfig
from cogworx.testing.doubles import InMemoryJournal

_PRICE = PriceTable(
    pro_input_usd_per_mtok=1.0,
    pro_output_usd_per_mtok=1.0,
    flash_input_usd_per_mtok=1.0,
    flash_output_usd_per_mtok=1.0,
)
_GIT_SHA = "test-fixed-sha-driver"
_SEED = 424242

# Small pair counts (subprocess-pytest-per-item corpus build) + few trials — bounds wall-clock cost
# across this file's several full end-to-end runs (mirrors test_live_corpus.py's discipline).
_N_O_PAIRS = 1
_N_DETK_PAIRS = 1
_R = 2
_CRN_N_OUTER = 20
_GATE_N_OUTER = 20
_GATE_N_SHUFFLES = 5


def _settings(*, mode: GateRunMode = "bring-up") -> GateRunSettings:
    arm_family = ProviderConfig(
        model_pro="stub-pro", model_flash="stub-flash", price_per_mtok=_PRICE
    )
    return GateRunSettings(
        mode=mode,
        price_basis="list",
        resolved_price_tables={"deepseek": _PRICE},
        phase_a_max_usd=10.0,
        phase_b_max_usd=10.0,
        arm_family=arm_family,
        role_families={"arm_family": "deepseek"},
    )


class _StubModel:
    """A deterministic ``Model`` double: never "broke", never a K-error signal, on every call —
    counts invocations so the L3-cache-resume test can assert zero re-calls."""

    def __init__(self) -> None:
        self.calls = 0

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
        self.calls += 1
        system = messages[0].content if messages else ""
        if "rigorous experiment judge" in system:
            payload: dict[str, Any] = {
                "experiment_validly_tests_solution": True,
                "predicted_solution_holds": True,
                "reasoning": "stub judge — never flags",
            }
        else:
            payload = {"disposition": "could_not_break", "breakage": None, "confidence": 0.5}
        return ModelResponse(text=json.dumps(payload), model_id="stub", finish_reason="stop")

    def count_tokens(self, text: str) -> int:
        return len(text.split())


async def _run(
    out_dir: Path, *, model: _StubModel | None = None
) -> tuple[BringUpManifest, _StubModel, InMemoryJournal]:
    stub_model = model if model is not None else _StubModel()
    journal = InMemoryJournal()
    manifest = await run_bring_up(
        _settings(),
        model=stub_model,
        journal=journal,
        out_dir=out_dir,
        seed=_SEED,
        git_sha=_GIT_SHA,
        R=_R,
        n_o_pairs=_N_O_PAIRS,
        n_detk_pairs=_N_DETK_PAIRS,
        crn_n_outer=_CRN_N_OUTER,
        gate_n_outer=_GATE_N_OUTER,
        gate_n_shuffles=_GATE_N_SHUFFLES,
    )
    return manifest, stub_model, journal


# ===========================================================================
# 1. full end-to-end: every bring-up arm emitted, verdict is INSTRUMENT_INVALID
# ===========================================================================


async def test_run_bring_up_emits_every_arm_and_is_instrument_invalid(tmp_path: Path) -> None:
    manifest, _model, _journal = await _run(tmp_path)

    arms_present = {t.arm for t in manifest.arm_telemetry}
    assert arms_present == {"A", "PC", "B", "C", "C_stripped", "D"}
    assert manifest.cell_count > 0
    assert manifest.verdict.verdict_status == "INSTRUMENT_INVALID"
    assert manifest.verdict.passed is False


# ===========================================================================
# 2. the manifest is written and carries the roster banner + verdict + fingerprint
# ===========================================================================


async def test_manifest_written_with_roster_banner_verdict_and_fingerprint(tmp_path: Path) -> None:
    _manifest, _model, _journal = await _run(tmp_path)

    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    loaded = BringUpManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))

    assert loaded.roster.mode == "bring-up"
    assert loaded.roster.binding_blocked_reasons != ()  # the bring-up banner
    assert loaded.verdict.verdict_status == "INSTRUMENT_INVALID"
    assert loaded.fingerprint_digest and isinstance(loaded.fingerprint_digest, str)

    assert (tmp_path / "cells.json").exists()


# ===========================================================================
# 3. CRN diagnostic: D-D' reported "not measured", never a crash
# ===========================================================================


async def test_crn_reports_d_d_prime_not_measured(tmp_path: Path) -> None:
    manifest, _model, _journal = await _run(tmp_path)

    assert any("D>D'" in note and "D'" in note for note in manifest.crn_not_measured)
    measured_labels = {p.label for p in manifest.crn_diagnostic.per_pair}
    assert "D>D'" not in measured_labels
    # D>A and D>C' ARE measured (both arms present in the bring-up map).
    assert "D>A" in measured_labels
    assert "D>C'" in measured_labels


# ===========================================================================
# 4. mode="binding" on a blocked single-family roster raises RosterUnsound — no run
# ===========================================================================


async def test_binding_mode_blocked_raises_roster_unsound_no_run(tmp_path: Path) -> None:
    stub_model = _StubModel()
    journal = InMemoryJournal()

    with pytest.raises(RosterUnsound):
        await run_bring_up(
            _settings(mode="binding"),
            model=stub_model,
            journal=journal,
            out_dir=tmp_path,
            seed=_SEED,
            git_sha=_GIT_SHA,
            R=_R,
            n_o_pairs=_N_O_PAIRS,
            n_detk_pairs=_N_DETK_PAIRS,
        )

    assert stub_model.calls == 0  # no model call ever happened
    assert not (tmp_path / "manifest.json").exists()  # no run completed
    assert not (tmp_path / "cells.json").exists()


# ===========================================================================
# 5. exactly ONE append_design_look on the journal (S6)
# ===========================================================================


async def test_exactly_one_design_look_appended(tmp_path: Path) -> None:
    _manifest, _model, journal = await _run(tmp_path)

    # Reach into the double's ledger directly: exactly one key, one distinct fingerprint.
    assert len(journal._design_looks) == 1
    (fingerprints,) = journal._design_looks.values()
    assert len(fingerprints) == 1


# ===========================================================================
# 6. mid-run kill + re-run resumes from the L3 cache — the second model is never called
# ===========================================================================


async def test_rerun_with_same_out_dir_resumes_from_cache(tmp_path: Path) -> None:
    first_manifest, first_model, _first_journal = await _run(tmp_path)
    assert first_model.calls > 0  # the first run actually called the model

    second_model = _StubModel()
    second_manifest, _second_model, _second_journal = await _run(tmp_path, model=second_model)

    assert second_model.calls == 0  # every cell was already answered in the on-disk L3 cache
    assert second_manifest.cell_count == first_manifest.cell_count
    assert second_manifest.fingerprint_digest == first_manifest.fingerprint_digest
