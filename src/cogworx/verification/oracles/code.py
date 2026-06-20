"""Deterministic code-execution oracle — run the thesis under pytest to verify it (CANON S9, S10).

The strongest oracle for code domains: it replaces the LLM-judge's *prediction* with *execution*.
The thesis's ``proposed_solution`` is written as ``solution.py`` and its ``experiment_design`` as
``test_solution.py``; pytest's exit code is the verdict. Because we ran a real tool first-hand, the
resulting ``Verdict.source`` is ``"tool"`` — ground truth that MAY back a CONFIRMED claim and a
procedural-KG Beta update (unlike a model-judge verdict, which is ``"inference"`` and may not).

Exit-code → (holds, valid_check), pytest's documented codes:
  0  → holds=True,  valid_check=True   (all tests passed)
  1  → holds=False, valid_check=True   (a real refutation — tests ran and failed)
  5  → holds=False, valid_check=False  (no tests collected — nothing was actually tested)
  2/3/4/137/timeout/other → holds=False, valid_check=False  (the experiment did not run cleanly)
``valid_check=False`` keeps a noise result out of the Beta posterior (F1).

THREAT MODEL + F7 (the load-bearing safety rule). This oracle EXECUTES model-written code. The local
subprocess tier is hardened against ACCIDENTS (hard timeout + process-group kill, fresh tempdir,
scrubbed env, no bytecode bleed) but — like the tess original — is explicitly NOT a security
sandbox (unrestricted network; on Windows the group-kill is cooperative). So the tier is bound to
the durable taint latch (Pod 3.2, ``RunState.tainted``): once a run's drive is tainted, model
output is adversary-influenced, and the unsandboxed tier **REFUSES** to execute it — returning a
zero-execution refusal ``Verdict`` (``source="system"``, ``valid_check=False``). On a tainted
drive, execution requires the docker-hardened tier (``--network none``, dropped caps, non-root,
read-only, mem/pids/fsize caps); when docker is unavailable the oracle refuses rather than run
untrusted code unsandboxed (S10 fail-safe). The docker tier is lesion-clean: with docker absent, a
clean (untainted) run still verifies via the local tier. Generalised from tess
``oracles/pytest_runner.py`` + ``docker_pytest_runner.py``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, cast

from cogworx.verification.contracts import OracleFrame, Thesis, Verdict
from cogworx.verification.quarantine import quarantine

if TYPE_CHECKING:
    from cogworx.loop.stage import StageContext

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S: Final = 30.0
DEFAULT_REASONING_TAIL: Final = 500
SANDBOX_IMAGE: Final = "cogworx-sandbox:py3.13"

_FENCE_RE: Final = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_DOCKER_KILL_TIMEOUT_S: Final = 5.0

_PYTEST_ARGS: Final = (
    "test_solution.py",
    "-q",
    "--no-header",
    "-p",
    "no:cacheprovider",
    "-p",
    "no:xdist",
    "-o",
    "addopts=",  # ignore the project's pyproject addopts (--strict-markers etc.)
)


def _extract_python(text: str) -> str:
    """Return the first ```python fenced block if present, else the text verbatim (port tess)."""
    match = _FENCE_RE.search(text)
    return match.group(1) if match is not None else text


def _tail(s: str, n: int) -> str:
    return s if len(s) <= n else "...\n" + s[-n:]


def _scrubbed_env(tmp: str) -> dict[str, str]:
    """Minimal env for the subprocess: keep PATH (so ``sys.executable`` resolves), point TEMP/TMP
    into the tempdir to bound stray writes, disable ``__pycache__`` to stop bytecode bleed."""
    env: dict[str, str] = {
        "PATH": os.environ.get("PATH", ""),
        "TEMP": tmp,
        "TMP": tmp,
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if sys.platform == "win32":
        env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
    return env


@dataclass(frozen=True)
class _RunResult:
    """Raw outcome of one pytest invocation (local or docker)."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


def _write_trial(tmp_path: Path, solution_code: str, test_code: str) -> None:
    """Lay down solution.py + test_solution.py + an empty conftest.py (anchors pytest's rootdir in
    the tempdir so it does not walk up into the project's pyproject)."""
    (tmp_path / "solution.py").write_text(solution_code, encoding="utf-8")
    (tmp_path / "test_solution.py").write_text(test_code, encoding="utf-8")
    (tmp_path / "conftest.py").write_text("", encoding="utf-8")


def _kill_group(proc: subprocess.Popen[str]) -> None:
    """Take down the whole process tree on timeout. Windows group-kill is cooperative (a detached
    grandchild can survive — documented limitation, the reason the docker tier exists)."""
    if sys.platform == "win32":
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        except (ValueError, OSError):
            proc.kill()
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            proc.kill()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()


def _run_pytest_local(solution_code: str, test_code: str, *, timeout_s: float) -> _RunResult:
    """Run the trial under pytest in a fresh tempdir (local, UNSANDBOXED tier). Blocking — call via
    ``asyncio.to_thread``. New process group so the timeout kill takes the whole tree."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        _write_trial(Path(tmp), solution_code, test_code)
        cmd = [sys.executable, "-m", "pytest", *_PYTEST_ARGS, f"--rootdir={tmp}"]
        kwargs: dict[str, Any] = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        proc = subprocess.Popen(
            cmd,
            cwd=tmp,
            env=_scrubbed_env(tmp),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **kwargs,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
            return _RunResult(proc.returncode, stdout or "", stderr or "", timed_out=False)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            stdout, stderr = proc.communicate()
            return _RunResult(-1, stdout or "", stderr or "", timed_out=True)


def _docker_bin(explicit: str | None) -> str | None:
    """Resolve the docker binary, or ``None`` if docker is not on PATH (tier unavailable → S8)."""
    return explicit or shutil.which("docker")


def _image_present(docker_bin: str, image: str) -> bool:  # pragma: no cover - needs docker
    try:
        result = subprocess.run(
            [docker_bin, "image", "inspect", image],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10.0,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    return result.returncode == 0


def build_docker_cmd(
    *, tmp_str: str, container_name: str, image: str, docker_bin: str
) -> list[str]:
    """Build the hardened ``docker run`` argv (PURE — unit-testable without docker).

    The hardening is the security review surface (red-teamer owns the full review): no network, no
    kernel caps, non-root, read-only rootfs, bounded memory/pids/fsize/fds, a size-capped noexec
    tmpfs for pytest scratch. Ported from tess ``docker_pytest_runner._build_run_cmd``.
    """
    return [
        docker_bin,
        "run",
        "--rm",
        "--name",
        container_name,
        "--network",
        "none",
        "--memory=512m",
        "--memory-swap=512m",  # swap=memory hard-caps total addressable memory
        "--cpus=1",
        "--pids-limit=64",  # fork-bomb defense
        "--cap-drop=ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        "1000:1000",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m,mode=1777",
        "--ulimit",
        "fsize=10485760",  # single-file write cap (bytes)
        "--ulimit",
        "nofile=128:128",  # fd-exhaustion cap
        "-v",
        f"{tmp_str}:/work:rw",
        "-w",
        "/work",
        image,
        "python",
        "-m",
        "pytest",
        *_PYTEST_ARGS,
        "--rootdir=/work",
        "--basetemp=/tmp/pytest",
    ]


def _kill_container(docker_bin: str, name: str) -> None:  # pragma: no cover - needs docker
    for args in ([docker_bin, "kill", "--signal=KILL", name], [docker_bin, "rm", "-f", name]):
        try:
            subprocess.run(args, capture_output=True, timeout=_DOCKER_KILL_TIMEOUT_S, check=False)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            log.warning("code_oracle.docker.cleanup_failed", extra={"container": name})


def _run_pytest_docker(  # pragma: no cover - needs docker
    solution_code: str, test_code: str, *, timeout_s: float, image: str, docker_bin: str
) -> _RunResult:
    """Run the trial under pytest in a hardened container (docker tier). Blocking — call via
    ``asyncio.to_thread``. Not exercised in CI (no docker); the security argv is verified by
    ``build_docker_cmd`` unit tests + review."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        _write_trial(Path(tmp), solution_code, test_code)
        name = f"cogworx-sandbox-{uuid.uuid4().hex[:12]}"
        cmd = build_docker_cmd(
            tmp_str=str(tmp), container_name=name, image=image, docker_bin=docker_bin
        )
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
            return _RunResult(proc.returncode, stdout or "", stderr or "", timed_out=False)
        except subprocess.TimeoutExpired:
            _kill_container(docker_bin, name)
            try:
                stdout, stderr = proc.communicate(timeout=_DOCKER_KILL_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
            return _RunResult(-1, stdout or "", stderr or "", timed_out=True)


def verdict_from_result(
    result: _RunResult,
    *,
    timeout_s: float,
    test_provenance: Literal["thesis", "frozen"] = "thesis",
) -> Verdict:
    """Map a pytest ``_RunResult`` to a ``Verdict`` (source ``"tool"`` — first-hand execution).

    The subprocess stdout/stderr is attacker-influenceable (model-emitted test code prints whatever
    it likes), so it is wrapped via the quarantine channel before becoming ``reasoning`` — a
    downstream antithesis/refine prompt cannot be hijacked by injected instructions (S10).

    ``test_provenance`` is stamped from the caller's ``test_source`` selector — NEVER from model
    output (S9 audit-only discipline; mirrors ``Verdict.source``).
    """
    if result.timed_out:
        return Verdict(
            holds=False,
            valid_check=False,
            reasoning=quarantine(f"timeout after {timeout_s}s (process killed)"),
            source="tool",
            test_provenance=test_provenance,
        )
    match result.returncode:
        case 0:
            holds, valid_check = True, True
        case 1:
            holds, valid_check = False, True
        case _:  # 5 (no tests), 2/3/4 (collection/usage), 137 (OOM), anything else
            holds, valid_check = False, False
    reasoning = quarantine(_tail(result.stdout + "\n" + result.stderr, DEFAULT_REASONING_TAIL))
    return Verdict(
        holds=holds,
        valid_check=valid_check,
        reasoning=reasoning,
        source="tool",
        test_provenance=test_provenance,
    )


class CodeOracle:
    """Executable oracle: verify a thesis by running ``solution.py`` + ``test_solution.py`` under
    pytest. Implements the :class:`~cogworx.verification.oracle.Oracle` protocol.

    F7 tier selection (read from the durable taint latch each call):
      * clean drive (``RunState.tainted`` is False) → the fast LOCAL subprocess tier;
      * tainted drive (or ``always_sandbox=True``) → the docker-hardened tier; if docker is
        unavailable, REFUSE — return a zero-execution ``Verdict`` (``source="system"``,
        ``valid_check=False``) rather than execute adversary-influenced code unsandboxed (S10).

    ``test_source`` selector (Pod 4.4b):
      * ``"thesis"`` (default) — test body extracted from ``thesis.experiment_design`` via
        ``_extract_python``, preserving all pre-4.4b behaviour byte-for-byte.
      * ``"frozen"`` — test body is ``frozen_test_code`` (raw Python, corpus-author-supplied at
        construction; no ``_extract_python`` call on it because the caller already holds raw
        Python, not a fenced markdown block). This is misconfiguration if ``frozen_test_code`` is
        ``None``; ``__init__`` raises :exc:`ValueError` immediately (not a runtime S8 degrade).

    The solution is ALWAYS extracted from ``thesis.proposed_solution`` regardless of
    ``test_source``; only the test body changes.

    Semver note (Pod 4.4b): ``test_source`` + ``frozen_test_code`` are new optional kwargs;
    all existing construction sites keep their current behaviour (additive, minor bump → 0.5.0).
    """

    def __init__(
        self,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        always_sandbox: bool = False,
        image: str = SANDBOX_IMAGE,
        docker_path: str | None = None,
        test_source: Literal["thesis", "frozen"] = "thesis",
        frozen_test_code: str | None = None,
    ) -> None:
        if test_source == "frozen" and frozen_test_code is None:
            raise ValueError(
                "CodeOracle: test_source='frozen' requires frozen_test_code to be provided; "
                "a frozen oracle with no frozen test is misconfiguration."
            )
        self._timeout_s = timeout_s
        self._always_sandbox = always_sandbox
        self._image = image
        self._docker_path = docker_path
        self._test_source: Literal["thesis", "frozen"] = test_source
        # After the guard above, frozen_test_code is str when test_source=="frozen".
        # Store as str | None; evaluate() narrows below via the same selector.
        self._frozen_test_code: str | None = frozen_test_code

    async def evaluate(self, *, frame: OracleFrame, thesis: Thesis, ctx: StageContext) -> Verdict:
        run = await ctx.journal.load_run(ctx.run_id)
        tainted = bool(run.tainted) if run is not None else False

        solution_code = _extract_python(thesis.proposed_solution)
        if self._test_source == "frozen":
            # __init__ guarantees non-None when test_source=="frozen"; cast narrows for mypy.
            test_code = cast(str, self._frozen_test_code)
        else:
            test_code = _extract_python(thesis.experiment_design)

        if tainted or self._always_sandbox:
            docker_bin = _docker_bin(self._docker_path)
            if docker_bin is None or not _image_present(docker_bin, self._image):
                return self._refusal(tainted=tainted)
            result = await asyncio.to_thread(
                _run_pytest_docker,
                solution_code,
                test_code,
                timeout_s=self._timeout_s,
                image=self._image,
                docker_bin=docker_bin,
            )
        else:
            result = await asyncio.to_thread(
                _run_pytest_local, solution_code, test_code, timeout_s=self._timeout_s
            )
        return verdict_from_result(
            result, timeout_s=self._timeout_s, test_provenance=self._test_source
        )

    def _refusal(self, *, tainted: bool) -> Verdict:
        """A zero-execution refusal Verdict (F7/S10). ``source="system"`` — a deterministic control
        event, not a tool result; ``valid_check=False`` so it stamps no Beta and records no
        evidence. NOTHING was executed."""
        why = "the drive is tainted" if tainted else "a sandbox was required"
        log.warning("code_oracle.refused", extra={"tainted": tainted})
        return Verdict(
            holds=False,
            valid_check=False,
            reasoning=(
                f"refused: {why} and no docker-hardened sandbox is available; the unsandboxed tier "
                "will not execute model-written code without a sandbox here (F7/S10)."
            ),
            source="system",
        )


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "SANDBOX_IMAGE",
    "CodeOracle",
    "build_docker_cmd",
    "verdict_from_result",
]
