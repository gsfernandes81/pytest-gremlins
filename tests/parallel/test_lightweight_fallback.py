"""Tests for the lightweight runner's fallback to the real pytest command.

The lightweight runner cannot execute fixture-taking, parametrized or async
tests.  When it says so (``LIGHTWEIGHT_CANNOT_RUN_EXIT_CODE``) the caller must
re-run the *original* test command — a full pytest invocation with the same
node IDs and environment — so the gremlin is scored on a real run instead of
on the runner's inability to do one.
"""

from __future__ import annotations

import re
import subprocess
from typing import TYPE_CHECKING

import pytest

from pytest_gremlins.parallel.lightweight import (
    LIGHTWEIGHT_CANNOT_RUN_EXIT_CODE,
    run_lightweight_with_fallback,
)
from pytest_gremlins.plugin import _get_lightweight_runner_script

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

TEST_COMMAND = ['python', 'bootstrap.py', '-x', 'tests/test_foo.py::test_bar']


@pytest.fixture
def env_vars(tmp_path: Path) -> dict[str, str]:
    """Environment that makes ``build_lightweight_command`` return a command."""
    (tmp_path / 'gremlin_lightweight_runner.py').write_text('# runner', encoding='utf-8')
    return {'PYTEST_GREMLINS_SOURCES_FILE': str(tmp_path / 'sources.json')}


def _recorder(*return_codes: int) -> tuple[list[list[str]], Callable[[list[str]], subprocess.CompletedProcess[bytes]]]:
    """Build a fake command runner that returns the given codes in order."""
    calls: list[list[str]] = []
    codes = list(return_codes)

    def run_command(command: list[str]) -> subprocess.CompletedProcess[bytes]:
        calls.append(command)
        return subprocess.CompletedProcess(args=command, returncode=codes.pop(0), stdout=b'', stderr=b'')

    return calls, run_command


@pytest.mark.small
class DescribeLightweightFallback:
    """run_lightweight_with_fallback re-runs the real command on cannot-run."""

    def it_reruns_the_original_command_when_the_runner_cannot_run_the_tests(
        self,
        env_vars: dict[str, str],
    ) -> None:
        """The cannot-run code must never reach the caller's exit-code mapping."""
        calls, run_command = _recorder(LIGHTWEIGHT_CANNOT_RUN_EXIT_CODE, 0)

        outcome = run_lightweight_with_fallback(TEST_COMMAND, env_vars, run_command)

        assert len(calls) == 2
        assert calls[1] == TEST_COMMAND
        assert outcome.returncode == 0

    @pytest.mark.parametrize(
        'return_code',
        [
            pytest.param(0, id='survived'),
            pytest.param(1, id='zapped'),
            pytest.param(2, id='error'),
        ],
    )
    def it_keeps_the_lightweight_verdict_for_every_other_code(
        self,
        env_vars: dict[str, str],
        return_code: int,
    ) -> None:
        """Only the cannot-run code costs a second run; the fast path stays fast."""
        calls, run_command = _recorder(return_code)

        outcome = run_lightweight_with_fallback(TEST_COMMAND, env_vars, run_command)

        assert len(calls) == 1
        assert outcome.returncode == return_code

    def it_runs_the_original_command_when_no_lightweight_command_is_available(self) -> None:
        """No runner script means the original command, exactly as before."""
        calls, run_command = _recorder(0)

        outcome = run_lightweight_with_fallback(TEST_COMMAND, {}, run_command)

        assert calls == [TEST_COMMAND]
        assert outcome.returncode == 0


@pytest.mark.small
class DescribeCannotRunExitCode:
    """The constant and the generated runner script must agree."""

    def it_is_outside_the_range_pytest_uses(self) -> None:
        """pytest exits 0-5; the cannot-run code must not collide with any of them."""
        assert LIGHTWEIGHT_CANNOT_RUN_EXIT_CODE > 5

    def it_matches_the_code_in_the_generated_runner_script(self) -> None:
        """The runner is a standalone script, so the value is declared twice."""
        script = _get_lightweight_runner_script()
        match = re.search(r'^CANNOT_RUN_EXIT_CODE = (\d+)$', script, re.MULTILINE)

        assert match is not None, 'the runner script no longer declares CANNOT_RUN_EXIT_CODE'
        assert int(match.group(1)) == LIGHTWEIGHT_CANNOT_RUN_EXIT_CODE
