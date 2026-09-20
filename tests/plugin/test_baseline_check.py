"""Tests for the baseline check that runs before any gremlin is scored.

A mutation score is only meaningful if the tests pass with no mutation active.
When they do not, every gremlin is "zapped" for a reason that is not the
mutation, and the report reads as a perfect score.  The baseline check runs the
selected tests once with no gremlin active and refuses to score anything unless
they pass.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from _pytest.config.argparsing import (
    OptionGroup,
    Parser,
)
import pytest

from pytest_gremlins.instrumentation.switcher import ACTIVE_GREMLIN_ENV_VAR
from pytest_gremlins.plugin import (
    GremlinSession,
    _run_baseline_check,
    pytest_addoption,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def gremlin_session() -> GremlinSession:
    """A session with two collected tests and nothing else configured."""
    return GremlinSession(
        enabled=True,
        test_node_ids={'test_one': 'tests/test_a.py::test_one', 'test_two': 'tests/test_a.py::test_two'},
    )


@pytest.mark.small
class DescribeBaselineCheckOption:
    """The check is on by default and can be switched off."""

    def it_registers_the_opt_out_flag(self) -> None:
        parser = MagicMock(spec=Parser)
        group = MagicMock(spec=OptionGroup)
        parser.getgroup.return_value = group

        pytest_addoption(parser)

        added_option_names = [call.args[0] for call in group.addoption.call_args_list]
        assert '--gremlin-no-baseline-check' in added_option_names

    def it_defaults_the_session_to_checking(self) -> None:
        assert GremlinSession().baseline_check is True

    def it_starts_with_no_recorded_failure(self) -> None:
        assert GremlinSession().baseline_failure is None


@pytest.mark.medium
class DescribeBaselineCheck:
    """_run_baseline_check reports the reason the baseline is not green.

    Medium, not small: patching ``subprocess.run`` from a small test leaves
    pytest-test-categories' spawn guard installed for later tests.
    """

    def it_passes_when_the_tests_pass_with_no_gremlin_active(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        gremlin_session: GremlinSession,
    ) -> None:
        monkeypatch.setattr(
            'pytest_gremlins.plugin.subprocess.run',
            lambda *_args, **_kwargs: subprocess.CompletedProcess(args=[], returncode=0, stdout=b'', stderr=b''),
        )

        assert _run_baseline_check(gremlin_session, tmp_path) is None

    @pytest.mark.parametrize(
        'return_code',
        [
            pytest.param(1, id='tests-failed'),
            pytest.param(2, id='interrupted'),
            pytest.param(4, id='usage-error'),
        ],
    )
    def it_fails_for_any_non_zero_exit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        gremlin_session: GremlinSession,
        return_code: int,
    ) -> None:
        monkeypatch.setattr(
            'pytest_gremlins.plugin.subprocess.run',
            lambda *_args, **_kwargs: subprocess.CompletedProcess(
                args=[],
                returncode=return_code,
                stdout=b'1 failed',
                stderr=b'',
            ),
        )

        reason = _run_baseline_check(gremlin_session, tmp_path)

        assert reason is not None
        assert str(return_code) in reason

    def it_fails_when_the_baseline_run_times_out(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        gremlin_session: GremlinSession,
    ) -> None:
        def timeout(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
            raise subprocess.TimeoutExpired(cmd='pytest', timeout=1)

        monkeypatch.setattr('pytest_gremlins.plugin.subprocess.run', timeout)

        reason = _run_baseline_check(gremlin_session, tmp_path)

        assert reason is not None
        assert 'timed out' in reason

    def it_runs_pytest_with_no_gremlin_active(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        gremlin_session: GremlinSession,
    ) -> None:
        """The baseline must be the unmutated run, whatever the parent environment holds."""
        recorded: dict[str, object] = {}

        def record(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
            recorded['command'] = command
            recorded['env'] = kwargs['env']
            return subprocess.CompletedProcess(args=command, returncode=0, stdout=b'', stderr=b'')

        monkeypatch.setattr('pytest_gremlins.plugin.subprocess.run', record)
        monkeypatch.setenv(ACTIVE_GREMLIN_ENV_VAR, 'g001')

        _run_baseline_check(gremlin_session, tmp_path)

        env = recorded['env']
        assert isinstance(env, dict)
        assert ACTIVE_GREMLIN_ENV_VAR not in env

    def it_runs_the_selected_tests_through_pytest(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        gremlin_session: GremlinSession,
    ) -> None:
        """Never the lightweight runner: it cannot run a fixture-using suite."""
        recorded: dict[str, object] = {}

        def record(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
            recorded['command'] = command
            return subprocess.CompletedProcess(args=command, returncode=0, stdout=b'', stderr=b'')

        monkeypatch.setattr('pytest_gremlins.plugin.subprocess.run', record)

        _run_baseline_check(gremlin_session, tmp_path)

        command = recorded['command']
        assert isinstance(command, list)
        assert 'pytest' in command
        assert 'gremlin_lightweight_runner.py' not in ' '.join(command)
        assert 'tests/test_a.py::test_one' in command
        assert 'tests/test_a.py::test_two' in command

    def it_skips_the_check_when_no_tests_were_collected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Nothing to run is not a red baseline; the empty-session path reports it."""

        def fail(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
            pytest.fail('the baseline check must not spawn a subprocess with no tests')

        monkeypatch.setattr('pytest_gremlins.plugin.subprocess.run', fail)

        assert _run_baseline_check(GremlinSession(enabled=True), tmp_path) is None


_FAILING_PROJECT_CONFTEST = """
import pytest

def pytest_configure(config):
    config.addinivalue_line('markers', 'small: fast unit tests')


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    for item in items:
        if not any(m.name in ('small', 'medium', 'large') for m in item.iter_markers()):
            item.add_marker(pytest.mark.small)
"""


def _write_failing_project(pytester: pytest.Pytester) -> None:
    """A project whose test fails before any mutation is applied."""
    pytester.makepyfile(target='def flag(value):\n    return value > 10\n')
    pytester.makepyfile(
        test_target='from target import flag\n\n\ndef test_flag():\n    assert flag(3)\n',
    )
    pytester.makepyprojecttoml(
        """
[tool.pytest-gremlins]
paths = ["target.py"]
"""
    )
    pytester.makeconftest(_FAILING_PROJECT_CONFTEST)


@pytest.mark.medium
class DescribeRedBaseline:
    """A red baseline stops the run instead of producing a flattering score."""

    def it_refuses_to_score_and_says_why(self, pytester: pytest.Pytester) -> None:
        _write_failing_project(pytester)

        result = pytester.runpytest('-p', 'pytest_gremlins', '--gremlins')

        output = result.stdout.str()
        assert 'baseline' in output.lower(), output
        assert 'Zapped:' not in output, output
        assert result.ret != 0, output

    def it_can_be_skipped_on_request(self, pytester: pytest.Pytester) -> None:
        """The opt-out exists for people who know their suite is red."""
        _write_failing_project(pytester)

        result = pytester.runpytest('-p', 'pytest_gremlins', '--gremlins', '--gremlin-no-baseline-check')

        assert 'Zapped:' in result.stdout.str(), result.stdout.str()
