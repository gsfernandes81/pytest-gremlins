"""Tests that the lightweight runner never spells "cannot run" as "test failed".

The runner imports test modules and calls the test functions directly, so a
whole class of tests — fixture-taking, parametrized, async, class-based with
setup — cannot be executed by it at all.  Reporting those as exit 1 makes the
wrapper score them ZAPPED, which is the tool claiming a kill for a test that
never ran (issue #486).

Everything here drives the *generated* runner script in a subprocess, which is
exactly how it runs in production.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from pytest_gremlins.parallel.lightweight import LIGHTWEIGHT_CANNOT_RUN_EXIT_CODE
from pytest_gremlins.plugin import _get_lightweight_runner_script

if TYPE_CHECKING:
    from pathlib import Path


TARGET_SOURCE = """
def flag(value):
    return value > 10
"""

TEST_SOURCE = """
import pytest

from target import flag


def test_plain():
    assert flag(11)


def test_failing():
    assert flag(9)


def test_fixture(tmp_path):
    assert flag(11)


@pytest.mark.parametrize('value', [11, 12])
def test_parametrized(value):
    assert flag(value)


async def test_async():
    assert flag(11)


class TestPlainClass:
    def test_method(self):
        assert flag(11)


class TestClassWithSetup:
    def setup_method(self):
        self.threshold = 10

    def test_method(self):
        assert flag(self.threshold + 1)
"""


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Write a target module, a test module and the generated runner script."""
    (tmp_path / 'target.py').write_text(TARGET_SOURCE, encoding='utf-8')
    (tmp_path / 'test_target.py').write_text(TEST_SOURCE, encoding='utf-8')
    (tmp_path / 'runner.py').write_text(_get_lightweight_runner_script(), encoding='utf-8')
    return tmp_path


def _run_many(project: Path, node_ids: list[str]) -> subprocess.CompletedProcess[str]:
    """Run the lightweight runner against several node IDs, as the wrapper does."""
    return subprocess.run(
        [sys.executable, str(project / 'runner.py'), *node_ids],
        cwd=str(project),
        env={**os.environ, 'GREMLIN_ROOTDIR': str(project)},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _run(project: Path, node_id: str) -> subprocess.CompletedProcess[str]:
    """Run the lightweight runner against a single node ID."""
    return _run_many(project, [node_id])


@pytest.mark.medium
class DescribeRunnableTests:
    """Tests the runner can genuinely execute keep their old verdicts."""

    def it_exits_zero_when_the_test_passes(self, project: Path) -> None:
        """A zero-argument test that passes still means "mutation survived"."""
        assert _run(project, 'test_target.py::test_plain').returncode == 0

    def it_exits_one_when_the_test_raises(self, project: Path) -> None:
        """A zero-argument test that raises is a genuine kill: exit 1."""
        assert _run(project, 'test_target.py::test_failing').returncode == 1

    def it_exits_zero_for_a_plain_class_based_test(self, project: Path) -> None:
        """A class with no __init__ and no setup protocol is still runnable."""
        assert _run(project, 'test_target.py::TestPlainClass::test_method').returncode == 0


@pytest.mark.medium
class DescribeTestsTheRunnerCannotRun:
    """Anything the runner cannot execute exits with the cannot-run code."""

    @pytest.mark.parametrize(
        'node_id',
        [
            pytest.param('test_target.py::test_fixture', id='fixture'),
            pytest.param('test_target.py::test_parametrized[11]', id='parametrized'),
            pytest.param('test_target.py::test_async', id='async'),
            pytest.param('test_target.py::TestClassWithSetup::test_method', id='class-setup'),
            pytest.param('test_target.py::test_missing', id='unknown-function'),
            pytest.param('test_target.py::Missing::test_method', id='unknown-class'),
            pytest.param('no_such_file.py::test_plain', id='unknown-module'),
            pytest.param('test_target.py', id='malformed-node-id'),
        ],
    )
    def it_exits_with_the_cannot_run_code(self, project: Path, node_id: str) -> None:
        """Never exit 1 here — exit 1 is the wrapper's word for ZAPPED."""
        outcome = _run(project, node_id)

        assert outcome.returncode == LIGHTWEIGHT_CANNOT_RUN_EXIT_CODE, outcome.stderr

    def it_names_the_test_and_the_reason_on_stderr(self, project: Path) -> None:
        """The operator has to be able to see why the fast path was refused."""
        outcome = _run(project, 'test_target.py::test_fixture')

        assert 'test_target.py::test_fixture' in outcome.stderr
        assert 'tmp_path' in outcome.stderr

    def it_refuses_the_whole_batch_when_one_test_cannot_run(self, project: Path) -> None:
        """One ineligible test sends the whole gremlin down the pytest fallback.

        Without this the batch would run the remaining tests bare and score the
        gremlin on a partial, pytest-unlike run.
        """
        outcome = _run_many(project, ['test_target.py::test_fixture', 'test_target.py::test_failing'])

        assert outcome.returncode == LIGHTWEIGHT_CANNOT_RUN_EXIT_CODE, outcome.stderr

    def it_still_reports_an_earlier_genuine_failure(self, project: Path) -> None:
        """-x semantics are unchanged: the first verdict in order still wins."""
        outcome = _run_many(project, ['test_target.py::test_failing', 'test_target.py::test_fixture'])

        assert outcome.returncode == 1, outcome.stderr
