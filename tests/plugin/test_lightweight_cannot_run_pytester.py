"""End-to-end proof that a test the fast path cannot run is not scored as a kill.

A project whose only test takes a fixture reported ``Zapped: 100%`` with zero
survivors: the lightweight runner called the test function bare, got a
``TypeError`` for the missing fixture argument, exited 1, and the wrapper reads
exit 1 as ZAPPED (issue #486).  The mutation was never exercised.

The second scenario is the guard on the fix: a zero-argument test that the
runner *can* execute must still zap a mutation it detects, so routing
"cannot run" away from exit 1 does not cost real kills.
"""

from __future__ import annotations

import re

import pytest

_SCORE_RE = re.compile(r'^(Zapped|Survived): (\d+) gremlins', re.MULTILINE)

_CONFTEST = """
import pytest

def pytest_configure(config):
    config.addinivalue_line('markers', 'small: fast unit tests')


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    for item in items:
        if not any(m.name in ('small', 'medium', 'large') for m in item.iter_markers()):
            item.add_marker(pytest.mark.small)
"""


def _write_project(pytester: pytest.Pytester, test_source: str) -> None:
    """Write a one-function target plus the given test module."""
    pytester.makepyfile(target='def flag(value):\n    return value > 10\n')
    pytester.makepyfile(test_target=test_source)
    pytester.makepyprojecttoml(
        """
[tool.pytest-gremlins]
paths = ["target.py"]
"""
    )
    pytester.makeconftest(_CONFTEST)


def _scores(output: str) -> dict[str, int]:
    """Parse the ``Zapped``/``Survived`` counts out of the mutation report."""
    return {label: int(count) for label, count in _SCORE_RE.findall(output)}


@pytest.mark.medium
class DescribeFixtureTakingTests:
    """A fixture-taking test must not turn every mutation into a kill."""

    def it_does_not_zap_a_mutation_its_test_never_exercised(self, pytester: pytest.Pytester) -> None:
        """``assert flag(11)`` cannot detect ``> 10`` becoming ``>= 10``.

        The honest report has survivors.  Before the fix this project reported
        100% zapped with zero survivors, on a test that never ran.
        """
        _write_project(
            pytester,
            'from target import flag\n\n\ndef test_flag(tmp_path):\n    assert flag(11)\n',
        )

        result = pytester.runpytest('-p', 'pytest_gremlins', '--gremlins')

        output = result.stdout.str()
        scores = _scores(output)
        assert scores, output
        assert scores['Survived'] > 0, output


@pytest.mark.medium
class DescribeRunnableTests:
    """Tests the fast path can run keep killing what they killed before."""

    def it_still_zaps_a_mutation_its_test_detects(self, pytester: pytest.Pytester) -> None:
        """A zero-argument test that pins both sides of the boundary kills every mutant."""
        _write_project(
            pytester,
            'from target import flag\n'
            '\n'
            '\n'
            'def test_flag():\n'
            '    assert flag(11)\n'
            '    assert not flag(10)\n'
            '    assert not flag(9)\n',
        )

        result = pytester.runpytest('-p', 'pytest_gremlins', '--gremlins')

        output = result.stdout.str()
        scores = _scores(output)
        assert scores, output
        assert scores['Zapped'] > 0, output
        assert scores['Survived'] == 0, output
