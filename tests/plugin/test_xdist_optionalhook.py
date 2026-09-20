"""Tests that the xdist hooks are declared optional.

``_XDIST_AVAILABLE`` only asks whether xdist is *importable*.  ``-p no:xdist``
leaves xdist importable while deactivating it, so its hookspecs are never
registered: the guard passes, the hooks are defined, and pluggy's
``check_pending()`` aborts collection with

    PluginValidationError: unknown hook 'pytest_configure_node'

Marking both hooks ``optionalhook=True`` tells pluggy that a missing spec is
expected, which is exactly the "xdist installed but switched off" case.
"""

from __future__ import annotations

import pytest

from pytest_gremlins import plugin


@pytest.mark.small
class DescribeXdistHookOptionality:
    """Both xdist hookimpls opt out of spec validation."""

    @pytest.mark.parametrize(
        'hook_name',
        [
            pytest.param('pytest_configure_node', id='configure-node'),
            pytest.param('pytest_xdist_node_collection_finished', id='node-collection-finished'),
        ],
    )
    def it_marks_the_hook_as_optional(self, hook_name: str) -> None:
        """Every xdist hookspec we implement must be registered as optional."""
        hook = getattr(plugin, hook_name)
        opts = getattr(hook, 'pytest_impl', None)

        assert opts is not None, f'{hook_name} carries no @pytest.hookimpl marker'
        assert opts['optionalhook'] is True, f'{hook_name} is not marked optionalhook=True'


@pytest.mark.medium
class DescribeCollectionWithXdistDisabled:
    """A session run with ``-p no:xdist`` must collect and run normally."""

    def it_collects_without_a_plugin_validation_error(self, pytester: pytest.Pytester) -> None:
        """``-p no:xdist`` deactivates xdist; the gremlins plugin must survive it."""
        pytester.makepyfile(
            test_target=('import pytest\n\n\n@pytest.mark.small\ndef test_trivial():\n    assert True\n'),
        )
        pytester.makeconftest(
            """
import pytest

def pytest_configure(config):
    config.addinivalue_line('markers', 'small: fast unit tests')
"""
        )

        result = pytester.runpytest_subprocess('-p', 'pytest_gremlins', '-p', 'no:xdist')

        output = result.stdout.str() + result.stderr.str()
        assert 'PluginValidationError' not in output, output
        assert 'INTERNALERROR' not in output, output
        result.assert_outcomes(passed=1)
