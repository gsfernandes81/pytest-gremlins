"""Lightweight test runner command builder.

Shared utility for constructing lightweight runner commands that skip
full pytest startup overhead. Used by pool.py, persistent_pool.py,
and plugin.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    import subprocess

LIGHTWEIGHT_CANNOT_RUN_EXIT_CODE = 77
"""Exit code the lightweight runner uses for "I cannot run this test at all".

The runner calls test functions directly, so fixture-taking, parametrized and
async tests are out of its reach.  Those must not be reported as ``1``, which
every caller reads as "the tests failed, so the mutation was zapped"; they get
this code instead and are re-run through the real pytest command.

pytest itself only ever exits 0-5, so this cannot collide with a verdict.  The
generated runner script declares the same value; ``tests/parallel/
test_lightweight_fallback.py`` keeps the two in step.
"""


def build_lightweight_command(
    test_command: list[str],
    env_vars: dict[str, str],
) -> list[str] | None:
    """Build a lightweight runner command if the runner script exists.

    Extracts test node IDs from the full test command and builds a
    command using the lightweight runner (no pytest overhead).

    Args:
        test_command: Original test command (e.g. [python, bootstrap.py, -x, ...]).
        env_vars: Environment variables that may contain sources file path.

    Returns:
        Lightweight command list, or None if the runner is not available.
    """
    sources_file = env_vars.get('PYTEST_GREMLINS_SOURCES_FILE', '')
    if not sources_file:
        return None

    runner_path = Path(sources_file).parent / 'gremlin_lightweight_runner.py'
    if not runner_path.exists():
        return None

    # Extract test node IDs from test_command (args containing '::')
    test_ids = [arg for arg in test_command[2:] if '::' in arg]
    if not test_ids:
        return None

    return [test_command[0], str(runner_path), *test_ids]


def run_lightweight_with_fallback(
    test_command: list[str],
    env_vars: dict[str, str],
    run_command: Callable[[list[str]], subprocess.CompletedProcess[bytes]],
) -> subprocess.CompletedProcess[bytes]:
    """Run the tests through the lightweight runner, falling back to pytest.

    The fast path is tried first.  When the runner reports that it cannot run
    the selected tests (:data:`LIGHTWEIGHT_CANNOT_RUN_EXIT_CODE`) the original
    ``test_command`` — a full pytest invocation with the same node IDs — is run
    instead, so the gremlin is scored on a real test run rather than on the
    runner's inability to perform one.  Every other exit code is returned
    untouched and costs no second run.

    ``run_command`` is supplied by the caller so that each execution path keeps
    its own subprocess settings (cwd, env, timeout) and its own error handling.

    Args:
        test_command: The full pytest command, including test node IDs.
        env_vars: Environment variables that may contain the sources file path.
        run_command: Runs one command and returns its completed process.

    Returns:
        The completed process whose exit code should be scored.
    """
    lightweight_command = build_lightweight_command(test_command, env_vars)
    if lightweight_command is None:
        return run_command(test_command)

    outcome = run_command(lightweight_command)
    if outcome.returncode != LIGHTWEIGHT_CANNOT_RUN_EXIT_CODE:
        return outcome

    return run_command(test_command)
