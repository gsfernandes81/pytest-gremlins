"""The transformer embeds the cargo-mutants set: every gremlin compiles and behaves as described."""

from __future__ import annotations

import ast
from typing import Any

import pytest

from pytest_gremlins.instrumentation.transformer import transform_source

_SOURCE = '''
from typing import Iterator


def score(x: int | None = 1 + 2) -> int:
    """Doc."""
    total = x * 2
    total += 1
    match total:
        case -1:
            return 0
        case 7 if x > 0:
            return {**{'a': 1}, 'b': 2}['b']
        case _:
            return -total


def flags() -> Iterator[bool]:
    yield True
'''


def _run(source: str, active: str | None, call: str) -> Any:
    _, tree = transform_source(source, 'm.py')
    namespace: dict[str, Any] = {'__gremlin_active__': active}
    exec(compile(ast.fix_missing_locations(tree), 'm.py', 'exec'), namespace)  # noqa: S102
    try:
        return eval(call, namespace)  # noqa: S307
    except Exception as error:
        return type(error).__name__


@pytest.mark.small
class DescribeCargoMutantsTransform:
    """The default operators are cargo-mutants' and each gremlin runs."""

    def it_makes_the_expected_gremlins(self) -> None:
        gremlins, _ = transform_source(_SOURCE, 'm.py')

        assert [(g.line_number, g.operator_name, g.description) for g in gremlins] == [
            (5, 'binary_operator', '+ to -'),
            (5, 'binary_operator', '+ to *'),
            (7, 'binary_operator', '* to +'),
            (7, 'binary_operator', '* to /'),
            (8, 'binary_operator', '+= to -='),
            (8, 'binary_operator', '+= to *='),
            (12, 'binary_operator', '> to =='),
            (12, 'binary_operator', '> to <'),
            (12, 'binary_operator', '> to >='),
            (13, 'struct_field', "delete key 'b' over **{'a': 1}"),
            (15, 'unary_operator', 'delete -'),
            (10, 'match_arm', 'delete match arm -1'),
            (12, 'match_arm_guard', 'replace match guard x > 0 with True'),
            (12, 'match_arm_guard', 'replace match guard x > 0 with False'),
            (7, 'fn_value', 'replace score -> int with 0'),
            (7, 'fn_value', 'replace score -> int with 1'),
            (7, 'fn_value', 'replace score -> int with -1'),
            (19, 'fn_value', 'replace flags -> Iterator[bool] with iter([])'),
            (19, 'fn_value', 'replace flags -> Iterator[bool] with iter([False])'),
        ]

    def it_runs_every_gremlin(self) -> None:
        gremlins, _ = transform_source(_SOURCE, 'm.py')

        results = {g.description: _run(_SOURCE, g.gremlin_id, '(score(3), list(flags()))') for g in gremlins}

        assert _run(_SOURCE, None, '(score(3), list(flags()))') == (2, [True])
        assert results == {
            'replace score -> int with 0': (0, [True]),
            'replace score -> int with 1': (1, [True]),
            'replace score -> int with -1': (-1, [True]),
            '+ to -': (2, [True]),
            '+ to *': (2, [True]),
            '* to +': (-6, [True]),
            '* to /': (-2.5, [True]),
            '+= to -=': (-5, [True]),
            '+= to *=': (-6, [True]),
            '> to ==': (-7, [True]),
            '> to <': (-7, [True]),
            '> to >=': (2, [True]),
            "delete key 'b' over **{'a': 1}": 'KeyError',
            'delete -': (2, [True]),
            'delete match arm -1': (2, [True]),
            'replace match guard x > 0 with True': (2, [True]),
            'replace match guard x > 0 with False': (-7, [True]),
            'replace flags -> Iterator[bool] with iter([])': (2, []),
            'replace flags -> Iterator[bool] with iter([False])': (2, [False]),
        }

    def it_keeps_the_docstring_under_every_gremlin(self) -> None:
        gremlins, _ = transform_source(_SOURCE, 'm.py')

        assert {_run(_SOURCE, g.gremlin_id, 'score.__doc__') for g in gremlins} == {'Doc.'}

    def it_leaves_annotations_alone(self) -> None:
        gremlins, _ = transform_source(
            'def f(x: int | None) -> int | None:\n    y: int | str = x\n    return y', 'm.py'
        )

        assert {g.operator_name for g in gremlins} == {'fn_value'}

    def it_instruments_a_function_nested_deeper_than_copying_recurses(self) -> None:
        branches = ''.join(f'    elif x == {i}:\n        return {i}\n' for i in range(1, 400))
        source = f'def lookup(x):\n    if x == 0:\n        return 0\n{branches}    return x + 1\n'

        gremlins, _ = transform_source(source, 'm.py')

        assert sum(g.operator_name == 'fn_value' for g in gremlins) == 1
        assert _run(source, None, 'lookup(1000)') == 1001
