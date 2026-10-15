"""Tests for the cargo-mutants operator set: one operator per cargo-mutants mutation genre."""

from __future__ import annotations

import ast

import pytest

from pytest_gremlins.operators import (
    DEFAULT_OPERATORS,
    BinaryOperator,
    FnValueOperator,
    GremlinOperator,
    MatchArmGuardOperator,
    MatchArmOperator,
    StructFieldOperator,
    UnaryOperator,
)
from pytest_gremlins.operators.fn_value import type_values

ALL = [
    FnValueOperator(),
    BinaryOperator(),
    UnaryOperator(),
    MatchArmOperator(),
    MatchArmGuardOperator(),
    StructFieldOperator(),
]


def _expr(source: str) -> ast.expr:
    return ast.parse(source, mode='eval').body


def _stmt(source: str) -> ast.stmt:
    return ast.parse(source).body[0]


def _unparsed(operator: GremlinOperator, node: ast.AST) -> list[str]:
    return [ast.unparse(mutated) for mutated in operator.mutate(node)]


@pytest.mark.small
class DescribeTheCargoMutantsSet:
    """Every operator is a GremlinOperator and the default set names each genre once."""

    @pytest.mark.parametrize('operator', ALL, ids=lambda op: op.name)
    def it_implements_the_protocol(self, operator: GremlinOperator) -> None:
        assert isinstance(operator, GremlinOperator)

    def it_defaults_to_one_operator_per_genre(self) -> None:
        assert tuple(op.name for op in ALL) == DEFAULT_OPERATORS


@pytest.mark.small
class DescribeBinaryOperator:
    """Replacements follow cargo-mutants' table in src/visit.rs."""

    @pytest.mark.parametrize(
        ('source', 'expected'),
        [
            ('a == b', ['a != b']),
            ('a != b', ['a == b']),
            ('a < b', ['a == b', 'a > b', 'a <= b']),
            ('a > b', ['a == b', 'a < b', 'a >= b']),
            ('a <= b', ['a > b']),
            ('a >= b', ['a < b']),
            ('a + b', ['a - b', 'a * b']),
            ('a - b', ['a + b', 'a / b']),
            ('a * b', ['a + b', 'a / b']),
            ('a / b', ['a % b', 'a * b']),
            ('a // b', ['a % b', 'a * b']),
            ('a % b', ['a / b', 'a + b']),
            ('a << b', ['a >> b']),
            ('a >> b', ['a << b']),
            ('a & b', ['a | b', 'a ^ b']),
            ('a | b', ['a & b', 'a ^ b']),
            ('a ^ b', ['a & b', 'a | b']),
            ('a and b', ['a or b']),
            ('a or b', ['a and b']),
        ],
    )
    def it_replaces_an_expression_operator(self, source: str, expected: list[str]) -> None:
        assert _unparsed(BinaryOperator(), _expr(source)) == expected

    @pytest.mark.parametrize(
        ('source', 'expected'),
        [
            ('a += b', ['a -= b', 'a *= b']),
            ('a -= b', ['a += b', 'a /= b']),
            ('a *= b', ['a += b', 'a /= b']),
            ('a /= b', ['a %= b', 'a *= b']),
            ('a %= b', ['a /= b', 'a += b']),
            ('a <<= b', ['a >>= b']),
            ('a >>= b', ['a <<= b']),
            ('a &= b', ['a |= b']),
            ('a |= b', ['a &= b']),
            ('a ^= b', ['a |= b', 'a &= b']),
        ],
    )
    def it_replaces_an_augmented_assignment(self, source: str, expected: list[str]) -> None:
        assert _unparsed(BinaryOperator(), _stmt(source)) == expected

    @pytest.mark.parametrize('source', ['a ** b', 'a @ b', 'a is b', 'a in b', 'a is not b', 'a not in b', 'a **= b'])
    def it_leaves_operators_rust_lacks(self, source: str) -> None:
        node = _stmt(source) if '=' in source and '==' not in source else _expr(source)
        assert not BinaryOperator().can_mutate(node)

    @pytest.mark.parametrize('source', ["'%s' % x", "'-' * n", "f'{x}' + y", 'x + [1]', 'x * (1,)', "b'a' + x"])
    def it_leaves_sequence_arithmetic(self, source: str) -> None:
        assert not BinaryOperator().can_mutate(_expr(source))

    def it_leaves_augmented_sequence_arithmetic(self) -> None:
        assert not BinaryOperator().can_mutate(_stmt("s += 'x'"))

    def it_still_mutates_bitwise_operators_on_literal_collections(self) -> None:
        assert _unparsed(BinaryOperator(), _expr('{1} | x')) == ['{1} & x', '{1} ^ x']

    def it_flips_one_operator_of_a_chain_at_a_time(self) -> None:
        assert _unparsed(BinaryOperator(), _expr('a or b or c or d')) == [
            'a and b or c or d',
            '(a or b) and c or d',
            '(a or b or c) and d',
        ]

    def it_mutates_each_comparison_of_a_chain(self) -> None:
        assert _unparsed(BinaryOperator(), _expr('a < b == c')) == [
            'a == b == c',
            'a > b == c',
            'a <= b == c',
            'a < b != c',
        ]

    @pytest.mark.parametrize(
        ('source', 'expected'),
        [('a < b', '< to <='), ('a and b', 'and to or'), ('a |= b', '|= to &=')],
    )
    def it_describes_the_replacement(self, source: str, expected: str) -> None:
        node = _stmt(source)
        node = node.value if isinstance(node, ast.Expr) else node
        operator = BinaryOperator()
        assert operator.describe(node, operator.mutate(node)[-1]) == expected


@pytest.mark.small
class DescribeUnaryOperator:
    """Unary operators are deleted."""

    @pytest.mark.parametrize(('source', 'expected'), [('not x', 'x'), ('-x', 'x'), ('~x', 'x'), ('-1', '1')])
    def it_deletes_the_operator(self, source: str, expected: str) -> None:
        assert _unparsed(UnaryOperator(), _expr(source)) == [expected]

    def it_leaves_unary_plus(self) -> None:
        assert not UnaryOperator().can_mutate(_expr('+x'))


@pytest.mark.small
class DescribeMatchArmOperator:
    """Arms are deleted only when a ``case _`` arm catches what they matched."""

    def it_deletes_each_unguarded_arm_but_the_wildcard(self) -> None:
        node = _stmt('match x:\n case 1: a()\n case 2 if y: b()\n case 3: c()\n case _: d()')
        operator = MatchArmOperator()
        mutated = operator.mutate(node)
        assert [operator.describe(node, m) for m in mutated] == ['delete match arm 1', 'delete match arm 3']
        assert [operator.locate(node, m) for m in mutated] == [2, 4]

    @pytest.mark.parametrize('catch_all', ['case other: pass', 'case 9: pass'])
    def it_leaves_a_match_without_a_wildcard(self, catch_all: str) -> None:
        assert not MatchArmOperator().can_mutate(_stmt(f'match x:\n case 1: pass\n {catch_all}'))


@pytest.mark.small
class DescribeMatchArmGuardOperator:
    """Guards are replaced with True and with False."""

    def it_replaces_each_guard_both_ways(self) -> None:
        node = _stmt('match x:\n case 1 if y:\n  pass\n case 2:\n  pass\n case 3 if z:\n  pass')
        operator = MatchArmGuardOperator()
        assert [operator.describe(node, m) for m in operator.mutate(node)] == [
            'replace match guard y with True',
            'replace match guard y with False',
            'replace match guard z with True',
            'replace match guard z with False',
        ]
        assert [operator.locate(node, m) for m in operator.mutate(node)] == [2, 2, 6, 6]


@pytest.mark.small
class DescribeStructFieldOperator:
    """Keys written over a ``**`` spread are deleted, as struct fields over ``..base`` are."""

    def it_deletes_keys_after_the_spread_only(self) -> None:
        node = _expr("{'a': 0, **base, 'b': 1,\n 'c': 2}")
        operator = StructFieldOperator()
        mutated = operator.mutate(node)
        assert [ast.unparse(m) for m in mutated] == ["{'a': 0, **base, 'c': 2}", "{'a': 0, **base, 'b': 1}"]
        assert [operator.describe(node, m) for m in mutated] == [
            "delete key 'b' over **base",
            "delete key 'c' over **base",
        ]
        assert [operator.locate(node, m) for m in mutated] == [1, 2]

    def it_leaves_a_dict_without_a_spread(self) -> None:
        assert not StructFieldOperator().can_mutate(_expr("{'a': 1}"))


@pytest.mark.small
class DescribeFnValueTypeValues:
    """Return annotations map to cargo-mutants' replacement values."""

    @pytest.mark.parametrize(
        ('annotation', 'expected'),
        [
            (None, ['None']),
            ('None', ['None']),
            ('bool', ['True', 'False']),
            ('int', ['0', '1', '-1']),
            ('float', ['0.0', '1.0', '-1.0']),
            ('str', ["''", "'xyzzy'"]),
            ('bytes', ["b''", "b'\\x00'", "b'\\x01'"]),
            ('Optional[bool]', ['None', 'True', 'False']),
            ('typing.Union[None, bool]', ['None', 'True', 'False']),
            ("'bool | None'", ['None', 'True', 'False']),
            ('list[bool]', ['[]', '[True]', '[False]']),
            ('Sequence[str]', ['[]', "['']", "['xyzzy']"]),
            ('set[bool]', ['set()', '{True}', '{False}']),
            ('frozenset[bool]', ['frozenset()', 'frozenset({True})', 'frozenset({False})']),
            ('dict[str, bool]', ['{}', "{'': True}", "{'': False}", "{'xyzzy': True}", "{'xyzzy': False}"]),
            ('tuple[bool, ...]', ['()', '(True,)', '(False,)']),
            ('tuple[bool]', ['(True,)', '(False,)']),
            ('tuple[()]', ['()']),
            ('Iterator[bool]', ['iter([])', 'iter([True])', 'iter([False])']),
            ('list', ['[]', '[None]']),
            ('MyClass', ['None']),
            ('int | str', ['None']),
            ('NoReturn', []),
        ],
    )
    def it_maps_the_annotation(self, annotation: str | None, expected: list[str]) -> None:
        assert type_values(None if annotation is None else _expr(annotation)) == expected


@pytest.mark.small
class DescribeFnValueOperator:
    """Function bodies are replaced with a value of the return type."""

    def it_keeps_the_docstring_and_replaces_the_rest(self) -> None:
        node = _stmt('def f() -> bool:\n    """Doc."""\n    x = 1\n    return x > 0')
        assert [ast.unparse(m.body) for m in FnValueOperator().mutate(node)] == [
            "'Doc.'\nreturn True",
            "'Doc.'\nreturn False",
        ]

    def it_sits_on_the_first_statement_that_runs(self) -> None:
        node = _stmt('def f():\n    """Doc."""\n    global g\n    g = 1')
        operator = FnValueOperator()
        assert operator.locate(node, operator.mutate(node)[0]) == 4

    @pytest.mark.parametrize(
        'source',
        [
            'def f() -> int:\n    """Doc."""',
            'def f():\n    pass',
            'def f():\n    ...',
            'def __init__(self):\n    self.x = 1',
            'def __new__(cls):\n    return object()',
            '@abstractmethod\ndef f(self) -> int:\n    raise NotImplementedError',
            '@typing.overload\ndef f(x: int) -> int:\n    return x',
            'def f() -> NoReturn:\n    raise SystemExit',
            'def f():\n    return None',
        ],
    )
    def it_leaves_functions_without_a_useful_replacement(self, source: str) -> None:
        assert not FnValueOperator().can_mutate(_stmt(source))

    def it_skips_a_replacement_identical_to_the_body(self) -> None:
        node = _stmt('def f() -> int:\n    return 1')
        operator = FnValueOperator()
        assert [operator.describe(node, m) for m in operator.mutate(node)] == [
            'replace f -> int with 0',
            'replace f -> int with -1',
        ]

    def it_empties_a_generator_or_makes_it_yield_one_value(self) -> None:
        node = _stmt('def f() -> Iterator[bool]:\n    yield True\n    yield False')
        operator = FnValueOperator()
        assert [operator.describe(node, m) for m in operator.mutate(node)] == [
            'replace f -> Iterator[bool] with iter([])',
            'replace f -> Iterator[bool] with iter([True])',
            'replace f -> Iterator[bool] with iter([False])',
        ]

    def it_treats_a_function_with_a_nested_generator_as_plain(self) -> None:
        node = _stmt('def f():\n    def g():\n        yield 1\n    return g')
        assert [ast.unparse(m.body) for m in FnValueOperator().mutate(node)] == ['return None']
