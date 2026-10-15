"""Binary operator mutation operator, after cargo-mutants' ``BinaryOperator`` genre.

The replacement table is cargo-mutants' own, with Rust's ``&&``/``||`` read as
Python's ``and``/``or``, Rust's integer ``/`` read as both ``/`` and ``//``, and
Rust's ``+=`` family read as Python's augmented assignments. Python operators
with no Rust counterpart (``**``, ``@``, ``is``, ``in``) are not mutated.
"""

from __future__ import annotations

import ast
import copy
from typing import ClassVar

_SYMBOLS: dict[type[ast.AST], str] = {
    ast.Add: '+',
    ast.Sub: '-',
    ast.Mult: '*',
    ast.Div: '/',
    ast.FloorDiv: '//',
    ast.Mod: '%',
    ast.LShift: '<<',
    ast.RShift: '>>',
    ast.BitAnd: '&',
    ast.BitOr: '|',
    ast.BitXor: '^',
    ast.Eq: '==',
    ast.NotEq: '!=',
    ast.Lt: '<',
    ast.LtE: '<=',
    ast.Gt: '>',
    ast.GtE: '>=',
    ast.And: 'and',
    ast.Or: 'or',
}

_ARITHMETIC: frozenset[type[ast.operator]] = frozenset({ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod})


def _is_sequence_literal(node: ast.expr) -> bool:
    """Return True for a str, bytes, f-string, list or tuple literal.

    Examples:
        >>> _is_sequence_literal(ast.parse("'x'", mode='eval').body)
        True
        >>> _is_sequence_literal(ast.parse('1', mode='eval').body)
        False
    """
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (str, bytes))
    return isinstance(node, (ast.JoinedStr, ast.List, ast.Tuple))


class BinaryOperator:
    """Replace binary, comparison, boolean and augmented-assignment operators.

    Mutations (cargo-mutants' table):
        - ``==`` -> ``!=``; ``!=`` -> ``==``
        - ``<`` -> ``==``, ``>``, ``<=``; ``>`` -> ``==``, ``<``, ``>=``
        - ``<=`` -> ``>``; ``>=`` -> ``<``
        - ``and`` -> ``or``; ``or`` -> ``and``
        - ``+`` -> ``-``, ``*``; ``-`` -> ``+``, ``/``; ``*`` -> ``+``, ``/``
        - ``/`` and ``//`` -> ``%``, ``*``; ``%`` -> ``/``, ``+``
        - ``<<`` -> ``>>``; ``>>`` -> ``<<``
        - ``&`` -> ``|``, ``^``; ``|`` -> ``&``, ``^``; ``^`` -> ``&``, ``|``
        - ``op=`` -> the replacements of ``op``, except ``&=`` -> ``|=`` and ``|=`` -> ``&=`` only

    Arithmetic with a string, bytes, f-string, list or tuple literal operand is
    concatenation, repetition or formatting, where every replacement is a
    ``TypeError``: cargo-mutants would count those unviable, so they are not made.
    """

    BINOP_MUTATIONS: ClassVar[dict[type[ast.operator], list[type[ast.operator]]]] = {
        ast.Add: [ast.Sub, ast.Mult],
        ast.Sub: [ast.Add, ast.Div],
        ast.Mult: [ast.Add, ast.Div],
        ast.Div: [ast.Mod, ast.Mult],
        ast.FloorDiv: [ast.Mod, ast.Mult],
        ast.Mod: [ast.Div, ast.Add],
        ast.LShift: [ast.RShift],
        ast.RShift: [ast.LShift],
        ast.BitAnd: [ast.BitOr, ast.BitXor],
        ast.BitOr: [ast.BitAnd, ast.BitXor],
        ast.BitXor: [ast.BitAnd, ast.BitOr],
    }

    AUGASSIGN_MUTATIONS: ClassVar[dict[type[ast.operator], list[type[ast.operator]]]] = {
        **BINOP_MUTATIONS,
        ast.BitAnd: [ast.BitOr],
        ast.BitOr: [ast.BitAnd],
        ast.BitXor: [ast.BitOr, ast.BitAnd],
    }

    COMPARE_MUTATIONS: ClassVar[dict[type[ast.cmpop], list[type[ast.cmpop]]]] = {
        ast.Eq: [ast.NotEq],
        ast.NotEq: [ast.Eq],
        ast.Lt: [ast.Eq, ast.Gt, ast.LtE],
        ast.Gt: [ast.Eq, ast.Lt, ast.GtE],
        ast.LtE: [ast.Gt],
        ast.GtE: [ast.Lt],
    }

    @property
    def name(self) -> str:
        """Return unique identifier for this operator."""
        return 'binary_operator'

    @property
    def description(self) -> str:
        """Return human-readable description for reports."""
        return 'Replace binary operators (cargo-mutants table: comparisons, and/or, arithmetic, bitwise, op=)'

    def can_mutate(self, node: ast.AST) -> bool:
        """Return True if this operator can mutate the given AST node.

        Examples:
            >>> BinaryOperator().can_mutate(ast.parse('a < b', mode='eval').body)
            True
            >>> BinaryOperator().can_mutate(ast.parse('a ** b', mode='eval').body)
            False
            >>> BinaryOperator().can_mutate(ast.parse("'%s' % b", mode='eval').body)
            False
        """
        return bool(self.mutate(node))

    def mutate(self, node: ast.AST) -> list[ast.AST]:
        """Return all mutated variants of this node, one per gremlin.

        Examples:
            >>> [ast.unparse(m) for m in BinaryOperator().mutate(ast.parse('a < b', mode='eval').body)]
            ['a == b', 'a > b', 'a <= b']
            >>> [ast.unparse(m) for m in BinaryOperator().mutate(ast.parse('a and b and c', mode='eval').body)]
            ['(a or b) and c', 'a and b or c']
        """
        if isinstance(node, ast.BinOp):
            return self._mutate_binop(node)
        if isinstance(node, ast.AugAssign):
            return self._mutate_augassign(node)
        if isinstance(node, ast.Compare):
            return self._mutate_compare(node)
        if isinstance(node, ast.BoolOp):
            return self._mutate_boolop(node)
        return []

    def describe(self, original: ast.AST, mutated: ast.AST) -> str:
        """Return ``'<original op> to <mutated op>'`` for one gremlin.

        Examples:
            >>> op = BinaryOperator()
            >>> node = ast.parse('a += b').body[0]
            >>> op.describe(node, op.mutate(node)[0])
            '+= to -='
        """
        before, after = self._changed_ops(original, mutated)
        suffix = '=' if isinstance(original, ast.AugAssign) else ''
        return f'{_SYMBOLS[type(before)]}{suffix} to {_SYMBOLS[type(after)]}{suffix}'

    def _changed_ops(self, original: ast.AST, mutated: ast.AST) -> tuple[ast.AST, ast.AST]:
        if isinstance(original, ast.Compare) and isinstance(mutated, ast.Compare):
            return next((a, b) for a, b in zip(original.ops, mutated.ops, strict=True) if type(a) is not type(b))
        if isinstance(original, ast.BoolOp):
            return original.op, ast.Or() if isinstance(original.op, ast.And) else ast.And()
        if isinstance(original, (ast.BinOp, ast.AugAssign)) and isinstance(mutated, (ast.BinOp, ast.AugAssign)):
            return original.op, mutated.op
        raise TypeError(f'not a binary operator mutation: {type(original).__name__}')

    def _mutate_binop(self, node: ast.BinOp) -> list[ast.AST]:
        op_type = type(node.op)
        if op_type in _ARITHMETIC and (_is_sequence_literal(node.left) or _is_sequence_literal(node.right)):
            return []
        mutations: list[ast.AST] = []
        for replacement in self.BINOP_MUTATIONS.get(op_type, []):
            mutated = copy.deepcopy(node)
            mutated.op = replacement()
            mutations.append(mutated)
        return mutations

    def _mutate_augassign(self, node: ast.AugAssign) -> list[ast.AST]:
        op_type = type(node.op)
        if op_type in _ARITHMETIC and _is_sequence_literal(node.value):
            return []
        mutations: list[ast.AST] = []
        for replacement in self.AUGASSIGN_MUTATIONS.get(op_type, []):
            mutated = copy.deepcopy(node)
            mutated.op = replacement()
            mutations.append(mutated)
        return mutations

    def _mutate_compare(self, node: ast.Compare) -> list[ast.AST]:
        mutations: list[ast.AST] = []
        for index, op in enumerate(node.ops):
            for replacement in self.COMPARE_MUTATIONS.get(type(op), []):
                mutated = copy.deepcopy(node)
                mutated.ops[index] = replacement()
                mutations.append(mutated)
        return mutations

    def _mutate_boolop(self, node: ast.BoolOp) -> list[ast.AST]:
        """Flip one operator of the chain at a time, reading it left-associatively as Rust does.

        ``a and b and c`` is ``(a and b) and c``: flipping the first operator gives
        ``(a or b) and c`` and flipping the second gives ``(a and b) or c``.
        """
        flipped: type[ast.boolop] = ast.Or if isinstance(node.op, ast.And) else ast.And
        values = copy.deepcopy(node.values)
        mutations: list[ast.AST] = []
        for position in range(1, len(values)):
            left = values[0] if position == 1 else ast.BoolOp(op=type(node.op)(), values=values[:position])
            mutated: ast.expr = ast.BoolOp(op=flipped(), values=[left, values[position]])
            rest = values[position + 1 :]
            if rest:
                mutated = ast.BoolOp(op=type(node.op)(), values=[mutated, *rest])
            mutations.append(ast.copy_location(copy.deepcopy(mutated), node))
        return mutations
