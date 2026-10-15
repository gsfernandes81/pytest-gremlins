"""Unary operator mutation operator, after cargo-mutants' ``UnaryOperator`` genre."""

from __future__ import annotations

import ast
import copy
from typing import ClassVar


class UnaryOperator:
    """Delete a unary operator.

    Mutations (cargo-mutants deletes Rust's ``!`` and ``-``; Rust's ``!`` is
    Python's ``not`` on a bool and ``~`` on an integer):
        - ``not x`` -> ``x``
        - ``-x`` -> ``x``
        - ``~x`` -> ``x``

    A unary ``+`` is not mutated.
    """

    SYMBOLS: ClassVar[dict[type[ast.unaryop], str]] = {ast.Not: 'not ', ast.USub: '-', ast.Invert: '~'}

    @property
    def name(self) -> str:
        """Return unique identifier for this operator."""
        return 'unary_operator'

    @property
    def description(self) -> str:
        """Return human-readable description for reports."""
        return 'Delete unary operators (not, -, ~)'

    def can_mutate(self, node: ast.AST) -> bool:
        """Return True if this operator can mutate the given AST node.

        Examples:
            >>> UnaryOperator().can_mutate(ast.parse('-x', mode='eval').body)
            True
            >>> UnaryOperator().can_mutate(ast.parse('+x', mode='eval').body)
            False
        """
        return isinstance(node, ast.UnaryOp) and type(node.op) in self.SYMBOLS

    def mutate(self, node: ast.AST) -> list[ast.AST]:
        """Return the operand with the operator deleted.

        Examples:
            >>> [ast.unparse(m) for m in UnaryOperator().mutate(ast.parse('not ok', mode='eval').body)]
            ['ok']
        """
        if not self.can_mutate(node):
            return []
        assert isinstance(node, ast.UnaryOp)  # noqa: S101 - narrowed by can_mutate
        return [copy.deepcopy(node.operand)]

    def describe(self, original: ast.AST, mutated: ast.AST) -> str:  # noqa: ARG002 - one mutation per node
        """Return ``'delete <op>'`` for one gremlin.

        Examples:
            >>> node = ast.parse('~x', mode='eval').body
            >>> UnaryOperator().describe(node, node)
            'delete ~'
        """
        assert isinstance(original, ast.UnaryOp)  # noqa: S101 - only UnaryOp nodes are mutated
        return f'delete {self.SYMBOLS[type(original.op)].strip()}'
