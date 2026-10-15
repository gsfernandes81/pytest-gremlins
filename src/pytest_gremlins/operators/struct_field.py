"""Dict field mutation operator, after cargo-mutants' ``StructField`` genre.

cargo-mutants deletes each field of a struct literal that has a base,
``Cat { name: "Felix", ..base }``, so the field falls back to the base's value.
Python's counterpart is a dict display over a ``**`` spread, ``{**base, 'name': 'Felix'}``:
each key written after a spread is deleted in turn. Keys written before every
spread are not, because the spread overrides them.
"""

from __future__ import annotations

import ast
import copy


def _deletable(node: ast.Dict) -> list[int]:
    """Return the indexes of the entries written after the first ``**`` spread.

    Examples:
        >>> _deletable(ast.parse("{'a': 1, **b, 'c': 2, **d, 'e': 3}", mode='eval').body)
        [2, 4]
    """
    first_spread = next((index for index, key in enumerate(node.keys) if key is None), None)
    if first_spread is None:
        return []
    return [index for index in range(first_spread + 1, len(node.keys)) if node.keys[index] is not None]


class StructFieldOperator:
    """Delete one keyed entry from a dict display that spreads a base dict."""

    @property
    def name(self) -> str:
        """Return unique identifier for this operator."""
        return 'struct_field'

    @property
    def description(self) -> str:
        """Return human-readable description for reports."""
        return 'Delete keys written over a ** spread in dict displays'

    def can_mutate(self, node: ast.AST) -> bool:
        """Return True if this operator can mutate the given AST node.

        Examples:
            >>> StructFieldOperator().can_mutate(ast.parse("{**base, 'a': 1}", mode='eval').body)
            True
            >>> StructFieldOperator().can_mutate(ast.parse("{'a': 1}", mode='eval').body)
            False
        """
        return isinstance(node, ast.Dict) and bool(_deletable(node))

    def mutate(self, node: ast.AST) -> list[ast.AST]:
        """Return one copy of the dict per deletable key, with that entry removed.

        Examples:
            >>> node = ast.parse("{**b, 'x': 1, 'y': 2}", mode='eval').body
            >>> [ast.unparse(m) for m in StructFieldOperator().mutate(node)]
            ["{**b, 'y': 2}", "{**b, 'x': 1}"]
        """
        if not isinstance(node, ast.Dict):
            return []
        mutations: list[ast.AST] = []
        for index in _deletable(node):
            mutated = copy.deepcopy(node)
            del mutated.keys[index]
            del mutated.values[index]
            mutations.append(mutated)
        return mutations

    def describe(self, original: ast.AST, mutated: ast.AST) -> str:
        """Return ``'delete key <key> over **<base>'`` for one gremlin."""
        key = self._deleted_key(original, mutated)
        assert isinstance(original, ast.Dict)  # noqa: S101 - only Dict nodes are mutated
        base = original.values[original.keys.index(None)]
        return f'delete key {ast.unparse(key)} over **{ast.unparse(base)}'

    def locate(self, original: ast.AST, mutated: ast.AST) -> int:
        """Return the line of the deleted key."""
        return self._deleted_key(original, mutated).lineno

    @staticmethod
    def _deleted_key(original: ast.AST, mutated: ast.AST) -> ast.expr:
        assert isinstance(original, ast.Dict)  # noqa: S101 - only Dict nodes are mutated
        assert isinstance(mutated, ast.Dict)  # noqa: S101
        kept = [ast.dump(key) if key is not None else None for key in mutated.keys]
        for index, key in enumerate(original.keys):
            if index >= len(kept) or (key is not None and ast.dump(key) != kept[index]):
                assert key is not None  # noqa: S101 - spreads are never deleted
                return key
        raise ValueError('mutated dict is identical to the original')
