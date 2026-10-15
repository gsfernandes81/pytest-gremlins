"""Match statement mutation operators, after cargo-mutants' ``MatchArm`` and ``MatchArmGuard`` genres."""

from __future__ import annotations

import ast
import copy


def _is_wildcard(case: ast.match_case) -> bool:
    """Return True for ``case _:``, the only pattern cargo-mutants treats as a catch-all.

    A capture pattern such as ``case other:`` is irrefutable too, but it is Rust's
    ``other =>``, which cargo-mutants does not count.
    """
    return isinstance(case.pattern, ast.MatchAs) and case.pattern.pattern is None and case.pattern.name is None


def _differing_case(original: ast.AST, mutated: ast.AST) -> ast.match_case:
    """Return the case of *original* that *mutated* deleted or whose guard it replaced."""
    assert isinstance(original, ast.Match)  # noqa: S101 - only Match nodes are mutated
    assert isinstance(mutated, ast.Match)  # noqa: S101
    for index, case in enumerate(original.cases):
        if index >= len(mutated.cases) or ast.dump(case) != ast.dump(mutated.cases[index]):
            return case
    raise ValueError('mutated match is identical to the original')


class MatchArmOperator:
    """Delete one arm of a ``match`` that has a ``case _:`` arm.

    The ``case _:`` arm itself and arms with a guard are never deleted: deleting the
    first would leave values unmatched, and deleting a guarded arm is what replacing
    its guard with ``False`` already does.
    """

    @property
    def name(self) -> str:
        """Return unique identifier for this operator."""
        return 'match_arm'

    @property
    def description(self) -> str:
        """Return human-readable description for reports."""
        return 'Delete match arms when a case _ arm is present'

    def can_mutate(self, node: ast.AST) -> bool:
        r"""Return True if this operator can mutate the given AST node.

        Examples:
            >>> MatchArmOperator().can_mutate(ast.parse('match x:\n case 1: pass\n case _: pass').body[0])
            True
            >>> MatchArmOperator().can_mutate(ast.parse('match x:\n case 1: pass\n case y: pass').body[0])
            False
        """
        return bool(self.mutate(node))

    def mutate(self, node: ast.AST) -> list[ast.AST]:
        r"""Return one copy of the match per deletable arm, with that arm removed.

        Examples:
            >>> tree = ast.parse('match x:\n case 1: a()\n case _: b()').body[0]
            >>> print(ast.unparse(MatchArmOperator().mutate(tree)[0]))
            match x:
                case _:
                    b()
        """
        if not isinstance(node, ast.Match) or not any(_is_wildcard(case) for case in node.cases):
            return []
        mutations: list[ast.AST] = []
        for index, case in enumerate(node.cases):
            if _is_wildcard(case) or case.guard is not None:
                continue
            mutated = copy.deepcopy(node)
            del mutated.cases[index]
            mutations.append(mutated)
        return mutations

    def describe(self, original: ast.AST, mutated: ast.AST) -> str:
        """Return ``'delete match arm <pattern>'`` for one gremlin."""
        return f'delete match arm {ast.unparse(_differing_case(original, mutated).pattern)}'

    def locate(self, original: ast.AST, mutated: ast.AST) -> int:
        """Return the line of the deleted arm's pattern."""
        return _differing_case(original, mutated).pattern.lineno


class MatchArmGuardOperator:
    """Replace each match arm guard with ``True`` and with ``False``."""

    @property
    def name(self) -> str:
        """Return unique identifier for this operator."""
        return 'match_arm_guard'

    @property
    def description(self) -> str:
        """Return human-readable description for reports."""
        return 'Replace match arm guards with True and False'

    def can_mutate(self, node: ast.AST) -> bool:
        r"""Return True if this operator can mutate the given AST node.

        Examples:
            >>> MatchArmGuardOperator().can_mutate(ast.parse('match x:\n case 1 if y: pass').body[0])
            True
        """
        return isinstance(node, ast.Match) and any(case.guard is not None for case in node.cases)

    def mutate(self, node: ast.AST) -> list[ast.AST]:
        r"""Return two copies of the match per guarded arm, its guard ``True`` and ``False``.

        Examples:
            >>> tree = ast.parse('match x:\n case 1 if y: pass').body[0]
            >>> [ast.unparse(m.cases[0].guard) for m in MatchArmGuardOperator().mutate(tree)]
            ['True', 'False']
        """
        if not isinstance(node, ast.Match):
            return []
        mutations: list[ast.AST] = []
        for index, case in enumerate(node.cases):
            if case.guard is None:
                continue
            for value in (True, False):
                mutated = copy.deepcopy(node)
                mutated.cases[index].guard = ast.copy_location(ast.Constant(value=value), case.guard)
                mutations.append(mutated)
        return mutations

    def describe(self, original: ast.AST, mutated: ast.AST) -> str:
        """Return ``'replace match guard <guard> with <value>'`` for one gremlin."""
        case = _differing_case(original, mutated)
        assert isinstance(mutated, ast.Match)  # noqa: S101
        assert isinstance(original, ast.Match)  # noqa: S101
        replaced = mutated.cases[original.cases.index(case)].guard
        assert case.guard is not None  # noqa: S101 - a guard was replaced
        assert replaced is not None  # noqa: S101
        return f'replace match guard {ast.unparse(case.guard)} with {ast.unparse(replaced)}'

    def locate(self, original: ast.AST, mutated: ast.AST) -> int:
        """Return the line of the replaced guard."""
        guard = _differing_case(original, mutated).guard
        assert guard is not None  # noqa: S101 - a guard was replaced
        return guard.lineno
