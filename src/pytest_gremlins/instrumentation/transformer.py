"""AST transformer for mutation switching.

This module provides the AST transformations needed to embed mutations
into Python source code with environment variable-controlled switching.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import logging
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    from collections.abc import Callable

from pytest_gremlins.instrumentation.gremlin import Gremlin
from pytest_gremlins.instrumentation.pragma import parse_pardoned_lines
from pytest_gremlins.operators import (
    DEFAULT_OPERATORS,
    ArithmeticOperator,
    BinaryOperator,
    BooleanOperator,
    BoundaryOperator,
    ComparisonOperator,
    FnValueOperator,
    GremlinOperator,
    MatchArmGuardOperator,
    MatchArmOperator,
    OperatorRegistry,
    ReturnOperator,
    StructFieldOperator,
    UnaryOperator,
)
from pytest_gremlins.operators.fn_value import (
    function_shell,
    split_docstring,
)

_comparison_operator = ComparisonOperator()

COMPARISON_MUTATIONS: dict[type[ast.cmpop], list[type[ast.cmpop]]] = ComparisonOperator.MUTATIONS

OP_TO_SYMBOL: dict[type[ast.cmpop], str] = ComparisonOperator.OP_TO_SYMBOL


@runtime_checkable
class DescribingOperator(Protocol):
    """An operator that describes its own gremlins."""

    def describe(self, original: ast.AST, mutated: ast.AST) -> str:
        """Return a human-readable description of one gremlin."""
        ...


@runtime_checkable
class LocatingOperator(Protocol):
    """An operator whose gremlins sit on a line other than the mutated node's first."""

    def locate(self, original: ast.AST, mutated: ast.AST) -> int:
        """Return the source line coverage and pardons attribute one gremlin to."""
        ...


def _create_default_registry() -> OperatorRegistry:
    """Create and populate the default operator registry with every operator.

    Only the cargo-mutants set (``DEFAULT_OPERATORS``) runs unless configured otherwise.
    """
    registry = OperatorRegistry()
    registry.register(FnValueOperator)
    registry.register(BinaryOperator)
    registry.register(UnaryOperator)
    registry.register(MatchArmOperator)
    registry.register(MatchArmGuardOperator)
    registry.register(StructFieldOperator)
    registry.register(ComparisonOperator)
    registry.register(ArithmeticOperator)
    registry.register(BooleanOperator)
    registry.register(BoundaryOperator)
    registry.register(ReturnOperator)
    return registry


_default_registry = _create_default_registry()


def get_default_registry() -> OperatorRegistry:
    """Get the operator registry with every operator registered.

    Returns:
        OperatorRegistry with the cargo-mutants operators (fn_value, binary_operator,
        unary_operator, match_arm, match_arm_guard, struct_field) and the opt-in
        comparison, arithmetic, boolean, boundary, and return operators.
    """
    return _default_registry


def get_default_operators() -> list[GremlinOperator]:
    """Get the operators that run when none are configured: cargo-mutants' set.

    Returns:
        A fresh instance of each operator named in ``DEFAULT_OPERATORS``, in order.
    """
    return _default_registry.get_all(enabled=list(DEFAULT_OPERATORS))


def create_gremlins_for_compare(
    node: ast.Compare,
    file_path: str,
    id_generator: Callable[[], str],
) -> list[Gremlin]:
    """Create gremlins for a comparison node.

    This is the shared logic for creating gremlins from comparison AST nodes.
    Used by both GremlinCollector and MutationSwitchingTransformer.

    Args:
        node: The comparison AST node.
        file_path: Path to the source file (for gremlin metadata).
        id_generator: Callable that returns the next gremlin ID.

    Returns:
        List of Gremlin objects for each possible mutation.
    """
    gremlins: list[Gremlin] = []
    mutations = generate_comparison_mutations(node)
    for mutated_node in mutations:
        original_op = _comparison_operator.get_symbol(node.ops[0])
        mutated_op = _comparison_operator.get_symbol(mutated_node.ops[0])
        gremlin = Gremlin(
            gremlin_id=id_generator(),
            file_path=file_path,
            line_number=node.lineno,
            original_node=node,
            mutated_node=mutated_node,
            operator_name='comparison',
            description=f'{original_op} to {mutated_op}',
        )
        gremlins.append(gremlin)
    return gremlins


def generate_comparison_mutations(node: ast.Compare) -> list[ast.Compare]:
    """Generate mutated variants of a comparison node.

    This function delegates to the ComparisonOperator for mutation generation.

    Args:
        node: A comparison AST node (e.g., x < 10).

    Returns:
        List of mutated comparison nodes, one for each possible mutation.
    """
    mutations = _comparison_operator.mutate(node)
    return [m for m in mutations if isinstance(m, ast.Compare)]


def create_gremlins_for_node(
    node: ast.expr | ast.stmt,
    operator: GremlinOperator,
    file_path: str,
    id_generator: Callable[[], str],
) -> list[Gremlin]:
    """Create gremlins for any AST node using a specific operator.

    Args:
        node: The AST node to mutate.
        operator: The operator to use for mutation.
        file_path: Path to the source file (for gremlin metadata).
        id_generator: Callable that returns the next gremlin ID.

    Returns:
        List of Gremlin objects for each possible mutation.
    """
    if not operator.can_mutate(node):
        return []

    mutations = operator.mutate(node)
    gremlins: list[Gremlin] = []

    for mutated_node in mutations:
        if not isinstance(mutated_node, (ast.expr, ast.stmt)):
            continue  # pragma: no cover
        if isinstance(operator, DescribingOperator):
            description = operator.describe(node, mutated_node)
        else:
            description = _get_mutation_description(node, mutated_node, operator)
        if isinstance(operator, LocatingOperator):
            line_number = operator.locate(node, mutated_node)
        else:
            line_number = getattr(node, 'lineno', 0)
        gremlin = Gremlin(
            gremlin_id=id_generator(),
            file_path=file_path,
            line_number=line_number,
            original_node=node,
            mutated_node=mutated_node,
            operator_name=operator.name,
            description=description,
        )
        gremlins.append(gremlin)

    return gremlins


def _get_comparison_description(original: ast.Compare, mutated: ast.Compare) -> str:
    """Get description for comparison mutation."""
    original_op = _comparison_operator.get_symbol(original.ops[0])
    mutated_op = _comparison_operator.get_symbol(mutated.ops[0])
    return f'{original_op} to {mutated_op}'


def _get_arithmetic_description(original: ast.BinOp, mutated: ast.BinOp) -> str:
    """Get description for arithmetic mutation."""
    arithmetic_op = ArithmeticOperator()
    original_sym = arithmetic_op.get_symbol(original.op)
    mutated_sym = arithmetic_op.get_symbol(mutated.op)
    return f'{original_sym} to {mutated_sym}'


def _get_boolean_description(original: ast.expr | ast.stmt, mutated: ast.expr | ast.stmt) -> str | None:
    """Get description for boolean mutation."""
    if isinstance(original, ast.BoolOp) and isinstance(mutated, ast.BoolOp):
        orig = 'and' if isinstance(original.op, ast.And) else 'or'
        mut = 'and' if isinstance(mutated.op, ast.And) else 'or'
        return f'{orig} to {mut}'
    if isinstance(original, ast.UnaryOp) and isinstance(original.op, ast.Not):
        return 'not x to x'
    if isinstance(original, ast.Constant) and isinstance(mutated, ast.Constant):
        return f'{original.value!r} to {mutated.value!r}'
    return None  # pragma: no cover


def _get_return_description(original: ast.expr | ast.stmt, mutated: ast.expr | ast.stmt) -> str | None:
    """Get description for return mutation."""
    if isinstance(mutated, ast.Return) and mutated.value is None:
        return 'return value to None'
    if (
        isinstance(original, ast.Return)
        and isinstance(mutated, ast.Return)
        and isinstance(original.value, ast.Constant)
        and isinstance(mutated.value, ast.Constant)
    ):
        return f'return {original.value.value!r} to {mutated.value.value!r}'
    return None  # pragma: no cover


def _get_mutation_description(
    original: ast.expr | ast.stmt,
    mutated: ast.expr | ast.stmt,
    operator: GremlinOperator,
) -> str:
    """Generate a human-readable description of a mutation.

    Args:
        original: The original AST node.
        mutated: The mutated AST node.
        operator: The operator that created the mutation.

    Returns:
        A description string for the mutation.
    """
    if operator.name == 'comparison' and isinstance(original, ast.Compare) and isinstance(mutated, ast.Compare):
        return _get_comparison_description(original, mutated)

    if operator.name == 'arithmetic' and isinstance(original, ast.BinOp) and isinstance(mutated, ast.BinOp):
        return _get_arithmetic_description(original, mutated)

    if operator.name == 'boolean':
        desc = _get_boolean_description(original, mutated)
        if desc:  # pragma: no branch
            return desc

    if operator.name == 'boundary' and isinstance(original, ast.Compare) and isinstance(mutated, ast.Compare):
        return 'boundary shift +/-1'

    if operator.name == 'return':
        desc = _get_return_description(original, mutated)
        if desc:
            return desc

    return f'{operator.name} mutation'  # pragma: no cover


class GremlinCollector(ast.NodeVisitor):
    """Collects gremlins from comparison nodes in an AST."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.gremlins: list[Gremlin] = []
        self._gremlin_counter = 0
        # Create a short unique prefix from file path to avoid ID collisions
        # when processing multiple files
        self._file_prefix = self._make_file_prefix(file_path)

    @staticmethod
    def _make_file_prefix(file_path: str) -> str:
        """Create a short unique prefix from file path.

        Uses the file stem (name without extension) combined with a short hash
        to ensure uniqueness even for files with the same name in different dirs.
        """
        stem = Path(file_path).stem
        # Use first 4 chars of MD5 hash for uniqueness
        path_hash = hashlib.md5(file_path.encode(), usedforsecurity=False).hexdigest()[:4]
        return f'{stem}_{path_hash}'

    def _next_gremlin_id(self) -> str:
        self._gremlin_counter += 1
        return f'{self._file_prefix}_g{self._gremlin_counter:03d}'

    def visit_Compare(self, node: ast.Compare) -> None:
        """Collect gremlins for comparison nodes."""
        gremlins = create_gremlins_for_compare(node, self.file_path, self._next_gremlin_id)
        self.gremlins.extend(gremlins)
        self.generic_visit(node)


def collect_gremlins(source: str, file_path: str) -> tuple[list[Gremlin], ast.Module]:
    """Collect gremlins from source code without modifying it.

    This function parses the source and identifies all potential mutation
    points, returning the gremlins found and the original (unmodified) AST.

    Note: This does NOT instrument the code. For instrumentation with
    mutation switching embedded, use transform_source() instead.

    Args:
        source: The Python source code to analyze.
        file_path: The path to the source file (for gremlin metadata).

    Returns:
        Tuple of (list of gremlins, original unmodified AST).
    """
    tree = ast.parse(source)
    collector = GremlinCollector(file_path)
    collector.visit(tree)
    return collector.gremlins, tree


def build_switching_expression(original: ast.expr, gremlins: list[Gremlin]) -> ast.IfExp:
    """Build an AST expression that switches between original and mutated code.

    Creates a nested IfExp (ternary expression) that checks __gremlin_active__
    and returns the appropriate expression based on which gremlin is active.

    The generated code is equivalent to:
        mutated1 if __gremlin_active__ == 'g001' else (
            mutated2 if __gremlin_active__ == 'g002' else (
                original
            )
        )

    Args:
        original: The original AST expression node.
        gremlins: List of gremlins that apply to this expression.

    Returns:
        An IfExp AST node implementing the switching logic.
    """
    gremlin_active = ast.Name(id='__gremlin_active__', ctx=ast.Load())

    result: ast.expr = copy.deepcopy(original)

    for gremlin in reversed(gremlins):
        condition = ast.Compare(
            left=gremlin_active,
            ops=[ast.Eq()],
            comparators=[ast.Constant(value=gremlin.gremlin_id)],
        )
        # mutated_node is ast.expr | ast.stmt; we know only expr gremlins reach here
        mutated_expr: ast.expr = copy.deepcopy(gremlin.mutated_node)  # type: ignore[assignment]
        result = ast.IfExp(
            test=condition,
            body=mutated_expr,
            orelse=result,
        )

    return result  # type: ignore[return-value]


def build_switching_statement(
    original: ast.stmt,
    gremlins: list[Gremlin],
) -> ast.If:
    """Build an AST statement that switches between original and mutated statements.

    Creates a nested If statement that checks __gremlin_active__
    and executes the appropriate statement based on which gremlin is active.

    Args:
        original: The original AST statement node.
        gremlins: List of gremlins that apply to this statement.

    Returns:
        An If AST node implementing the switching logic.
    """
    gremlin_active = ast.Name(id='__gremlin_active__', ctx=ast.Load())

    result: ast.stmt = copy.deepcopy(original)

    for gremlin in reversed(gremlins):
        condition = ast.Compare(
            left=gremlin_active,
            ops=[ast.Eq()],
            comparators=[ast.Constant(value=gremlin.gremlin_id)],
        )
        # mutated_node is ast.expr | ast.stmt; we know only stmt gremlins reach here
        mutated_stmt: ast.stmt = copy.deepcopy(gremlin.mutated_node)  # type: ignore[assignment]
        result = ast.If(
            test=condition,
            body=[mutated_stmt],
            orelse=[result],
        )

    return result  # type: ignore[return-value]


def build_switching_body(original: list[ast.stmt], gremlins: list[Gremlin]) -> list[ast.stmt]:
    """Build a function body that runs the original body or one gremlin's replacement body.

    Each gremlin's ``mutated_node`` is the whole function with its body replaced; its
    statements after the docstring become the branch taken when that gremlin is active.

    Args:
        original: The function's (already instrumented) body.
        gremlins: Gremlins whose mutated nodes are functions.

    Returns:
        The docstring, if any, then one If statement switching between the bodies.
    """
    docstring, statements = split_docstring(original)
    gremlin_active = ast.Name(id='__gremlin_active__', ctx=ast.Load())
    result: list[ast.stmt] = statements

    for gremlin in reversed(gremlins):
        mutated = gremlin.mutated_node
        if not isinstance(mutated, (ast.FunctionDef, ast.AsyncFunctionDef)):  # pragma: no cover
            raise TypeError(f'Expected a function gremlin, got {type(mutated).__name__}')
        _, replacement = split_docstring(copy.deepcopy(mutated.body))
        condition = ast.Compare(
            left=gremlin_active,
            ops=[ast.Eq()],
            comparators=[ast.Constant(value=gremlin.gremlin_id)],
        )
        switch = ast.If(test=condition, body=replacement, orelse=result)
        result = [ast.copy_location(switch, statements[0])]

    return [*docstring, *result]


def _is_main_guard(node: ast.Compare) -> bool:
    """Return True for ``__name__ == '__main__'`` (or ``!=``, either way round).

    Flipped, importing the module runs it as a script: tests cannot tell, and a
    script that starts a mutation run of its own module recurses without end.

    Examples:
        >>> _is_main_guard(ast.parse("'__main__' == __name__", mode='eval').body)
        True
    """
    if len(node.ops) != 1 or not isinstance(node.ops[0], (ast.Eq, ast.NotEq)):
        return False
    sides = (node.left, node.comparators[0])
    has_name = any(isinstance(side, ast.Name) and side.id == '__name__' for side in sides)
    has_main = any(isinstance(side, ast.Constant) and side.value == '__main__' for side in sides)
    return has_name and has_main


class MutationSwitchingTransformer(ast.NodeTransformer):
    """AST transformer that replaces mutation points with switching expressions.

    This transformer walks the AST and replaces each mutation point (e.g.,
    comparison expression) with a switching expression that selects the
    appropriate mutation based on the __gremlin_active__ variable.
    """

    def __init__(
        self,
        file_path: str,
        operators: list[GremlinOperator] | None = None,
    ) -> None:
        self.file_path = file_path
        self.gremlins: list[Gremlin] = []
        self._gremlin_counter = 0
        self._operators = operators if operators is not None else get_default_operators()
        # Create a short unique prefix from file path to avoid ID collisions
        # when processing multiple files in parallel
        self._file_prefix = self._make_file_prefix(file_path)

    @staticmethod
    def _make_file_prefix(file_path: str) -> str:
        """Create a short unique prefix from file path.

        Uses the file stem (name without extension) combined with a short hash
        to ensure uniqueness even for files with the same name in different dirs.
        """
        stem = Path(file_path).stem
        # Use first 4 chars of MD5 hash for uniqueness
        path_hash = hashlib.md5(file_path.encode(), usedforsecurity=False).hexdigest()[:4]
        return f'{stem}_{path_hash}'

    def _next_gremlin_id(self) -> str:
        self._gremlin_counter += 1
        return f'{self._file_prefix}_g{self._gremlin_counter:03d}'

    def _create_gremlins_for_compare(self, node: ast.Compare) -> list[Gremlin]:
        """Create gremlins for a comparison node."""
        return create_gremlins_for_compare(node, self.file_path, self._next_gremlin_id)

    def _get_operators_for_node(self, node: ast.expr | ast.stmt) -> list[GremlinOperator]:
        """Get all operators that can mutate the given node."""
        return [op for op in self._operators if op.can_mutate(node)]

    def _create_gremlins_for_node(self, node: ast.expr | ast.stmt) -> list[Gremlin]:
        """Create gremlins for any node using all applicable operators."""
        all_gremlins: list[Gremlin] = []
        for operator in self._get_operators_for_node(node):
            gremlins = create_gremlins_for_node(
                node,
                operator,
                self.file_path,
                self._next_gremlin_id,
            )
            all_gremlins.extend(gremlins)
        return all_gremlins

    def _switch_expression_with_pristine_mutations(self, node: ast.expr) -> ast.expr:
        """Instrument ``node``'s operands, then switch ``node`` itself between its mutations.

        Only one gremlin is active at a time, so while a mutation of ``node`` is active none of the gremlins
        inside its operands are. The mutated variants are therefore built from the operands as they were
        before instrumentation; only the original branch keeps the nested switches. Building the variants
        from the instrumented operands instead would copy the whole subtree into every variant and make the
        tree grow exponentially with the depth of a chain such as ``x + x + x + ...``.
        """
        pristine = copy.deepcopy(node)
        self.generic_visit(node)

        gremlins = self._create_gremlins_for_node(pristine)
        if not gremlins:
            return node

        self.gremlins.extend(gremlins)
        return build_switching_expression(node, gremlins)

    def visit_Compare(self, node: ast.Compare) -> ast.expr:
        """Replace comparison nodes with mutation switching expressions, except a ``__main__`` guard."""
        if _is_main_guard(node):
            return node
        return self._switch_expression_with_pristine_mutations(node)

    def visit_BinOp(self, node: ast.BinOp) -> ast.expr:
        """Replace binary operation nodes with mutation switching expressions."""
        return self._switch_expression_with_pristine_mutations(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.expr:
        """Replace boolean operation nodes with mutation switching expressions."""
        return self._switch_expression_with_pristine_mutations(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.expr:
        """Replace unary operation nodes (including 'not') with mutation switching."""
        return self._switch_expression_with_pristine_mutations(node)

    def visit_Constant(self, node: ast.Constant) -> ast.expr:
        """Replace boolean constants with mutation switching expressions."""
        gremlins = self._create_gremlins_for_node(node)
        if not gremlins:
            return node

        self.gremlins.extend(gremlins)
        return build_switching_expression(node, gremlins)

    def visit_Dict(self, node: ast.Dict) -> ast.expr:
        """Replace dict displays with mutation switching expressions."""
        return self._switch_expression_with_pristine_mutations(node)

    def _switch_statement_with_pristine_mutations(self, node: ast.stmt) -> ast.stmt:
        """Instrument ``node``'s children, then switch ``node`` itself between its mutations.

        The statement counterpart of ``_switch_expression_with_pristine_mutations``.
        """
        pristine = copy.deepcopy(node)
        self.generic_visit(node)

        gremlins = self._create_gremlins_for_node(pristine)
        if not gremlins:
            return node

        self.gremlins.extend(gremlins)
        return build_switching_statement(node, gremlins)

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.stmt:
        """Replace augmented assignments with mutation switching statements."""
        return self._switch_statement_with_pristine_mutations(node)

    def visit_Match(self, node: ast.Match) -> ast.stmt:
        """Replace match statements with mutation switching statements."""
        return self._switch_statement_with_pristine_mutations(node)

    def visit_match_case(self, node: ast.match_case) -> ast.match_case:
        """Instrument a case's guard and body; a pattern admits only literals, never a switch."""
        if node.guard is not None:
            node.guard = self.visit(node.guard)
        node.body = self._visit_statements(node.body)
        return node

    def _visit_statements(self, statements: list[ast.stmt]) -> list[ast.stmt]:
        visited: list[ast.stmt] = []
        for statement in statements:
            result = self.visit(statement)
            if isinstance(result, ast.AST):
                visited.append(result)  # type: ignore[arg-type]
            elif result is not None:  # pragma: no cover - no visitor here returns a list
                visited.extend(result)
        return visited

    def visit_arg(self, node: ast.arg) -> ast.arg:
        """Leave parameter annotations alone: an annotation is a type, not code under test."""
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.stmt:
        """Instrument an annotated assignment's target and value, never its annotation."""
        node.target = self.visit(node.target)
        if node.value is not None:
            node.value = self.visit(node.value)
        return node

    def visit_TypeAlias(self, node: ast.stmt) -> ast.stmt:
        """Leave ``type X = ...`` alone: it is a type, not code under test."""
        return node

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.stmt:
        """Instrument a function, then switch its body between the original and each replacement.

        The return annotation and type parameters are not instrumented. The docstring
        stays the first statement, so ``__doc__`` is the same with any gremlin active.
        """
        # Made before the children are instrumented in place, from the pristine body; the
        # gremlins keep only a shell of the function, never its body, which can be nested
        # deeper than copying or unparsing recurses.
        shell = function_shell(node, [ast.copy_location(ast.Expr(ast.Constant(...)), node)])
        gremlins = [
            dataclasses.replace(gremlin, original_node=shell) for gremlin in self._create_gremlins_for_node(node)
        ]
        node.decorator_list = [self.visit(decorator) for decorator in node.decorator_list]
        node.args = self.visit(node.args)
        node.body = self._visit_statements(node.body)
        if not gremlins:
            return node

        self.gremlins.extend(gremlins)
        node.body = build_switching_body(node.body, gremlins)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.stmt:
        """Instrument a function and switch its body (see ``_visit_function``)."""
        return self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.stmt:
        """Instrument an async function and switch its body (see ``_visit_function``)."""
        return self._visit_function(node)

    def visit_Return(self, node: ast.Return) -> ast.stmt:
        """Replace return statements with mutation switching."""
        self.generic_visit(node)

        gremlins = self._create_gremlins_for_node(node)
        if not gremlins:
            return node

        self.gremlins.extend(gremlins)
        return build_switching_statement(node, gremlins)


def transform_source(
    source: str,
    file_path: str,
    operators: list[GremlinOperator] | None = None,
) -> tuple[list[Gremlin], ast.Module]:
    """Transform source code by embedding mutation switching.

    This is the main entry point for instrumenting Python source code.
    It parses the source, identifies mutation points, and replaces them
    with switching expressions that can toggle between original and
    mutated behavior based on the ACTIVE_GREMLIN environment variable.

    Args:
        source: The Python source code to transform.
        file_path: The path to the source file (for gremlin metadata).
        operators: Optional list of operators to use. If None, uses all 5 default operators.

    Returns:
        Tuple of (list of gremlins, transformed AST with embedded switches).
    """
    tree = ast.parse(source)
    transformer = MutationSwitchingTransformer(file_path, operators=operators)
    new_tree = transformer.visit(tree)
    if not isinstance(new_tree, ast.Module):  # pragma: no cover
        raise TypeError(f'Expected ast.Module, got {type(new_tree).__name__}')

    pardoned_lines = parse_pardoned_lines(source, file_path=file_path)
    gremlins_by_line: dict[int, list[int]] = {}
    for i, g in enumerate(transformer.gremlins):
        gremlins_by_line.setdefault(g.line_number, []).append(i)

    _logger = logging.getLogger(__name__)
    for line_number in pardoned_lines:
        if line_number not in gremlins_by_line:
            _logger.warning(
                'Dead gremlin pragma at line %d of %s: no gremlins on that line',
                line_number,
                file_path,
            )

    resolved_gremlins: list[Gremlin] = []
    for gremlin in transformer.gremlins:
        if gremlin.line_number in pardoned_lines:
            reason_code, justification = pardoned_lines[gremlin.line_number]
            resolved_gremlins.append(
                dataclasses.replace(
                    gremlin,
                    pardoned=True,
                    pardon_reason=f'{reason_code}: {justification}',
                )
            )
        else:
            resolved_gremlins.append(gremlin)

    return resolved_gremlins, new_tree
