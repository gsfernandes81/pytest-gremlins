r"""Function body mutation operator, after cargo-mutants' ``FnValue`` genre.

The body of a function is replaced by a return of a value guessed from its
return annotation, checking that the tests observe the function's side effects
and tell its return values apart. cargo-mutants' type table, read for Python:

======================================  =============================================
Return annotation                       Replacement values
======================================  =============================================
none, ``None``, or any type not below   ``None`` (Rust's ``()`` and ``Default::default()``)
``bool``                                ``True``, ``False``
``int``                                 ``0``, ``1``, ``-1``
``float``                               ``0.0``, ``1.0``, ``-1.0``
``str``                                 ``''``, ``'xyzzy'``
``bytes``                               ``b''``, ``b'\\x00'``, ``b'\\x01'`` (Rust's ``Vec<u8>``)
``T | None``, ``Optional[T]``           ``None``, then every value of ``T``
``list[T]``, ``Sequence[T]``            ``[]``, then ``[v]`` for every value of ``T``
``set[T]``, ``frozenset[T]``            empty, then one element of every value of ``T``
``dict[K, V]``, ``Mapping[K, V]``       ``{}``, then ``{k: v}`` for every pair
``tuple[A, B]``                         the product of every value of ``A``, ``B``
``tuple[T, ...]``                       ``()``, then ``(v,)`` for every value of ``T``
``Iterator[T]``, ``Iterable[T]``        ``iter([])``, then ``iter([v])`` for every ``T``
``NoReturn``, ``Never``                 no mutation
======================================  =============================================

A generator (or async generator) instead becomes empty, then yields one value
of its element type. As in cargo-mutants, a function with an empty body (only a
docstring, ``pass`` or ``...``) is not mutated, nor is a replacement identical to
the body; constructors (``__init__``, ``__new__``) are skipped as cargo-mutants
skips ``new``, and ``@abstractmethod`` and ``@overload`` functions, which have no
body to speak of, as it skips trait methods without one.
"""

from __future__ import annotations

import ast
import copy
import itertools
from typing import TypeAlias

FunctionNode: TypeAlias = ast.FunctionDef | ast.AsyncFunctionDef

_SKIPPED_NAMES = frozenset({'__init__', '__new__'})
_SKIPPED_DECORATORS = frozenset({'abstractmethod', 'overload'})
_SEQUENCES = frozenset({'list', 'List', 'Sequence', 'MutableSequence', 'Collection'})
_SETS = frozenset({'set', 'Set', 'AbstractSet', 'MutableSet'})
_FROZENSETS = frozenset({'frozenset', 'FrozenSet'})
_MAPPINGS = frozenset({'dict', 'Dict', 'Mapping', 'MutableMapping'})
_ITERATORS = frozenset(
    {'Iterator', 'Iterable', 'Generator', 'AsyncIterator', 'AsyncIterable', 'AsyncGenerator', 'Reversible'}
)
_NEVER = frozenset({'NoReturn', 'Never'})
_SCALARS: dict[str, list[str]] = {
    'None': ['None'],
    'bool': ['True', 'False'],
    'int': ['0', '1', '-1'],
    'float': ['0.0', '1.0', '-1.0'],
    'str': ["''", "'xyzzy'"],
    'bytes': ["b''", "b'\\x00'", "b'\\x01'"],
}


def _type_name(node: ast.expr) -> str | None:
    """Return the last component of a dotted type name, or None.

    Examples:
        >>> _type_name(ast.parse('typing.Optional', mode='eval').body)
        'Optional'
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Constant) and node.value is None:
        return 'None'
    return None


def _type_args(node: ast.Subscript) -> list[ast.expr]:
    return list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]


def _optional_inner(node: ast.expr) -> ast.expr | None:
    """Return ``T`` from ``T | None``, ``None | T``, ``Optional[T]`` or ``Union[T, None]``, else None."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        if _type_name(node.right) == 'None':
            return node.left
        if _type_name(node.left) == 'None':
            return node.right
        return None
    if isinstance(node, ast.Subscript):
        name = _type_name(node.value)
        args = _type_args(node)
        if name == 'Optional' and len(args) == 1:
            return args[0]
        if name == 'Union' and len(args) == 2:  # noqa: PLR2004 - exactly T and None
            others = [arg for arg in args if _type_name(arg) != 'None']
            if len(others) == 1:
                return others[0]
    return None


def iterator_item(annotation: ast.expr | None) -> ast.expr | None:
    """Return the element type of an iterator or generator annotation, else None.

    Examples:
        >>> ast.unparse(iterator_item(ast.parse('Iterator[int]', mode='eval').body))
        'int'
    """
    if isinstance(annotation, ast.Subscript) and _type_name(annotation.value) in _ITERATORS:
        return _type_args(annotation)[0]
    return None


def type_values(annotation: ast.expr | None) -> list[str]:  # noqa: C901, PLR0911, PLR0912 - one branch per row of the table
    """Return the source of every replacement value for a return annotation.

    Examples:
        >>> type_values(ast.parse('int | None', mode='eval').body)
        ['None', '0', '1', '-1']
        >>> type_values(ast.parse('tuple[bool, str]', mode='eval').body)
        ["(True, '')", "(True, 'xyzzy')", "(False, '')", "(False, 'xyzzy')"]
        >>> type_values(None)
        ['None']
    """
    if annotation is None:
        return ['None']
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            return type_values(ast.parse(annotation.value, mode='eval').body)
        except SyntaxError:
            return ['None']
    inner = _optional_inner(annotation)
    if inner is not None:
        return ['None', *(value for value in type_values(inner) if value != 'None')]
    if isinstance(annotation, ast.Subscript):
        name = _type_name(annotation.value)
        args = _type_args(annotation)
        if name in _SEQUENCES:
            return ['[]', *(f'[{value}]' for value in type_values(args[0]))]
        if name in _SETS:
            return ['set()', *(f'{{{value}}}' for value in type_values(args[0]))]
        if name in _FROZENSETS:
            return ['frozenset()', *(f'frozenset({{{value}}})' for value in type_values(args[0]))]
        if name in _MAPPINGS and len(args) == 2:  # noqa: PLR2004 - key and value
            pairs = itertools.product(type_values(args[0]), type_values(args[1]))
            return ['{}', *(f'{{{key}: {value}}}' for key, value in pairs)]
        if name in _ITERATORS:
            return ['iter([])', *(f'iter([{value}])' for value in type_values(args[0]))]
        if name in {'tuple', 'Tuple'}:
            if len(args) == 2 and isinstance(args[1], ast.Constant) and args[1].value is Ellipsis:  # noqa: PLR2004
                return ['()', *(f'({value},)' for value in type_values(args[0]))]
            if len(args) == 1 and isinstance(args[0], ast.Tuple) and not args[0].elts:
                return ['()']
            products = itertools.product(*(type_values(arg) for arg in args))
            return [f'({", ".join(values)}{"," if len(args) == 1 else ""})' for values in products]
        return ['None']
    name = _type_name(annotation)
    if name in _NEVER:
        return []
    if name in _SCALARS:
        return _SCALARS[name]
    if name in _SEQUENCES:
        return ['[]', '[None]']
    if name in _SETS:
        return ['set()', '{None}']
    if name in _FROZENSETS:
        return ['frozenset()', 'frozenset({None})']
    if name in _MAPPINGS:
        return ['{}', '{None: None}']
    if name in {'tuple', 'Tuple'}:
        return ['()', '(None,)']
    return ['None']


def _has_docstring(body: list[ast.stmt]) -> bool:
    first = body[0]
    return isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)


def split_docstring(body: list[ast.stmt]) -> tuple[list[ast.stmt], list[ast.stmt]]:
    """Return a function body as (its docstring, if any, as a list; the statements after it)."""
    count = 1 if body and _has_docstring(body) else 0
    return body[:count], body[count:]


def _is_empty(statements: list[ast.stmt]) -> bool:
    return all(
        isinstance(statement, ast.Pass)
        or (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant))
        for statement in statements
    )


def is_generator(node: FunctionNode) -> bool:
    r"""Return True if *node* yields, outside any function, lambda or class nested in it.

    Examples:
        >>> is_generator(ast.parse('def f():\n    yield 1').body[0])
        True
        >>> is_generator(ast.parse('def f():\n    def g():\n        yield 1\n    return g').body[0])
        False
    """
    pending: list[ast.AST] = list(node.body)
    while pending:
        current = pending.pop()
        if isinstance(current, (ast.Yield, ast.YieldFrom)):
            return True
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        pending.extend(ast.iter_child_nodes(current))
    return False


def function_shell(node: FunctionNode, body: list[ast.stmt]) -> FunctionNode:
    """Return a copy of *node*'s signature around *body*, never copying *node*'s own body.

    A function body can be nested deeper than ``copy.deepcopy`` can recurse (a long
    ``elif`` chain), and a gremlin needs only the signature and its own body.
    """
    shell = copy.copy(node)
    shell.decorator_list = copy.deepcopy(node.decorator_list)
    shell.args = copy.deepcopy(node.args)
    shell.returns = copy.deepcopy(node.returns)
    shell.body = body
    return shell


def _decorator_name(decorator: ast.expr) -> str | None:
    return _type_name(decorator.func if isinstance(decorator, ast.Call) else decorator)


class FnValueOperator:
    """Replace a function's body with a return of a value of its return type."""

    @property
    def name(self) -> str:
        """Return unique identifier for this operator."""
        return 'fn_value'

    @property
    def description(self) -> str:
        """Return human-readable description for reports."""
        return 'Replace function bodies with a value of the return type'

    def can_mutate(self, node: ast.AST) -> bool:
        r"""Return True if this operator can mutate the given AST node.

        Examples:
            >>> FnValueOperator().can_mutate(ast.parse('def f() -> int:\n    return g()').body[0])
            True
            >>> FnValueOperator().can_mutate(ast.parse('def f() -> int:\n    ...').body[0])
            False
        """
        return bool(self.mutate(node))

    def mutate(self, node: ast.AST) -> list[ast.AST]:
        r"""Return one copy of the function per replacement, its body replaced.

        A docstring stays, so the function's ``__doc__`` is the same in every gremlin.

        Examples:
            >>> tree = ast.parse('def f(x) -> bool:\n    return x > 0').body[0]
            >>> [ast.unparse(m.body) for m in FnValueOperator().mutate(tree)]
            ['return True', 'return False']
        """
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or self._skipped(node):
            return []
        docstring, statements = split_docstring(node.body)
        if _is_empty(statements):
            return []
        mutations: list[ast.AST] = []
        for replacement in self._replacements(node):
            if self._same(statements, replacement):
                continue
            for statement in replacement:
                ast.copy_location(statement, statements[0])
                ast.fix_missing_locations(statement)
            mutations.append(function_shell(node, [*copy.deepcopy(docstring), *replacement]))
        return mutations

    def describe(self, original: ast.AST, mutated: ast.AST) -> str:
        r"""Return ``'replace <name> [-> <annotation>] with <value>'`` for one gremlin.

        Examples:
            >>> op = FnValueOperator()
            >>> tree = ast.parse('def f() -> int:\n    return g()').body[0]
            >>> op.describe(tree, op.mutate(tree)[0])
            'replace f -> int with 0'
        """
        assert isinstance(original, (ast.FunctionDef, ast.AsyncFunctionDef))  # noqa: S101
        assert isinstance(mutated, (ast.FunctionDef, ast.AsyncFunctionDef))  # noqa: S101
        signature = original.name if original.returns is None else f'{original.name} -> {ast.unparse(original.returns)}'
        _, replacement = split_docstring(mutated.body)
        if is_generator(original):
            yielded = [s.value.value for s in replacement if isinstance(s, ast.Expr) and isinstance(s.value, ast.Yield)]
            value = f'iter([{ast.unparse(yielded[0])}])' if yielded and yielded[0] is not None else 'iter([])'
        else:
            returned = replacement[0]
            assert isinstance(returned, ast.Return)  # noqa: S101 - a plain function returns its value
            value = 'None' if returned.value is None else ast.unparse(returned.value)
        return f'replace {signature} with {value}'

    def locate(self, original: ast.AST, mutated: ast.AST) -> int:  # noqa: ARG002 - every gremlin of a function shares it
        """Return the line of the first statement that runs on every call.

        Coverage maps each test to the lines it ran, and a ``def`` line runs once,
        at import: the gremlin sits on a body line so the tests that call the
        function are the ones selected for it.
        """
        assert isinstance(original, (ast.FunctionDef, ast.AsyncFunctionDef))  # noqa: S101
        _, statements = split_docstring(original.body)
        runs = [statement for statement in statements if not isinstance(statement, (ast.Global, ast.Nonlocal))]
        return (runs or statements)[0].lineno

    @staticmethod
    def _same(statements: list[ast.stmt], replacement: list[ast.stmt]) -> bool:
        if len(statements) != len(replacement):
            return False
        return ast.dump(ast.Module(body=statements, type_ignores=[])) == ast.dump(
            ast.Module(body=replacement, type_ignores=[])
        )

    @staticmethod
    def _skipped(node: FunctionNode) -> bool:
        if node.name in _SKIPPED_NAMES:
            return True
        return any(_decorator_name(decorator) in _SKIPPED_DECORATORS for decorator in node.decorator_list)

    @staticmethod
    def _replacements(node: FunctionNode) -> list[list[ast.stmt]]:
        if is_generator(node):
            item = iterator_item(node.returns)
            values = type_values(item) if item is not None else ['None']
            yields: list[list[ast.stmt]] = [[ast.Expr(ast.Yield(_expr(value)))] for value in values]
            return [[ast.Return()], *yields]
        return [[ast.Return(_expr(value))] for value in type_values(node.returns)]


def _expr(source: str) -> ast.expr:
    return ast.parse(source, mode='eval').body
