"""Mutation operators for pytest-gremlins.

This package provides the operator system for defining and managing
mutation operators that inject gremlins into Python code.
"""

from pytest_gremlins.operators.arithmetic import ArithmeticOperator
from pytest_gremlins.operators.binary_operator import BinaryOperator
from pytest_gremlins.operators.boolean import BooleanOperator
from pytest_gremlins.operators.boundary import BoundaryOperator
from pytest_gremlins.operators.comparison import ComparisonOperator
from pytest_gremlins.operators.fn_value import FnValueOperator
from pytest_gremlins.operators.match_arm import (
    MatchArmGuardOperator,
    MatchArmOperator,
)
from pytest_gremlins.operators.protocol import GremlinOperator
from pytest_gremlins.operators.registry import OperatorRegistry
from pytest_gremlins.operators.return_value import ReturnOperator
from pytest_gremlins.operators.struct_field import StructFieldOperator
from pytest_gremlins.operators.unary_operator import UnaryOperator

# The operators that run when none are configured: cargo-mutants' mutation genres.
DEFAULT_OPERATORS = (
    'fn_value',
    'binary_operator',
    'unary_operator',
    'match_arm',
    'match_arm_guard',
    'struct_field',
)

# The operators that were the default before cargo-mutants' set; each can still be enabled by name.
LEGACY_OPERATORS = ('comparison', 'arithmetic', 'boolean', 'boundary', 'return')

__all__ = [
    'DEFAULT_OPERATORS',
    'LEGACY_OPERATORS',
    'ArithmeticOperator',
    'BinaryOperator',
    'BooleanOperator',
    'BoundaryOperator',
    'ComparisonOperator',
    'FnValueOperator',
    'GremlinOperator',
    'MatchArmGuardOperator',
    'MatchArmOperator',
    'OperatorRegistry',
    'ReturnOperator',
    'StructFieldOperator',
    'UnaryOperator',
]
