"""
Stub definitions for functions defined in the spec

These are used to test function signatures.

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.
"""

__all__ = []

from .array_object import __abs__, __add__, __and__, __array_namespace__, __bool__, __dlpack__, __dlpack_device__, __eq__, __float__, __floordiv__, __ge__, __getitem__, __gt__, __index__, __int__, __invert__, __le__, __lshift__, __lt__, __matmul__, __mod__, __mul__, __ne__, __neg__, __or__, __pos__, __pow__, __rshift__, __setitem__, __sub__, __truediv__, __xor__, to_device, __iadd__, __radd__, __iand__, __rand__, __ifloordiv__, __rfloordiv__, __ilshift__, __rlshift__, __imatmul__, __rmatmul__, __imod__, __rmod__, __imul__, __rmul__, __ior__, __ror__, __ipow__, __rpow__, __irshift__, __rrshift__, __isub__, __rsub__, __itruediv__, __rtruediv__, __ixor__, __rxor__, dtype, device, mT, ndim, shape, size, T

__all__ += ['__abs__', '__add__', '__and__', '__array_namespace__', '__bool__', '__dlpack__', '__dlpack_device__', '__eq__', '__float__', '__floordiv__', '__ge__', '__getitem__', '__gt__', '__index__', '__int__', '__invert__', '__le__', '__lshift__', '__lt__', '__matmul__', '__mod__', '__mul__', '__ne__', '__neg__', '__or__', '__pos__', '__pow__', '__rshift__', '__setitem__', '__sub__', '__truediv__', '__xor__', 'to_device', '__iadd__', '__radd__', '__iand__', '__rand__', '__ifloordiv__', '__rfloordiv__', '__ilshift__', '__rlshift__', '__imatmul__', '__rmatmul__', '__imod__', '__rmod__', '__imul__', '__rmul__', '__ior__', '__ror__', '__ipow__', '__rpow__', '__irshift__', '__rrshift__', '__isub__', '__rsub__', '__itruediv__', '__rtruediv__', '__ixor__', '__rxor__', 'dtype', 'device', 'mT', 'ndim', 'shape', 'size', 'T']

from .constants import e, inf, nan, pi

__all__ += ['e', 'inf', 'nan', 'pi']

from .creation_functions import arange, asarray, empty, empty_like, eye, from_dlpack, full, full_like, linspace, meshgrid, ones, ones_like, tril, triu, zeros, zeros_like

__all__ += ['arange', 'asarray', 'empty', 'empty_like', 'eye', 'from_dlpack', 'full', 'full_like', 'linspace', 'meshgrid', 'ones', 'ones_like', 'tril', 'triu', 'zeros', 'zeros_like']

from .data_type_functions import astype, broadcast_arrays, broadcast_to, can_cast, finfo, iinfo, result_type

__all__ += ['astype', 'broadcast_arrays', 'broadcast_to', 'can_cast', 'finfo', 'iinfo', 'result_type']

from .elementwise_functions import abs, acos, acosh, add, asin, asinh, atan, atan2, atanh, bitwise_and, bitwise_left_shift, bitwise_invert, bitwise_or, bitwise_right_shift, bitwise_xor, ceil, cos, cosh, divide, equal, exp, expm1, floor, floor_divide, greater, greater_equal, isfinite, isinf, isnan, less, less_equal, log, log1p, log2, log10, logaddexp, logical_and, logical_not, logical_or, logical_xor, multiply, negative, not_equal, positive, pow, remainder, round, sign, sin, sinh, square, sqrt, subtract, tan, tanh, trunc

__all__ += ['abs', 'acos', 'acosh', 'add', 'asin', 'asinh', 'atan', 'atan2', 'atanh', 'bitwise_and', 'bitwise_left_shift', 'bitwise_invert', 'bitwise_or', 'bitwise_right_shift', 'bitwise_xor', 'ceil', 'cos', 'cosh', 'divide', 'equal', 'exp', 'expm1', 'floor', 'floor_divide', 'greater', 'greater_equal', 'isfinite', 'isinf', 'isnan', 'less', 'less_equal', 'log', 'log1p', 'log2', 'log10', 'logaddexp', 'logical_and', 'logical_not', 'logical_or', 'logical_xor', 'multiply', 'negative', 'not_equal', 'positive', 'pow', 'remainder', 'round', 'sign', 'sin', 'sinh', 'square', 'sqrt', 'subtract', 'tan', 'tanh', 'trunc']

from .linear_algebra_functions import matmul, matrix_transpose, tensordot, vecdot

__all__ += ['matmul', 'matrix_transpose', 'tensordot', 'vecdot']

from .manipulation_functions import concat, expand_dims, flip, permute_dims, reshape, roll, squeeze, stack

__all__ += ['concat', 'expand_dims', 'flip', 'permute_dims', 'reshape', 'roll', 'squeeze', 'stack']

from .searching_functions import argmax, argmin, nonzero, where

__all__ += ['argmax', 'argmin', 'nonzero', 'where']

from .set_functions import unique_all, unique_counts, unique_inverse, unique_values

__all__ += ['unique_all', 'unique_counts', 'unique_inverse', 'unique_values']

from .sorting_functions import argsort, sort

__all__ += ['argsort', 'sort']

from .statistical_functions import max, mean, min, prod, std, sum, var

__all__ += ['max', 'mean', 'min', 'prod', 'std', 'sum', 'var']

from .utility_functions import all, any

__all__ += ['all', 'any']

from . import linalg

__all__ += ['linalg']
