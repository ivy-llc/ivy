"""
This file defines the types for type annotations.

The type variables should be replaced with the actual types for a given
library, e.g., for NumPy TypeVar('array') would be replaced with ndarray.
"""

from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Sequence, Tuple, TypeVar, Union

array = TypeVar('array')
device = TypeVar('device')
dtype = TypeVar('dtype')
SupportsDLPack = TypeVar('SupportsDLPack')
SupportsBufferProtocol = TypeVar('SupportsBufferProtocol')
PyCapsule = TypeVar('PyCapsule')
# ellipsis cannot actually be imported from anywhere, so include a dummy here
# to keep pyflakes happy. https://github.com/python/typeshed/issues/3556
ellipsis = TypeVar('ellipsis')

@dataclass
class finfo_object:
    bits: int
    eps: float
    max: float
    min: float
    smallest_normal: float

@dataclass
class iinfo_object:
    bits: int
    max: int
    min: int

# This should really be recursive, but that isn't supported yet.
NestedSequence = Sequence[Sequence[Any]]

__all__ = ['Any', 'List', 'Literal', 'NestedSequence', 'Optional',
'PyCapsule', 'SupportsBufferProtocol', 'SupportsDLPack', 'Tuple', 'Union',
'array', 'device', 'dtype', 'ellipsis', 'finfo_object', 'iinfo_object']

