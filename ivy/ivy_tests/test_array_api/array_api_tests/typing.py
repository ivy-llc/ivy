from typing import Any, Tuple, Type, Union

__all__ = [
    "DataType",
    "Scalar",
    "ScalarType",
    "Array",
    "Shape",
    "AtomicIndex",
    "Index",
    "Param",
]

DataType = Type[Any]
Scalar = Union[bool, int, float]
ScalarType = Union[Type[bool], Type[int], Type[float]]
Array = Any
Shape = Tuple[int, ...]
AtomicIndex = Union[int, "ellipsis", slice, None]  # noqa
Index = Union[AtomicIndex, Tuple[AtomicIndex, ...]]
Param = Tuple
