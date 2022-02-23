from typing import Tuple, Type, Union, Any

__all__ = [
    "DataType",
    "Scalar",
    "ScalarType",
    "Array",
    "Shape",
    "Param",
]

DataType = Type[Any]
Scalar = Union[bool, int, float]
ScalarType = Union[Type[bool], Type[int], Type[float]]
Array = Any
Shape = Tuple[int, ...]
Param = Tuple
