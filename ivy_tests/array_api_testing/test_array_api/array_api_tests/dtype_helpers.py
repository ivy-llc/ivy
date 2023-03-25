import re
from collections.abc import Mapping
from functools import lru_cache
from inspect import signature
from typing import Any, Dict, NamedTuple, Sequence, Tuple, Union
from warnings import warn

from . import _array_module as xp
from ._array_module import _UndefinedStub
from .stubs import name_to_func
from .typing import DataType, ScalarType

__all__ = [
    "int_dtypes",
    "uint_dtypes",
    "all_int_dtypes",
    "float_dtypes",
    "numeric_dtypes",
    "all_dtypes",
    "dtype_to_name",
    "bool_and_all_int_dtypes",
    "dtype_to_scalars",
    "is_int_dtype",
    "is_float_dtype",
    "get_scalar_type",
    "dtype_ranges",
    "default_int",
    "default_uint",
    "default_float",
    "promotion_table",
    "dtype_nbits",
    "dtype_signed",
    "func_in_dtypes",
    "func_returns_bool",
    "binary_op_to_symbol",
    "unary_op_to_symbol",
    "inplace_op_to_symbol",
    "op_to_func",
    "fmt_types",
]


class EqualityMapping(Mapping):
    """
    Mapping that uses equality for indexing

    Typical mappings (e.g. the built-in dict) use hashing for indexing. This
    isn't ideal for the Array API, as no __hash__() method is specified for
    dtype objects - but __eq__() is!

    See https://data-apis.org/array-api/latest/API_specification/data_types.html#data-type-objects
    """

    def __init__(self, key_value_pairs: Sequence[Tuple[Any, Any]]):
        keys = [k for k, _ in key_value_pairs]
        for i, key in enumerate(keys):
            if not (key == key):  # specifically checking __eq__, not __neq__
                raise ValueError(f"Key {key!r} does not have equality with itself")
            other_keys = keys[:]
            other_keys.pop(i)
            for other_key in other_keys:
                if key == other_key:
                    raise ValueError(f"Key {key!r} has equality with key {other_key!r}")
        self._key_value_pairs = key_value_pairs

    def __getitem__(self, key):
        for k, v in self._key_value_pairs:
            if key == k:
                return v
        else:
            raise KeyError(f"{key!r} not found")

    def __iter__(self):
        return (k for k, _ in self._key_value_pairs)

    def __len__(self):
        return len(self._key_value_pairs)

    def __str__(self):
        return "{" + ", ".join(f"{k!r}: {v!r}" for k, v in self._key_value_pairs) + "}"

    def __repr__(self):
        return f"EqualityMapping({self})"


_uint_names = ("uint8", "uint16", "uint32", "uint64")
_int_names = ("int8", "int16", "int32", "int64")
_float_names = ("float32", "float64")
_dtype_names = ("bool",) + _uint_names + _int_names + _float_names


uint_dtypes = tuple(getattr(xp, name) for name in _uint_names)
int_dtypes = tuple(getattr(xp, name) for name in _int_names)
float_dtypes = tuple(getattr(xp, name) for name in _float_names)
all_int_dtypes = uint_dtypes + int_dtypes
numeric_dtypes = all_int_dtypes + float_dtypes
all_dtypes = (xp.bool,) + numeric_dtypes
bool_and_all_int_dtypes = (xp.bool,) + all_int_dtypes


dtype_to_name = EqualityMapping([(getattr(xp, name), name) for name in _dtype_names])


dtype_to_scalars = EqualityMapping(
    [
        (xp.bool, [bool]),
        *[(d, [int]) for d in all_int_dtypes],
        *[(d, [int, float]) for d in float_dtypes],
    ]
)


def is_int_dtype(dtype):
    return dtype in all_int_dtypes


def is_float_dtype(dtype):
    # None equals NumPy's xp.float64 object, so we specifically check it here.
    # xp.float64 is in fact an alias of np.dtype('float64'), and its equality
    # with None is meant to be deprecated at some point.
    # See https://github.com/numpy/numpy/issues/18434
    if dtype is None:
        return False
    return dtype in float_dtypes


def get_scalar_type(dtype: DataType) -> ScalarType:
    if is_int_dtype(dtype):
        return int
    elif is_float_dtype(dtype):
        return float
    else:
        return bool


class MinMax(NamedTuple):
    min: Union[int, float]
    max: Union[int, float]


dtype_ranges = EqualityMapping(
    [
        (xp.int8, MinMax(-128, +127)),
        (xp.int16, MinMax(-32_768, +32_767)),
        (xp.int32, MinMax(-2_147_483_648, +2_147_483_647)),
        (xp.int64, MinMax(-9_223_372_036_854_775_808, +9_223_372_036_854_775_807)),
        (xp.uint8, MinMax(0, +255)),
        (xp.uint16, MinMax(0, +65_535)),
        (xp.uint32, MinMax(0, +4_294_967_295)),
        (xp.uint64, MinMax(0, +18_446_744_073_709_551_615)),
        (xp.float32, MinMax(-3.4028234663852886e38, 3.4028234663852886e38)),
        (xp.float64, MinMax(-1.7976931348623157e308, 1.7976931348623157e308)),
    ]
)

dtype_nbits = EqualityMapping(
    [(d, 8) for d in [xp.int8, xp.uint8]]
    + [(d, 16) for d in [xp.int16, xp.uint16]]
    + [(d, 32) for d in [xp.int32, xp.uint32, xp.float32]]
    + [(d, 64) for d in [xp.int64, xp.uint64, xp.float64]]
)


dtype_signed = EqualityMapping(
    [(d, True) for d in int_dtypes] + [(d, False) for d in uint_dtypes]
)


if isinstance(xp.asarray, _UndefinedStub):
    default_int = xp.int32
    default_float = xp.float32
    warn(
        "array module does not have attribute asarray. "
        "default int is assumed int32, default float is assumed float32"
    )
else:
    default_int = xp.asarray(int()).dtype
    if default_int not in int_dtypes:
        warn(f"inferred default int is {default_int!r}, which is not an int")
    default_float = xp.asarray(float()).dtype
    if default_float not in float_dtypes:
        warn(f"inferred default float is {default_float!r}, which is not a float")
if dtype_nbits[default_int] == 32:
    default_uint = xp.uint32
else:
    default_uint = xp.uint64


_numeric_promotions = [
    # ints
    ((xp.int8, xp.int8), xp.int8),
    ((xp.int8, xp.int16), xp.int16),
    ((xp.int8, xp.int32), xp.int32),
    ((xp.int8, xp.int64), xp.int64),
    ((xp.int16, xp.int16), xp.int16),
    ((xp.int16, xp.int32), xp.int32),
    ((xp.int16, xp.int64), xp.int64),
    ((xp.int32, xp.int32), xp.int32),
    ((xp.int32, xp.int64), xp.int64),
    ((xp.int64, xp.int64), xp.int64),
    # uints
    ((xp.uint8, xp.uint8), xp.uint8),
    ((xp.uint8, xp.uint16), xp.uint16),
    ((xp.uint8, xp.uint32), xp.uint32),
    ((xp.uint8, xp.uint64), xp.uint64),
    ((xp.uint16, xp.uint16), xp.uint16),
    ((xp.uint16, xp.uint32), xp.uint32),
    ((xp.uint16, xp.uint64), xp.uint64),
    ((xp.uint32, xp.uint32), xp.uint32),
    ((xp.uint32, xp.uint64), xp.uint64),
    ((xp.uint64, xp.uint64), xp.uint64),
    # ints and uints (mixed sign)
    ((xp.int8, xp.uint8), xp.int16),
    ((xp.int8, xp.uint16), xp.int32),
    ((xp.int8, xp.uint32), xp.int64),
    ((xp.int16, xp.uint8), xp.int16),
    ((xp.int16, xp.uint16), xp.int32),
    ((xp.int16, xp.uint32), xp.int64),
    ((xp.int32, xp.uint8), xp.int32),
    ((xp.int32, xp.uint16), xp.int32),
    ((xp.int32, xp.uint32), xp.int64),
    ((xp.int64, xp.uint8), xp.int64),
    ((xp.int64, xp.uint16), xp.int64),
    ((xp.int64, xp.uint32), xp.int64),
    # floats
    ((xp.float32, xp.float32), xp.float32),
    ((xp.float32, xp.float64), xp.float64),
    ((xp.float64, xp.float64), xp.float64),
]
_numeric_promotions += [((d2, d1), res) for (d1, d2), res in _numeric_promotions]
_promotion_table = list(set(_numeric_promotions))
_promotion_table.insert(0, ((xp.bool, xp.bool), xp.bool))
promotion_table = EqualityMapping(_promotion_table)


def result_type(*dtypes: DataType):
    if len(dtypes) == 0:
        raise ValueError()
    elif len(dtypes) == 1:
        return dtypes[0]
    result = promotion_table[dtypes[0], dtypes[1]]
    for i in range(2, len(dtypes)):
        result = promotion_table[result, dtypes[i]]
    return result


r_alias = re.compile("[aA]lias")
r_in_dtypes = re.compile("x1?: array\n.+have an? (.+) data type.")
r_int_note = re.compile(
    "If one or both of the input arrays have integer data types, "
    "the result is implementation-dependent"
)
category_to_dtypes = {
    "boolean": (xp.bool,),
    "integer": all_int_dtypes,
    "floating-point": float_dtypes,
    "numeric": numeric_dtypes,
    "integer or boolean": bool_and_all_int_dtypes,
}
func_in_dtypes: Dict[str, Tuple[DataType, ...]] = {}
for name, func in name_to_func.items():
    if m := r_in_dtypes.search(func.__doc__):
        dtype_category = m.group(1)
        if dtype_category == "numeric" and r_int_note.search(func.__doc__):
            dtype_category = "floating-point"
        dtypes = category_to_dtypes[dtype_category]
        func_in_dtypes[name] = dtypes
    elif any("x" in name for name in signature(func).parameters.keys()):
        func_in_dtypes[name] = all_dtypes
# See https://github.com/data-apis/array-api/pull/413
func_in_dtypes["expm1"] = float_dtypes


func_returns_bool = {
    # elementwise
    "abs": False,
    "acos": False,
    "acosh": False,
    "add": False,
    "asin": False,
    "asinh": False,
    "atan": False,
    "atan2": False,
    "atanh": False,
    "bitwise_and": False,
    "bitwise_invert": False,
    "bitwise_left_shift": False,
    "bitwise_or": False,
    "bitwise_right_shift": False,
    "bitwise_xor": False,
    "ceil": False,
    "cos": False,
    "cosh": False,
    "divide": False,
    "equal": True,
    "exp": False,
    "expm1": False,
    "floor": False,
    "floor_divide": False,
    "greater": True,
    "greater_equal": True,
    "isfinite": True,
    "isinf": True,
    "isnan": True,
    "less": True,
    "less_equal": True,
    "log": False,
    "logaddexp": False,
    "log10": False,
    "log1p": False,
    "log2": False,
    "logical_and": True,
    "logical_not": True,
    "logical_or": True,
    "logical_xor": True,
    "multiply": False,
    "negative": False,
    "not_equal": True,
    "positive": False,
    "pow": False,
    "remainder": False,
    "round": False,
    "sign": False,
    "sin": False,
    "sinh": False,
    "sqrt": False,
    "square": False,
    "subtract": False,
    "tan": False,
    "tanh": False,
    "trunc": False,
    # searching
    "where": False,
    # linalg
    "matmul": False,
}


unary_op_to_symbol = {
    "__invert__": "~",
    "__neg__": "-",
    "__pos__": "+",
}


binary_op_to_symbol = {
    "__add__": "+",
    "__and__": "&",
    "__eq__": "==",
    "__floordiv__": "//",
    "__ge__": ">=",
    "__gt__": ">",
    "__le__": "<=",
    "__lshift__": "<<",
    "__lt__": "<",
    "__matmul__": "@",
    "__mod__": "%",
    "__mul__": "*",
    "__ne__": "!=",
    "__or__": "|",
    "__pow__": "**",
    "__rshift__": ">>",
    "__sub__": "-",
    "__truediv__": "/",
    "__xor__": "^",
}


op_to_func = {
    "__abs__": "abs",
    "__add__": "add",
    "__and__": "bitwise_and",
    "__eq__": "equal",
    "__floordiv__": "floor_divide",
    "__ge__": "greater_equal",
    "__gt__": "greater",
    "__le__": "less_equal",
    "__lt__": "less",
    "__matmul__": "matmul",
    "__mod__": "remainder",
    "__mul__": "multiply",
    "__ne__": "not_equal",
    "__or__": "bitwise_or",
    "__pow__": "pow",
    "__lshift__": "bitwise_left_shift",
    "__rshift__": "bitwise_right_shift",
    "__sub__": "subtract",
    "__truediv__": "divide",
    "__xor__": "bitwise_xor",
    "__invert__": "bitwise_invert",
    "__neg__": "negative",
    "__pos__": "positive",
}


for op, elwise_func in op_to_func.items():
    func_in_dtypes[op] = func_in_dtypes[elwise_func]
    func_returns_bool[op] = func_returns_bool[elwise_func]


inplace_op_to_symbol = {}
for op, symbol in binary_op_to_symbol.items():
    if op == "__matmul__" or func_returns_bool[op]:
        continue
    iop = f"__i{op[2:]}"
    inplace_op_to_symbol[iop] = f"{symbol}="
    func_in_dtypes[iop] = func_in_dtypes[op]
    func_returns_bool[iop] = func_returns_bool[op]


func_in_dtypes["__bool__"] = (xp.bool,)
func_in_dtypes["__int__"] = all_int_dtypes
func_in_dtypes["__index__"] = all_int_dtypes
func_in_dtypes["__float__"] = float_dtypes
func_in_dtypes["from_dlpack"] = numeric_dtypes
func_in_dtypes["__dlpack__"] = numeric_dtypes


@lru_cache
def fmt_types(types: Tuple[Union[DataType, ScalarType], ...]) -> str:
    f_types = []
    for type_ in types:
        try:
            f_types.append(dtype_to_name[type_])
        except KeyError:
            # i.e. dtype is bool, int, or float
            f_types.append(type_.__name__)
    return ", ".join(f_types)
