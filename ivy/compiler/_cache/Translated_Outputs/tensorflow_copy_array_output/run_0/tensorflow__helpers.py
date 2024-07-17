from collections import UserDict
from numbers import Number
from numpy.core.numeric import normalize_axis_tuple
from operator import mul
from .tensorflow_NestedSequence import tensorflow_NestedSequence
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TypeVar
from typing import Union
import functools
import inspect
import itertools
import math
import numpy as np
import re
import tensorflow
import tensorflow as tf


promotion_table = {
    ("bool", "bool"): "bool",
    ("int8", "int8"): "int8",
    ("int8", "int16"): "int16",
    ("int8", "int32"): "int32",
    ("int8", "int64"): "int64",
    ("int16", "int16"): "int16",
    ("int16", "int32"): "int32",
    ("int16", "int64"): "int64",
    ("int32", "int32"): "int32",
    ("int32", "int64"): "int64",
    ("int64", "int64"): "int64",
    ("uint8", "int8"): "int16",
    ("uint8", "int16"): "int16",
    ("uint8", "int32"): "int32",
    ("uint8", "int64"): "int64",
    ("uint8", "uint8"): "uint8",
    ("uint8", "uint16"): "uint16",
    ("uint8", "uint32"): "uint32",
    ("uint8", "uint64"): "uint64",
    ("uint16", "int8"): "int32",
    ("uint16", "int16"): "int32",
    ("uint16", "int32"): "int32",
    ("uint16", "int64"): "int64",
    ("uint16", "uint16"): "uint16",
    ("uint16", "uint32"): "uint32",
    ("uint16", "uint64"): "uint64",
    ("uint32", "int8"): "int64",
    ("uint32", "int16"): "int64",
    ("uint32", "int32"): "int64",
    ("uint32", "int64"): "int64",
    ("uint32", "uint32"): "uint32",
    ("uint32", "uint64"): "uint64",
    ("uint64", "uint64"): "uint64",
    ("float16", "float16"): "float16",
    ("float16", "float32"): "float32",
    ("float16", "float64"): "float64",
    ("float32", "float32"): "float32",
    ("float32", "float64"): "float64",
    ("float64", "float64"): "float64",
    ("bool", "int8"): "int8",
    ("bool", "int16"): "int16",
    ("bool", "int32"): "int32",
    ("bool", "int64"): "int64",
    ("bool", "uint8"): "uint8",
    ("bool", "uint16"): "uint16",
    ("bool", "uint32"): "uint32",
    ("bool", "uint64"): "uint64",
    ("bool", "float16"): "float16",
    ("bool", "float32"): "float32",
    ("bool", "float64"): "float64",
    ("bool", "bfloat16"): "bfloat16",
    ("bool", "complex64"): "complex64",
    ("bool", "complex128"): "complex128",
    ("int8", "float16"): "float16",
    ("int8", "float32"): "float32",
    ("int8", "float64"): "float64",
    ("int8", "bfloat16"): "bfloat16",
    ("int8", "complex64"): "complex64",
    ("int8", "complex128"): "complex128",
    ("int16", "float32"): "float32",
    ("int16", "float64"): "float64",
    ("int16", "complex64"): "complex64",
    ("int16", "complex128"): "complex128",
    ("int32", "float64"): "float64",
    ("int32", "complex128"): "complex128",
    ("int64", "float64"): "float64",
    ("int64", "complex128"): "complex128",
    ("uint8", "float16"): "float16",
    ("uint8", "float32"): "float32",
    ("uint8", "float64"): "float64",
    ("uint8", "bfloat16"): "bfloat16",
    ("uint8", "complex64"): "complex64",
    ("uint8", "complex128"): "complex128",
    ("uint16", "float32"): "float32",
    ("uint16", "float64"): "float64",
    ("uint16", "complex64"): "complex64",
    ("uint16", "complex128"): "complex128",
    ("uint32", "float64"): "float64",
    ("uint32", "complex128"): "complex128",
    ("uint64", "int8"): "float64",
    ("uint64", "int16"): "float64",
    ("uint64", "int32"): "float64",
    ("uint64", "int64"): "float64",
    ("uint64", "float64"): "float64",
    ("uint64", "complex128"): "complex128",
    ("float16", "bfloat16"): "float32",
    ("float16", "complex64"): "complex64",
    ("float16", "complex128"): "complex128",
    ("float32", "complex64"): "complex64",
    ("float32", "complex128"): "complex128",
    ("float64", "complex64"): "complex128",
    ("float64", "complex128"): "complex128",
    ("bfloat16", "float16"): "float32",
    ("bfloat16", "float32"): "float32",
    ("bfloat16", "float64"): "float64",
    ("bfloat16", "bfloat16"): "bfloat16",
    ("bfloat16", "complex64"): "complex64",
    ("bfloat16", "complex128"): "complex128",
    ("complex64", "float64"): "complex128",
    ("complex64", "complex64"): "complex64",
    ("complex64", "complex128"): "complex128",
    ("complex128", "complex128"): "complex128",
    ("float16", "int16"): "float32",
    ("float16", "int32"): "float64",
    ("float16", "int64"): "float64",
    ("float16", "uint16"): "float32",
    ("float16", "uint32"): "float64",
    ("float16", "uint64"): "float64",
    ("float32", "int32"): "float64",
    ("float32", "int64"): "float64",
    ("float32", "uint32"): "float64",
    ("float32", "uint64"): "float64",
    ("bfloat16", "int16"): "float32",
    ("bfloat16", "int32"): "float64",
    ("bfloat16", "int64"): "float64",
    ("bfloat16", "uint16"): "float32",
    ("bfloat16", "uint32"): "float64",
    ("bfloat16", "uint64"): "float64",
    ("complex64", "int32"): "complex128",
    ("complex64", "int64"): "complex128",
    ("complex64", "uint32"): "complex128",
    ("complex64", "uint64"): "complex128",
}
array_api_promotion_table = {
    ("bool", "bool"): "bool",
    ("int8", "int8"): "int8",
    ("int8", "int16"): "int16",
    ("int8", "int32"): "int32",
    ("int8", "int64"): "int64",
    ("int16", "int16"): "int16",
    ("int16", "int32"): "int32",
    ("int16", "int64"): "int64",
    ("int32", "int32"): "int32",
    ("int32", "int64"): "int64",
    ("int64", "int64"): "int64",
    ("uint8", "int8"): "int16",
    ("uint8", "int16"): "int16",
    ("uint8", "int32"): "int32",
    ("uint8", "int64"): "int64",
    ("uint8", "uint8"): "uint8",
    ("uint8", "uint16"): "uint16",
    ("uint8", "uint32"): "uint32",
    ("uint8", "uint64"): "uint64",
    ("uint16", "int8"): "int32",
    ("uint16", "int16"): "int32",
    ("uint16", "int32"): "int32",
    ("uint16", "int64"): "int64",
    ("uint16", "uint16"): "uint16",
    ("uint16", "uint32"): "uint32",
    ("uint16", "uint64"): "uint64",
    ("uint32", "int8"): "int64",
    ("uint32", "int16"): "int64",
    ("uint32", "int32"): "int64",
    ("uint32", "int64"): "int64",
    ("uint32", "uint32"): "uint32",
    ("uint32", "uint64"): "uint64",
    ("uint64", "uint64"): "uint64",
    ("float16", "float16"): "float16",
    ("float16", "float32"): "float32",
    ("float16", "float64"): "float64",
    ("float32", "float32"): "float32",
    ("float32", "float64"): "float64",
    ("float64", "float64"): "float64",
}
tf.experimental.numpy.experimental_enable_numpy_behavior(True)
default_complex_dtype_stack = []
default_dtype_stack = []
default_float_dtype_stack = []
ivy_dtype_dict = {
    tensorflow.int8: "int8",
    tensorflow.int16: "int16",
    tensorflow.int32: "int32",
    tensorflow.int64: "int64",
    tensorflow.uint8: "uint8",
    tensorflow.uint16: "uint16",
    tensorflow.uint32: "uint32",
    tensorflow.uint64: "uint64",
    tensorflow.bfloat16: "bfloat16",
    tensorflow.float16: "float16",
    tensorflow.float32: "float32",
    tensorflow.float64: "float64",
    tensorflow.complex64: "complex64",
    tensorflow.complex128: "complex128",
    tensorflow.bool: "bool",
}
default_int_dtype_stack = []
backend = ""
native_dtype_dict = {
    "int8": tensorflow.int8,
    "int16": tensorflow.int16,
    "int32": tensorflow.int32,
    "int64": tensorflow.int64,
    "uint8": tensorflow.uint8,
    "uint16": tensorflow.uint16,
    "uint32": tensorflow.uint32,
    "uint64": tensorflow.uint64,
    "bfloat16": tensorflow.bfloat16,
    "float16": tensorflow.float16,
    "float32": tensorflow.float32,
    "float64": tensorflow.float64,
    "complex64": tensorflow.complex64,
    "complex128": tensorflow.complex128,
    "bool": tensorflow.bool,
}
default_device_stack = []
SupportsBufferProtocol = TypeVar("SupportsBufferProtocol")
default_uint_dtype_stack = []


def tensorflow_handle_array_like_without_promotion(fn: Callable):
    @functools.wraps(fn)
    def _handle_array_like_without_promotion(*args, **kwargs):
        args = list(args)
        num_args = len(args)
        try:
            type_hints = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return fn(*args, **kwargs)
        parameters = list(type_hints.keys())
        annotations = [param.annotation for param in type_hints.values()]
        device = tensorflow__get_preferred_device(args, kwargs)
        for i, (annotation, parameter, arg) in enumerate(
            zip(annotations, parameters, args)
        ):
            annotation_str = str(annotation)
            if (
                ("rray" in annotation_str or "Tensor" in annotation_str)
                and parameter != "out"
                and all(
                    sq not in annotation_str
                    for sq in ["Sequence", "List", "Tuple", "float", "int", "bool"]
                )
            ):
                if i < num_args:
                    if tensorflow__check_in_nested_sequence(
                        arg, value=Ellipsis, _type=slice
                    ):
                        continue
                    if not tensorflow_is_array(arg):
                        args = tensorflow_set_item(
                            args, i, tensorflow_asarray(arg, device=device)
                        )
                elif parameters in kwargs:
                    kwarg = tensorflow_get_item(kwargs, parameter)
                    if not tensorflow_is_array(kwarg):
                        kwargs = tensorflow_set_item(
                            kwargs, parameter, tensorflow_asarray(kwarg, device=device)
                        )
        return fn(*args, **kwargs)

    _handle_array_like_without_promotion.handle_array_like_without_promotion = True
    return _handle_array_like_without_promotion


def tensorflow_stack(
    arrays: Union[Tuple[tensorflow.Tensor], List[tensorflow.Tensor]],
    /,
    *,
    axis: int = 0,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    try:
        return tensorflow.experimental.numpy.stack(arrays, axis)
    except ValueError as e:
        raise Exception(e) from e


def tensorflow_stack_1(
    self: tensorflow.Tensor,
    /,
    arrays: Union[
        Tuple[Union[tensorflow.Tensor, tf.Tensor]],
        List[Union[tensorflow.Tensor, tf.Tensor]],
    ],
    *,
    axis: int = 0,
    out: Optional[tensorflow.Tensor] = None,
):
    if not isinstance(arrays, (tuple, list)):
        arrays = [arrays]
    if isinstance(arrays, tuple):
        x = (self,) + arrays
    else:
        x = [self] + arrays
    return tensorflow_stack(x, axis=axis, out=out)


def tensorflow_is_native_array(x, /, *, exclusive=False):
    if isinstance(x, (tensorflow.Tensor, tensorflow.Variable, tensorflow.TensorArray)):
        if exclusive and isinstance(x, tensorflow.Variable):
            return False
        return True
    return False


def tensorflow_is_ivy_array(
    x: Union[tensorflow.Tensor, tf.Tensor], /, *, exclusive: Optional[bool] = False
):
    return isinstance(x, tensorflow.Tensor) and tensorflow_is_native_array(
        x, exclusive=exclusive
    )


def tensorflow_is_array(x: Any, /, *, exclusive: bool = False):
    return tensorflow_is_ivy_array(
        x, exclusive=exclusive
    ) or tensorflow_is_native_array(x, exclusive=exclusive)


def tensorflow_exists(x: Any, /):
    return x is not None


def tensorflow_default(
    x: Any,
    /,
    default_val: Any,
    *,
    catch_exceptions: bool = False,
    rev: bool = False,
    with_callable: bool = False,
):
    with_callable = catch_exceptions or with_callable
    if rev:
        x, default_val = default_val, x
    if with_callable:
        x_callable = callable(x)
        default_callable = callable(default_val)
    else:
        x_callable = False
        default_callable = False
    if catch_exceptions:
        try:
            x = x() if x_callable else x
        except Exception:
            return default_val() if default_callable else default_val
    else:
        x = x() if x_callable else x
    return (
        x
        if tensorflow_exists(x)
        else default_val()
        if default_callable
        else default_val
    )


def tensorflow_nested_argwhere(
    nest: Iterable,
    fn: Callable,
    check_nests: bool = False,
    to_ignore: Optional[Union[type, Tuple[type]]] = None,
    _index: Optional[List] = None,
    _base: bool = True,
    stop_after_n_found: Optional[int] = None,
):
    to_ignore = tensorflow_default(to_ignore, ())
    _index = [] if _index is None else _index
    if isinstance(nest, (tuple, list)) and not isinstance(nest, to_ignore):
        n = 0
        _indices = []
        for i, item in enumerate(nest):
            ind = (
                tensorflow_nested_argwhere(
                    item,
                    fn,
                    check_nests,
                    to_ignore,
                    _index + [i],
                    False,
                    stop_after_n_found - n,
                )
                if stop_after_n_found is not None
                else tensorflow_nested_argwhere(
                    item, fn, check_nests, to_ignore, _index + [i], False, None
                )
            )
            if stop_after_n_found is not None and ind:
                if n >= stop_after_n_found:
                    break
                n = n + len(ind)
            _indices = _indices + [ind]
            if stop_after_n_found is not None and n >= stop_after_n_found:
                break
        _indices = [idx for idxs in _indices if idxs for idx in idxs]
        if check_nests and fn(nest):
            _indices.append(_index)
    elif isinstance(nest, (dict, UserDict)) and not isinstance(nest, to_ignore):
        n = 0
        _indices = []
        for k, v in nest.items():
            ind = (
                tensorflow_nested_argwhere(
                    v,
                    fn,
                    check_nests,
                    to_ignore,
                    _index + [k],
                    False,
                    stop_after_n_found - n,
                )
                if stop_after_n_found is not None
                else tensorflow_nested_argwhere(
                    v, fn, check_nests, to_ignore, _index + [k], False, None
                )
            )
            if stop_after_n_found is not None and ind:
                if n >= stop_after_n_found:
                    break
                n = n + len(ind)
            _indices = _indices + [ind]
        _indices = [idx for idxs in _indices if idxs for idx in idxs]
        if check_nests and fn(nest):
            _indices.append(_index)
    else:
        cond_met = fn(nest)
        if cond_met:
            return [_index]
        return False
    return [index for index in _indices if index]


def tensorflow__check_float64(input):
    if tensorflow_is_array(input):
        return tensorflow_dtype(input) == "float64"
    if math.isfinite(input):
        m, e = math.frexp(input)
        return abs(input) > 3.4028235e38 or e < -126 or e > 128
    return False


def tensorflow_as_ivy_dtype(dtype_in: Union[str, str], /):
    return tensorflow_as_ivy_dtype_1(dtype_in)


def tensorflow_is_complex_dtype(
    dtype_in: Union[str, str, tensorflow.Tensor, tf.Tensor, Number], /
):
    if tensorflow_is_array(dtype_in):
        dtype_in = tensorflow_dtype(dtype_in)
    elif isinstance(dtype_in, tuple):
        dtype_in = tensorflow_default_int_dtype()
    elif isinstance(dtype_in, np.ndarray):
        return "complex" in dtype_in.dtype.name
    elif isinstance(dtype_in, Number):
        return isinstance(dtype_in, (complex, np.complexfloating))
    elif isinstance(dtype_in, (list, tuple, dict)):
        return tensorflow_nested_argwhere(
            dtype_in,
            lambda x: isinstance(x, (complex, np.complexfloating))
            or tensorflow_is_array(x)
            and "complex" in tensorflow_dtype(x),
        )
    return "complex" in tensorflow_as_ivy_dtype(dtype_in)


def tensorflow_real(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    return tensorflow.math.real(x)


def tensorflow_real_1(self):
    return tensorflow_real(self)


def tensorflow_imag(
    val: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    return tensorflow.math.imag(val, name=None)


def tensorflow_imag_1(self):
    return tensorflow_imag(self)


def tensorflow__check_complex128(input):
    if tensorflow_is_array(input):
        return tensorflow_dtype(input) == "complex128"
    elif isinstance(input, np.ndarray):
        return str(input.dtype) == "complex128"
    if hasattr(input, "real") and hasattr(input, "imag"):
        return tensorflow__check_float64(
            tensorflow_real_1(input)
        ) and tensorflow__check_float64(tensorflow_imag_1(input))
    return False


def tensorflow_default_complex_dtype(
    *,
    input: Optional[Union[tensorflow.Tensor, tf.Tensor]] = None,
    complex_dtype: Optional[Union[str, tf.DType]] = None,
    as_native: bool = False,
):
    global default_complex_dtype_stack
    if tensorflow_exists(complex_dtype):
        if as_native is True:
            return tensorflow_as_native_dtype(complex_dtype)
        return str(tensorflow_as_ivy_dtype_1(complex_dtype))
    as_native = tensorflow_default(as_native, False)
    if tensorflow_exists(input):
        if tensorflow_is_array(input):
            ret = tensorflow_dtype(input)
        elif isinstance(input, np.ndarray):
            ret = str(input.dtype)
        elif isinstance(input, (list, tuple, dict)):
            if tensorflow_nested_argwhere(
                input, lambda x: tensorflow__check_complex128(x), stop_after_n_found=1
            ):
                ret = tf.complex128
            elif not default_complex_dtype_stack:
                def_dtype = tensorflow_default_dtype()
                if tensorflow_is_complex_dtype(def_dtype):
                    ret = def_dtype
                else:
                    ret = "complex64"
            else:
                ret = default_complex_dtype_stack[-1]
        elif isinstance(input, Number):
            if tensorflow__check_complex128(input):
                ret = tf.complex128
            elif not default_complex_dtype_stack:
                def_dtype = tensorflow_default_dtype()
                if tensorflow_is_complex_dtype(def_dtype):
                    ret = def_dtype
                else:
                    ret = "complex64"
            else:
                ret = default_complex_dtype_stack[-1]
    elif not default_complex_dtype_stack:
        def_dtype = tensorflow_default_dtype()
        if tensorflow_is_complex_dtype(def_dtype):
            ret = def_dtype
        else:
            ret = "complex64"
    else:
        ret = default_complex_dtype_stack[-1]
    if as_native:
        return tensorflow_as_native_dtype(ret)
    return str(tensorflow_as_ivy_dtype_1(ret))


def tensorflow_is_float_dtype(
    dtype_in: Union[str, str, tensorflow.Tensor, tf.Tensor, Number], /
):
    if tensorflow_is_array(dtype_in):
        dtype_in = tensorflow_dtype(dtype_in)
    elif isinstance(dtype_in, tuple):
        dtype_in = tensorflow_default_int_dtype()
    elif isinstance(dtype_in, np.ndarray):
        return "float" in dtype_in.dtype.name
    elif isinstance(dtype_in, Number):
        return isinstance(dtype_in, (float, np.floating))
    elif isinstance(dtype_in, (list, tuple, dict)):
        return bool(
            tensorflow_nested_argwhere(
                dtype_in,
                lambda x: isinstance(x, (float, np.floating))
                or tensorflow_is_array(x)
                and "float" in tensorflow_dtype(x),
            )
        )
    return "float" in tensorflow_as_ivy_dtype(dtype_in)


def tensorflow_is_uint_dtype(
    dtype_in: Union[str, str, tensorflow.Tensor, tf.Tensor, Number], /
):
    if tensorflow_is_array(dtype_in):
        dtype_in = tensorflow_dtype(dtype_in)
    elif isinstance(dtype_in, tuple):
        dtype_in = tensorflow_default_int_dtype()
    elif isinstance(dtype_in, np.ndarray):
        return "uint" in dtype_in.dtype.name
    elif isinstance(dtype_in, Number):
        return isinstance(dtype_in, np.unsignedinteger)
    elif isinstance(dtype_in, (list, tuple, dict)):
        return tensorflow_nested_argwhere(
            dtype_in,
            lambda x: isinstance(x, np.unsignedinteger)
            or tensorflow_is_array(x)
            and "uint" in tensorflow_dtype(x),
        )
    return "uint" in tensorflow_as_ivy_dtype(dtype_in)


def tensorflow_is_int_dtype(
    dtype_in: Union[str, str, tensorflow.Tensor, tf.Tensor, Number], /
):
    if tensorflow_is_array(dtype_in):
        dtype_in = tensorflow_dtype(dtype_in)
    elif isinstance(dtype_in, tuple):
        dtype_in = tensorflow_default_int_dtype()
    elif isinstance(dtype_in, np.ndarray):
        return "int" in dtype_in.dtype.name
    elif isinstance(dtype_in, Number):
        return isinstance(dtype_in, (int, np.integer)) and not isinstance(
            dtype_in, bool
        )
    elif isinstance(dtype_in, (list, tuple, dict)):

        def nested_fun(x):
            return (
                isinstance(x, (int, np.integer))
                or tensorflow_is_array(x)
                and "int" in tensorflow_dtype(x)
            ) and x is not bool

        return bool(tensorflow_nested_argwhere(dtype_in, nested_fun))
    return "int" in tensorflow_as_ivy_dtype_1(dtype_in)


def tensorflow_default_dtype(
    *,
    dtype: Optional[Union[str, str]] = None,
    item: Optional[Union[tensorflow.Tensor, tf.Tensor]] = None,
    as_native: bool = False,
):
    if tensorflow_exists(dtype):
        if as_native is True:
            return tensorflow_as_native_dtype(dtype)
        return tensorflow_as_ivy_dtype_1(dtype)
    as_native = tensorflow_default(as_native, False)
    if tensorflow_exists(item):
        if hasattr(item, "override_dtype_check"):
            return item.override_dtype_check()
        elif isinstance(item, (list, tuple, dict)) and len(item) == 0:
            pass
        elif tensorflow_is_complex_dtype(item):
            return tensorflow_default_complex_dtype(input=item, as_native=as_native)
        elif tensorflow_is_float_dtype(item):
            return tensorflow_default_float_dtype(input=item, as_native=as_native)
        elif tensorflow_is_uint_dtype(item):
            return tensorflow_default_int_dtype(input=item, as_native=as_native)
        elif tensorflow_is_int_dtype(item):
            return tensorflow_default_int_dtype(input=item, as_native=as_native)
        elif as_native:
            return tensorflow_as_native_dtype("bool")
        else:
            return "bool"
    global default_dtype_stack
    if not default_dtype_stack:
        global default_float_dtype_stack
        if default_float_dtype_stack:
            ret = default_float_dtype_stack[-1]
        else:
            ret = "float32"
    else:
        ret = default_dtype_stack[-1]
    if as_native:
        return tensorflow_as_native_dtype(ret)
    return tensorflow_as_ivy_dtype_1(ret)


def tensorflow_default_float_dtype(
    *,
    input: Optional[Union[tensorflow.Tensor, tf.Tensor]] = None,
    float_dtype: Optional[Union[str, tf.DType]] = None,
    as_native: bool = False,
):
    global default_float_dtype_stack
    if tensorflow_exists(float_dtype):
        if as_native is True:
            return tensorflow_as_native_dtype(float_dtype)
        return str(tensorflow_as_ivy_dtype_1(float_dtype))
    as_native = tensorflow_default(as_native, False)
    if tensorflow_exists(input):
        if tensorflow_is_array(input):
            ret = tensorflow_dtype(input)
        elif isinstance(input, np.ndarray):
            ret = str(input.dtype)
        elif isinstance(input, (list, tuple, dict)):
            if tensorflow_nested_argwhere(
                input, lambda x: tensorflow__check_float64(x), stop_after_n_found=1
            ):
                ret = tf.float64
            elif not default_float_dtype_stack:
                def_dtype = tensorflow_default_dtype()
                if tensorflow_is_float_dtype(def_dtype):
                    ret = def_dtype
                else:
                    ret = "float32"
            else:
                ret = default_float_dtype_stack[-1]
        elif isinstance(input, Number):
            if tensorflow__check_float64(input):
                ret = tf.float64
            elif not default_float_dtype_stack:
                def_dtype = tensorflow_default_dtype()
                if tensorflow_is_float_dtype(def_dtype):
                    ret = def_dtype
                else:
                    ret = "float32"
            else:
                ret = default_float_dtype_stack[-1]
    elif not default_float_dtype_stack:
        def_dtype = tensorflow_default_dtype()
        if tensorflow_is_float_dtype(def_dtype):
            ret = def_dtype
        else:
            ret = "float32"
    else:
        ret = default_float_dtype_stack[-1]
    if as_native:
        return tensorflow_as_native_dtype(ret)
    return str(tensorflow_as_ivy_dtype_1(ret))


def tensorflow_as_ivy_dtype_1(
    dtype_in: Union[tensorflow.DType, str, int, float, complex, bool, np.dtype], /
):
    if dtype_in is int:
        return tensorflow_default_int_dtype()
    if dtype_in is float:
        return tensorflow_default_float_dtype()
    if dtype_in is complex:
        return tensorflow_default_complex_dtype()
    if dtype_in is bool:
        return str("bool")
    if isinstance(dtype_in, np.dtype):
        dtype_in = dtype_in.name
    if isinstance(dtype_in, str):
        if dtype_in in native_dtype_dict:
            dtype_str = dtype_in
        else:
            raise Exception(
                f"Cannot convert to ivy dtype. {dtype_in} is not supported by TensorFlow backend."
            )
    else:
        dtype_str = ivy_dtype_dict[dtype_in]
    if "uint" in dtype_str:
        return str(dtype_str)
    elif "int" in dtype_str:
        return str(dtype_str)
    elif "float" in dtype_str:
        return str(dtype_str)
    elif "complex" in dtype_str:
        return str(dtype_str)
    elif "bool" in dtype_str:
        return str("bool")
    else:
        raise Exception(f"Cannot recognize {dtype_str} as a valid Dtype.")


def tensorflow_default_int_dtype(
    *,
    input: Optional[Union[tensorflow.Tensor, tf.Tensor]] = None,
    int_dtype: Optional[Union[str, tf.DType]] = None,
    as_native: bool = False,
):
    global default_int_dtype_stack
    if tensorflow_exists(int_dtype):
        if as_native is True:
            return tensorflow_as_native_dtype(int_dtype)
        return str(tensorflow_as_ivy_dtype_1(int_dtype))
    as_native = tensorflow_default(as_native, False)
    if tensorflow_exists(input):
        if tensorflow_is_array(input):
            ret = tensorflow_dtype(input)
        elif isinstance(input, tuple):
            ret = tensorflow_default_int_dtype()
        elif isinstance(input, np.ndarray):
            ret = str(input.dtype)
        elif isinstance(input, (list, tuple, dict)):
            if tensorflow_nested_argwhere(
                input,
                lambda x: tensorflow_dtype(x) == "uint64"
                if tensorflow_is_array(x)
                else x > 9223372036854775807 and x != math.inf,
                stop_after_n_found=1,
            ):
                ret = tf.uint64
            elif tensorflow_nested_argwhere(
                input,
                lambda x: tensorflow_dtype(x) == "int64"
                if tensorflow_is_array(x)
                else x > 2147483647 and x != math.inf,
                stop_after_n_found=1,
            ):
                ret = tf.int64
            elif not default_int_dtype_stack:
                def_dtype = tensorflow_default_dtype()
                if tensorflow_is_int_dtype(def_dtype):
                    ret = def_dtype
                else:
                    ret = "int32"
            else:
                ret = default_int_dtype_stack[-1]
        elif isinstance(input, Number):
            if input > 9223372036854775807 and input != math.inf and backend != "torch":
                ret = tf.uint64
            elif input > 2147483647 and input != math.inf:
                ret = tf.int64
            elif not default_int_dtype_stack:
                def_dtype = tensorflow_default_dtype()
                if tensorflow_is_int_dtype(def_dtype):
                    ret = def_dtype
                else:
                    ret = "int32"
            else:
                ret = default_int_dtype_stack[-1]
    elif not default_int_dtype_stack:
        def_dtype = tensorflow_default_dtype()
        if tensorflow_is_int_dtype(def_dtype):
            ret = def_dtype
        else:
            ret = "int32"
    else:
        ret = default_int_dtype_stack[-1]
    if as_native:
        return tensorflow_as_native_dtype(ret)
    return str(tensorflow_as_ivy_dtype_1(ret))


def tensorflow_as_native_dtype(
    dtype_in: Union[tensorflow.DType, str, bool, int, float, np.dtype],
):
    if dtype_in is int:
        return tensorflow_default_int_dtype(as_native=True)
    if dtype_in is float:
        return tensorflow_default_float_dtype(as_native=True)
    if dtype_in is complex:
        return tensorflow_default_complex_dtype(as_native=True)
    if dtype_in is bool:
        return tensorflow.bool
    if isinstance(dtype_in, np.dtype):
        dtype_in = dtype_in.name
    if not isinstance(dtype_in, str):
        return dtype_in
    if dtype_in in native_dtype_dict:
        return native_dtype_dict[str(dtype_in)]
    else:
        raise Exception(
            f"Cannot convert to TensorFlow dtype. {dtype_in} is not supported by TensorFlow."
        )


def tensorflow_dtype(
    x: Union[tensorflow.Tensor, tensorflow.Variable, np.ndarray],
    *,
    as_native: bool = False,
):
    if as_native:
        return tensorflow_as_native_dtype(x.dtype)
    return tensorflow_as_ivy_dtype_1(x.dtype)


def tensorflow_is_bool_dtype(
    dtype_in: Union[str, str, tensorflow.Tensor, tf.Tensor, Number], /
):
    if tensorflow_is_array(dtype_in):
        dtype_in = tensorflow_dtype(dtype_in)
    elif isinstance(dtype_in, np.ndarray):
        return "bool" in dtype_in.dtype.name
    elif isinstance(dtype_in, Number):
        return isinstance(dtype_in, (bool, np.bool_)) and not isinstance(dtype_in, bool)
    elif isinstance(dtype_in, (list, tuple, dict)):
        return bool(
            tensorflow_nested_argwhere(
                dtype_in, lambda x: isinstance(x, (bool, np.bool_)) and x is not int
            )
        )
    return "bool" in tensorflow_as_ivy_dtype_1(dtype_in)


def tensorflow_handle_methods(fn):
    def extract_function_name(s):
        match = re.search("_(.+?)(?:_\\d+)?$", s)
        if match:
            return match.group(1)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if tensorflow_is_array(args[0]):
            return fn(*args, **kwargs)
        else:
            fn_name = extract_function_name(fn.__name__)
            new_fn = getattr(args[0], fn_name)
            return new_fn(*args[1:], **kwargs)

    return wrapper


@tensorflow_handle_methods
def tensorflow___getitem__(self, query):
    return tensorflow_get_item(self, query)


def tensorflow_handle_get_item(fn):
    @functools.wraps(fn)
    def wrapper(inp, query, **kwargs):
        try:
            res = tensorflow___getitem__(inp, query)
        except IndexError:
            raise
        except Exception:
            res = fn(inp, query, **kwargs)
        return res

    return wrapper


@tensorflow_handle_get_item
def tensorflow_get_item(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    query: Union[tensorflow.Tensor, tensorflow.Variable, Tuple],
    *,
    copy: Optional[bool] = None,
):
    if (
        tensorflow_is_array(query)
        and tensorflow_is_bool_dtype(query)
        and not len(query.shape)
    ):
        return tensorflow.expand_dims(x, 0)
    return x[query]


def tensorflow_index_nest(
    nest: Union[List, Tuple, Dict, tensorflow.Tensor, tf.Tensor, dict],
    index: Union[List[int], Tuple[int], Iterable[int]],
    /,
):
    ret = nest
    for i in index:
        ret = tensorflow_get_item(ret, i)
    return ret


def tensorflow__get_first_array(*args, **kwargs):
    def array_fn(x):
        return (
            tensorflow_is_array(x)
            if not hasattr(x, "_ivy_array")
            else tensorflow_is_array(x.ivy_array)
        )

    array_fn = array_fn if "array_fn" not in kwargs else kwargs["array_fn"]
    arr = None
    if args:
        arr_idxs = tensorflow_nested_argwhere(args, array_fn, stop_after_n_found=1)
        if arr_idxs:
            arr = tensorflow_index_nest(args, arr_idxs[0])
        else:
            arr_idxs = tensorflow_nested_argwhere(
                kwargs, array_fn, stop_after_n_found=1
            )
            if arr_idxs:
                arr = tensorflow_index_nest(kwargs, arr_idxs[0])
    elif kwargs:
        arr_idxs = tensorflow_nested_argwhere(kwargs, array_fn, stop_after_n_found=1)
        if arr_idxs:
            arr = tensorflow_index_nest(kwargs, arr_idxs[0])
    return arr


def tensorflow_as_native_dev(device: str, /):
    if isinstance(device, str) and "/" in device:
        return device
    ret = f"/{str(device).upper()}"
    if not ret[-1].isnumeric():
        ret += ":0"
    return ret


@tensorflow_handle_methods
def tensorflow_split(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    copy: Optional[bool] = None,
    num_or_size_splits: Optional[
        Union[int, Sequence[int], Union[tensorflow.Tensor, tensorflow.Variable]]
    ] = None,
    axis: int = 0,
    with_remainder: bool = False,
):
    if x.shape == ():
        if num_or_size_splits is not None and num_or_size_splits != 1:
            raise Exception(
                f"input array had no shape, but num_sections specified was {num_or_size_splits}"
            )
        return [x]
    if num_or_size_splits is None:
        dim_size = tensorflow.shape(x)[axis]
        num_or_size_splits = int(dim_size)
    if isinstance(num_or_size_splits, (tensorflow.Tensor, tensorflow.Variable)):
        num_or_size_splits = tensorflow.cast(num_or_size_splits, tensorflow.int32)
    elif isinstance(num_or_size_splits, int) and with_remainder:
        num_chunks = x.shape[axis] / num_or_size_splits
        num_chunks_int = math.floor(num_chunks)
        remainder = num_chunks - num_chunks_int
        if remainder != 0:
            num_or_size_splits = [num_or_size_splits] * num_chunks_int + [
                int(remainder * num_or_size_splits)
            ]
    return tensorflow.split(x, num_or_size_splits, axis)


@tensorflow_handle_methods
def tensorflow_split_1(
    self: tensorflow.Tensor,
    /,
    *,
    copy: Optional[bool] = None,
    num_or_size_splits: Optional[
        Union[int, Sequence[int], tensorflow.Tensor, tf.Tensor]
    ] = None,
    axis: int = 0,
    with_remainder: bool = False,
):
    return tensorflow_split(
        self,
        copy=copy,
        num_or_size_splits=num_or_size_splits,
        axis=axis,
        with_remainder=with_remainder,
    )


def tensorflow_as_ivy_dev(device: str, /):
    if isinstance(device, str) and "/" not in device:
        return str(device)
    dev_in_split = tensorflow_split_1(device[1:], ":")[-2:]
    if len(dev_in_split) == 1:
        return str(dev_in_split[0])
    dev_type, dev_idx = dev_in_split[0], dev_in_split[1]
    dev_type = dev_type.lower()
    if dev_type == "cpu":
        return str(dev_type)
    return str(f"{dev_type}:{dev_idx}")


def tensorflow_dev(
    x: Union[tensorflow.Tensor, tensorflow.Variable, tensorflow.TensorArray],
    /,
    *,
    as_native: bool = False,
):
    if isinstance(x, tensorflow.TensorArray):
        x = tensorflow_stack_1(x)
    dv = x.device
    if as_native:
        return dv
    dv = dv if dv else tensorflow_default_device(as_native=False)
    return tensorflow_as_ivy_dev(dv)


def tensorflow_default_device(
    device: Optional[Union[str, str]] = None,
    /,
    *,
    item: Optional[Union[list, tuple, dict, tensorflow.Tensor, tf.Tensor]] = None,
    as_native: Optional[bool] = None,
):
    if tensorflow_exists(device):
        if as_native is True:
            return tensorflow_as_native_dev(device)
        elif as_native is False:
            return tensorflow_as_ivy_dev(device)
        return device
    as_native = tensorflow_default(as_native, False)
    if tensorflow_exists(item):
        if isinstance(item, (list, tuple, dict)) and len(item) == 0:
            pass
        elif tensorflow_is_array(item):
            return tensorflow_dev(item, as_native=as_native)
    global default_device_stack
    if not default_device_stack:
        ret = "cpu"
    else:
        ret = default_device_stack[-1]
    if as_native:
        return tensorflow_as_native_dev(ret)
    return tensorflow_as_ivy_dev(ret)


def tensorflow__get_preferred_device(args, kwargs):
    device = None
    if "device" in kwargs and kwargs["device"] is not None:
        return device
    if not False:
        arr_arg = tensorflow__get_first_array(*args, **kwargs)
        return tensorflow_default_device(item=arr_arg, as_native=True)
    return tensorflow_default_device(as_native=True)


def tensorflow__check_in_nested_sequence(sequence, value=None, _type=None):
    if sequence is value or isinstance(sequence, _type):
        return True
    elif isinstance(sequence, (tuple, list)):
        if any(isinstance(_val, _type) or _val is value for _val in sequence):
            return True
        else:
            return any(
                tensorflow__check_in_nested_sequence(sub_sequence, value, _type)
                for sub_sequence in sequence
                if isinstance(sub_sequence, (tuple, list))
            )


def tensorflow_nested_map(
    fn: Callable,
    x: Union[tensorflow.Tensor, tf.Tensor, Iterable],
    /,
    include_derived: Optional[Union[Dict[str, bool], bool]] = None,
    to_ignore: Optional[Union[type, Tuple[type]]] = None,
    to_mutable: bool = False,
    _tuple_check_fn: Optional[Callable] = None,
    _list_check_fn: Optional[Callable] = None,
    _dict_check_fn: Optional[Callable] = None,
    shallow: bool = True,
):
    to_ignore = tensorflow_default(to_ignore, ())
    if include_derived is True:
        include_derived = {"tuple": True, "list": True, "dict": True}
    elif not include_derived:
        include_derived = {}
    for t in ("tuple", "list", "dict"):
        if t not in include_derived:
            include_derived = tensorflow_set_item(include_derived, t, False)
    class_instance = type(x)
    if (
        hasattr(x, "is_tracked_proxy")
        and hasattr(class_instance, "__bases__")
        and not set(class_instance.__bases__).intersection(set(to_ignore))
    ):
        to_ignore = to_ignore + (class_instance,)
    tuple_check_fn = tensorflow_default(
        _tuple_check_fn,
        (lambda x_, t_: isinstance(x_, t_))
        if include_derived["tuple"]
        else lambda x_, t_: type(x_) is t_,
    )
    list_check_fn = tensorflow_default(
        _list_check_fn,
        (lambda x_, t_: isinstance(x_, t_))
        if include_derived["list"]
        else lambda x_, t_: type(x_) is t_,
    )
    dict_check_fn = tensorflow_default(
        _dict_check_fn,
        (lambda x_, t_: isinstance(x_, t_))
        if include_derived["dict"]
        else lambda x_, t_: type(x_) is t_,
    )
    if tuple_check_fn(x, tuple) and not isinstance(x, to_ignore):
        ret_list = [
            tensorflow_nested_map(
                fn,
                i,
                include_derived,
                to_ignore,
                to_mutable,
                tuple_check_fn,
                list_check_fn,
                dict_check_fn,
                shallow,
            )
            for i in x
        ]
        if to_mutable:
            return ret_list
        elif hasattr(x, "_fields"):
            return class_instance(**dict(zip(x._fields, ret_list)))
        else:
            return class_instance(ret_list)
    elif list_check_fn(x, list) and not isinstance(x, to_ignore):
        ret_list = [
            tensorflow_nested_map(
                fn,
                i,
                include_derived,
                to_ignore,
                to_mutable,
                tuple_check_fn,
                list_check_fn,
                dict_check_fn,
                shallow,
            )
            for i in x
        ]
        if shallow:
            x = tensorflow_set_item(x, slice(None, None, None), ret_list[:])
            return x
        return class_instance(ret_list)
    elif (dict_check_fn(x, dict) or isinstance(x, UserDict)) and not isinstance(
        x, to_ignore
    ):
        class_instance = type(x)
        ret = {
            k: tensorflow_nested_map(
                fn,
                v,
                include_derived,
                to_ignore,
                to_mutable,
                tuple_check_fn,
                list_check_fn,
                dict_check_fn,
                shallow,
            )
            for k, v in x.items()
        }
        if shallow:
            x.update(ret)
            return x
        return class_instance(ret)
    elif isinstance(x, slice):
        return slice(*tensorflow_nested_map(fn, [x.start, x.stop, x.step]))
    return fn(x)


def tensorflow__to_ivy(x: Any):
    if isinstance(x, tensorflow.Tensor):
        return x
    elif isinstance(x, tf.TensorShape):
        return tuple(x)
    elif isinstance(x, dict):
        return x.to_ivy()
    if tensorflow_is_native_array(x) or isinstance(x, np.ndarray):
        return tensorflow.convert_to_tensor(x)
    return x


def tensorflow_to_ivy(
    x: Union[tensorflow.Tensor, tf.Tensor, Iterable],
    nested: bool = False,
    include_derived: Optional[Dict[str, bool]] = None,
):
    if nested:
        return tensorflow_nested_map(
            tensorflow__to_ivy, x, include_derived, shallow=False
        )
    return tensorflow__to_ivy(x)


def tensorflow__asarray_to_native_arrays_and_back(fn: Callable):
    @functools.wraps(fn)
    def _asarray_to_native_arrays_and_back_wrapper(*args, dtype=None, **kwargs):
        new_arg = args[0]
        new_args = (new_arg,) + args[1:]
        if dtype is not None:
            dtype = tensorflow_default_dtype(dtype=dtype, as_native=True)
        return tensorflow_to_ivy(fn(*new_args, dtype=dtype, **kwargs))

    _asarray_to_native_arrays_and_back_wrapper._asarray_to_native_arrays_and_back = True
    return _asarray_to_native_arrays_and_back_wrapper


def tensorflow__flatten_nest(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from tensorflow__flatten_nest(x)
        else:
            yield x


def tensorflow_promote_types(
    type1: Union[str, tf.DType],
    type2: Union[str, tf.DType],
    /,
    *,
    array_api_promotion: bool = False,
):
    if not (type1 and type2):
        return type1 if type1 else type2
    query = [tensorflow_as_ivy_dtype_1(type1), tensorflow_as_ivy_dtype_1(type2)]
    query = tuple(query)
    if query not in promotion_table:
        query = query[1], query[0]

    def _promote(query):
        if array_api_promotion:
            return tensorflow_get_item(array_api_promotion_table, query)
        return tensorflow_get_item(promotion_table, query)

    return _promote(query)


def tensorflow__asarray_infer_dtype(fn: Callable):
    @functools.wraps(fn)
    def _asarray_infer_dtype_wrapper(*args, dtype=None, **kwargs):
        def _infer_dtype(obj):
            if isinstance(obj, tf.TensorShape):
                obj = list(obj)
            if hasattr(obj, "dtype"):
                return obj.dtype.name if isinstance(obj, np.ndarray) else obj.dtype
            else:
                return tensorflow_default_dtype(item=obj)

        if not tensorflow_exists(dtype):
            arr = args[0]
            dtype_list = [
                tensorflow_nested_map(lambda x: _infer_dtype(x), arr, shallow=False)
            ]
            dtype_list = tensorflow__flatten_nest(dtype_list)
            dtype_list = list(set(dtype_list))
            if len(dtype_list) != 0:
                dtype = dtype_list[0]
                for dt in dtype_list[1:]:
                    dtype = tensorflow_promote_types(dtype, dt)
            else:
                dtype = tensorflow_default_float_dtype()
            dtype = tensorflow_as_native_dtype(dtype)
        return fn(*args, dtype=dtype, **kwargs)

    _asarray_infer_dtype_wrapper.infer_dtype = True
    return _asarray_infer_dtype_wrapper


@tensorflow_handle_array_like_without_promotion
@tensorflow__asarray_to_native_arrays_and_back
@tensorflow__asarray_infer_dtype
def tensorflow_asarray(
    obj: Union[
        tensorflow.Tensor,
        tensorflow.Variable,
        tensorflow.TensorShape,
        bool,
        int,
        float,
        tensorflow_NestedSequence,
        SupportsBufferProtocol,
        np.ndarray,
    ],
    /,
    *,
    copy: Optional[bool] = None,
    dtype: Optional[tensorflow.DType] = None,
    device: Optional[str] = None,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    with tensorflow.device(device):
        if tensorflow.is_tensor(obj):
            ret = tensorflow.cast(obj, dtype) if obj.dtype != dtype else obj
        elif (
            dtype is not None
            and dtype.is_integer
            and np.issubdtype(np.array(obj).dtype, np.floating)
        ):
            obj_np = np.array(obj)
            ret = tensorflow.convert_to_tensor(obj_np, dtype)
        else:
            ret = tensorflow.convert_to_tensor(obj, dtype)
        return tensorflow.identity(ret) if copy or ret.device != device else ret


@tensorflow_handle_array_like_without_promotion
def tensorflow_unstack(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    copy: Optional[bool] = None,
    axis: int = 0,
    keepdims: bool = False,
):
    if x.shape == ():
        return [x]
    ret = tensorflow.unstack(x, axis=axis)
    if keepdims:
        return [tensorflow.expand_dims(r, axis) for r in ret]
    return ret


def tensorflow_unstack_1(
    self: tensorflow.Tensor,
    /,
    *,
    copy: Optional[bool] = None,
    axis: int = 0,
    keepdims: bool = False,
):
    return tensorflow_unstack(self, copy=copy, axis=axis, keepdims=keepdims)


@tensorflow_handle_array_like_without_promotion
def tensorflow_copy_array(
    x: Union[tensorflow.Tensor, tensorflow.Variable, tensorflow.TensorArray],
    *,
    to_ivy_array: bool = True,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    if isinstance(x, tensorflow.TensorArray):
        x_wrapped = tensorflow_stack_1(x)
        y = tensorflow.TensorArray(x.dtype, tensorflow_size_1(x)())
        x = tensorflow_unstack_1(y, tensorflow_copy_array(x_wrapped))
    else:
        x = tensorflow.identity(x)
    if to_ivy_array:
        return tensorflow_to_ivy(x)
    return x


def tensorflow_tile(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    repeats: Sequence[int],
    *,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    if x.shape == ():
        x = tensorflow.reshape(x, (-1,))
    if isinstance(repeats, Number):
        repeats = [repeats]
    if isinstance(repeats, tensorflow.Tensor) and repeats.shape == ():
        repeats = tensorflow.reshape(repeats, (-1,))
    if len(x.shape) < len(repeats):
        while len(x.shape) != len(repeats):
            x = tensorflow.expand_dims(x, 0)
    elif len(x.shape) > len(repeats):
        repeats = list(repeats)
        while len(x.shape) != len(repeats):
            repeats = [1] + repeats
    return tensorflow.tile(x, repeats)


@tensorflow_handle_array_like_without_promotion
def tensorflow_nonzero(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    as_tuple: bool = True,
    size: Optional[int] = None,
    fill_value: Number = 0,
):
    res = tensorflow.experimental.numpy.nonzero(x)
    if size is not None:
        dtype = tensorflow.int64
        if isinstance(fill_value, float):
            dtype = tensorflow.float64
        res = tensorflow.cast(res, dtype)
        diff = size - res[0].shape[0]
        if diff > 0:
            res = tensorflow.pad(res, [[0, 0], [0, diff]], constant_values=fill_value)
        elif diff < 0:
            res = tensorflow.slice(res, [0, 0], [-1, size])
    if as_tuple:
        return tuple(res)
    return tensorflow.stack(res, axis=1)


@tensorflow_handle_array_like_without_promotion
def tensorflow_diff(
    x: Union[tensorflow.Tensor, tensorflow.Variable, list, tuple],
    /,
    *,
    n: int = 1,
    axis: int = -1,
    prepend: Optional[
        Union[tensorflow.Tensor, tensorflow.Variable, int, float, list, tuple]
    ] = None,
    append: Optional[
        Union[tensorflow.Tensor, tensorflow.Variable, int, float, list, tuple]
    ] = None,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    if n == 0:
        return x
    if prepend is not None:
        x = tensorflow.experimental.numpy.append(
            prepend, x, axis=axis if axis != -1 else None
        )
    if append is not None:
        x = tensorflow.experimental.numpy.append(
            x, append, axis=axis if axis != -1 else None
        )
    return tensorflow.experimental.numpy.diff(x, n=n, axis=axis)


def tensorflow__parse_ellipsis(so, ndims):
    pre = list()
    for s in so:
        if s is Ellipsis:
            break
        pre.append(s)
    post = list()
    for s in reversed(so):
        if s is Ellipsis:
            break
        post.append(s)
    ret = list(
        pre
        + [slice(None, None, None) for _ in range(ndims - len(pre) - len(post))]
        + list(reversed(post))
    )
    return ret, (len(pre), ndims - len(post))


def tensorflow_broadcast_arrays(*arrays: Union[tensorflow.Tensor, tensorflow.Variable]):
    if len(arrays) > 1:
        try:
            desired_shape = tensorflow.broadcast_dynamic_shape(
                tensorflow.shape(arrays[0]), tensorflow.shape(arrays[1])
            )
        except tensorflow.errors.InvalidArgumentError as e:
            raise Exception(e) from e
        if len(arrays) > 2:
            for i in range(2, len(arrays)):
                try:
                    desired_shape = tensorflow.broadcast_dynamic_shape(
                        desired_shape, tensorflow.shape(arrays[i])
                    )
                except tensorflow.errors.InvalidArgumentError as e:
                    raise Exception(e) from e
    else:
        return [arrays[0]]
    result = []
    for tensor in arrays:
        result.append(tensorflow.broadcast_to(tensor, desired_shape))
    return result


@tensorflow_handle_array_like_without_promotion
def tensorflow_astype(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    dtype: Union[tf.DType, str],
    /,
    *,
    copy: bool = True,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    dtype = tensorflow_as_native_dtype(dtype)
    if x.dtype == dtype:
        return tensorflow.experimental.numpy.copy(x) if copy else x
    return tensorflow.cast(x, dtype)


def tensorflow_astype_1(
    self: tensorflow.Tensor,
    dtype: str,
    /,
    *,
    copy: bool = True,
    out: Optional[tensorflow.Tensor] = None,
):
    return tensorflow_astype(self, dtype, copy=copy, out=out)


@tensorflow_handle_array_like_without_promotion
def tensorflow_where(
    condition: Union[tensorflow.Tensor, tensorflow.Variable],
    x1: Union[tensorflow.Tensor, tensorflow.Variable],
    x2: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    oirg_x1 = x1
    oirg_x2 = x2
    try:
        dtype = (
            x1.dtype
            if hasattr(x1, "dtype")
            else x2.dtype
            if hasattr(x2, "dtype")
            else tensorflow_default_dtype()
        )
        if not tensorflow_is_array(x1):
            x1 = tensorflow_asarray(x1, dtype=dtype)
        if not tensorflow_is_array(x2):
            x2 = tensorflow_asarray(x2, dtype=dtype)
    except:
        x1 = oirg_x1
        x2 = oirg_x2
    return tensorflow.cast(
        tensorflow.experimental.numpy.where(condition, x1, x2), x1.dtype
    )


@tensorflow_handle_array_like_without_promotion
def tensorflow_arange(
    start: float,
    /,
    stop: Optional[float] = None,
    step: float = 1,
    *,
    dtype: Optional[tensorflow.DType] = None,
    device: Optional[str] = None,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    if stop is None:
        stop = start
        start = 0
    if step > 0 and start > stop or step < 0 and start < stop:
        if isinstance(stop, float):
            stop = float(start)
        else:
            stop = start
    if isinstance(start, (float, int)):
        start = tensorflow.convert_to_tensor(start)
    if isinstance(stop, (float, int)):
        stop = tensorflow.convert_to_tensor(stop)
    if isinstance(step, (float, int)):
        step = tensorflow.convert_to_tensor(step)
    if dtype is None:
        if isinstance(start, int) and isinstance(stop, int) and isinstance(step, int):
            return tensorflow.cast(
                tensorflow.range(start, stop, delta=step, dtype=tensorflow.int64),
                tensorflow.int32,
            )
        else:
            return tensorflow.range(start, stop, delta=step)
    else:
        dtype = tensorflow_as_native_dtype(tensorflow_default_dtype(dtype=dtype))
        if dtype in [
            tensorflow.int8,
            tensorflow.uint8,
            tensorflow.int16,
            tensorflow.uint16,
            tensorflow.uint32,
            tensorflow.uint64,
        ]:
            return tensorflow.cast(
                tensorflow.range(start, stop, delta=step, dtype=tensorflow.int64), dtype
            )
        else:
            return tensorflow.range(start, stop, delta=step, dtype=dtype)


def tensorflow__parse_slice(idx, s):
    step = 1 if idx.step is None else idx.step
    if step > 0:
        start = 0 if idx.start is None else idx.start
        if start >= s:
            stop = start
        else:
            if start <= -s:
                start = 0
            elif start < 0:
                start = start + s
            stop = s if idx.stop is None else idx.stop
            if stop > s:
                stop = s
            elif start <= -s:
                stop = 0
            elif stop < 0:
                stop = stop + s
    else:
        start = s - 1 if idx.start is None else idx.start
        if start < -s:
            stop = start
        else:
            if start >= s:
                start = s - 1
            elif start < 0:
                start = start + s
            if idx.stop is None:
                stop = -1
            else:
                stop = idx.stop
                if stop > s:
                    stop = s
                elif stop < -s:
                    stop = -1
                elif stop == -s:
                    stop = 0
                elif stop < 0:
                    stop = stop + s
    q_i = tensorflow_arange(start, stop, step)
    ag__result_list_0 = []
    for q in q_i:
        if 0 <= q < s:
            res = q
            ag__result_list_0.append(res)
    q_i = ag__result_list_0
    q_i = (
        tensorflow_asarray(q_i)
        if len(q_i) or start == stop or idx.stop is not None
        else tensorflow_arange(0, s, 1)
    )
    return q_i


@tensorflow_handle_array_like_without_promotion
def tensorflow_shape(
    x: Union[tensorflow.Tensor, tensorflow.Variable], /, *, as_array: bool = False
):
    if as_array:
        return tensorflow_asarray(
            tensorflow.shape(x), dtype=tensorflow_default_int_dtype()
        )
    else:
        return tuple(x.shape)


def tensorflow__deep_flatten(iterable):
    def _flatten_gen(iterable):
        for item in iterable:
            if isinstance(item, list):
                yield from _flatten_gen(item)
            else:
                yield item

    return list(_flatten_gen(iterable))


def tensorflow__calculate_out_shape(axis, array_shape):
    if type(axis) not in (tuple, list):
        axis = (axis,)
    out_dims = len(axis) + len(array_shape)
    norm_axis = normalize_axis_tuple(axis, out_dims)
    shape_iter = iter(array_shape)
    ag__result_list_0 = []
    for current_ax in range(out_dims):
        res = 1 if current_ax in norm_axis else next(shape_iter)
        ag__result_list_0.append(res)
    out_shape = ag__result_list_0
    return out_shape


@tensorflow_handle_array_like_without_promotion
def tensorflow_expand_dims(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    copy: Optional[bool] = None,
    axis: Union[int, Sequence[int]] = 0,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    try:
        out_shape = tensorflow__calculate_out_shape(axis, tensorflow.shape(x))
        ret = tensorflow.reshape(x, shape=out_shape)
        return ret
    except (tensorflow.errors.InvalidArgumentError, np.AxisError) as error:
        raise Exception(error) from error


def tensorflow_check_elem_in_list(elem, list, inverse=False, message=""):
    if inverse and elem in list:
        raise Exception(
            message if message != "" else f"{elem} must not be one of {list}"
        )
    elif not inverse and elem not in list:
        raise Exception(message if message != "" else f"{elem} must be one of {list}")


def tensorflow__reshape_fortran_tf(x, shape):
    if len(x.shape) > 0:
        x = tensorflow.transpose(x)
    return tensorflow.transpose(tensorflow.reshape(x, shape[::-1]))


@tensorflow_handle_array_like_without_promotion
def tensorflow_reshape(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    shape: Union[tf.TensorShape, Sequence[int]],
    *,
    copy: Optional[bool] = None,
    order: str = "C",
    allowzero: bool = True,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    tensorflow_check_elem_in_list(order, ["C", "F"])
    if not allowzero:
        shape = [
            (new_s if con else old_s)
            for new_s, con, old_s in zip(
                shape, tensorflow.constant(shape) != 0, x.shape
            )
        ]
    if order == "F":
        return tensorflow__reshape_fortran_tf(x, shape)
    return tensorflow.reshape(x, shape)


def tensorflow_reshape_1(
    self: tensorflow.Tensor,
    /,
    shape: Union[tuple, tf.TensorShape, Sequence[int]],
    *,
    copy: Optional[bool] = None,
    order: str = "C",
    allowzero: bool = True,
    out: Optional[tensorflow.Tensor] = None,
):
    return tensorflow_reshape(
        self, shape, copy=copy, allowzero=allowzero, out=out, order=order
    )


@tensorflow_handle_array_like_without_promotion
def tensorflow_meshgrid(
    *arrays: Union[tensorflow.Tensor, tensorflow.Variable],
    sparse: bool = False,
    indexing: str = "xy",
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    if not sparse:
        return tensorflow.meshgrid(*arrays, indexing=indexing)
    sd = (1,) * len(arrays)
    ag__result_list_0 = []
    for i, a in enumerate(arrays):
        res = tensorflow.reshape(
            tensorflow.convert_to_tensor(a), sd[:i] + (-1,) + sd[i + 1 :]
        )
        ag__result_list_0.append(res)
    res = ag__result_list_0
    if indexing == "xy" and len(arrays) > 1:
        res[0] = tensorflow.reshape(res[0], (1, -1) + sd[2:])
        res[1] = tensorflow.reshape(res[1], (-1, 1) + sd[2:])
    return res


def tensorflow_infer_dtype(fn: Callable):
    @functools.wraps(fn)
    def _infer_dtype(*args, dtype=None, **kwargs):
        arr = (
            None
            if tensorflow_exists(dtype)
            else tensorflow__get_first_array(*args, **kwargs)
        )
        dtype = tensorflow_default_dtype(dtype=dtype, item=arr, as_native=True)
        return fn(*args, dtype=dtype, **kwargs)

    _infer_dtype.infer_dtype = True
    return _infer_dtype


@tensorflow_infer_dtype
@tensorflow_handle_array_like_without_promotion
def tensorflow_empty(
    shape: Union[tf.TensorShape, Sequence[int]],
    *,
    dtype: tensorflow.DType,
    device: Optional[str] = None,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    return tensorflow.experimental.numpy.empty(shape, dtype=tensorflow.float32)


def tensorflow__parse_query(query, x_shape, scatter=False):
    query = (query,) if not isinstance(query, tuple) else query
    ag__result_list_0 = []
    for q in query:
        res = tensorflow_asarray(q) if isinstance(q, (tuple, list, int)) else q
        ag__result_list_0.append(res)
    query = ag__result_list_0
    ag__result_list_1 = []
    for i, q in enumerate(query):
        if tensorflow_is_array(q):
            res = i
            ag__result_list_1.append(res)
    non_slice_q_idxs = ag__result_list_1
    to_front = (
        len(non_slice_q_idxs) > 1
        and any(tensorflow_diff(non_slice_q_idxs) != 1)
        and non_slice_q_idxs[-1] < len(x_shape)
    )
    ag__result_list_2 = []
    for i, q in enumerate(query):
        if q is None:
            res = i
            ag__result_list_2.append(res)
    new_axes = ag__result_list_2
    ag__result_list_3 = []
    for q in query:
        if q is not None:
            res = q
            ag__result_list_3.append(res)
    query = ag__result_list_3
    query = [Ellipsis] if query == [] else query
    ellipsis_inds = None
    if any(q is Ellipsis for q in query):
        query, ellipsis_inds = tensorflow__parse_ellipsis(query, len(x_shape))
    ag__result_list_4 = []
    for i, v in enumerate(query):
        if tensorflow_is_array(v):
            res = i
            ag__result_list_4.append(res)
    array_inds = ag__result_list_4
    if array_inds:
        array_queries = tensorflow_broadcast_arrays(
            *[v for i, v in enumerate(query) if i in array_inds]
        )
        array_queries = [
            (
                tensorflow_nonzero(q, as_tuple=False)[0]
                if tensorflow_is_bool_dtype(q)
                else q
            )
            for q in array_queries
        ]
        array_queries = [
            (
                tensorflow_astype_1(
                    tensorflow_where(
                        arr < 0, arr + tensorflow_get_item(x_shape, i), arr
                    ),
                    tf.int64,
                )
                if tensorflow_size_1(arr)
                else tensorflow_astype_1(arr, tf.int64)
            )
            for arr, i in zip(array_queries, array_inds)
        ]
        for idx, arr in zip(array_inds, array_queries):
            query = tensorflow_set_item(query, idx, arr)
    ag__result_list_5 = []
    for i, q in enumerate(query):
        res = (
            tensorflow_astype_1(
                tensorflow__parse_slice(q, tensorflow_get_item(x_shape, i)), tf.int64
            )
            if isinstance(q, slice)
            else q
        )
        ag__result_list_5.append(res)
    query = ag__result_list_5
    if len(query) < len(x_shape):
        query = query + [
            tensorflow_astype_1(tensorflow_arange(0, s, 1), tf.int64)
            for s in tensorflow_get_item(x_shape, slice(len(query), None, None))
        ]
    if len(array_inds) and to_front:
        target_shape = (
            [list(array_queries[0].shape)]
            + [
                list(tensorflow_get_item(query, i).shape)
                for i in range(len(query))
                if i not in array_inds
            ]
            + [[] for _ in range(len(array_inds) - 1)]
        )
    elif len(array_inds):
        target_shape = (
            [list(tensorflow_get_item(query, i).shape) for i in range(0, array_inds[0])]
            + [list(tensorflow_shape(array_queries[0], as_array=True))]
            + [[] for _ in range(len(array_inds) - 1)]
            + [
                list(tensorflow_shape(tensorflow_get_item(query, i), as_array=True))
                for i in range(array_inds[-1] + 1, len(query))
            ]
        )
    else:
        target_shape = [list(q.shape) for q in query]
    if ellipsis_inds is not None:
        target_shape = (
            tensorflow_get_item(target_shape, slice(None, ellipsis_inds[0], None))
            + [
                tensorflow_get_item(
                    target_shape, slice(ellipsis_inds[0], ellipsis_inds[1], None)
                )
            ]
            + tensorflow_get_item(target_shape, slice(ellipsis_inds[1], None, None))
        )
    for i, ax in enumerate(new_axes):
        if len(array_inds) and to_front:
            ax = ax - (sum(1 for x in array_inds if x < ax) - 1)
            ax = ax + i
        target_shape = [
            *tensorflow_get_item(target_shape, slice(None, ax, None)),
            1,
            *tensorflow_get_item(target_shape, slice(ax, None, None)),
        ]
    target_shape = tensorflow__deep_flatten(target_shape)
    ag__result_list_6 = []
    for q in query:
        res = tensorflow_expand_dims(q) if not len(q.shape) else q
        ag__result_list_6.append(res)
    query = ag__result_list_6
    if len(array_inds):
        array_queries = [
            (
                tensorflow_reshape_1(arr, (-1,))
                if len(arr.shape) > 1
                else tensorflow_expand_dims(arr)
                if not len(arr.shape)
                else arr
            )
            for arr in array_queries
        ]
        array_queries = tensorflow_stack(array_queries, axis=1)
    if len(array_inds) == len(query):
        indices = tensorflow_reshape_1(array_queries, (*target_shape, len(x_shape)))
    elif len(array_inds) == 0:
        indices = tensorflow_reshape_1(
            tensorflow_stack(tensorflow_meshgrid(*query, indexing="ij"), axis=-1),
            (*target_shape, len(x_shape)),
        )
    elif to_front:
        post_array_queries = (
            tensorflow_reshape_1(
                tensorflow_stack(
                    tensorflow_meshgrid(
                        *[v for i, v in enumerate(query) if i not in array_inds],
                        indexing="ij",
                    ),
                    axis=-1,
                ),
                (-1, len(query) - len(array_inds)),
            )
            if len(array_inds) < len(query)
            else tensorflow_empty((1, 0))
        )
        indices = tensorflow_reshape_1(
            tensorflow_asarray(
                [
                    (*arr, *post)
                    for arr, post in itertools.product(
                        array_queries, post_array_queries
                    )
                ]
            ),
            (*target_shape, len(x_shape)),
        )
    else:
        pre_array_queries = (
            tensorflow_reshape_1(
                tensorflow_stack(
                    tensorflow_meshgrid(
                        *[v for i, v in enumerate(query) if i < array_inds[0]],
                        indexing="ij",
                    ),
                    axis=-1,
                ),
                (-1, array_inds[0]),
            )
            if array_inds[0] > 0
            else tensorflow_empty((1, 0))
        )
        post_array_queries = (
            tensorflow_reshape_1(
                tensorflow_stack(
                    tensorflow_meshgrid(
                        *[v for i, v in enumerate(query) if i > array_inds[-1]],
                        indexing="ij",
                    ),
                    axis=-1,
                ),
                (-1, len(query) - 1 - array_inds[-1]),
            )
            if array_inds[-1] < len(query) - 1
            else tensorflow_empty((1, 0))
        )
        indices = tensorflow_reshape_1(
            tensorflow_asarray(
                [
                    (*pre, *arr, *post)
                    for pre, arr, post in itertools.product(
                        pre_array_queries, array_queries, post_array_queries
                    )
                ]
            ),
            (*target_shape, len(x_shape)),
        )
    return (
        tensorflow_astype_1(indices, tf.int64),
        target_shape,
        array_inds if len(array_inds) and to_front else None,
    )


def tensorflow_get_num_dims(x, /, *, as_array=False):
    return (
        tensorflow.cast(tensorflow.shape(tensorflow.shape(x))[0], tensorflow.int64)
        if as_array
        else int(tensorflow.shape(tensorflow.shape(x)))
    )


def tensorflow_to_numpy(
    x: Union[tensorflow.Tensor, tensorflow.Variable], /, *, copy: bool = True
):
    if (
        tensorflow_is_array(x)
        and tensorflow_get_num_dims(x) == 0
        and tensorflow_as_native_dtype(x.dtype) is tensorflow.bfloat16
    ):
        x = tensorflow.expand_dims(x, 0)
        if copy:
            return np.squeeze(np.array(tensorflow.convert_to_tensor(x)), 0)
        else:
            return np.squeeze(np.asarray(tensorflow.convert_to_tensor(x)), 0)
    if copy:
        return np.array(tensorflow.convert_to_tensor(x))
    else:
        return np.asarray(tensorflow.convert_to_tensor(x))


def tensorflow_to_scalar_1(x: Union[tensorflow.Tensor, tensorflow.Variable], /):
    ret = tensorflow_to_numpy(x).item()
    if x.dtype == tensorflow.bfloat16:
        return float(ret)
    return ret


@tensorflow_handle_array_like_without_promotion
def tensorflow_to_scalar(x: Union[tensorflow.Tensor, tf.Tensor], /):
    return tensorflow_to_scalar_1(x)


def tensorflow_default_uint_dtype(
    *,
    input: Optional[Union[tensorflow.Tensor, tf.Tensor]] = None,
    uint_dtype: Optional[Union[str, tf.DType]] = None,
    as_native: bool = False,
):
    global default_uint_dtype_stack
    if tensorflow_exists(uint_dtype):
        if as_native is True:
            return tensorflow_as_native_dtype(uint_dtype)
        return str(tensorflow_as_ivy_dtype_1(uint_dtype))
    as_native = tensorflow_default(as_native, False)
    if tensorflow_exists(input):
        if tensorflow_is_array(input):
            ret = tensorflow_dtype(input)
        elif isinstance(input, np.ndarray):
            ret = input.dtype
        elif isinstance(input, (list, tuple, dict)):

            def is_native(x):
                return tensorflow_is_native_array(x)

            if tensorflow_nested_argwhere(
                input,
                lambda x: tensorflow_dtype(x) == "uint64"
                if is_native(x)
                else x > 9223372036854775807 and x != math.inf,
                stop_after_n_found=1,
            ):
                ret = tf.uint64
            elif default_uint_dtype_stack:
                ret = default_uint_dtype_stack[-1]
            else:
                def_dtype = tensorflow_default_dtype()
                if tensorflow_is_uint_dtype(def_dtype):
                    ret = def_dtype
                else:
                    ret = "uint32"
        elif isinstance(input, Number):
            if input > 4294967295 and input != math.inf and backend != "torch":
                ret = tf.uint64
            elif default_uint_dtype_stack:
                ret = default_uint_dtype_stack[-1]
            else:
                def_dtype = tensorflow_default_dtype()
                if tensorflow_is_uint_dtype(def_dtype):
                    ret = def_dtype
                else:
                    ret = "uint32"
    elif default_uint_dtype_stack:
        ret = default_uint_dtype_stack[-1]
    else:
        def_dtype = tensorflow_default_dtype()
        if tensorflow_is_uint_dtype(def_dtype):
            ret = def_dtype
        else:
            ret = "uint32"
    if as_native:
        return tensorflow_as_native_dtype(ret)
    return str(tensorflow_as_ivy_dtype_1(ret))


def tensorflow_infer_default_dtype(
    dtype: Union[str, tf.DType, str], as_native: bool = False
):
    if tensorflow_is_complex_dtype(dtype):
        default_dtype = tensorflow_default_complex_dtype(as_native=as_native)
    elif tensorflow_is_float_dtype(dtype):
        default_dtype = tensorflow_default_float_dtype(as_native=as_native)
    elif tensorflow_is_uint_dtype(dtype):
        default_dtype = tensorflow_default_uint_dtype(as_native=as_native)
    elif tensorflow_is_int_dtype(dtype):
        default_dtype = tensorflow_default_int_dtype(as_native=as_native)
    elif as_native:
        default_dtype = tensorflow_as_native_dtype("bool")
    else:
        default_dtype = tensorflow_as_ivy_dtype_1("bool")
    return default_dtype


def tensorflow_dtype_bits(dtype_in: Union[tensorflow.DType, str, np.dtype], /):
    dtype_str = tensorflow_as_ivy_dtype_1(dtype_in)
    if "bool" in dtype_str:
        return 1
    return int(
        dtype_str.replace("tf.", "")
        .replace("uint", "")
        .replace("int", "")
        .replace("bfloat", "")
        .replace("float", "")
        .replace("complex", "")
    )


def tensorflow__infer_dtype(dtype: tensorflow.DType):
    default_dtype = tensorflow_infer_default_dtype(dtype)
    if tensorflow_dtype_bits(dtype) < tensorflow_dtype_bits(default_dtype):
        return default_dtype
    return dtype


@tensorflow_handle_array_like_without_promotion
def tensorflow_prod(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    dtype: Optional[tensorflow.DType] = None,
    keepdims: bool = False,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    dtype = tensorflow_as_native_dtype(dtype)
    if dtype is None:
        dtype = tensorflow__infer_dtype(x.dtype)
    axis = tuple(axis) if isinstance(axis, list) else axis
    return tensorflow.experimental.numpy.prod(
        x, axis=axis, dtype=dtype, keepdims=keepdims
    )


def tensorflow__numel(shape):
    shape = tuple(shape)
    return tensorflow_to_scalar(tensorflow_prod(shape)) if shape != () else 1


def tensorflow_check_one_way_broadcastable(x1, x2):
    if len(x1) > len(x2):
        return False
    for a, b in zip(x1[::-1], x2[::-1]):
        if a in (1, b):
            pass
        else:
            return False
    return True


def tensorflow_check_shapes_broadcastable(var, data):
    if not tensorflow_check_one_way_broadcastable(var, data):
        raise Exception(f"Could not broadcast shape {data} to shape {var}.")


@tensorflow_handle_array_like_without_promotion
def tensorflow_broadcast_to(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    shape: Union[tf.TensorShape, Sequence[int]],
    *,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    tensorflow_check_shapes_broadcastable(x.shape, shape)
    if tensorflow.rank(x) > len(shape):
        return tensorflow.broadcast_to(tensorflow.reshape(x, -1), shape)
    return tensorflow.broadcast_to(x, shape)


def tensorflow__broadcast_to(input, target_shape):
    if tensorflow__numel(tuple(input.shape)) == tensorflow__numel(tuple(target_shape)):
        return tensorflow_reshape(input, target_shape)
    else:
        input = input if len(input.shape) else tensorflow_expand_dims(input, axis=0)
        return tensorflow_broadcast_to(input, target_shape)


@tensorflow_handle_array_like_without_promotion
def tensorflow_any(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    try:
        return tensorflow.reduce_any(
            tensorflow.cast(x, tensorflow.bool), axis=axis, keepdims=keepdims
        )
    except tensorflow.errors.InvalidArgumentError as e:
        raise Exception(e) from e


def tensorflow__broadcast_inputs(x1, x2):
    x1_, x2_ = x1, x2
    iterables = list, tuple, tuple
    if not isinstance(x1_, iterables):
        x1_, x2_ = x2, x1
    if not isinstance(x1_, iterables):
        return [x1], [x2]
    if not isinstance(x2_, iterables):
        x1 = [x1] * len(x2)
    return x1, x2


def tensorflow_check_equal(x1, x2, inverse=False, message="", as_array=True):
    def eq_fn(x1, x2):
        return x1 == x2 if inverse else x1 != x2

    def comp_fn(x1, x2):
        return tensorflow_any(eq_fn(x1, x2))

    if not as_array:

        def iter_comp_fn(x1_, x2_):
            return any(eq_fn(x1, x2) for x1, x2 in zip(x1_, x2_))

        def comp_fn(x1, x2):
            return iter_comp_fn(*tensorflow__broadcast_inputs(x1, x2))

    eq = comp_fn(x1, x2)
    if inverse and eq:
        raise Exception(f"{x1} must not be equal to {x2}" if message == "" else message)
    elif not inverse and eq:
        raise Exception(f"{x1} must be equal to {x2}" if message == "" else message)


def tensorflow_multiply(
    x1: Union[float, tensorflow.Tensor, tensorflow.Variable],
    x2: Union[float, tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    oirg_x1 = x1
    oirg_x2 = x2
    try:
        dtype = (
            x1.dtype
            if hasattr(x1, "dtype")
            else x2.dtype
            if hasattr(x2, "dtype")
            else tensorflow_default_dtype()
        )
        if not tensorflow_is_array(x1):
            x1 = tensorflow_asarray(x1, dtype=dtype)
        if not tensorflow_is_array(x2):
            x2 = tensorflow_asarray(x2, dtype=dtype)
    except:
        x1 = oirg_x1
        x2 = oirg_x2
    return tensorflow.math.multiply(x1, x2)


def tensorflow_check_gather_nd_input_valid(params, indices, batch_dims):
    if batch_dims >= len(params.shape):
        raise Exception(
            f"batch_dims = {batch_dims} must be less than rank(`params`) = {len(params.shape)}."
        )
    if batch_dims >= len(indices.shape):
        raise Exception(
            f"batch_dims = {batch_dims}  must be less than rank(`indices`) = {len(indices.shape)}."
        )
    if tensorflow_get_item(
        params.shape, slice(0, batch_dims, None)
    ) != tensorflow_get_item(indices.shape, slice(0, batch_dims, None)):
        raise Exception(
            f"batch dimensions must match in `params` and `indices`; saw {tensorflow_get_item(params.shape, slice(0, batch_dims, None))} vs. {tensorflow_get_item(indices.shape, slice(0, batch_dims, None))}"
        )
    if indices.shape[-1] > len(
        tensorflow_get_item(params.shape, slice(batch_dims, None, None))
    ):
        raise Exception(
            f"index innermost dimension length must be <= rank(`params[batch_dims:]`); saw: {indices.shape[-1]} vs. {len(tensorflow_get_item(params.shape, slice(batch_dims, None, None)))} ."
        )


def tensorflow_gather_nd_helper(params, indices):
    indices_shape = tensorflow.shape(indices)
    params_shape = tensorflow.shape(params)
    num_index_dims = indices_shape[-1]
    result_dim_sizes_list = [
        tensorflow.math.reduce_prod(params_shape[i + 1 :])
        for i in range(len(params_shape) - 1)
    ] + [1]
    result_dim_sizes = tensorflow.convert_to_tensor(
        result_dim_sizes_list, dtype=indices.dtype
    )
    implicit_indices_factor = result_dim_sizes[num_index_dims - 1]
    flat_params = tensorflow.reshape(params, (-1,))
    new_shape = [1] * (len(indices_shape) - 1) + [num_index_dims]
    indices_scales = tensorflow.reshape(result_dim_sizes[0:num_index_dims], new_shape)
    indices_for_flat_tiled = tensorflow.reshape(
        tensorflow.reduce_sum(indices * indices_scales, -1, keepdims=True), (-1, 1)
    )
    indices_for_flat_tiled = tensorflow.repeat(
        indices_for_flat_tiled, implicit_indices_factor, axis=1
    )
    implicit_indices = tensorflow.repeat(
        tensorflow.expand_dims(tensorflow.range(implicit_indices_factor), 0),
        indices_for_flat_tiled.shape[0],
        axis=0,
    )
    indices_for_flat = indices_for_flat_tiled + implicit_indices
    flat_indices_for_flat = tensorflow.reshape(indices_for_flat, (-1,))
    flat_gather = tensorflow.gather(flat_params, flat_indices_for_flat)
    res = tensorflow.reshape(
        flat_gather,
        tensorflow.concat([indices_shape[:-1], params_shape[num_index_dims:]], 0),
    )
    return res


@tensorflow_handle_array_like_without_promotion
def tensorflow_gather_nd(
    params: Union[tensorflow.Tensor, tensorflow.Variable],
    indices: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    batch_dims: int = 0,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    tensorflow_check_gather_nd_input_valid(params, indices, batch_dims)
    try:
        return tensorflow.gather_nd(params, indices, batch_dims=batch_dims)
    except Exception:
        batch_dims %= len(params.shape)
        result = []
        if batch_dims == 0:
            result = tensorflow_gather_nd_helper(params, indices)
        else:
            for b in range(batch_dims):
                if b == 0:
                    zip_list = list(zip(params, indices))
                else:
                    zip_list = [
                        (p, i)
                        for z in [zip(p1, i1) for p1, i1 in zip_list]
                        for p, i in z
                    ]
            for z in zip_list:
                p, i = z[0], z[1]
                r = tensorflow_gather_nd_helper(p, i)
                result.append(r)
            result = tensorflow.stack(result)
            result = tensorflow.reshape(
                result,
                tensorflow.concat([params.shape[0:batch_dims], result.shape[1:]], 0),
            )
        return result


def tensorflow_is_variable(x, /, *, exclusive=False):
    return isinstance(x, tensorflow.Variable)


def tensorflow__is_variable(x, exclusive=False, to_ignore=None):
    x = x
    return tensorflow_nested_map(
        lambda x: tensorflow_is_variable(x, exclusive=exclusive),
        x,
        include_derived=True,
        shallow=False,
        to_ignore=to_ignore,
    )


def tensorflow_inplace_update(
    x: Union[tensorflow.Tensor, tensorflow.Tensor],
    val: Union[tensorflow.Tensor, tensorflow.Tensor],
    /,
    *,
    ensure_in_backend: bool = False,
    keep_input_dtype: bool = False,
):
    if tensorflow_is_array(x) and tensorflow_is_array(val):
        if keep_input_dtype:
            val = tensorflow_astype(val, x.dtype)
        (x_native, val_native), _ = (x, val), "_"
        if tensorflow__is_variable(x_native):
            x_native.assign(val_native)
            if tensorflow_is_ivy_array(x):
                x = x_native
            else:
                x = tensorflow.convert_to_tensor(x_native)
        else:
            x = x_native
        return x
    else:
        return val


def tensorflow_scatter_nd(
    indices: Union[tensorflow.Tensor, tensorflow.Variable],
    updates: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    shape: Optional[Union[tf.TensorShape, Sequence[int]]] = None,
    *,
    reduction: str = "sum",
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    updates_dtype = updates.dtype
    if tensorflow_exists(out):
        dtype = tensorflow_promote_types(out.dtype, updates_dtype)
    updates = tensorflow.cast(
        updates,
        tensorflow_as_native_dtype(dtype) if tensorflow_exists(out) else updates_dtype,
    )
    expected_shape = (
        list(tensorflow.shape(indices)[:-1])
        + list(out.shape[tensorflow.shape(indices)[-1] :])
        if tensorflow_exists(out)
        else list(tensorflow.shape(indices)[:-1])
        + list(shape[tensorflow.shape(indices)[-1] :])
    )
    updates = tensorflow__broadcast_to(updates, expected_shape)
    if len(updates.shape) == 0:
        indices = tensorflow.expand_dims(indices, 0)
        updates = tensorflow.expand_dims(updates, 0)
    target = out
    target_given = tensorflow_exists(target)
    if tensorflow_exists(shape) and target_given:
        tensorflow_check_equal(tuple(target.shape), tuple(shape), as_array=False)
    if not target_given:
        shape = list(shape) if tensorflow_exists(shape) else list(out.shape)
        target = tensorflow.zeros(shape, dtype=updates.dtype)
    if reduction == "sum":
        res = tensorflow.tensor_scatter_nd_add(target, indices, updates)
    elif reduction == "min":
        res = tensorflow.tensor_scatter_nd_min(target, indices, updates)
    elif reduction == "max":
        res = tensorflow.tensor_scatter_nd_max(target, indices, updates)
    elif reduction == "mul":
        updates = tensorflow_multiply(tensorflow_gather_nd(target, indices), updates)
        res = tensorflow.tensor_scatter_nd_update(target, indices, updates)
    elif reduction == "replace":
        res = tensorflow.tensor_scatter_nd_update(target, indices, updates)
    else:
        raise Exception(
            f'reduction is {reduction}, but it must be one of "sum", "min", "max", "mul" or "replace"'
        )
    if tensorflow_exists(out):
        return tensorflow_inplace_update(out, res)
    return res


@tensorflow_handle_methods
def tensorflow___setitem__(self, query, val):
    self = tensorflow_set_item(self, query, val)


def tensorflow_handle_set_item(fn):
    @functools.wraps(fn)
    def wrapper(inp, query, val, **kwargs):
        try:
            tensorflow___setitem__(inp, query, val)
            res = inp
        except IndexError:
            raise
        except Exception:
            res = fn(inp, query, val, **kwargs)
        return res

    return wrapper


@tensorflow_handle_set_item
def tensorflow_set_item(
    x: Union[tensorflow.Tensor, tf.Tensor],
    query: Union[tensorflow.Tensor, tf.Tensor, Tuple],
    val: Union[tensorflow.Tensor, tf.Tensor],
    /,
    *,
    copy: Optional[bool] = False,
):
    if isinstance(query, (list, tuple)) and any(
        [(q is Ellipsis or isinstance(q, slice) and q.stop is None) for q in query]
    ):
        np_array = x.numpy()
        np_array = tensorflow_set_item(np_array, query, np.asarray(val))
        return tensorflow_asarray(np_array)
    if copy:
        x = tensorflow_copy_array(x)
    if not tensorflow_is_array(val):
        val = tensorflow_asarray(val)
    if 0 in x.shape or 0 in val.shape:
        return x
    if tensorflow_is_array(query) and tensorflow_is_bool_dtype(query):
        if not len(query.shape):
            query = tensorflow_tile(query, (x.shape[0],))
        indices = tensorflow_nonzero(query, as_tuple=False)
    else:
        indices, target_shape, _ = tensorflow__parse_query(
            query, tensorflow_shape(x, as_array=True), scatter=True
        )
        if indices is None:
            return x
    val = tensorflow_astype_1(val, x.dtype)
    ret = tensorflow_scatter_nd(indices, val, reduction="replace", out=x)
    return ret


@tensorflow_handle_array_like_without_promotion
def tensorflow_size(x: tensorflow.Tensor, /):
    return functools.reduce(mul, x.shape) if len(x.shape) > 0 else 1


def tensorflow_size_1(self):
    return tensorflow_size(self)
