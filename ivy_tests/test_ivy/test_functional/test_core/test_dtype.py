"""Collection of tests for unified dtype functions."""

# global
import numpy as np
from hypothesis import given, assume, strategies as st


# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
import ivy.functional.backends.jax as ivy_jax
import ivy.functional.backends.tensorflow as ivy_tf
import ivy.functional.backends.torch as ivy_torch
import ivy.functional.backends.mxnet as ivy_mxn
from functools import reduce  # for making strategy
from operator import mul  # for making strategy
from typing import Tuple
import pytest


# dtype objects
def test_dtype_instances(device, call):
    assert ivy.exists(ivy.int8)
    assert ivy.exists(ivy.int16)
    assert ivy.exists(ivy.int32)
    assert ivy.exists(ivy.int64)
    assert ivy.exists(ivy.uint8)
    if ivy.current_backend_str() != "torch":
        assert ivy.exists(ivy.uint16)
        assert ivy.exists(ivy.uint32)
        assert ivy.exists(ivy.uint64)
    assert ivy.exists(ivy.float32)
    assert ivy.exists(ivy.float64)
    assert ivy.exists(ivy.bool)


# Functions to use in strategy #
# ---------------------------- #

# from array-api repo
def _broadcast_shapes(shape1, shape2):
    """Broadcasts `shape1` and `shape2`"""
    N1 = len(shape1)
    N2 = len(shape2)
    N = max(N1, N2)
    shape = [None for _ in range(N)]
    i = N - 1
    while i >= 0:
        n1 = N1 - N + i
        if N1 - N + i >= 0:
            d1 = shape1[n1]
        else:
            d1 = 1
        n2 = N2 - N + i
        if N2 - N + i >= 0:
            d2 = shape2[n2]
        else:
            d2 = 1

        if d1 == 1:
            shape[i] = d2
        elif d2 == 1:
            shape[i] = d1
        elif d1 == d2:
            shape[i] = d1
        else:
            raise Exception("Broadcast error")

        i = i - 1

    return tuple(shape)


# from array-api repo
def broadcast_shapes(*shapes):
    if len(shapes) == 0:
        raise ValueError("shapes=[] must be non-empty")
    elif len(shapes) == 1:
        return shapes[0]
    result = _broadcast_shapes(shapes[0], shapes[1])
    for i in range(2, len(shapes)):
        result = _broadcast_shapes(result, shapes[i])
    return result


# np.prod and others have overflow and math.prod is Python 3.8+ only
def prod(seq):
    return reduce(mul, seq, 1)


# from array-api repo
def mutually_broadcastable_shapes(
    num_shapes: int,
    *,
    base_shape: Tuple[int, ...] = (),
    min_dims: int = 1,
    max_dims: int = 4,
    min_side: int = 1,
    max_side: int = 4,
):
    if max_dims is None:
        max_dims = min(max(len(base_shape), min_dims) + 5, 32)
    if max_side is None:
        max_side = max(base_shape[-max_dims:] + (min_side,)) + 5
    return (
        helpers.nph.mutually_broadcastable_shapes(
            num_shapes=num_shapes,
            base_shape=base_shape,
            min_dims=min_dims,
            max_dims=max_dims,
            min_side=min_side,
            max_side=max_side,
        )
        .map(lambda BS: BS.input_shapes)
        .filter(lambda shapes: all(prod(i for i in s if i > 0) < 1000 for s in shapes))
    )


# for data generation in multiple tests
dtype_shared = st.shared(st.sampled_from(ivy_np.valid_dtypes), key="dtype")


@st.composite
def dtypes_shared(draw, num_dtypes):
    if isinstance(num_dtypes, str):
        num_dtypes = draw(st.shared(helpers.ints(), key=num_dtypes))
    return draw(
        st.shared(
            st.lists(
                st.sampled_from(ivy_np.valid_dtypes),
                min_size=num_dtypes,
                max_size=num_dtypes,
            ),
            key="dtypes",
        )
    )


# Array API Standard Function Tests #
# --------------------------------- #


# astype
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_dtypes, num_arrays=1
    ),
    dtype=st.sampled_from(ivy_np.valid_dtypes),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="astype"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_astype(
    dtype_and_x,
    dtype,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="astype",
        x=np.asarray(x, dtype=input_dtype),
        dtype=dtype,
    )


# broadcast arrays
@st.composite
def broadcastable_arrays(draw, dtypes):
    num_arrays = st.shared(helpers.ints(min_value=2, max_value=5), key="num_arrays")
    shapes = draw(num_arrays.flatmap(mutually_broadcastable_shapes))
    dtypes = draw(dtypes)
    arrays = []
    for c, (shape, dtype) in enumerate(zip(shapes, dtypes), 1):
        x = draw(helpers.nph.arrays(dtype=dtype, shape=shape), label=f"x{c}").tolist()
        arrays.append(x)
    return arrays


@given(
    arrays=broadcastable_arrays(dtypes_shared("num_arrays")),
    input_dtypes=dtypes_shared("num_arrays"),
    as_variable=helpers.array_bools(),
    native_array=helpers.array_bools(),
    container=helpers.array_bools(),
    instance_method=helpers.array_bools(),
)
def test_broadcast_arrays(
    arrays,
    input_dtypes,
    as_variable,
    native_array,
    container,
    instance_method,
    fw,
):
    kw = {}
    for i, (array, dtype) in enumerate(zip(arrays, input_dtypes)):
        kw["x{}".format(i)] = np.asarray(array, dtype=dtype)
    num_positional_args = len(kw)
    print("input: ", kw)
    helpers.test_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="broadcast_arrays",
        **kw,
    )


# broadcast_to
@st.composite
def array_and_broadcastable_shape(draw, dtype):
    in_shape = draw(helpers.nph.array_shapes(min_dims=1, max_dims=4))
    x = draw(helpers.nph.arrays(shape=in_shape, dtype=dtype))
    to_shape = draw(
        mutually_broadcastable_shapes(1, base_shape=in_shape)
        .map(lambda S: S[0])
        .filter(lambda s: broadcast_shapes(in_shape, s) == s),
        label="shape",
    )
    return (x, to_shape)


@given(
    array_and_shape=array_and_broadcastable_shape(dtype_shared),
    input_dtype=dtype_shared,
    as_variable_flags=st.booleans(),
    with_out=st.booleans(),
    native_array_flags=st.booleans(),
    container_flags=st.booleans(),
    instance_method=st.booleans(),
)
def test_broadcast_to(
    array_and_shape,
    input_dtype,
    as_variable_flags,
    with_out,
    native_array_flags,
    container_flags,
    instance_method,
    fw,
):
    array, to_shape = array_and_shape
    num_positional_args = len(array)
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable_flags,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array_flags,
        container_flags=container_flags,
        instance_method=instance_method,
        fw=fw,
        fn_name="broadcast_to",
        x=array,
        shape=to_shape,
    )


# can_cast
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_dtypes, num_arrays=1
    ),
    to_dtype=st.sampled_from(ivy_np.valid_dtypes),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="can_cast"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_can_cast(
    dtype_and_x,
    to_dtype,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="can_cast",
        from_=np.array(x, dtype=input_dtype),
        to=to_dtype,
    )


@st.composite
def _array_or_type(draw, float_or_int):
    valid_dtypes = {"float": ivy_np.valid_float_dtypes, "int": ivy_np.valid_int_dtypes}[
        float_or_int
    ]
    return draw(
        st.sampled_from(
            (
                draw(
                    helpers.dtype_and_values(
                        available_dtypes=valid_dtypes, num_arrays=1
                    )
                ),
                draw(st.sampled_from(valid_dtypes)),
            )
        )
    )


# finfo
@given(
    type=_array_or_type("float"),
    num_positional_args=helpers.num_positional_args(fn_name="finfo"),
)
def test_finfo(
    type,
    num_positional_args,
    fw,
):
    if isinstance(type, str):
        input_dtype = type
    else:
        input_dtype, x = type
        type = np.array(x, dtype=input_dtype)
    ret = helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=False,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=False,
        container_flags=False,
        instance_method=False,
        fw=fw,
        fn_name="finfo",
        type=type,
        test_values=False,
    )
    if not ivy.exists(ret):
        return
    mach_lims, mach_lims_np = ret
    assert mach_lims.min == mach_lims_np.min
    assert mach_lims.max == mach_lims_np.max
    assert mach_lims.eps == mach_lims_np.eps
    assert mach_lims.bits == mach_lims_np.bits


# iinfo
@given(
    type=_array_or_type("int"),
    num_positional_args=helpers.num_positional_args(fn_name="iinfo"),
)
def test_iinfo(
    type,
    num_positional_args,
    fw,
):
    if isinstance(type, str):
        input_dtype = type
    else:
        input_dtype, x = type
        type = np.array(x, dtype=input_dtype)
    ret = helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=False,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=False,
        container_flags=False,
        instance_method=False,
        fw=fw,
        fn_name="iinfo",
        type=type,
        test_values=False,
    )

    if not ivy.exists(ret):
        return
    mach_lims, mach_lims_np = ret
    assert mach_lims.min == mach_lims_np.min
    assert mach_lims.max == mach_lims_np.max
    assert mach_lims.dtype == mach_lims_np.dtype
    assert mach_lims.bits == mach_lims_np.bits


# result_type
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy.valid_dtypes,
        num_arrays=st.shared(helpers.ints(min_value=2, max_value=5), key="num_arrays"),
        shared_dtype=False,
    ),
    as_variable=st.booleans(),
    num_positional_args=st.shared(
        helpers.ints(min_value=2, max_value=5), key="num_arrays"
    ),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_result_type(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x = helpers.as_lists(*dtype_and_x)
    kw = {}
    for i, (dtype_, x_) in enumerate(zip(dtype, x)):
        kw["x{}".format(i)] = np.asarray(x_, dtype=dtype_)
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="result_type",
        **kw,
    )


# Extra Ivy Function Tests #
# ------------------------ #


# as_ivy_dtype
@given(
    input_dtype=st.sampled_from(ivy.valid_dtypes),
)
def test_as_ivy_dtype(
    input_dtype,
):
    res = ivy.as_ivy_dtype(input_dtype)
    if isinstance(input_dtype, str):
        assert isinstance(res, str)
        return

    assert isinstance(input_dtype, ivy.Dtype) or isinstance(
        input_dtype, str
    ), f"input_dtype={input_dtype!r}, but should be str or ivy.Dtype"
    assert isinstance(res, str), f"result={res!r}, but should be str"


_valid_dtype_in_all_frameworks = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "float16",
    "float32",
    "float64",
    "bool",
]


# as_native_dtype
@given(input_dtype=st.sampled_from(_valid_dtype_in_all_frameworks))
def test_as_native_dtype(
    input_dtype,
):
    res = ivy.as_native_dtype(input_dtype)
    if isinstance(input_dtype, ivy.NativeDtype):
        assert isinstance(res, ivy.NativeDtype)
        return

    assert isinstance(input_dtype, ivy.Dtype) or isinstance(
        input_dtype, str
    ), f"input_dtype={input_dtype!r}, but should be str or ivy.Dtype"
    assert isinstance(
        res, ivy.NativeDtype
    ), f"result={res!r}, but should be ivy.NativeDtype"


# closest_valid_dtypes
@given(input_dtype=st.sampled_from(_valid_dtype_in_all_frameworks))
def test_closest_valid_dtype(
    input_dtype,
):
    res = ivy.closest_valid_dtype(input_dtype)
    assert isinstance(input_dtype, ivy.Dtype) or isinstance(input_dtype, str)
    assert isinstance(res, ivy.Dtype) or isinstance(
        res, str
    ), f"result={res!r}, but should be str or ivy.Dtype"


# default_dtype
@given(
    input_dtype=st.sampled_from(ivy_np.valid_dtypes),
    as_native=st.booleans(),
)
def test_default_dtype(
    input_dtype,
    as_native,
):
    assume(input_dtype in ivy.valid_dtypes)

    res = ivy.default_dtype(dtype=input_dtype, as_native=as_native)
    assert (
        isinstance(input_dtype, ivy.Dtype)
        or isinstance(input_dtype, str)
        or isinstance(input_dtype, ivy.NativeDtype)
    )
    assert isinstance(res, ivy.Dtype) or isinstance(
        input_dtype, str
    ), f"input_dtype={input_dtype!r}, but should be str or ivy.Dtype"


# dtype
@given(
    array=helpers.nph.arrays(
        dtype=dtype_shared,
        shape=helpers.lists(
            arg=helpers.ints(min_value=1, max_value=5),
            min_size="num_dims",
            max_size="num_dims",
            size_bounds=[1, 5],
        ),
    ),
    input_dtype=dtype_shared,
    as_native=st.booleans(),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="dtype"),
    native_array=st.booleans(),
    container=st.booleans(),
)
def test_dtype(
    array,
    input_dtype,
    as_native,
    as_variable,
    num_positional_args,
    native_array,
    container,
    fw,
):
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=False,
        fw=fw,
        fn_name="dtype",
        x=array,
        as_native=as_native,
        test_values=False,
    )


# dtype_bits
@given(
    input_dtype=st.sampled_from(ivy_np.valid_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="dtype_bits"),
)
def test_dtype_bits(
    input_dtype,
    num_positional_args,
    fw,
):
    ret = helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=False,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=True,
        container_flags=False,
        instance_method=False,
        fw=fw,
        fn_name="dtype_bits",
        dtype_in=input_dtype,
        test_values=False,
    )
    if not ivy.exists(ret):
        return
    num_bits, num_bits_np = ret
    assert num_bits == num_bits_np


# is_bool_dtype
@given(
    array=helpers.nph.arrays(
        dtype=dtype_shared,
        shape=helpers.lists(
            arg=helpers.ints(min_value=1, max_value=5),
            min_size="num_dims",
            max_size="num_dims",
            size_bounds=[1, 5],
        ),
    ),
    input_dtype=dtype_shared,
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="is_bool_dtype"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_is_bool_dtype(
    array,
    input_dtype,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="is_bool_dtype",
        dtype_in=array,
        test_values=False,
    )


# is_float_dtype
@given(
    array=helpers.nph.arrays(
        dtype=dtype_shared,
        shape=helpers.lists(
            arg=helpers.ints(min_value=1, max_value=5),
            min_size="num_dims",
            max_size="num_dims",
            size_bounds=[1, 5],
        ),
    ),
    input_dtype=dtype_shared,
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="is_float_dtype"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_is_float_dtype(
    array,
    input_dtype,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="is_float_dtype",
        dtype_in=array,
    )


# is_int_dtype
@given(
    array=helpers.nph.arrays(
        dtype=dtype_shared,
        shape=helpers.lists(
            arg=helpers.ints(min_value=1, max_value=5),
            min_size="num_dims",
            max_size="num_dims",
            size_bounds=[1, 5],
        ),
    ),
    input_dtype=dtype_shared,
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="is_int_dtype"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_is_int_dtype(
    array,
    input_dtype,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="is_int_dtype",
        dtype_in=array,
    )


# promote_types
@given(
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=ivy.valid_dtypes,
        num_arrays=2,
        shared_dtype=False,
    ),
    as_variable=helpers.list_of_length(x=st.booleans(), length=2),
    num_positional_args=helpers.num_positional_args(fn_name="promote_types"),
    native_array=helpers.list_of_length(x=st.booleans(), length=2),
    container=helpers.list_of_length(x=st.booleans(), length=2),
)
def test_promote_types(
    dtype_and_values,
    as_variable,
    num_positional_args,
    native_array,
    container,
    fw,
):
    types, arrays = dtype_and_values
    type1, type2 = types
    input_dtype = [type1, type2]
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=False,
        fw=fw,
        fn_name="promote_types",
        type1=type1,
        type2=type2,
        test_values=False,
    )


# type_promote_arrays
@given(
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_dtypes,
        num_arrays=2,
        shared_dtype=False,
    ),
    as_variable=helpers.list_of_length(x=st.booleans(), length=2),
    num_positional_args=helpers.num_positional_args(fn_name="type_promote_arrays"),
    native_array=helpers.list_of_length(x=st.booleans(), length=2),
)
def test_type_promote_arrays(
    dtype_and_values,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    types, arrays = dtype_and_values
    type1, type2 = types
    x1, x2 = arrays
    input_dtype = [type1, type2]
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=False,
        instance_method=False,
        fw=fw,
        fn_name="type_promote_arrays",
        x1=np.array(x1),
        x2=np.array(x2),
        test_values=True,
    )


# default_float_dtype
@pytest.mark.parametrize("float_dtype", [ivy.float16, ivy.float32, ivy.float64, None])
@pytest.mark.parametrize(
    "input",
    [
        [(5.0, 25.0), (6.0, 36.0), (7.0, 49.0)],
        np.array([10.0, 0.0, -3.0]),
        10,
        None,
    ],
)
@pytest.mark.parametrize("as_native", [True, False])
def test_default_float_dtype(input, float_dtype, as_native):
    res = ivy.default_float_dtype(
        input=input, float_dtype=float_dtype, as_native=as_native
    )
    assert (
        isinstance(res, ivy.Dtype)
        or isinstance(res, ivy.NativeDtype)
        or isinstance(res, str)
    )
    assert (
        ivy.default_float_dtype(input=None, float_dtype=None, as_native=False)
        == ivy.float32
    )
    assert ivy.default_float_dtype(float_dtype=ivy.float16) == ivy.float16
    assert ivy.default_float_dtype() == ivy.float32


# default_int_dtype
@pytest.mark.parametrize("int_dtype", [ivy.int16, ivy.int32, ivy.int64, None])
@pytest.mark.parametrize(
    "input",
    [
        [(5, 25), (6, 36), (7, 49)],
        np.array([10, 0, -3]),
        10,
        10.0,
        None,
    ],
)
@pytest.mark.parametrize("as_native", [True, False])
def test_default_int_dtype(input, int_dtype, as_native):
    res = ivy.default_int_dtype(input=input, int_dtype=int_dtype, as_native=as_native)
    assert (
        isinstance(res, ivy.Dtype)
        or isinstance(res, ivy.NativeDtype)
        or isinstance(res, str)
    )
    assert (
        ivy.default_int_dtype(input=None, int_dtype=None, as_native=False) == ivy.int32
    )
    assert ivy.default_int_dtype(int_dtype=ivy.int16) == ivy.int16
    assert ivy.default_int_dtype() == ivy.int32


@st.composite
def dtypes_list(draw):
    num = draw(st.one_of(helpers.ints(min_value=1, max_value=5)))
    return draw(
        st.lists(
            st.sampled_from(ivy.valid_dtypes),
            min_size=num,
            max_size=num,
        )
    )


# function_unsupported_dtypes
@given(supported_dtypes=dtypes_list())
def test_function_supported_dtypes(
    supported_dtypes,
):
    def func():
        return

    func.supported_dtypes = tuple(supported_dtypes)
    res = ivy.function_supported_dtypes(func)
    supported_dtypes_true = tuple(set(func.supported_dtypes))
    assert sorted(supported_dtypes_true) == sorted(res)


# function_unsupported_dtypes
@given(unsupported_dtypes=dtypes_list())
def test_function_unsupported_dtypes(
    unsupported_dtypes,
):
    def func():
        return

    func.unsupported_dtypes = tuple(unsupported_dtypes)
    res = ivy.function_unsupported_dtypes(func)
    unsupported_dtypes_true = tuple(set(ivy.invalid_dtypes + func.unsupported_dtypes))
    assert sorted(unsupported_dtypes_true) == sorted(res)


# invalid_dtype
@given(
    dtype_in=st.sampled_from(ivy.valid_dtypes),
)
def test_invalid_dtype(dtype_in, fw):
    res = ivy.invalid_dtype(dtype_in)
    fw_invalid_dtypes = {
        "torch": ivy_torch.invalid_dtypes,
        "tensorflow": ivy_tf.invalid_dtypes,
        "jax": ivy_jax.invalid_dtypes,
        "mxnet": ivy_mxn.invalid_dtypes,
        "numpy": ivy_np.invalid_dtypes,
    }
    if dtype_in in fw_invalid_dtypes[fw]:
        assert res is True, (
            f"fDtype = {dtype_in!r} is a valid dtype for {fw}, but" f"result = {res}"
        )
    else:
        assert res is False, (
            f"fDtype = {dtype_in!r} is not a valid dtype for {fw}, but"
            f"result = {res}"
        )


# unset_default_dtype()
@given(
    dtype=st.sampled_from(ivy.valid_dtypes),
)
def test_unset_default_dtype(dtype):
    stack_size_before = len(ivy.default_dtype_stack)
    ivy.set_default_dtype(dtype)
    ivy.unset_default_dtype()
    stack_size_after = len(ivy.default_dtype_stack)
    assert (
        stack_size_before == stack_size_after
    ), f"Default dtype not unset. Stack size= {stack_size_after!r}"


# unset_default_float_dtype()
@given(
    dtype=st.sampled_from(ivy.valid_float_dtypes),
)
def test_unset_default_float_dtype(dtype):
    stack_size_before = len(ivy.default_float_dtype_stack)
    ivy.set_default_float_dtype(dtype)
    ivy.unset_default_float_dtype()
    stack_size_after = len(ivy.default_float_dtype_stack)
    assert (
        stack_size_before == stack_size_after
    ), f"Default float dtype not unset. Stack size= {stack_size_after!r}"


# unset_default_int_dtype()
@given(
    dtype=st.sampled_from(ivy.valid_int_dtypes),
)
def test_unset_default_int_dtype(dtype):
    stack_size_before = len(ivy.default_int_dtype_stack)
    ivy.set_default_int_dtype(dtype)
    ivy.unset_default_int_dtype()
    stack_size_after = len(ivy.default_int_dtype_stack)
    assert (
        stack_size_before == stack_size_after
    ), f"Default int dtype not unset. Stack size= {stack_size_after!r}"


# valid_dtype
@given(
    dtype_in=st.sampled_from(ivy.valid_dtypes),
)
def test_valid_dtype(dtype_in, fw):
    res = ivy.valid_dtype(dtype_in)
    fw_valid_dtypes = {
        "torch": ivy_torch.valid_dtypes,
        "tensorflow": ivy_tf.valid_dtypes,
        "jax": ivy_jax.valid_dtypes,
        "mxnet": ivy_mxn.valid_dtypes,
        "numpy": ivy_np.valid_dtypes,
    }
    if dtype_in in fw_valid_dtypes[fw]:
        assert res is True, (
            f"fDtype = {dtype_in!r} is not a valid dtype for {fw}, but"
            f"result = {res}"
        )
    else:
        assert res is False, (
            f"fDtype = {dtype_in!r} is a valid dtype for {fw}, but" f"result = {res}"
        )
