"""Collection of tests for unified dtype functions."""

# global
import numpy as np
from hypothesis import given, strategies as st


# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
import ivy.functional.backends.jax
import ivy.functional.backends.tensorflow
import ivy.functional.backends.torch
import ivy.functional.backends.mxnet
from functools import reduce  # for making strategy
from operator import mul  # for making strategy
from typing import Tuple


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
# -------------------------- #

# taken from array-api repo
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


# taken from array-api repo
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


# taken from array-api repo
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


# For data generation in multiple tests
dtype_shared = st.shared(st.sampled_from(ivy_np.valid_dtypes), key="dtype")


# Ivy Unit Tests #
# -------------- #


# astype
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_dtypes, 1),
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
        input_dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "astype",
        x=np.asarray(x, dtype=input_dtype),
        dtype=dtype,
    )


# broadcast arrays
@st.composite
def broadcastable_arrays(draw, dtype):
    shapes = draw(st.integers(2, 5).flatmap(mutually_broadcastable_shapes))
    arrays = []
    for c, shape in enumerate(shapes, 1):
        x = draw(helpers.nph.arrays(dtype=dtype, shape=shape), label=f"x{c}")
        arrays.append(x)
    return helpers.as_lists(*arrays)


@given(
    arrays=broadcastable_arrays(dtype_shared),
    dtype=dtype_shared,
    as_variable=st.booleans(),
    native_array=st.booleans(),
    container=st.booleans(),
)
def test_broadcast_arrays(
    arrays,
    dtype,
    as_variable,
    native_array,
    container,
    fw,
):
    kw = {}
    for i, array in enumerate(zip(arrays)):
        kw["x{}".format(i)] = ivy.asarray(array)
    num_positional_args = len(kw)
    helpers.test_function(
        dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        False,
        fw,
        "broadcast_arrays",
        **kw,
    )


# broadcast_to
@st.composite
def array_and_broadcastable_shape(draw, in_dtype):
    dtype = in_dtype
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
    in_dtype=dtype_shared,
    as_variable_flags=st.booleans(),
    with_out=st.booleans(),
    native_array_flags=st.booleans(),
    container_flags=st.booleans(),
    instance_method=st.booleans(),
)
def test_broadcast_to(
    array_and_shape,
    in_dtype,
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
        input_dtypes=in_dtype,
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
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_dtypes, 1),
    dtype=st.sampled_from(ivy_np.valid_dtypes),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="can_cast"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_can_cast(
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
        input_dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "can_cast",
        from_=np.array(x, dtype=input_dtype),
        to=dtype,
    )


# dtype_bits
@given(
    dtype=st.sampled_from(ivy_np.valid_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="dtype_bits"),
)
def test_dtype_bits(
    dtype,
    num_positional_args,
    fw,
):
    ret = helpers.test_function(
        dtype,
        False,
        False,
        num_positional_args,
        True,
        False,
        False,
        fw,
        "dtype_bits",
        dtype_in=dtype,
        test_values=False,
    )
    if not ivy.exists(ret):
        return
    num_bits, num_bits_np = ret
    assert num_bits == num_bits_np


@st.composite
def _array_or_type(draw, float_or_int):
    valid_dtypes = {"float": ivy_np.valid_float_dtypes, "int": ivy_np.valid_int_dtypes}[
        float_or_int
    ]
    return draw(
        st.sampled_from(
            (
                draw(helpers.dtype_and_values(valid_dtypes, 1)),
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
        input_dtype,
        False,
        False,
        num_positional_args,
        False,
        False,
        False,
        fw,
        "finfo",
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
        input_dtype,
        False,
        False,
        num_positional_args,
        False,
        False,
        False,
        fw,
        "iinfo",
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


# is_float_dtype
@given(
    array=helpers.nph.arrays(
        dtype=dtype_shared,
        shape=helpers.lists(
            st.integers(1, 5),
            min_size="num_dims",
            max_size="num_dims",
            size_bounds=[1, 5],
        ),
    ),
    dtype_in=dtype_shared,
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="is_float_dtype"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_is_float_dtype(
    array,
    dtype_in,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    helpers.test_function(
        dtype_in,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "is_float_dtype",
        dtype_in=array,
    )


# is_int_dtype
@given(
    array=helpers.nph.arrays(
        dtype=dtype_shared,
        shape=helpers.lists(
            st.integers(1, 5),
            min_size="num_dims",
            max_size="num_dims",
            size_bounds=[1, 5],
        ),
    ),
    dtype_in=dtype_shared,
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="is_int_dtype"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_is_int_dtype(
    array,
    dtype_in,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    helpers.test_function(
        dtype_in,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "is_int_dtype",
        dtype_in=array,
    )


# promote_types
@given(
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=ivy.valid_dtypes,
        num_arrays=2,
        shared_dtype=False,
    ),
    as_variable=helpers.list_of_length(st.booleans(), 2),
    num_positional_args=helpers.num_positional_args(fn_name="promote_types"),
    native_array=helpers.list_of_length(st.booleans(), 2),
    container=helpers.list_of_length(st.booleans(), 2),
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
        input_dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        False,
        fw,
        "promote_types",
        type1=type1,
        type2=type2,
        test_values=False,
    )


# result_type
@given(
    dtype_and_x=helpers.dtype_and_values(
        ivy.valid_dtypes,
        st.shared(st.integers(2, 5), key="num_arrays"),
        shared_dtype=False,
    ),
    as_variable=st.booleans(),
    num_positional_args=st.shared(st.integers(2, 5), key="num_arrays"),
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
        dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "result_type",
        **kw,
    )


# type_promote_arrays
@given(
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_dtypes,
        num_arrays=2,
        shared_dtype=False,
    ),
    as_variable=helpers.list_of_length(st.booleans(), 2),
    num_positional_args=helpers.num_positional_args(fn_name="type_promote_arrays"),
    native_array=helpers.list_of_length(st.booleans(), 2),
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
        input_dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        False,
        False,
        fw,
        "type_promote_arrays",
        x1=np.array(x1),
        x2=np.array(x2),
        test_values=True,
    )
