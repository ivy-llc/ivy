"""Collection of tests for unified dtype functions."""

# global
from hypothesis import given, strategies as st


# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
import ivy.functional.backends.numpy
import ivy.functional.backends.jax
import ivy.functional.backends.tensorflow
import ivy.functional.backends.torch
import ivy.functional.backends.mxnet


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


# astype
@given(
    dtype_and_x=helpers.dtype_and_values(ivy.valid_dtypes, 2),
    as_variable=helpers.list_of_length(st.booleans(), 2),
    num_positional_args=st.integers(2, 2),
    native_array=helpers.list_of_length(st.booleans(), 2),
    container=helpers.list_of_length(st.booleans(), 2),
    instance_method=st.booleans(),
)
def test_astype(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x = dtype_and_x
    if (v == [] for v in x):
        return
    helpers.test_array_function(
        dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "astype",
        x=x,
    )


# broadcast_to
@given(
    array_shape=helpers.lists(
        st.integers(1, 5), min_size="num_dims", max_size="num_dims", size_bounds=[1, 5]
    ),
    dtype=st.sampled_from(ivy_np.valid_dtypes),
    data=st.data(),
    as_variable=st.booleans(),
    num_positional_args=st.integers(0, 2),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_broadcast_to(
    array_shape,
    dtype,
    data,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    # smoke this for torch
    if fw == "torch" and dtype in ["uint16", "uint32", "uint64"]:
        return
    x = data.draw(helpers.nph.arrays(shape=array_shape, dtype=dtype))
    helpers.test_array_function(
        dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "broadcast_to",
        x=x,
        shape=array_shape,
    )


# can_cast
@given(
    dtype_and_x=helpers.dtype_and_values(ivy.valid_dtypes, 2),
    as_variable=helpers.list_of_length(st.booleans(), 2),
    num_positional_args=st.integers(2, 2),
    native_array=helpers.list_of_length(st.booleans(), 2),
    container=helpers.list_of_length(st.booleans(), 2),
    instance_method=st.booleans(),
)
def test_can_cast(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x = dtype_and_x
    if (v == [] for v in x):
        return
    helpers.test_array_function(
        dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "can_cast",
        x=x,
    )


# dtype_bits
@given(
    dtype_and_x=helpers.dtype_and_values(ivy.valid_dtypes, 2),
    as_variable=helpers.list_of_length(st.booleans(), 2),
    num_positional_args=st.integers(1, 1),
    native_array=helpers.list_of_length(st.booleans(), 2),
    container=helpers.list_of_length(st.booleans(), 2),
    instance_method=st.booleans(),
)
def test_dtype_bits(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x = dtype_and_x
    if (v == [] for v in x):
        return
    helpers.test_array_function(
        dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "dtype_bits",
        x=x,
    )


# dtype_from_str
@given(
    dtype_and_x=helpers.dtype_and_values(ivy.valid_dtypes, 2),
    as_variable=helpers.list_of_length(st.booleans(), 2),
    num_positional_args=st.integers(1, 1),
    native_array=helpers.list_of_length(st.booleans(), 2),
    container=helpers.list_of_length(st.booleans(), 2),
    instance_method=st.booleans(),
)
def test_dtype_from_str(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x = dtype_and_x
    if (v == [] for v in x):
        return
    helpers.test_array_function(
        dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "dtype_from_str",
        x=x,
    )


# dtype_to_str
@given(
    dtype_and_x=helpers.dtype_and_values(ivy.valid_dtypes, 2),
    as_variable=helpers.list_of_length(st.booleans(), 2),
    num_positional_args=st.integers(1, 1),
    native_array=helpers.list_of_length(st.booleans(), 2),
    container=helpers.list_of_length(st.booleans(), 2),
    instance_method=st.booleans(),
)
def test_dtype_to_str(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x = dtype_and_x
    if (v == [] for v in x):
        return
    helpers.test_array_function(
        dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "dtype_to_str",
        x=x,
    )


# finfo
@given(
    dtype_and_x=helpers.dtype_and_values(ivy.valid_float_dtypes, 2),
    as_variable=helpers.list_of_length(st.booleans(), 2),
    num_positional_args=st.integers(1, 1),
    native_array=helpers.list_of_length(st.booleans(), 2),
    container=helpers.list_of_length(st.booleans(), 2),
    instance_method=st.booleans(),
)
def test_finfo(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x = dtype_and_x
    if (v == [] for v in x):
        return
    helpers.test_array_function(
        dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "finfo",
        x=x,
    )


# iinfo
@given(
    dtype_and_x=helpers.dtype_and_values(ivy.valid_int_dtypes, 2),
    as_variable=helpers.list_of_length(st.booleans(), 2),
    num_positional_args=st.integers(1, 1),
    native_array=helpers.list_of_length(st.booleans(), 2),
    container=helpers.list_of_length(st.booleans(), 2),
    instance_method=st.booleans(),
)
def test_iinfo(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x = dtype_and_x
    if (v == [] for v in x):
        return
    helpers.test_array_function(
        dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "iinfo",
        x=x,
    )


# is_float_dtype
@given(
    array_shape=helpers.lists(
        st.integers(1, 5), min_size="num_dims", max_size="num_dims", size_bounds=[1, 5]
    ),
    dtype=st.sampled_from(ivy_np.valid_dtypes),
    data=st.data(),
    as_variable=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_is_float_dtype(
    array_shape,
    dtype,
    data,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    # smoke this for torch
    if fw == "torch" and dtype in ["uint16", "uint32", "uint64"]:
        return
    x = data.draw(helpers.nph.arrays(shape=array_shape, dtype=dtype))
    helpers.test_array_function(
        dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "is_float_dtype",
        dtype_in=x,
    )


# is_int_dtype
@given(
    array_shape=helpers.lists(
        st.integers(1, 5), min_size="num_dims", max_size="num_dims", size_bounds=[1, 5]
    ),
    dtype=st.sampled_from(ivy_np.valid_dtypes),
    data=st.data(),
    as_variable=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_is_int_dtype(
    array_shape,
    dtype,
    data,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    # smoke this for torch
    if fw == "torch" and dtype in ["uint16", "uint32", "uint64"]:
        return
    x = data.draw(helpers.nph.arrays(shape=array_shape, dtype=dtype))
    helpers.test_array_function(
        dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "is_int_dtype",
        dtype_in=x,
    )


# Still to Add #
# ---------------#

# broadcast_arrays
# result_type
