"""Collection of tests for creation functions."""

# global
import numpy as np
from numbers import Number
from hypothesis import given, strategies as st
from pyparsing import one_of

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np


# native_array
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_numeric_dtypes).filter(lambda x: isinstance(x[1], list)),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="native_array"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_native_array(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x = dtype_and_x
    helpers.test_array_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "native_array",
        x=x
    )

# linspace
@given(
    start = st.integers(-50, 50) | st.floats(-50, 50) | helpers.lists(st.one_of(st.integers(-50, 50),st.floats(-50, 50)) , min_size=1),
    stop = st.integers(-50, 50) | st.floats(-50, 50) | helpers.lists(st.one_of(st.integers(-50, 50),st.floats(-50, 50)), min_size=1),
    num = st.integers(1, 50),
    axis = st.none() | st.integers(1, 50),
    endpoint = st.booleans(),
    dtype = st.sampled_from(ivy_np.valid_numeric_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="linspace"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_linspace(
    start,
    stop,
    num,
    axis,
    endpoint,
    dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    helpers.test_array_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "linspace",
        start = start,
        stop = stop,
        num = num,
        axis = axis,
        endpoint = endpoint
    )



# logspace
@given(
    start_ = st.integers() | st.lists(st.integers(), min_size=1), 
    stop = st.integers() | st.lists(st.integers(), min_size=1),
    num = st.integers(1, 50),
    base = st.floats().filter(lambda x: True if x != 0 else False),
    axis = st.none() | st.integers(1, 50),
    dtype = st.sampled_from(ivy_np.valid_numeric_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="logspace"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_logspace(
    start,
    stop,
    num,
    base,
    axis,
    dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    helpers.test_array_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "logspace",
        start = start,
        stop = stop,
        num = num,
        base = base,
        axis = axis,
    )


# arange()
@given(
    start = st.integers(min_value = 0), 
    stop = st.integers(min_value = 0) | st.none(),
    step = st.integers().filter(lambda x : True if x != 0 else False),
    dtype = st.sampled_from(ivy_np.valid_int_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="arange"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_arange(
    start,
    stop,
    step,
    dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    helpers.test_array_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "arange",
        start = start,
        stop = stop,
        step = step,
    )


# asarray()
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_int_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="asarray"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_asarray(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x = dtype_and_x
    helpers.test_array_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "asarray",
        object_in = np.asarray(x),
    )


# empty()
@given(
    shape = st.integers(0,5) | st.lists(st.integers(0,5), min_size=1, max_size = 5),
    dtype = st.sampled_from(ivy_np.valid_int_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="empty"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_empty(
    shape,
    dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):    
    helpers.test_array_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "empty",
        shape=shape,
    )


# empty_like()
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_numeric_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="empty_like"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_empty_like(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x = dtype_and_x
    helpers.test_array_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "empty_like",
        x=ivy.asarray(x, dtype=dtype),
    )


# eye()
@given(
    n_rows = st.integers(min_value = 0, max_value = 5),
    n_cols = st.none() | st.integers(min_value = 0, max_value = 5),
    k = st.integers(min_value = -5, max_value = 5),
    dtype = st.sampled_from(ivy_np.valid_int_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="eye"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_eye(
    n_rows,
    n_cols,
    k,
    dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
     
    helpers.test_array_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "eye",
        n_rows = n_rows,
        n_cols = n_cols,
        k = k,
    )


# from_dlpack()
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_int_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="from_dlpack"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_from_dlpack(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "from_dlpack",
        x=np.asarray(x, dtype=dtype),
    )


# full()
@given(
    shape = st.integers(1,5) | st.lists(st.integers(1,5), min_size=1, max_size = 5),
    fill_value= st.integers() | st.floats(),
    dtype = st.sampled_from(ivy_np.valid_int_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="full"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_full(
    shape,
    fill_value,
    dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    helpers.test_array_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "full",
        shape=shape,
        fill_value=fill_value,
    )


# full_like()
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_int_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="full_like"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    fill_value= st.integers() | st.floats(),
)
def test_full_like(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    fill_value,
):
    dtype, x = dtype_and_x
     
    helpers.test_array_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "full_like",
        x=np.asarray(x, dtype=dtype),
        fill_value = fill_value,
    )


# meshgrid
@given(
    dtype_and_x=helpers.dtype_and_values(
        ivy_np.valid_int_dtypes,
        st.shared(st.integers(1, 3), key="num_arrays"),
        shared_dtype=True,
    ),
    as_variable=st.booleans(),
    num_positional_args=st.shared(st.integers(1, 3), key="num_arrays"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_meshgrid(
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
        "meshgrid",
        **kw
    )


# ones()
@given(
shape = st.integers(1,5) | st.lists(st.integers(1,5), min_size=1, max_size = 5),
dtype = st.sampled_from(ivy_np.valid_int_dtypes),
as_variable = st.booleans(),
with_out=st.booleans(),
num_positional_args=helpers.num_positional_args(fn_name="ones"),
native_array = st.booleans(),
container=st.booleans(),
instance_method=st.booleans(),
)
def test_ones(
    shape,
    dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    helpers.test_array_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "ones",
        shape=shape,
    )


# ones_like()
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_int_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_ones_like(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "ones_like",
        x=np.asarray(x, dtype=dtype),
    )


# tril()
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_int_dtypes),
    k=st.integers(-5, 5),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="tril"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_tril(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    k,
):
    dtype, x = dtype_and_x
    helpers.test_array_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "tril",
        x=np.asarray(x, dtype=dtype),
        k=0,
    )


# triu()
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_int_dtypes),
    k=st.integers(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="triu"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_triu(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    k,
):
    dtype, x = dtype_and_x
     
    helpers.test_array_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "triu",
        x=np.asarray(x, dtype=dtype),
        k = k,
    )


# zeros()
@given(
shape = st.integers(1,5) | st.lists(st.integers(1,5), min_size=1, max_size = 5),
dtype = st.sampled_from(ivy_np.valid_int_dtypes),
as_variable = st.booleans(),
with_out=st.booleans(),
num_positional_args=helpers.num_positional_args(fn_name="zeros"),
native_array = st.booleans(),
container=st.booleans(),
instance_method=st.booleans(),
)
def test_zeros(
    shape,
    dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    helpers.test_array_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "zeros",
        shape=shape,
    )


# zeros_like()
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_int_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_zeros_like(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "zeros_like",
        x=np.asarray(x, dtype=dtype),
    )
    