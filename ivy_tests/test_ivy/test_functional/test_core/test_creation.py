"""Collection of tests for creation functions."""

# global
from enum import Flag
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
    x = helpers.lists(st.integers(0,5) | st.floats(0,5), min_size=1, max_size=5),
    as_tuple = st.booleans(),
    dtype = st.sampled_from(ivy_np.valid_numeric_dtypes),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="native_array"),
    native_array=st.booleans(),
    container=st.booleans(),
)
def test_native_array(
    x,
    as_tuple,
    dtype,
    as_variable,
    num_positional_args,
    native_array,
    container,
    fw,
    device
):
    
    instance_method = False
    with_out = False

    #as tuple if generated as True
    if isinstance(x, list) and as_tuple is True:
        x = tuple(x)


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
        x=x,
        dtype=dtype,
        device=device
    )

# linspace - Tensor tuple index
@given(
    start = st.integers(1,5) | st.lists(st.integers(1,5), min_size=1, max_size=5), 
    stop = st.integers(1,5) | st.lists(st.integers(1,5), min_size=1, max_size=5),
    num = st.integers(1, 5),
    #axis = st.none() | st.integers(1, 5),
    dtype = st.sampled_from(ivy_np.valid_float_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="logspace"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_linspace(
    start,
    stop,
    num,
    #axis,
    device,
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
        False,
        False,
        fw,
        "linspace",
        start=start,
        stop=stop,
        num=num,
        axis=None,
        device=device
    )



# logspace - same as linspace
@given(
    start = st.integers(1,5) | st.lists(st.integers(1,5), min_size=1, max_size=5), 
    stop = st.integers(1,5) | st.lists(st.integers(1,5), min_size=1, max_size=5),
    num = st.integers(1, 5),
    base = st.floats().filter(lambda x: True if x != 0 else False),
    #axis = st.none() | st.integers(1, 5),
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
    #axis,
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
        False,
        False,
        fw,
        "logspace",
        start = start,
        stop = stop,
        num = num,
        base = base,
        #axis = axis,
    )


# arange() - passing right now
@given(
    start = st.integers(0,5), 
    stop = st.integers(0,5) | st.none(),
    step = st.integers(0,5).filter(lambda x : True if x != 0 else False),
    dtype = st.sampled_from(ivy_np.valid_int_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="arange"),
)
def test_arange(
    start,
    stop,
    step,
    dtype,
    device,
    num_positional_args,
    fw,
):
    helpers.test_array_function(
        dtype,
        False,
        False,
        num_positional_args,
        False,
        False,
        False,
        fw,
        "arange",
        start = start,
        stop = stop,
        step = step,
        dtype=dtype,
        device=device
    )


# asarray() - passing right now
@given(
    x = helpers.lists(st.one_of(st.integers(-5, 5), st.floats(-5, 5)), min_size=1, max_size=5),
    dtype = st.sampled_from(ivy_np.valid_numeric_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="asarray"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_asarray(
    x,
    dtype,
    device,
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
        False,
        False,
        fw,
        "asarray",
        object_in=x,
        dtype=dtype,
        device=device
    )


# empty() - failing assertion of zero values
@given(
    shape = st.lists(st.integers(min_value = 0, max_value=5), min_size = 1, max_size = 3),
    as_tuple = st.booleans(),
    dtype = st.sampled_from(ivy_np.valid_float_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="empty"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_empty(
    shape,
    as_tuple,
    dtype,
    device,
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
        False,
        with_out,
        num_positional_args,
        False,
        False,
        False,
        fw,
        "empty",
        shape=shape,
        dtype=dtype,
        device=device,
    )

# empty_like() - Failing assertion of 0 values
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
        False,
        False,
        fw,
        "empty_like",
        x=ivy.asarray(x, dtype=dtype),
    )


# eye() - passing
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
    device,
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
        False,
        False,
        fw,
        "eye",
        n_rows = n_rows,
        n_cols = n_cols,
        k = k,
        dtype=dtype,
        device=device,
    )


# from_dlpack() - jax and tf fail
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
    helpers.test_array_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "from_dlpack",
        x=np.asarray(x),
    )


# full() - eager tensor error
@given(
shape = st.lists(st.integers(min_value = 0, max_value=10), min_size = 1, max_size = 3),
as_tuple = st.booleans(),
fill_value = st.integers(0,5) | st.floats(0,5),
dtype = st.sampled_from(ivy_np.valid_int_dtypes),
num_positional_args=helpers.num_positional_args(fn_name="full"),
container = st.booleans(),
)
def test_full(
    shape,
    as_tuple,
    fill_value,
    dtype,
    device,
    num_positional_args,
    container,
    fw,
):
    #as tuple if generated as True
    if isinstance(shape, list) and as_tuple is True:
        shape = tuple(shape)

    instance_method = False
    as_variable = False
    native_array = False
    with_out = False
    
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
        fill_value = fill_value,
        dtype=dtype,
        device=device,
    )


# full_like() - passing with np array
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


# meshgrid - jax must be 1D
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
    helpers.test_array_function(
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


# ones() - dtype error for jax
@given(
shape = st.integers(0,10) | st.lists(st.integers(min_value = 0, max_value=10), min_size = 0, max_size = 3),
as_tuple = st.booleans(),
dtype = st.sampled_from(ivy_np.valid_int_dtypes),
num_positional_args=helpers.num_positional_args(fn_name="ones"),
container = st.booleans(),
)
def test_ones(
    shape,
    as_tuple,
    dtype,
    device,
    num_positional_args,
    container,
    fw,
):
    #as tuple if generated as True
    if isinstance(shape, list) and as_tuple is True:
        shape = tuple(shape)

    instance_method = False
    as_variable = False
    native_array = False
    with_out = False
    
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
        dtype=dtype,
        device=device,
    )


# ones_like() - passing with np.array, needs reformat
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
    helpers.test_array_function(
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


# tril() - not passing
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


# triu() - Not passing
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


# zeros() - Passing, int, tuple, list. Needs reformat
@given(
shape = st.integers(0,10) | st.lists(st.integers(min_value = 0, max_value=10), min_size = 0, max_size = 3),
as_tuple = st.booleans(),
dtype = st.sampled_from(ivy_np.valid_int_dtypes),
num_positional_args=helpers.num_positional_args(fn_name="zeros"),
container = st.booleans(),
)
def test_zeros(
    shape,
    as_tuple,
    dtype,
    device,
    num_positional_args,
    container,
    fw,
):
    #as tuple if generated as True
    if isinstance(shape, list) and as_tuple is True:
        shape = tuple(shape)

    instance_method = False
    as_variable = False
    native_array = False
    with_out = False
    
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
        dtype=dtype,
        device=device,
    )


# zeros_like() - Passing but with np.array
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
    helpers.test_array_function(
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
    