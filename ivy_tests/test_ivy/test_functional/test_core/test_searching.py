"""Collection of tests for searching functions."""

# global
from hypothesis import given, strategies as st
import numpy as np

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np

# argmax
@given(
    array_shape=helpers.lists(
        st.integers(1, 10), min_size="num_dims", max_size="num_dims", size_bounds=[1, 10]
    ),
    input_dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
    data=st.data(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="argmax"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_argmax(
    array_shape,
    input_dtype,
    data,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    x = data.draw(
        helpers.nph.arrays(shape=array_shape, dtype=input_dtype).filter(
            lambda x: not np.any(np.isnan(x))
        )
    )

    ndim = len(x.shape)
    axis = data.draw(st.integers(-ndim, ndim - 1))
    keepdims = data.draw(st.booleans())

    helpers.test_array_function(
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "argmax",
        x=x,
        axis=axis,
        keepdims=keepdims
    )
    
# argmin
@given(
    array_shape=helpers.lists(
        st.integers(1, 10), min_size="num_dims", max_size="num_dims", size_bounds=[1, 10]
    ),
    input_dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
    data=st.data(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="argmin"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_argmin(
    array_shape,
    input_dtype,
    data,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    x = data.draw(
        helpers.nph.arrays(shape=array_shape, dtype=input_dtype).filter(
            lambda x: not np.any(np.isnan(x))
        )
    )

    ndim = len(x.shape)
    axis = data.draw(st.integers(-ndim, ndim - 1))
    keepdims = data.draw(st.booleans())

    helpers.test_array_function(
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "argmin",
        x=x,
        axis=axis,
        keepdims=keepdims
    )
    
# nonzero
@given(
    array_shape=helpers.lists(
        st.integers(1, 10), min_size="num_dims", max_size="num_dims", size_bounds=[1, 10]
    ),
    input_dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
    data=st.data(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="nonzero"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_nonzero(
    array_shape,
    input_dtype,
    data,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    x = data.draw(
        helpers.nph.arrays(shape=array_shape, dtype=input_dtype).filter(
            lambda x: not np.any(np.isnan(x))
        )
    )

    helpers.test_array_function(
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "nonzero",
        x=x
    )
    
# where
@given(
    array_shape=helpers.lists(
        st.integers(1, 10), min_size="num_dims", max_size="num_dims", size_bounds=[1, 10]
    ),
    input_dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
    data=st.data(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="where"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_where(
    array_shape,
    input_dtype,
    data,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    condition = data.draw(
        helpers.nph.arrays(shape=array_shape[0], dtype=st.booleans()))
    x1 = data.draw(
        helpers.nph.arrays(shape=array_shape[1], dtype=input_dtype).filter(
            lambda x: not np.any(np.isnan(x))
        )
    )
    x2 = data.draw(
        helpers.nph.arrays(shape=array_shape[1], dtype=input_dtype).filter(
            lambda x: not np.any(np.isnan(x))
        )
    )

    ndim = len(x1.shape)

    helpers.test_array_function(
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "where",
        condition=condition,
        x1=x1,
        x2=x2
    )