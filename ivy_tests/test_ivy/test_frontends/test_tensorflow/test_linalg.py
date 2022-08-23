# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.tensorflow as ivy_tf
import ivy.functional.backends.numpy as ivy_np


@st.composite
def _get_dtype_and_matrix(draw):
    arbitrary_dims = draw(helpers.get_shape(max_dim_size=5))
    random_size = draw(st.integers(min_value=1, max_value=4))
    shape = (*arbitrary_dims, random_size, random_size)
    return draw(
        helpers.dtype_and_values(
            available_dtypes=ivy_tf.valid_float_dtypes,
            shape=shape,
            min_value=-10,
            max_value=10,
        )
    )


@given(
    dtype_and_input=_get_dtype_and_matrix(),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.det"
    ),
    native_array=st.booleans(),
)
def test_tensorflow_det(
    dtype_and_input, as_variable, num_positional_args, native_array, fw
):
    input_dtype, x = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="linalg.det",
        input=np.asarray(x, dtype=input_dtype),


    )


@given(
    dtype_and_input=_get_dtype_and_matrix(),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.eigvalsh"
    ),
    native_array=st.booleans(),
)
def test_tensorflow_eigvalsh(
    dtype_and_input, as_variable, num_positional_args, native_array, fw
):
    input_dtype, x = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="linalg.eigvalsh",
        input=np.asarray(x, dtype=input_dtype),
    )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes[1:],
        min_num_dims=2,
        min_value=-1e+05,
        max_value=1e+05
    ),
    as_variables=st.booleans(),
    native_array=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.matrix_rank"
    ),
    tolr=st.floats(allow_nan=False, allow_infinity=False) | st.just(None),
    data=st.data()
)
def test_matrix_rank(
    *,
    data,
    dtype_x,
    as_variables,
    num_positional_args,
    native_array,
    tolr,
    fw
):
    input_dtype , x = dtype_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variables,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="linalg.matrix_rank",
        atol=1.0,
        a=np.asarray(x, dtype=input_dtype),
        tol=tolr
    )
