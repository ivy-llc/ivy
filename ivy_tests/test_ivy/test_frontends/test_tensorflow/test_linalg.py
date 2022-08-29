# global
import numpy as np
from hypothesis import given, strategies as st

import sys

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.tensorflow as ivy_tf
from ivy_tests.test_ivy.helpers import handle_cmd_line_args
from ivy_tests.test_ivy.test_functional.test_core.test_linalg import _get_dtype_value1_value2_axis_for_tensordot


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


@st.composite
def _solve_get_dtype_and_data(draw):
    batch = draw(st.integers(min_value=1, max_value=5))
    random_size = draw(st.integers(min_value=2, max_value=4))
    # shape = (batch, random_size, random_size)

    input_dtype = draw(
        st.shared(st.sampled_from(ivy_tf.valid_float_dtypes), key="shared_dtype")
    )
    shape = (random_size, random_size)
    tmp = []
    for i in range(batch):
        tmp.append(
            draw(
                helpers.array_values(
                    dtype=input_dtype,
                    shape=shape,
                    min_value=-10,
                    max_value=10,
                ).filter(lambda x: np.linalg.cond(x) < 1 / sys.float_info.epsilon)
            )
        )

    data1 = (input_dtype, tmp)

    shape = (batch, random_size, draw(st.integers(min_value=2, max_value=4)))
    data2 = draw(
        helpers.dtype_and_values(
            available_dtypes=ivy_tf.valid_float_dtypes,
            shape=shape,
            min_value=-10,
            max_value=10,
        )
    )

    return data1, data2


# solve
@handle_cmd_line_args
@given(
    dtype_and_x=_solve_get_dtype_and_data(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.solve"
    ),
)
def test_tensorflow_solve(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):

    data1, data2 = dtype_and_x
    input_dtype1, x = data1
    input_dtype2, y = data2

    helpers.test_frontend_function(
        input_dtypes=[input_dtype1, input_dtype2],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="linalg.solve",
        x=np.asarray(x, dtype=input_dtype1),
        y=np.asarray(y, dtype=input_dtype2),
    )


# slogdet
@given(
    dtype_and_x=_get_dtype_and_matrix(),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.slogdet"
    ),
    native_array=st.booleans(),
)
def test_tensorflow_slogdet(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="linalg.slogdet",
        x=np.asarray(x, dtype=input_dtype),
    )


# tensordot
@given(
    dtype_x1_x2_axis=_get_dtype_value1_value2_axis_for_tensordot(),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.tensordot"
    ),
    native_array=st.booleans(),
)
def test_tensorflow_tensordot(
    *,
    dtype_x1_x2_axis,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    (
        dtype,
        x1,
        x2,
        axis,
    ) = dtype_x1_x2_axis

    as_variable = [as_variable, as_variable]
    native_array = [native_array, native_array]

    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="tensordot",
        x=x1,
        x2=x2,
        axes=axis,
    )