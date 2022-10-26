from hypothesis import given
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.keras.activations.hard_sigmoid"
    ),
)
def test_tensorflow_hard_sigmoid(
    dtype_and_x, as_variable, num_positional_args, native_array
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="keras.activations.hard_sigmoid",
        rtol=1e-2,
        atol=1e-2,
        x=x[0],
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.keras.activations.linear"
    ),
)
def test_tensorflow_linear(dtype_and_x, as_variable, num_positional_args, native_array):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="keras.activations.linear",
        x=x[0],
    )


# sigmoid
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.keras.activations.sigmoid"
    ),
)
def test_tensorflow_sigmoid(
    dtype_and_x, as_variable, num_positional_args, native_array
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="keras.activations.sigmoid",
        rtol=1e-2,
        atol=1e-2,
        x=x[0],
    )


# tanh
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.keras.activations.tanh"
    ),
)
def test_tensorflow_tanh(dtype_and_x, as_variable, num_positional_args, native_array):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="keras.activations.tanh",
        rtol=1e-2,
        atol=1e-2,
        x=x[0],
    )


# softmax
@handle_cmd_line_args
@given(
    dtype_x_and_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        max_axes_size=1,
        force_int_axis=True,
        valid_axis=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.keras.activations.softmax"
    ),
)
def test_tensorflow_softmax(
    dtype_x_and_axis, as_variable, num_positional_args, native_array
):
    input_dtype, x, axis = dtype_x_and_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="keras.activations.softmax",
        x=x[0],
        axis=axis,
    )


# gelu test
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    approximate=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.keras.activations.gelu"
    ),
)
def test_tensorflow_gelu(
    dtype_and_x, approximate, as_variable, num_positional_args, native_array
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="keras.activations.gelu",
        x=x[0],
        approximate=approximate,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric")
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.keras.activations.relu"
    ),
)
def test_tensorflow_relu(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="keras.activations.relu",
        x=x[0],
    )


# softplus
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.keras.activations.softplus"
    ),
)
def test_tensorflow_softplus(
    dtype_and_x, as_variable, num_positional_args, native_array
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="keras.activations.softplus",
        rtol=1e-2,
        atol=1e-2,
        x=x[0],
    )


# softsign
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.keras.activations.softsign"
    ),
)
def test_tensorflow_softsign(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="keras.activations.softsign",
        rtol=1e-2,
        atol=1e-2,
        x=x[0],
    )


# swish
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.keras.activations.swish"
    ),
)
def test_tensorflow_swish(dtype_and_x, as_variable, num_positional_args, native_array):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="keras.activations.swish",
        rtol=1e-2,
        atol=1e-2,
        x=x[0],
    )


# elu
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_value=-3,
        max_value=3,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
    ),
    alpha=st.one_of(
        helpers.floats(
            min_value=-3,
            max_value=3,
        )
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.keras.activations.elu"
    ),
)
def test_tensorflow_elu(
    dtype_x,
    alpha,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        with_inplace=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="keras.activations.elu",
        x=x[0],
        alpha=alpha,
    )


# selu
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_value=-3,
        max_value=3,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.keras.activations.selu"
    ),
)
def test_tensorflow_selu(
    dtype_x,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        with_inplace=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="keras.activations.selu",
        x=x[0],
    )
