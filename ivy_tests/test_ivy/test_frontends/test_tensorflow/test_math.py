# global
import ivy
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
import ivy.functional.backends.tensorflow as ivy_tf
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# add
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.add"
    ),
)
def test_tensorflow_add(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
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
        fn_tree="add",
        x=np.asarray(x[0], dtype=input_dtype[0]),
        y=np.asarray(x[1], dtype=input_dtype[1]),
    )


# tan
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.tan"
    ),
)
def test_tensorflow_tan(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
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
        fn_tree="tan",
        x=np.asarray(x, dtype=input_dtype),
    )


# multiply
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.multiply"
    ),
)
def test_tensorflow_multiply(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
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
        fn_tree="multiply",
        x=np.asarray(x[0], dtype=input_dtype[0]),
        y=np.asarray(x[1], dtype=input_dtype[1]),
    )


# subtract
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.subtract"
    ),
)
def test_tensorflow_subtract(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
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
        fn_tree="subtract",
        x=np.asarray(x[0], dtype=input_dtype[0]),
        y=np.asarray(x[1], dtype=input_dtype[1]),
    )


# logical_xor
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple([ivy.bool]),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.logical_xor"
    ),
)
def test_tensorflow_logical_xor(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
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
        fn_tree="math.logical_xor",
        x=np.asarray(x[0], dtype=input_dtype[0]),
        y=np.asarray(x[1], dtype=input_dtype[1]),
    )


# divide
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.divide"
    ),
)
def test_tensorflow_divide(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
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
        fn_tree="divide",
        x=np.asarray(x[0], dtype=input_dtype[0]),
        y=np.asarray(x[1], dtype=input_dtype[1]),
    )


# negative
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_numeric_dtypes).intersection(
                set(ivy_tf.valid_numeric_dtypes)
            )
        ),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.negative"
    ),
)
def test_tensorflow_negative(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
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
        fn_tree="negative",
        x=np.asarray(x, dtype=input_dtype),
    )


# logical_and
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple([ivy.bool]),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.logical_and"
    ),
)
def test_tensorflow_logical_and(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
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
        fn_tree="math.logical_and",
        x=np.asarray(x[0], dtype=input_dtype[0]),
        y=np.asarray(x[1], dtype=input_dtype[1]),
    )


# log_sigmoid
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        small_value_safety_factor=1.0,
        large_value_safety_factor=1.0,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.log_sigmoid"
    ),
)
def test_tensorflow_log_sigmoid(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
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
        fn_tree="math.log_sigmoid",
        x=np.asarray(x, dtype=input_dtype),
    )


# reciprocal_no_nan()
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.reciprocal_no_nan"
    ),
)
def test_tensorflow_reciprocal_no_nan(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=1,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="math.reciprocal_no_nan",
        input_tensor=np.asarray(x, dtype=input_dtype),
    )


# reduce_all()
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple([ivy.bool]),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.reduce_all"
    ),
)
def test_tensorflow_reduce_all(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
):
    (
        input_dtype,
        x,
    ) = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="reduce_all",
        input_tensor=np.asarray(x, dtype=input_dtype),
    )


# reduce_any()
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple([ivy.bool]),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.reduce_any"
    ),
)
def test_tensorflow_reduce_any(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
):
    (
        input_dtype,
        x,
    ) = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="reduce_any",
        input_tensor=np.asarray(x, dtype=input_dtype),
    )


# reduce_euclidean_norm()
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.reduce_euclidean_norm"
    ),
)
def test_tensorflow_reduce_euclidean_norm(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
):
    (
        input_dtype,
        x,
    ) = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="math.reduce_euclidean_norm",
        input_tensor=np.asarray(x, dtype=input_dtype),
    )


# reduce_logsumexp()
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.reduce_logsumexp"
    ),
)
def test_tensorflow_reduce_logsumexp(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
):
    (
        input_dtype,
        x,
    ) = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="reduce_logsumexp",
        input_tensor=np.asarray(x, dtype=input_dtype),
    )


# argmax
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.statistical_dtype_values(function="argmax"),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.argmax"
    ),
)
def test_tensorflow_argmax(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x, axis = dtype_and_x
    if isinstance(axis, tuple):
        axis = axis[0]
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="math.argmax",
        input=np.asarray(x, dtype=input_dtype),
        axis=axis,
        output_type="int64",
    )


# reduce_max
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.reduce_max"
    ),
)
def test_tensorflow_reduce_max(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
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
        fn_tree="reduce_max",
        input_tensor=np.asarray(x, dtype=input_dtype),
    )


# reduce_min
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.reduce_min"
    ),
)
def test_tensorflow_reduce_min(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
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
        fn_tree="reduce_min",
        input_tensor=np.asarray(x, dtype=input_dtype),
    )


# reduce_prod
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.reduce_prod"
    ),
)
def test_tensorflow_reduce_prod(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
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
        fn_tree="reduce_prod",
        input_tensor=np.asarray(x, dtype=input_dtype),
    )


# reduce_std
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.reduce_std"
    ),
)
def test_tensorflow_reduce_std(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
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
        fn_tree="math.reduce_std",
        input_tensor=np.asarray(x, dtype=input_dtype),
    )


# asinh
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.asinh"
    ),
)
def test_tensorflow_asinh(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
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
        fn_tree="asinh",
        x=np.asarray(x, dtype=input_dtype),
    )


# reduce_sum
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.reduce_sum"
    ),
)
def test_tensorflow_reduce_sum(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
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
        fn_tree="reduce_sum",
        input_tensor=np.asarray(x, dtype=input_dtype),
    )


# reduce_variance
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.reduce_variance"
    ),
)
def test_tensorflow_reduce_variance(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
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
        fn_tree="math.reduce_variance",
        input_tensor=np.asarray(x, dtype=input_dtype),
    )


# scalar_mul
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        min_dim_size=2,
    ),
    scalar_val=helpers.list_of_length(x=st.floats(), length=1),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.scalar_mul"
    ),
)
def test_tensorflow_scalar_mul(
    dtype_and_x, scalar_val, as_variable, num_positional_args, native_array, fw
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
        fn_tree="scalar_mul",
        scalar=scalar_val[0],
        x=np.asarray(x, dtype=input_dtype),
    )


# divide_no_nan
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        num_arrays=2,
        available_dtypes=helpers.get_dtypes("float"),
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.divide_no_nan"
    ),
)
def test_tensorflow_divide_no_nan(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
):
    input_dtypes, xy = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="math.divide_no_nan",
        x=np.asarray(xy[0], dtype=input_dtypes[0]),
        y=np.asarray(xy[1], dtype=input_dtypes[1]),
    )


# erfcinv
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.erfcinv"
    ),
)
def test_tensorflow_erfcinv(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
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
        fn_tree="math.erfcinv",
        x=np.asarray(x, dtype=input_dtype),
    )


# is_non_decreasing
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.is_non_decreasing"
    ),
)
def test_tensorflow_is_non_decreasing(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
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
        fn_tree="math.is_non_decreasing",
        x=np.asarray(x, dtype=input_dtype),
    )


# is_strictly_increasing
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.is_strictly_increasing"
    ),
)
def test_tensorflow_is_strictly_increasing(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
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
        fn_tree="math.is_strictly_increasing",
        x=np.asarray(x, dtype=input_dtype),
    )


# count_nonzero
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), shape=(2, 3)
    ),
    axis=helpers.get_axis(shape=(2, 3), max_size=2),
    keepdims=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.count_nonzero"
    ),
)
def test_tensorflow_count_nonzero(
    dtype_and_x, axis, keepdims, as_variable, num_positional_args, native_array, fw
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
        fn_tree="math.count_nonzero",
        input=x,
        axis=axis,
        keepdims=keepdims,
        dtype=input_dtype,
    )


# confusion_matrix
@handle_cmd_line_args
@given(
    predictions=helpers.array_values(
        dtype=ivy.int32, shape=(3,), min_value=0, max_value=3
    ),
    labels=helpers.array_values(dtype=ivy.int32, shape=(3,), min_value=0, max_value=3),
    num_classes=st.integers(min_value=4, max_value=10),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.confusion_matrix"
    ),
)
def test_tensorflow_confusion_matrix(
    labels, predictions, num_classes, as_variable, num_positional_args, native_array, fw
):
    helpers.test_frontend_function(
        input_dtypes=ivy.int32,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="math.confusion_matrix",
        labels=labels,
        predictions=predictions,
        num_classes=num_classes,
    )


# polyval
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    x=helpers.array_values(shape=(3,), dtype=ivy.int32),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.polyval"
    ),
)
def test_tensorflow_polyval(
    dtype_and_x, x, as_variable, num_positional_args, native_array, fw
):
    input_dtype, coeffs = dtype_and_x
    coeffs = [coeffs]
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="math.polyval",
        coeffs=coeffs,
        x=x,
    )


# unsorted_segment_mean
@handle_cmd_line_args
@given(
    data=helpers.array_values(dtype=ivy.int32, shape=(5, 6), min_value=1, max_value=9),
    segment_ids=helpers.array_values(
        dtype=ivy.int32, shape=(5,), min_value=0, max_value=4
    ),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.unsorted_segment_mean"
    ),
    native_array=st.booleans(),
)
def test_tensorflow_unsorted_segment_mean(
    data, segment_ids, as_variable, num_positional_args, native_array, fw
):
    helpers.test_frontend_function(
        input_dtypes=[ivy.float32, ivy.int32],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="math.unsorted_segment_mean",
        data=np.asarray(data, dtype=np.float32),
        segment_ids=np.asarray(segment_ids, dtype=np.int32),
        num_segments=np.max(segment_ids) + 1,
    )


# unsorted_segment_sqrt_n
@handle_cmd_line_args
@given(
    data=helpers.array_values(dtype=ivy.int32, shape=(5, 6), min_value=1, max_value=9),
    segment_ids=helpers.array_values(
        dtype=ivy.int32, shape=(5,), min_value=0, max_value=4
    ),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.unsorted_segment_sqrt_n"
    ),
    native_array=st.booleans(),
)
def test_tensorflow_unsorted_segment_sqrt_n(
    data, segment_ids, as_variable, num_positional_args, native_array, fw
):
    helpers.test_frontend_function(
        input_dtypes=[ivy.float32, ivy.int32],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="math.unsorted_segment_sqrt_n",
        data=np.asarray(data, dtype=np.float32),
        segment_ids=np.asarray(segment_ids, dtype=np.int32),
        num_segments=np.max(segment_ids) + 1,
    )


# zero_fraction
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(set(ivy_tf.valid_float_dtypes))
        ),
        min_num_dims=1,
    ),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.zero_fraction"
    ),
    native_array=st.booleans(),
)
def test_tensorflow_zero_fraction(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
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
        fn_tree="math.zero_fraction",
        value=np.asarray(x, dtype=input_dtype),
    )
