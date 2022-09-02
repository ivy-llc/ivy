# global
import ivy
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
import ivy.functional.backends.tensorflow as ivy_tf
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# Acos
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(set(ivy_tf.valid_float_dtypes))
        ),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Acos"
    ),
)
def test_tensorflow_Acos(
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
        fn_tree="raw_ops.Acos",
        x=np.asarray(x, dtype=input_dtype),
    )


# Acosh
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(set(ivy_tf.valid_float_dtypes))
        ),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Acosh"
    ),
)
def test_tensorflow_Acosh(
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
        fn_tree="raw_ops.Acosh",
        x=np.asarray(x, dtype=input_dtype),
    )


# noinspection DuplicatedCode
@st.composite
def _arrays_idx_n_dtypes(draw):
    num_dims = draw(st.shared(helpers.ints(min_value=1, max_value=4), key="num_dims"))
    num_arrays = draw(
        st.shared(helpers.ints(min_value=2, max_value=4), key="num_arrays")
    )
    common_shape = draw(
        helpers.lists(
            arg=helpers.ints(min_value=2, max_value=3),
            min_size=num_dims - 1,
            max_size=num_dims - 1,
        )
    )
    unique_idx = draw(helpers.ints(min_value=0, max_value=num_dims - 1))
    unique_dims = draw(
        helpers.lists(
            arg=helpers.ints(min_value=2, max_value=3),
            min_size=num_arrays,
            max_size=num_arrays,
        )
    )
    xs = list()
    available_dtypes = tuple(
        set(ivy_np.valid_float_dtypes).intersection(ivy_tf.valid_float_dtypes)
    )
    input_dtypes = draw(
        helpers.array_dtypes(available_dtypes=available_dtypes, shared_dtype=True)
    )
    for ud, dt in zip(unique_dims, input_dtypes):
        x = draw(
            helpers.array_values(
                shape=common_shape[:unique_idx] + [ud] + common_shape[unique_idx:],
                dtype=dt,
            )
        )
        xs.append(x)
    return xs, input_dtypes, unique_idx


# concat
@handle_cmd_line_args
@given(
    xs_n_input_dtypes_n_unique_idx=_arrays_idx_n_dtypes(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.concat"
    ),
)
def test_tensorflow_concat(
    xs_n_input_dtypes_n_unique_idx,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    xs, input_dtypes, unique_idx = xs_n_input_dtypes_n_unique_idx
    xs = [np.asarray(x, dtype=dt) for x, dt in zip(xs, input_dtypes)]
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="concat",
        values=xs,
        axis=unique_idx,
    )


# Cos
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(set(ivy_tf.valid_float_dtypes))
        ),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Cos"
    ),
)
def test_tensorflow_Cos(
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
        fn_tree="raw_ops.Cos",
        x=np.asarray(x, dtype=input_dtype),
    )


# Cosh
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(set(ivy_tf.valid_float_dtypes))
        ),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Cosh"
    ),
)
def test_tensorflow_Cosh(
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
        fn_tree="raw_ops.Cosh",
        x=np.asarray(x, dtype=input_dtype),
    )


@st.composite
def _dtypes(draw):
    return draw(
        st.shared(
            helpers.list_of_length(
                x=st.sampled_from(ivy_tf.valid_numeric_dtypes), length=1
            ),
            key="dtype",
        )
    )


@st.composite
def _fill_value(draw):
    dtype = draw(_dtypes())[0]
    if ivy.is_uint_dtype(dtype):
        return draw(helpers.ints(min_value=0, max_value=5))
    elif ivy.is_int_dtype(dtype):
        return draw(helpers.ints(min_value=-5, max_value=5))
    return draw(helpers.floats(min_value=-5, max_value=5))


# fill
@handle_cmd_line_args
@given(
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    fill_value=_fill_value(),
    dtypes=_dtypes(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.fill"
    ),
)
def test_tensorflow_fill(
    shape,
    fill_value,
    dtypes,
    with_out,
    as_variable,
    native_array,
    num_positional_args,
    fw,
):
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="fill",
        dims=shape,
        value=fill_value,
        rtol=1e-05,
    )


# Asin
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(set(ivy_tf.valid_float_dtypes))
        )
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Asin"
    ),
)
def test_tensorflow_Asin(
    dtype_and_x, as_variable, num_positional_args, fw, native_array
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="raw_ops.Asin",
        x=np.asarray(x, dtype=dtype),
    )


# Atan
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(set(ivy_tf.valid_float_dtypes))
        )
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Atan"
    ),
)
def test_tensorflow_Atan(
    dtype_and_x, as_variable, num_positional_args, fw, native_array
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="raw_ops.Atan",
        x=np.asarray(x, dtype=dtype),
    )


# BitwiseAnd
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy.all_int_dtypes,
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.BitwiseAnd"
    ),
)
def test_tensorflow_BitwiseAnd(
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
        fn_tree="raw_ops.BitwiseAnd",
        x=np.asarray(x[0], dtype=input_dtype[0]),
        y=np.asarray(x[1], dtype=input_dtype[1]),
    )


# BitwiseOr
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy.all_int_dtypes,
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.BitwiseOr"
    ),
)
def test_tensorflow_BitwiseOr(
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
        fn_tree="raw_ops.BitwiseOr",
        x=np.asarray(x[0], dtype=input_dtype[0]),
        y=np.asarray(x[1], dtype=input_dtype[1]),
    )


# BitwiseXor
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy.all_int_dtypes,
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.BitwiseXor"
    ),
)
def test_tensorflow_BitwiseXor(
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
        fn_tree="raw_ops.BitwiseXor",
        x=np.asarray(x[0], dtype=input_dtype[0]),
        y=np.asarray(x[1], dtype=input_dtype[1]),
    )


# Atanh
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(set(ivy_tf.valid_float_dtypes))
        )
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Atanh"
    ),
)
def test_tensorflow_Atanh(
    dtype_and_x, as_variable, num_positional_args, fw, native_array
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="raw_ops.Atanh",
        x=np.asarray(x, dtype=dtype),
    )


# Tan
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(set(ivy_tf.valid_float_dtypes))
        ),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Tan"
    ),
)
def test_tensorflow_Tan(
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
        fn_tree="raw_ops.Tan",
        x=np.asarray(x, dtype=input_dtype),
    )


# Square
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
        fn_name="ivy.functional.frontends.tensorflow.Square"
    ),
)
def test_tensorflow_Square(
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
        fn_tree="raw_ops.Square",
        x=np.asarray(x, dtype=input_dtype),
    )


# Sqrt
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(set(ivy_tf.valid_float_dtypes))
        ),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Sqrt"
    ),
)
def test_tensorflow_Sqrt(
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
        fn_tree="raw_ops.Sqrt",
        x=np.asarray(x, dtype=input_dtype),
    )


# Tanh
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(set(ivy_tf.valid_float_dtypes))
        ),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Tanh"
    ),
)
def test_tensorflow_Tanh(
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
        fn_tree="raw_ops.Tanh",
        x=np.asarray(x, dtype=input_dtype),
    )


# Maximum
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(set(ivy_tf.valid_float_dtypes))
        ),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Maximum"
    ),
)
def test_tensorflow_Maximum(
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
        fn_tree="raw_ops.Maximum",
        x=np.asarray(x[0], dtype=input_dtype[0]),
        y=np.asarray(x[1], dtype=input_dtype[1]),
    )


# Minimum
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(set(ivy_tf.valid_float_dtypes))
        ),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Minimum"
    ),
)
def test_tensorflow_Minimum(
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
        fn_tree="raw_ops.Minimum",
        x=np.asarray(x[0], dtype=input_dtype[0]),
        y=np.asarray(x[1], dtype=input_dtype[1]),
    )


# Sub
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy.valid_numeric_dtypes, num_arrays=2, shared_dtype=True
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Sub"
    ),
)
def test_tensorflow_Sub(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="raw_ops.Sub",
        x=np.asarray(x[0], dtype=dtype[0]),
        y=np.asarray(x[1], dtype=dtype[1]),
    )


# Less
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy.valid_numeric_dtypes,
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Less"
    ),
)
def test_tensorflow_Less(
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
        fn_tree="raw_ops.Less",
        x=np.asarray(x[0], dtype=input_dtype[0]),
        y=np.asarray(x[1], dtype=input_dtype[1]),
    )


# Floor
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(set(ivy_tf.valid_float_dtypes))
        ),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Floor"
    ),
)
def test_tensorflow_Floor(
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
        fn_tree="raw_ops.Floor",
        x=np.asarray(x, dtype=input_dtype),
    )


# FloorDiv
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(set(ivy_tf.valid_float_dtypes))
        ),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.FloorDiv"
    ),
)
def test_tensorflow_FloorDiv(
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
        fn_tree="raw_ops.FloorDiv",
        x=np.asarray(x[0], dtype=input_dtype[0]),
        y=np.asarray(x[1], dtype=input_dtype[1]),
    )


# Exp
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(set(ivy_tf.valid_float_dtypes))
        ),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Tanh"
    ),
)
def test_tensorflow_Exp(
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
        fn_tree="raw_ops.Exp",
        x=np.asarray(x, dtype=input_dtype),
    )


# Expm1
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(set(ivy_tf.valid_float_dtypes))
        ),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Expm1"
    ),
)
def test_tensorflow_Expm1(
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
        fn_tree="raw_ops.Expm1",
        x=np.asarray(x, dtype=input_dtype),
    )
