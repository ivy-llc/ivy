# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
import ivy.functional.backends.tensorflow as ivy_tf
from ivy_tests.test_ivy.helpers import handle_frontend_method
from ivy_tests.test_ivy.test_frontends.test_tensorflow.test_raw_ops import (
    _pow_helper_shared_dtype,
)


# __add__
@handle_frontend_method(
    method_tree="tensorflow.Tensor.__add__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_add(dtype_and_x, as_variable, native_array):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[input_dtype[1]],
        method_as_variable_flags=as_variable,
        method_num_positional_args=1,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend="tensorflow",
        class_name="EagerTensor",
        method_name="__add__",
    )


@handle_frontend_method(
    method_tree="tensorflow.Tensor.__div__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_div(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[input_dtype[1]],
        method_as_variable_flags=as_variable,
        method_num_positional_args=1,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend="tensorflow",
        class_name="EagerTensor",
        method_name="__div__",
    )


@handle_frontend_method(
    method_tree="tensorflow.Tensor.get_shape",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        min_dim_size=1,
    ),
)
def test_tensorflow_instance_get_shape(dtype_and_x, as_variable, native_array):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_as_variable_flags=[],
        method_num_positional_args=0,
        method_native_array_flags=[],
        method_all_as_kwargs_np={},
        frontend="tensorflow",
        class_name="EagerTensor",
        method_name="get_shape",
    )


@handle_frontend_method(
    method_tree="tensorflow.Tensor.__eq__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
    ),
)
def test_tensorflow_instance_eq(dtype_and_x, as_variable, native_array):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=1,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend="tensorflow",
        class_name="EagerTensor",
        method_name="__eq__",
    )


@handle_frontend_method(
    method_tree="tensorflow.Tensor.__floordiv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_floordiv(dtype_and_x, as_variable, native_array):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[input_dtype[1]],
        method_as_variable_flags=as_variable,
        method_num_positional_args=1,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend="tensorflow",
        class_name="EagerTensor",
        method_name="__floordiv__",
    )


@handle_frontend_method(
    method_tree="tensorflow.Tensor.__ge__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_ge(dtype_and_x, as_variable, native_array):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[input_dtype[1]],
        method_as_variable_flags=as_variable,
        method_num_positional_args=1,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend="tensorflow",
        class_="EagerTensor",
        method_name="__ge__",
    )


@handle_frontend_method(
    method_tree="tensorflow.Tensor.__gt__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_gt(dtype_and_x, as_variable, native_array):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[input_dtype[1]],
        method_as_variable_flags=as_variable,
        method_num_positional_args=1,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend="tensorflow",
        class_="EagerTensor",
        method_name="__gt__",
    )


@handle_frontend_method(
    method_tree="tensorflow.Tensor.__le__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_le(dtype_and_x, as_variable, native_array):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[input_dtype[1]],
        method_as_variable_flags=as_variable,
        method_num_positional_args=1,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend="tensorflow",
        class_="EagerTensor",
        method_name="__le__",
    )


@handle_frontend_method(
    method_tree="tensorflow.Tensor.__lt__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_lt(dtype_and_x, as_variable, native_array):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[input_dtype[1]],
        method_as_variable_flags=as_variable,
        method_num_positional_args=1,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend="tensorflow",
        class_="EagerTensor",
        method_name="__lt__",
    )


@handle_frontend_method(
    method_tree="tensorflow.Tensor.__mul__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_mul(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[input_dtype[1]],
        method_as_variable_flags=as_variable,
        method_num_positional_args=1,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend="tensorflow",
        class_name="EagerTensor",
        method_name="__mul__",
    )


# __sub__
@handle_frontend_method(
    method_tree="tensorflow.Tensor.__sub__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(set(ivy_tf.valid_float_dtypes))
        ),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_sub(dtype_and_x, as_variable, native_array):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[input_dtype[1]],
        method_as_variable_flags=as_variable,
        method_num_positional_args=1,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend="tensorflow",
        class_name="EagerTensor",
        method_name="__sub__",
    )


# __ne__
@handle_frontend_method(
    method_tree="tensorflow.Tensor.__ne__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_ne(dtype_and_x, as_variable, native_array):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[input_dtype[1]],
        method_as_variable_flags=as_variable,
        method_num_positional_args=1,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend="tensorflow",
        class_name="EagerTensor",
        method_name="__ne__",
    )


# __radd__
@handle_frontend_method(
    method_tree="tensorflow.Tensor.__radd__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_radd(dtype_and_x, as_variable, native_array):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[input_dtype[1]],
        method_as_variable_flags=as_variable,
        method_num_positional_args=1,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "x": x[1],
        },
        frontend="tensorflow",
        class_name="EagerTensor",
        method_name="__radd__",
    )


# __rfloordiv__
@handle_frontend_method(
    method_tree="tensorflow.Tensor.__rfloordiv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_rfloordiv(dtype_and_x, as_variable, native_array):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[input_dtype[1]],
        method_as_variable_flags=as_variable,
        method_num_positional_args=1,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "x": x[1],
        },
        frontend="tensorflow",
        class_name="EagerTensor",
        method_name="__rfloordiv__",
    )


# __rsub__
@handle_frontend_method(
    method_tree="tensorflow.Tensor.__rsub__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_rsub(dtype_and_x, as_variable, native_array):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[input_dtype[1]],
        method_as_variable_flags=as_variable,
        method_num_positional_args=1,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "x": x[1],
        },
        frontend="tensorflow",
        class_name="EagerTensor",
        method_name="__rsub__",
    )


# __and__
@handle_frontend_method(
    method_tree="tensorflow.Tensor.__add__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_and(dtype_and_x, as_variable, native_array):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[input_dtype[1]],
        method_as_variable_flags=as_variable,
        method_num_positional_args=1,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend="tensorflow",
        class_name="EagerTensor",
        method_name="__and__",
    )


# __rand__
@handle_frontend_method(
    method_tree="tensorflow.Tensor.__rand__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_rand(dtype_and_x, as_variable, native_array):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[input_dtype[1]],
        method_as_variable_flags=as_variable,
        method_num_positional_args=1,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "x": x[1],
        },
        frontend="tensorflow",
        class_name="EagerTensor",
        method_name="__rand__",
    )


# __or__
@handle_frontend_method(
    method_tree="tensorflow.Tensor.__or__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_or(dtype_and_x, as_variable, native_array):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[input_dtype[1]],
        method_as_variable_flags=as_variable,
        method_num_positional_args=1,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend="tensorflow",
        class_name="EagerTensor",
        method_name="__or__",
    )


# __ror__
@handle_frontend_method(
    method_tree="tensorflow.Tensor.__ror__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_ror(dtype_and_x, as_variable, native_array):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[input_dtype[1]],
        method_as_variable_flags=as_variable,
        method_num_positional_args=1,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "x": x[1],
        },
        frontend="tensorflow",
        class_name="EagerTensor",
        method_name="__ror__",
    )


# __truediv__
@handle_frontend_method(
    method_tree="tensorflow.Tensor.__truediv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_truediv(dtype_and_x, as_variable, native_array):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[input_dtype[1]],
        method_as_variable_flags=as_variable,
        method_num_positional_args=1,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend="tensorflow",
        class_name="EagerTensor",
        method_name="__truediv__",
    )


# __rtruediv__
@handle_frontend_method(
    method_tree="tensorflow.Tensor.__rtruediv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_rtruediv(dtype_and_x, as_variable, native_array):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[input_dtype[1]],
        method_as_variable_flags=as_variable,
        method_num_positional_args=1,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "x": x[1],
        },
        frontend="tensorflow",
        class_name="EagerTensor",
        method_name="__rtruediv__",
    )


# __bool__
@handle_frontend_method(
    method_tree="tensorflow.Tensor.__bool__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        max_dim_size=1,
    ),
)
def test_tensorflow_instance_bool(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_as_variable_flags=[],
        method_num_positional_args=0,
        method_native_array_flags=[],
        method_all_as_kwargs_np={},
        frontend="tensorflow",
        class_name="EagerTensor",
        method_name="__bool__",
    )


# __nonzero__
@handle_frontend_method(
    method_tree="tensorflow.Tensor.__nonzero__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        max_dim_size=1,
    ),
)
def test_tensorflow_instance_nonzero(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_as_variable_flags=[],
        method_num_positional_args=0,
        method_native_array_flags=[],
        method_all_as_kwargs_np={},
        frontend="tensorflow",
        class_name="EagerTensor",
        method_name="__nonzero__",
    )


# __neg__
@handle_frontend_method(
    method_tree="tensorflow.Tensor.__neg__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=[
            "float32",
            "float64",
            "int8",
            "int16",
            "int32",
            "int64",
        ],
    ),
)
def test_tensorflow_instance_neg(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_as_variable_flags=[],
        method_num_positional_args=0,
        method_native_array_flags=[],
        method_all_as_kwargs_np={},
        frontend="tensorflow",
        class_name="EagerTensor",
        method_name="__neg__",
    )


# __rxor__
@handle_frontend_method(
    method_tree="tensorflow.Tensor.__rxor__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_rxor(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[input_dtype[1]],
        method_as_variable_flags=as_variable,
        method_num_positional_args=1,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "x": x[1],
        },
        frontend="tensorflow",
        class_name="EagerTensor",
        method_name="__rxor__",
    )


# __xor__
@handle_frontend_method(
    method_tree="tensorflow.Tensor.__xor__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_xor(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[input_dtype[1]],
        method_as_variable_flags=as_variable,
        method_num_positional_args=1,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend="tensorflow",
        class_name="EagerTensor",
        method_name="__xor__",
    )


# __matmul__
@handle_frontend_method(
    method_tree="tensorflow.Tensor.__matmul__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=[
            "float16",
            "float32",
            "float64",
            "int32",
            "int64",
        ],
        shape=(3, 3),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_matmul(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[input_dtype[1]],
        method_as_variable_flags=as_variable,
        method_num_positional_args=1,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend="tensorflow",
        class_name="EagerTensor",
        method_name="__matmul__",
    )


# __rmatmul__
@handle_frontend_method(
    method_tree="tensorflow.Tensor.__rmatmul__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=[
            "float16",
            "float32",
            "float64",
            "int32",
            "int64",
        ],
        shape=(3, 3),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_rmatmul(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[input_dtype[1]],
        method_as_variable_flags=as_variable,
        method_num_positional_args=1,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "x": x[1],
        },
        frontend="tensorflow",
        class_name="EagerTensor",
        method_name="__rmatmul__",
    )


# __array__
@handle_frontend_method(
    method_tree="tensorflow.Tensor.__array__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric")
    ),
)
def test_tensorflow_instance_array(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_as_variable_flags=[],
        method_num_positional_args=0,
        method_native_array_flags=[],
        method_all_as_kwargs_np={},
        frontend="tensorflow",
        class_name="EagerTensor",
        method_name="__array__",
    )


# __invert__
@handle_frontend_method(
    method_tree="tensorflow.Tensor.__invert__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer")
    ),
)
def test_tensorflow_instance_invert(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_as_variable_flags=[],
        method_num_positional_args=0,
        method_native_array_flags=[],
        method_all_as_kwargs_np={},
        frontend="tensorflow",
        class_name="EagerTensor",
        method_name="__invert__",
    )


# __rmul__
@handle_frontend_method(
    method_tree="tensorflow.Tensor.__rmul__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
        min_value=-100,
        max_value=100,
    ),
)
def test_tensorflow_instance_rmul(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[input_dtype[1]],
        method_as_variable_flags=as_variable,
        method_num_positional_args=1,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "x": x[1],
        },
        frontend="tensorflow",
        class_name="EagerTensor",
        method_name="__rmul__",
    )


# __rpow__
@handle_frontend_method(
    method_tree="tensorflow.Tensor.__rpow__",
    dtype_and_x=_pow_helper_shared_dtype(),
)
def test_tensorflow_instance_rpow(dtype_and_x, as_variable, native_array):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[input_dtype[1]],
        method_as_variable_flags=as_variable,
        method_num_positional_args=1,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend="tensorflow",
        class_name="EagerTensor",
        method_name="__rpow__",
    )


# __pow__
@handle_frontend_method(
    method_tree="tensorflow.EagerTensor.__pow__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=[
            "float16",
            "float32",
            "float64",
            "int32",
            "int64",
        ],
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_pow(dtype_and_x, as_variable, native_array, fw):
    input_dtype, x = dtype_and_x
    if x[1].dtype == "int32" or x[1].dtype == "int64":
        if x[1].ndim == 0:
            if x[1] < 0:
                x[1] *= -1
        else:
            x[1][(x[1] < 0).nonzero()] *= -1
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=[input_dtype[1]],
        as_variable_flags_method=as_variable,
        num_positional_args_method=1,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "y": x[1],
        },
        frontend="tensorflow",
        class_name="EagerTensor",
        method_name="__pow__",
    )


@st.composite
def _array_and_index(
    draw,
    *,
    available_dtypes=helpers.get_dtypes("numeric"),
    min_num_dims=1,
    max_num_dims=3,
    min_dim_size=1,
    max_dim_size=10,
    shape=None,
):
    if isinstance(min_dim_size, st._internal.SearchStrategy):
        min_dim_size = draw(min_dim_size)
    if isinstance(max_dim_size, st._internal.SearchStrategy):
        max_dim_size = draw(max_dim_size)
    if isinstance(available_dtypes, st._internal.SearchStrategy):
        available_dtypes = draw(available_dtypes)

    assert available_dtypes is not None, "Unspecified dtype or available_dtypes."
    dtype = draw(
        helpers.array_dtypes(
            num_arrays=1,
            available_dtypes=available_dtypes,
        )
    )
    dtype.append("int32")

    if shape is not None:
        if not isinstance(shape, (tuple, list)):
            shape = draw(shape)
    else:
        shape = draw(
            st.shared(
                helpers.get_shape(
                    min_num_dims=min_num_dims,
                    max_num_dims=max_num_dims,
                    min_dim_size=min_dim_size,
                    max_dim_size=max_dim_size,
                ),
                key="shape",
            )
        )

    array = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=shape,
        )
    )

    index = tuple([draw(helpers.ints(min_value=0, max_value=_ - 1)) for _ in shape])
    index = index if len(index) != 0 else index[0]
    return dtype, [array, index]


# __getitem__
@handle_frontend_method(
    method_tree="tensorflow.Tensor.__getitem__",
    dtype_and_x=_array_and_index(available_dtypes=helpers.get_dtypes("numeric")),
)
def test_tensorflow_instance_getitem(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    data = x[0]
    index = x[1]
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={"data": data},
        method_input_dtypes=[input_dtype[1]],
        method_as_variable_flags=as_variable,
        method_num_positional_args=1,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={"slice_spec": index},
        frontend="tensorflow",
        class_name="EagerTensor",
        method_name="__getitem__",
    )


@st.composite
def _array_and_shape(
    draw,
    *,
    min_num_dims=1,
    max_num_dims=3,
    min_dim_size=1,
    max_dim_size=10,
):
    if isinstance(min_dim_size, st._internal.SearchStrategy):
        min_dim_size = draw(min_dim_size)
    if isinstance(max_dim_size, st._internal.SearchStrategy):
        max_dim_size = draw(max_dim_size)

    available_dtypes = draw(helpers.get_dtypes("numeric"))
    dtype = draw(
        helpers.array_dtypes(
            num_arrays=1,
            available_dtypes=available_dtypes,
        )
    )
    dtype.append("int32")
    shape = draw(
        st.shared(
            helpers.get_shape(
                min_num_dims=min_num_dims,
                max_num_dims=max_num_dims,
                min_dim_size=min_dim_size,
                max_dim_size=max_dim_size,
            ),
            key="shape",
        )
    )
    array = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=shape,
        )
    )
    to_shape = [(None if draw(st.booleans()) else _) for _ in shape]

    return dtype, [array, to_shape]


@handle_frontend_method(
    method_tree="tensorflow.Tensor.set_shape",
    dtype_and_x=_array_and_shape(
        min_num_dims=0,
        max_num_dims=5,
    ),
)
def test_tensorflow_instance_set_shape(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=[input_dtype[1]],
        method_as_variable_flags=as_variable,
        method_num_positional_args=1,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={"shape": x[1]},
        frontend="tensorflow",
        class_name="EagerTensor",
        method_name="set_shape",
    )


# __len__
@handle_frontend_method(
    method_tree="tensorflow.Tensor.__len__",
    dtype_and_x=_array_and_shape(
        min_num_dims=1,
        max_num_dims=5,
    ),
)
def test_tensorflow_instance_len(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=[],
        as_variable_flags_method=[],
        num_positional_args_method=0,
        native_array_flags_method=[],
        all_as_kwargs_np_method={},
        frontend="tensorflow",
        class_name="EagerTensor",
        method_name="__len__",
    )
