# global
import pytest
from hypothesis import strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
import ivy.functional.backends.tensorflow as ivy_tf
from ivy_tests.test_ivy.helpers import handle_frontend_method, handle_frontend_test
from ivy_tests.test_ivy.test_frontends.test_tensorflow.test_raw_ops import (
    _pow_helper_shared_dtype,
)
import ivy_tests.test_ivy.helpers.test_parameter_flags as pf
from ivy.functional.frontends.tensorflow import EagerTensor


@handle_frontend_test(
    fn_tree="tensorflow.math.add",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
)
def test_array_property_device(
    dtype_x,
):
    _, data = dtype_x
    data = ivy.native_array(data[0])
    x = EagerTensor(data)
    ivy.assertions.check_equal(x.device, ivy.dev(data))


@handle_frontend_test(
    fn_tree="tensorflow.math.add",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_numpy_ndarray_property_dtype(
    dtype_x,
):
    dtype, data = dtype_x
    x = EagerTensor(data[0])
    ivy.assertions.check_equal(x.dtype, ivy.Dtype(dtype[0]))


@handle_frontend_test(
    fn_tree="tensorflow.math.add",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        ret_shape=True,
    ),
)
def test_numpy_ndarray_property_shape(
    dtype_x,
):
    dtype, data, shape = dtype_x
    x = EagerTensor(data[0])
    ivy.assertions.check_equal(x.ivy_array.shape, ivy.Shape(shape))


# __add__
@handle_frontend_method(
    init_name="constant",
    method_tree="tensorflow.EagerTensor.__add__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_add(
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


@handle_frontend_method(
    init_name="constant",
    method_tree="tensorflow.EagerTensor.__div__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_div(
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        method_name=method_name,
        init_name=init_name,
    )


@handle_frontend_method(
    init_name="constant",
    method_tree="tensorflow.EagerTensor.get_shape",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        min_dim_size=1,
    ),
)
def test_tensorflow_instance_get_shape(
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=[],
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=[],
        method_all_as_kwargs_np={},
        frontend=frontend,
        method_name=method_name,
        init_name=init_name,
    )


@handle_frontend_method(
    init_name="constant",
    method_tree="tensorflow.EagerTensor.__eq__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
    ),
)
def test_tensorflow_instance_eq(
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend=frontend,
        method_name=method_name,
        init_name=init_name,
    )


@pytest.mark.skip("Gets stuck.")  # TODO fix
@handle_frontend_method(
    init_name="constant",
    method_tree="tensorflow.EagerTensor.__floordiv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_floordiv(
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        method_name=method_name,
        init_name=init_name,
    )


@handle_frontend_method(
    init_name="constant",
    method_tree="tensorflow.EagerTensor.__ge__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_ge(
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        method_name=method_name,
        init_name=init_name,
    )


@handle_frontend_method(
    init_name="constant",
    method_tree="tensorflow.EagerTensor.__gt__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_gt(
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        method_name=method_name,
        init_name=init_name,
    )


@handle_frontend_method(
    init_name="constant",
    method_tree="tensorflow.EagerTensor.__le__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_le(
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        method_name=method_name,
        init_name=init_name,
    )


@handle_frontend_method(
    init_name="constant",
    method_tree="tensorflow.EagerTensor.__lt__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_lt(
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        method_name=method_name,
        init_name=init_name,
    )


@handle_frontend_method(
    init_name="constant",
    method_tree="tensorflow.EagerTensor.__mul__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_mul(
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        method_name=method_name,
        init_name=init_name,
    )


# __mod__
@handle_frontend_method(
    init_name="constant",
    method_tree="tensorflow.EagerTensor.__mod__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_mod(
    dtype_and_x,
    as_variable,
    native_array,
    frontend,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        method_name=method_name,
        init_name=init_name,
    )


# __sub__
@handle_frontend_method(
    init_name="constant",
    method_tree="tensorflow.EagerTensor.__sub__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(set(ivy_tf.valid_float_dtypes))
        ),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_sub(
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        method_name=method_name,
        init_name=init_name,
    )


# __ne__
@handle_frontend_method(
    init_name="constant",
    method_tree="tensorflow.EagerTensor.__ne__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_ne(
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend=frontend,
        method_name=method_name,
        init_name=init_name,
    )


# __radd__
@handle_frontend_method(
    init_name="constant",
    method_tree="tensorflow.EagerTensor.__radd__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_radd(
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "x": x[1],
        },
        frontend=frontend,
        method_name=method_name,
        init_name=init_name,
    )


# __rfloordiv__
@handle_frontend_method(
    init_name="constant",
    method_tree="tensorflow.EagerTensor.__rfloordiv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_rfloordiv(
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "x": x[1],
        },
        frontend=frontend,
        method_name=method_name,
        init_name=init_name,
    )


# __rsub__
@handle_frontend_method(
    init_name="constant",
    method_tree="tensorflow.EagerTensor.__rsub__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_rsub(
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "x": x[1],
        },
        frontend=frontend,
        method_name=method_name,
        init_name=init_name,
    )


# __and__
@handle_frontend_method(
    init_name="constant",
    method_tree="tensorflow.EagerTensor.__add__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_and(
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        method_name=method_name,
        init_name=init_name,
    )


# __rand__
@handle_frontend_method(
    init_name="constant",
    method_tree="tensorflow.EagerTensor.__rand__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_rand(
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "x": x[1],
        },
        frontend=frontend,
        method_name=method_name,
        init_name=init_name,
    )


# __or__
@handle_frontend_method(
    init_name="constant",
    method_tree="tensorflow.EagerTensor.__or__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_or(
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        method_name=method_name,
        init_name=init_name,
    )


# __ror__
@handle_frontend_method(
    init_name="constant",
    method_tree="tensorflow.EagerTensor.__ror__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_ror(
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "x": x[1],
        },
        frontend=frontend,
        method_name=method_name,
        init_name=init_name,
    )


# __truediv__
@handle_frontend_method(
    init_name="constant",
    method_tree="tensorflow.EagerTensor.__truediv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_truediv(
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        method_name=method_name,
        init_name=init_name,
    )


# __rtruediv__
@handle_frontend_method(
    init_name="constant",
    method_tree="tensorflow.EagerTensor.__rtruediv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_rtruediv(
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "x": x[1],
        },
        frontend=frontend,
        method_name=method_name,
        init_name=init_name,
    )


# __bool__
@handle_frontend_method(
    init_name="constant",
    method_tree="tensorflow.EagerTensor.__bool__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        max_dim_size=1,
    ),
)
def test_tensorflow_instance_bool(
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=[],
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=[],
        method_all_as_kwargs_np={},
        frontend=frontend,
        method_name=method_name,
        init_name=init_name,
    )


# __nonzero__
@handle_frontend_method(
    init_name="constant",
    method_tree="tensorflow.EagerTensor.__nonzero__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        max_dim_size=1,
    ),
)
def test_tensorflow_instance_nonzero(
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=[],
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=[],
        method_all_as_kwargs_np={},
        frontend=frontend,
        method_name=method_name,
        init_name=init_name,
    )


# __neg__
@handle_frontend_method(
    init_name="constant",
    method_tree="tensorflow.EagerTensor.__neg__",
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
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=[],
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=[],
        method_all_as_kwargs_np={},
        frontend=frontend,
        method_name=method_name,
        init_name=init_name,
    )


# __rxor__
@handle_frontend_method(
    init_name="constant",
    method_tree="tensorflow.EagerTensor.__rxor__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_rxor(
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "x": x[1],
        },
        frontend=frontend,
        method_name=method_name,
        init_name=init_name,
    )


# __xor__
@handle_frontend_method(
    init_name="constant",
    method_tree="tensorflow.EagerTensor.__xor__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_xor(
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        method_name=method_name,
        init_name=init_name,
    )


# __matmul__
@handle_frontend_method(
    init_name="constant",
    method_tree="tensorflow.EagerTensor.__matmul__",
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
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        method_name=method_name,
        init_name=init_name,
    )


# __rmatmul__
@handle_frontend_method(
    init_name="constant",
    method_tree="tensorflow.EagerTensor.__rmatmul__",
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
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "x": x[1],
        },
        frontend=frontend,
        method_name=method_name,
        init_name=init_name,
    )


# __array__
@handle_frontend_method(
    init_name="constant",
    method_tree="tensorflow.EagerTensor.__array__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric")
    ),
)
def test_tensorflow_instance_array(
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=[],
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=[],
        method_all_as_kwargs_np={},
        frontend=frontend,
        method_name=method_name,
        init_name=init_name,
    )


# __invert__
@handle_frontend_method(
    init_name="constant",
    method_tree="tensorflow.EagerTensor.__invert__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer")
    ),
)
def test_tensorflow_instance_invert(
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=[],
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=[],
        method_all_as_kwargs_np={},
        frontend=frontend,
        method_name=method_name,
        init_name=init_name,
    )


# __rmul__
@handle_frontend_method(
    init_name="constant",
    method_tree="tensorflow.EagerTensor.__rmul__",
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
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "x": x[1],
        },
        frontend=frontend,
        method_name=method_name,
        init_name=init_name,
    )


# __rpow__
@handle_frontend_method(
    init_name="constant",
    method_tree="tensorflow.EagerTensor.__rpow__",
    dtype_and_x=_pow_helper_shared_dtype(),
)
def test_tensorflow_instance_rpow(
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        method_name=method_name,
        init_name=init_name,
    )


# __pow__
@handle_frontend_method(
    init_name="constant",
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
def test_tensorflow_instance_pow(
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    if x[1].dtype == "int32" or x[1].dtype == "int64":
        if x[1].ndim == 0:
            if x[1] < 0:
                x[1] *= -1
        else:
            x[1][(x[1] < 0).nonzero()] *= -1

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        method_name=method_name,
        init_name=init_name,
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
    init_name="constant",
    method_tree="tensorflow.EagerTensor.__getitem__",
    dtype_and_x=_array_and_index(available_dtypes=helpers.get_dtypes("numeric")),
)
def test_tensorflow_instance_getitem(
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    data = x[0]
    index = x[1]
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={"value": data},
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={"slice_spec": index},
        frontend=frontend,
        method_name=method_name,
        init_name=init_name,
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
    init_name="constant",
    method_tree="tensorflow.EagerTensor.set_shape",
    dtype_and_x=_array_and_shape(
        min_num_dims=0,
        max_num_dims=5,
    ),
)
def test_tensorflow_instance_set_shape(
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={"value": x[0]},
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={"shape": x[1]},
        frontend=frontend,
        method_name=method_name,
        init_name=init_name,
    )


# __len__
@handle_frontend_method(
    init_name="constant",
    method_tree="tensorflow.EagerTensor.__len__",
    dtype_and_x=_array_and_shape(
        min_num_dims=1,
        max_num_dims=5,
    ),
)
def test_tensorflow_instance_len(
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=[],
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=[],
        method_all_as_kwargs_np={},
        frontend=frontend,
        method_name=method_name,
        init_name=init_name,
    )
