# global
import pytest
from hypothesis import strategies as st, given

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers

from ivy_tests.test_ivy.helpers import handle_frontend_method
from ivy_tests.test_ivy.test_frontends.test_tensorflow.test_raw_ops import (
    _pow_helper_shared_dtype,
)
from ivy.functional.frontends.tensorflow import EagerTensor


CLASS_TREE = "ivy.functional.frontends.tensorflow.EagerTensor"


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ),
)
def test_tensorflow_tensor_property_ivy_array(
    dtype_x,
):
    _, data = dtype_x
    x = EagerTensor(data[0])
    ret = helpers.flatten_and_to_np(ret=x.ivy_array.data)
    ret_gt = helpers.flatten_and_to_np(ret=data[0])
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="tensorflow",
    )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ),
)
def test_tensorflow_tensor_property_device(
    dtype_x,
):
    _, data = dtype_x
    data = ivy.native_array(data[0])
    x = EagerTensor(data)
    ivy.utils.assertions.check_equal(x.device, ivy.dev(data))


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False),
    ),
)
def test_tensorflow_tensor_property_dtype(
    dtype_x,
):
    dtype, data = dtype_x
    x = EagerTensor(data[0])
    ivy.utils.assertions.check_equal(x.dtype, ivy.Dtype(dtype[0]))


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False),
        ret_shape=True,
    ),
)
def test_tensorflow_tensor_property_shape(
    dtype_x,
):
    dtype, data, shape = dtype_x
    x = EagerTensor(data[0])
    ivy.utils.assertions.check_equal(x.ivy_array.shape, ivy.Shape(shape))


# __add__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="__add__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_add(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="__div__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_div(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="get_shape",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        min_dim_size=1,
    ),
)
def test_tensorflow_instance_get_shape(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="__eq__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
    ),
)
def test_tensorflow_instance_eq(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
    )


@pytest.mark.skip("Gets stuck.")  # TODO fix
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="__floordiv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_floordiv(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="__ge__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_ge(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="__gt__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_gt(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="__le__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_le(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="__lt__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_lt(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="__mul__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_mul(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
    )


# __mod__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="__mod__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_mod(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
    )


# __sub__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="__sub__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_sub(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
    )


# __ne__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="__ne__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_ne(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
    )


# __radd__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="__radd__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_radd(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "x": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
    )


# __rfloordiv__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="__rfloordiv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_rfloordiv(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "x": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
    )


# __rsub__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="__rsub__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_rsub(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "x": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
    )


# __and__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="__add__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_and(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
    )


# __rand__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="__rand__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_rand(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "x": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
    )


# __or__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="__or__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_or(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
    )


# __ror__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="__ror__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_ror(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "x": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
    )


# __truediv__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="__truediv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_truediv(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
    )


# __rtruediv__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="__rtruediv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_rtruediv(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "x": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
    )


# __bool__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="__bool__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        max_dim_size=1,
    ),
)
def test_tensorflow_instance_bool(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
    )


# __nonzero__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="__nonzero__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        max_dim_size=1,
    ),
)
def test_tensorflow_instance_nonzero(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
    )


# __neg__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="__neg__",
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
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
    )


# __rxor__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="__rxor__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_rxor(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "x": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
    )


# __xor__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="__xor__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_xor(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
    )


# __matmul__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="__matmul__",
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
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
    )


# __rmatmul__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="__rmatmul__",
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
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "x": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
    )


# __array__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="__array__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric")
    ),
    dtype=helpers.get_dtypes("valid", full=False),
)
def test_tensorflow_instance_array(
    dtype_and_x,
    dtype,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dtype": dtype[0],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
    )


# __invert__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="__invert__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer")
    ),
)
def test_tensorflow_instance_invert(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
    )


# __rmul__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="__rmul__",
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
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "x": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
    )


# __rpow__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="__rpow__",
    dtype_and_x=_pow_helper_shared_dtype(),
)
def test_tensorflow_instance_rpow(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
    )


# __pow__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="__pow__",
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
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
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
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
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
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="__getitem__",
    dtype_and_x=_array_and_index(available_dtypes=helpers.get_dtypes("numeric")),
)
def test_tensorflow_instance_getitem(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    data = x[0]
    index = x[1]
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        init_all_as_kwargs_np={"value": data},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"slice_spec": index},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
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
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="set_shape",
    dtype_and_x=_array_and_shape(
        min_num_dims=0,
        max_num_dims=5,
    ),
)
def test_tensorflow_instance_set_shape(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        init_all_as_kwargs_np={"value": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"shape": x[1]},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
    )


# __len__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.constant",
    method_name="__len__",
    dtype_and_x=_array_and_shape(
        min_num_dims=1,
        max_num_dims=5,
    ),
)
def test_tensorflow_instance_len(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
    )
