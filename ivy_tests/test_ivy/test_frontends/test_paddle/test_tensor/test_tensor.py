# flake8: noqa
# global
from hypothesis import strategies as st, given, assume
import numpy as np
import ivy

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy.functional.backends.paddle.general import _check_query
from ivy.functional.frontends.paddle import Tensor
from ivy_tests.test_ivy.helpers import handle_frontend_method
from ivy_tests.test_ivy.test_functional.test_core.test_statistical import (
    _statistical_dtype_values,
)
from ivy_tests.test_ivy.test_functional.test_experimental.test_core.test_manipulation import (  # noqa E501
    _get_dtype_values_k_axes_for_rot90,
)

CLASS_TREE = "ivy.functional.frontends.paddle.Tensor"


# Helpers #
# ------- #


@st.composite
def _reshape_helper(draw):
    # generate a shape s.t len(shape) > 0
    shape = draw(
        helpers.get_shape(
            allow_none=False,
            min_num_dims=1,
            max_num_dims=3,
            min_dim_size=1,
            max_dim_size=3,
        )
    )

    reshape_shape = draw(helpers.reshape_shapes(shape=shape))

    dtypes, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            shape=shape,
        )
    )
    return dtypes, x, reshape_shape


@st.composite
def _get_dtype_and_square_matrix(draw):
    dim_size = draw(helpers.ints(min_value=2, max_value=5))
    dtype = draw(helpers.get_dtypes("float", index=1, full=False))
    mat = draw(
        helpers.array_values(
            dtype=dtype[0], shape=(dim_size, dim_size), min_value=0, max_value=10
        )
    )
    return dtype, mat


@st.composite
def _get_clip_inputs(draw):
    shape = draw(
        helpers.get_shape(
            min_num_dims=1, max_num_dims=5, min_dim_size=1, max_dim_size=10
        )
    )
    x_dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            shape=shape,
            min_value=0,
            max_value=50,
        )
    )
    min = draw(
        helpers.array_values(dtype=x_dtype[0], shape=(1,), min_value=0, max_value=25)
    )
    max = draw(
        helpers.array_values(dtype=x_dtype[0], shape=(1,), min_value=26, max_value=50)
    )
    if draw(st.booleans()):
        min = None
    elif draw(st.booleans()):
        max = None
    return x_dtype, x, min, max


# Tests #
# ----- #


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ).filter(lambda x: "bfloat16" not in x[0]),
)
def test_paddle_device(
    dtype_x,
):
    _, data = dtype_x
    x = Tensor(data[0])
    x.ivy_array = data[0]
    ivy.utils.assertions.check_equal(
        x.place, ivy.dev(ivy.array(data[0])), as_array=False
    )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ).filter(lambda x: "bfloat16" not in x[0]),
)
def test_paddle_dtype(
    dtype_x,
):
    dtype, data = dtype_x
    x = Tensor(data[0])
    x.ivy_array = data[0]
    ivy.utils.assertions.check_equal(x.dtype, dtype[0], as_array=False)


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False),
        ret_shape=True,
    ).filter(lambda x: "bfloat16" not in x[0]),
)
def test_paddle_shape(dtype_x):
    _, data, shape = dtype_x
    x = Tensor(data[0])
    ivy.utils.assertions.check_equal(
        x.ivy_array.shape, ivy.Shape(shape), as_array=False
    )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False),
    ).filter(lambda x: "bfloat16" not in x[0]),
)
def test_paddle_ndim(
    dtype_x,
):
    _, data = dtype_x
    x = Tensor(data[0])
    ivy.utils.assertions.check_equal(x.ndim, data[0].ndim, as_array=False)


# reshape
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="reshape",
    dtype_x_shape=_reshape_helper(),
)
def test_paddle_reshape(
    dtype_x_shape,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtype, x, shape = dtype_x_shape
    assume(len(shape) != 0)
    shape = {
        "shape": shape,
    }
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np=shape,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __getitem__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="__getitem__",
    dtype_x_index=helpers.dtype_array_query(
        available_dtypes=helpers.get_dtypes("valid"),
        allow_neg_step=False,
    ).filter(lambda x: x[0][0] == x[0][-1] and _check_query(x[-2])),
)
def test_paddle_getitem(
    dtype_x_index,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtype, x, index = dtype_x_index
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x},
        method_input_dtypes=[*input_dtype[1:]],
        method_all_as_kwargs_np={"item": index},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# dim
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="dim",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_paddle_dim(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# abs
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="abs",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_abs(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# sin
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="sin",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_sin(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# sinh
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="sinh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_sinh(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# asin
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="asin",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_asin(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# asinh
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="asinh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_asinh(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        backend_to_test=backend_fw,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# cosh
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="cosh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_cosh(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# log
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="log",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_log(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# argmax
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="argmax",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=st.one_of(helpers.get_dtypes("float")),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    keep_dims=st.booleans(),
)
def test_paddle_argmax(
    dtype_x_axis,
    keep_dims,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x, axis = dtype_x_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "axis": axis,
            "keepdim": keep_dims,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# exp
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="exp",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_exp(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# cos
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="cos",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_cos(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# log10
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="log10",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_log10(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# argsort
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="argsort",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=st.one_of(helpers.get_dtypes("float")),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    descending=st.booleans(),
)
def test_paddle_argsort(
    dtype_x_axis,
    descending,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtypes, x, axis = dtype_x_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "axis": axis,
            "descending": descending,
        },
        frontend=frontend,
        backend_to_test=backend_fw,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# floor
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="floor",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_floor(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# sqrt
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="sqrt",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_paddle_sqrt(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# tanh
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="tanh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_paddle_tanh(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# __(add_)__


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="add_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_add_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# square
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="square",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_square(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# cholesky
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="cholesky",
    dtype_and_x=_get_dtype_and_square_matrix(),
    upper=st.booleans(),
)
def test_paddle_cholesky(
    dtype_and_x,
    upper,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    x = np.matmul(x.T, x) + np.identity(x.shape[0])

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"upper": upper},
        frontend=frontend,
        backend_to_test=backend_fw,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="multiply",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_paddle_multiply(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
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
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# all
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="all",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("bool"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    keep_dims=st.booleans(),
)
def test_paddle_all(
    dtype_x_axis,
    keep_dims,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtypes, x, axis = dtype_x_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "axis": axis,
            "keepdim": keep_dims,
        },
        frontend=frontend,
        backend_to_test=backend_fw,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# allclose
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="allclose",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    # rtol=1e-05,
    # atol=1e-08,
    # equal_nan=st.booleans(),
)
def test_paddle_allclose(
    dtype_and_x,
    # rtol,
    # atol,
    # equal_nan,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
            # "rtol": rtol,
            # "atol": atol,
            # "equal_nan": equal_nan,
        },
        frontend=frontend,
        backend_to_test=backend_fw,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# sort
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="sort",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=st.one_of(helpers.get_dtypes("float")),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    descending=st.booleans(),
)
def test_paddle_sort(
    dtype_x_axis,
    descending,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtypes, x, axis = dtype_x_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "axis": axis,
            "descending": descending,
        },
        frontend=frontend,
        backend_to_test=backend_fw,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# any
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="any",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=st.one_of(helpers.get_dtypes("float")),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    keep_dims=st.booleans(),
)
def test_paddle_any(
    dtype_x_axis,
    keep_dims,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x, axis = dtype_x_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "axis": axis,
            "keepdim": keep_dims,
        },
        frontend=frontend,
        backend_to_test=backend_fw,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


#  isinf
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="isinf",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_paddle_isinf(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# astype
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="astype",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    dtype=st.one_of(helpers.get_dtypes("valid")),
)
def test_paddle_astype(
    dtype_and_x,
    dtype,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    if dtype is None:
        dtype = input_dtype
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dtype": dtype,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


#  isfinite
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="isfinite",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_paddle_isfinite(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# erf
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="erf",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_paddle_erf(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# subtract
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="subtract",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), num_arrays=2, shared_dtype=True
    ),
)
def test_paddle_subtract(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# bitwise_xor
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="bitwise_xor",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=2, shared_dtype=True
    ),
)
def test_paddle_bitwise_xor(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# logical_xor
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="logical_xor",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=2, shared_dtype=True
    ),
)
def test_paddle_logical_xor(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# logical_or
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="logical_or",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=2, shared_dtype=True
    ),
)
def test_paddle_logical_or(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# rsqrt
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="rsqrt",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_paddle_rsqrt(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="bitwise_or",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=2, shared_dtype=True
    ),
)
def test_paddle_bitwise_or(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# ceil
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="ceil",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_ceil(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# bitwise_and
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="bitwise_and",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=2, shared_dtype=True
    ),
)
def test_paddle_bitwise_and(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# greater_than
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="greater_than",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        shared_dtype=True,
        safety_factor_scale="log",
        small_abs_safety_factor=32,
    ),
)
def test_paddle_greater_than(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


#  bitwise_not
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="bitwise_not",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_paddle_bitwise_not(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# reciprocal
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="reciprocal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_paddle_reciprocal(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# logical_and
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="logical_and",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=2, shared_dtype=True
    ),
)
def test_paddle_logical_and(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"self": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# divide
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="divide",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        shared_dtype=True,
        safety_factor_scale="log",
        small_abs_safety_factor=32,
    ),
)
def test_paddle_divide(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="cumprod",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        valid_axis=True,
        force_int_axis=True,
        min_num_dims=1,
        min_value=-5,
        max_value=5,
    ),
)
def test_paddle_cumprod(
    dtype_x_axis,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"dim": axis},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="cumsum",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        valid_axis=True,
        force_int_axis=True,
        min_num_dims=1,
        min_value=-5,
        max_value=5,
    ),
)
def test_paddle_cumsum(
    dtype_x_axis,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"axis": axis},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="angle",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=["float64", "complex64", "complex128"],
    ),
)
def test_paddle_angle(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# equal
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="equal",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_paddle_equal(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# rad2deg
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="rad2deg",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_rad2deg(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="fmax",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), num_arrays=2, shared_dtype=True
    ),
)
def test_paddle_fmax(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="fmin",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), num_arrays=2, shared_dtype=True
    ),
)
def test_paddle_fmin(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="minimum",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_paddle_minimum(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# clip
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="clip",
    input_and_ranges=_get_clip_inputs(),
)
def test_paddle_clip(
    input_and_ranges,
    frontend,
    backend_fw,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x, min, max = input_and_ranges
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"min": min, "max": max},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


#  less_than
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="less_than",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=2, shared_dtype=True
    ),
)
def test_paddle_less_than(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# max
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="max",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=st.one_of(helpers.get_dtypes("valid")),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=False,
    ),
    keep_dims=st.booleans(),
)
def test_paddle_max(
    dtype_x_axis,
    keep_dims,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtypes, x, axis = dtype_x_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "axis": axis,
            "keepdim": keep_dims,
        },
        frontend=frontend,
        backend_to_test=backend_fw,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# deg2rad
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="deg2rad",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_deg2rad(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# rot90
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="rot90",
    dtype_m_k_axes=_get_dtype_values_k_axes_for_rot90(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=3,
        max_num_dims=6,
        min_dim_size=1,
        max_dim_size=10,
    ),
)
def test_paddle_rot90(
    dtype_m_k_axes,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, values, k, axes = dtype_m_k_axes

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": values,
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "k": k,
            "axes": axes,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# imag
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="imag",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_paddle_imag(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# floor_divide
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="floor_divide",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        min_value=2,
        shared_dtype=True,
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
        safety_factor_scale="linear",
    ),
)
def test_paddle_floor_divide(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtypes_and_x
    # Absolute tolerance is 1,
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
        atol_=1,
    )


# is_tensor
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="is_tensor",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=1,
    ),
)
def test_paddle_is_tensor(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# isclose
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="isclose",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=2, shared_dtype=True
    ),
)
def test_paddle_isclose(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


#  equal_all
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="equal_all",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        min_value=-np.inf,
        max_value=np.inf,
        shared_dtype=True,
        safety_factor_scale="log",
        small_abs_safety_factor=32,
    ),
)
def test_paddle_equal_all(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# conj
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="conj",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_paddle_conj(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# floor_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="floor_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_floor_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# log2
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="log2",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_log2(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# neg
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="neg",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_paddle_neg(
    dtype_and_x,
    frontend,
    backend_fw,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# isnan
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="isnan",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_isnan(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


#  logical_not
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="logical_not",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_paddle_logical_not(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="sign",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_sign(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# acosh
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="acosh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_acosh(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# cond
@st.composite
def _get_dtype_and_matrix_non_singular(draw, dtypes):
    while True:
        matrix = draw(
            helpers.dtype_and_values(
                available_dtypes=dtypes,
                min_value=-10,
                max_value=10,
                min_num_dims=2,
                max_num_dims=2,
                min_dim_size=1,
                max_dim_size=5,
                shape=st.tuples(st.integers(1, 5), st.integers(1, 5)).filter(
                    lambda x: x[0] == x[1]
                ),
                allow_inf=False,
                allow_nan=False,
            )
        )
        if np.linalg.det(matrix[1][0]) != 0:
            break

    return matrix[0], matrix[1]


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="cond",
    dtype_and_x=_get_dtype_and_matrix_non_singular(dtypes=["float32", "float64"]),
    p=st.sampled_from([None, "fro", "nuc", np.inf, -np.inf, 1, -1, 2, -2]),
)
def test_paddle_cond(
    dtype_and_x,
    p,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"p": p},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="sgn",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_complex"),
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=1,
        max_dim_size=1,
        abs_smallest_val=1e-10,
        min_value=-10,
        max_value=10,
    ),
)
def test_paddle_sgn(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="tolist",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_paddle_tolist(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# min
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="min",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=st.one_of(helpers.get_dtypes("numeric")),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    keep_dims=st.booleans(),
)
def test_paddle_min(
    dtype_x_axis,
    keep_dims,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtypes, x, axis = dtype_x_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "axis": axis,
            "keepdim": keep_dims,
        },
        frontend=frontend,
        backend_to_test=backend_fw,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# maximum
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="maximum",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_paddle_instance_maximum(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"other": x[1]},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# std
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="std",
    dtypes_and_x=_statistical_dtype_values(function="std"),
    keep_dims=st.booleans(),
)
def test_paddle_instance_std(
    dtypes_and_x,
    keep_dims,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtypes, x, axis, _ = dtypes_and_x
    helpers.test_frontend_method(
        backend_to_test=backend_fw,
        init_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "axis": axis,
            "keepdim": keep_dims,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# atanh
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="atanh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_paddle_atanh(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# log1p
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="log1p",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        safety_factor_scale="log",
        small_abs_safety_factor=32,
    ),
)
def test_paddle_log1p(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        backend_to_test=backend_fw,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


#  greater_equal
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="greater_equal",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=2, shared_dtype=True
    ),
)
def test_paddle_greater_equal(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )
